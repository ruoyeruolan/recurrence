import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class GATLayer(nn.Module):
    """Graph Attention Layer"""
    def __init__(self, in_features, out_features, alpha=0.35, dropout=0.1):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.dropout = dropout
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, x, adj):
        # x: (N, M, in_features), adj: (M, M)
        h = torch.matmul(x, self.W)  # (N, M, out_features)
        N, M, _ = h.size()
        
        # Self-attention
        a_input = self._prepare_attentional_mechanism_input(h)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        
        # Mask attention with adjacency
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, h)
        return F.elu(h_prime)
    
    def _prepare_attentional_mechanism_input(self, h):
        N, M, out_features = h.size()
        h_repeat = h.repeat(1, 1, M).view(N, M * M, out_features)
        h_repeat_interleave = h.repeat(1, M, 1)
        all_combinations = torch.cat([h_repeat_interleave, h_repeat], dim=2)
        return all_combinations.view(N, M, M, 2 * out_features)


class CausalEncoder(nn.Module):
    """Causal Encoder using VAE structure"""
    def __init__(self, input_dim, hidden_dim):
        super(CausalEncoder, self).__init__()
        self.fc_mu = nn.Linear(input_dim, hidden_dim)
        self.fc_logvar = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, h):
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class DiffusionLayer(nn.Module):
    """Diffusion Layer to fuse causal and feature embeddings"""
    def __init__(self, causal_dim, feature_dim, output_dim):
        super(DiffusionLayer, self).__init__()
        self.fc = nn.Linear(causal_dim + feature_dim, output_dim)
        
    def forward(self, z, h):
        fused = torch.cat([z, h], dim=-1)
        cs = F.relu(self.fc(fused))
        return cs


class CausalDecoder(nn.Module):
    """Causal Decoder to reconstruct gene expression"""
    def __init__(self, cs_dim, output_dim):
        super(CausalDecoder, self).__init__()
        self.fc = nn.Linear(cs_dim, output_dim)
        
    def forward(self, cs):
        x_recon = torch.sigmoid(self.fc(cs))
        return x_recon


class GCNLayer(nn.Module):
    """Graph Convolutional Layer"""
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, x, adj):
        # Add self-loops
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        
        # Compute degree matrix
        rowsum = adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
        # Normalize adjacency
        support = torch.mm(x, self.weight)
        adj_normalized = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        output = torch.mm(adj_normalized, support)
        
        return F.relu(output)


class InferenceModel(nn.Module):
    """Inference Model for GRN prediction"""
    def __init__(self, input_dim, hidden_dim, cs_dim, output_dim=1):
        super(InferenceModel, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        
        # MLP for final prediction
        self.fc1 = nn.Linear(2 * (hidden_dim + cs_dim), 128)
        self.fc2 = nn.Linear(128, output_dim)
        
    def forward(self, x, adj, cs):
        # GCN encoding
        h = self.gcn1(x, adj)
        h = self.gcn2(h, adj)
        
        # Integrate with causal signals
        ie = torch.cat([h, cs], dim=-1)
        
        return h, ie


class GRACE(nn.Module):
    """Complete GRACE Model"""
    def __init__(self, n_genes, n_cells, hidden_dim=64, causal_dim=16, 
                 cs_dim=32, alpha=0.35, dropout=0.1):
        super(GRACE, self).__init__()
        
        self.n_genes = n_genes
        self.n_cells = n_cells
        
        # Feature Encoder (GAT)
        self.gat = GATLayer(n_cells, hidden_dim, alpha, dropout)
        
        # Causal Encoder
        self.causal_encoder = CausalEncoder(hidden_dim, causal_dim)
        
        # Diffusion Layer
        self.diffusion = DiffusionLayer(causal_dim, hidden_dim, cs_dim)
        
        # Causal Decoder
        self.causal_decoder = CausalDecoder(cs_dim, n_cells)
        
        # Inference Model
        self.inference_model = InferenceModel(n_cells, hidden_dim, cs_dim)
        
    def forward(self, x, adj):
        """
        Args:
            x: Gene expression matrix (N_cells, N_genes)
            adj: Adjacency matrix (N_genes, N_genes)
        Returns:
            x_recon: Reconstructed gene expression
            rs: Regulatory scores for gene pairs
            mu, logvar: Parameters for KL divergence
        """
        # Transpose for GAT: (1, N_genes, N_cells)
        x_input = x.t().unsqueeze(0)
        
        # Feature Encoder
        h = self.gat(x_input, adj).squeeze(0)  # (N_genes, hidden_dim)
        
        # Causal Encoder
        mu, logvar = self.causal_encoder(h)
        z = self.causal_encoder.reparameterize(mu, logvar)
        
        # Diffusion Layer
        cs = self.diffusion(z, h)
        
        # Causal Decoder
        x_recon = self.causal_decoder(cs)  # (N_genes, N_cells)
        
        # Inference Model
        h_prime, ie = self.inference_model(x_recon.t(), adj, cs)
        
        return x_recon, h_prime, ie, cs, mu, logvar
    
    def predict_links(self, ie_i, ie_j):
        """Predict regulatory relationship between gene i and gene j"""
        pair_emb = torch.cat([ie_i, ie_j], dim=-1)
        score = self.inference_model.fc1(pair_emb)
        score = F.relu(score)
        score = self.inference_model.fc2(score)
        score = torch.sigmoid(score)
        return score
    
    def compute_loss(self, x, x_recon, mu, logvar, rs, y, 
                     beta=1.0, gamma=0.1, delta=0.01):
        """
        Compute GRACE loss
        Args:
            x: Original gene expression
            x_recon: Reconstructed gene expression
            mu, logvar: VAE parameters
            rs: Regulatory scores
            y: Ground truth labels
            beta, gamma, delta: Hyperparameters
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon.t(), x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / mu.size(0)
        
        # Variance regularization
        var_loss = torch.sum(torch.exp(logvar))
        
        # Inference loss (BCE for link prediction)
        infer_loss = F.binary_cross_entropy(rs, y, reduction='mean')
        
        total_loss = beta * recon_loss + gamma * kl_loss + delta * var_loss + infer_loss
        
        return total_loss, recon_loss, kl_loss, var_loss, infer_loss


# Example usage
if __name__ == "__main__":
    # Hyperparameters
    n_genes = 500
    n_cells = 1000
    n_samples = 100  # Number of gene pairs for training
    
    # Create model
    model = GRACE(n_genes=n_genes, n_cells=n_cells, hidden_dim=64, 
                  causal_dim=16, cs_dim=32)
    
    # Sample data
    x = torch.randn(n_cells, n_genes)  # Gene expression
    adj = torch.randint(0, 2, (n_genes, n_genes)).float()  # Prior regulatory graph
    adj = (adj + adj.t()) / 2  # Make symmetric
    
    # Forward pass
    x_recon, h_prime, ie, cs, mu, logvar = model(x, adj)
    
    # Sample gene pairs for link prediction
    pairs_i = torch.randint(0, n_genes, (n_samples,))
    pairs_j = torch.randint(0, n_genes, (n_samples,))
    y = torch.randint(0, 2, (n_samples, 1)).float()  # Ground truth
    
    # Predict links
    rs = torch.zeros(n_samples, 1)
    for idx in range(n_samples):
        i, j = pairs_i[idx], pairs_j[idx]
        rs[idx] = model.predict_links(ie[i], ie[j])
    
    # Compute loss
    loss, recon_loss, kl_loss, var_loss, infer_loss = model.compute_loss(
        x, x_recon, mu, logvar, rs, y
    )
    
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Reconstruction Loss: {recon_loss.item():.4f}")
    print(f"KL Loss: {kl_loss.item():.4f}")
    print(f"Variance Loss: {var_loss.item():.4f}")
    print(f"Inference Loss: {infer_loss.item():.4f}")
    print(f"\nModel architecture created successfully!")
    print(f"Gene expression reconstructed: {x_recon.shape}")
    print(f"Causal signals generated: {cs.shape}")
