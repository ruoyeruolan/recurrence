// example: frontend/src/app.js (伪代码)
async function loadMeta(cluster, name) {
    const res = await fetch(`/api/pages/${cluster}/${name}`);
    if (!res.ok) return renderNotFound();
    const meta = await res.json();
    if (meta.name === 'encoder') renderEncoderComponent();
    else if (meta.name === 'movie') renderVisualizerComponent();
    else renderIndexComponent();
}