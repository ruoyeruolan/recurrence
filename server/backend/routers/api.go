// @Introduce  :
// @File       : api.go
// @Author     : ryrl
// @Email      : ryrl970311@gmail.com
// @Time       : 2025/10/12 22:59
// @Description:

package routers

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

var clusters = map[string]map[string]string{
	"default": {"index": "index.html"},
	"grace":   {"encoder": "causalencoderscm.html"},
	"pyg":     {"movie": "visualizerhetero.html"},
}

func RegisterAPIRoutes(rg *gin.RouterGroup) {
	rg.GET("/pages/:cluster/:name", func(c *gin.Context) {
		cluster := c.Param("cluster")
		name := c.Param("name")
		clusterMap, ok := clusters[cluster]
		if !ok {
			c.JSON(http.StatusNotFound, gin.H{"error": "cluster not found"})
			return
		}
		fname, ok := clusterMap[name]
		if !ok {
			c.JSON(http.StatusNotFound, gin.H{"error": "page not found"})
			return
		}
		c.JSON(http.StatusOK, gin.H{"cluster": cluster, "name": name, "file": fname})
	})
}
