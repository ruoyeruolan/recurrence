package routers

import (
	// "embed"
	"github.com/gin-gonic/gin"
	// "html/template"
	"net/http"
)

var pages = map[string]map[string]string{

	"default": {
		"index": "index.html",
	},

	"grace": {
		"encoder": "causalencoderscm.html",
	},

	"pyg": {
		"movie": "visualizerhetero.html",
	},
}

func RegisterPage(router *gin.Engine, page string, fname string) {
	router.GET(page, func(ctx *gin.Context) {
		ctx.HTML(http.StatusOK, fname, gin.H{
			"title": "Causal Encoder SCM",
		})
	})
}

func RegisterDynamicPages(router *gin.Engine) {
	router.GET("/", func(ctx *gin.Context) {
		ctx.Redirect(http.StatusFound, "/pages/default/index")
	})

	router.GET("/pages/:cluster/:name", func(ctx *gin.Context) {
		cluster := ctx.Param("cluster")
		name := ctx.Param("name")
		fname, ok := pages[cluster][name]

		if !ok {
			ctx.AbortWithStatus(http.StatusNotFound)
			return
		}
		ctx.HTML(http.StatusOK, fname, gin.H{
			"title": name,
		})
	})
}
