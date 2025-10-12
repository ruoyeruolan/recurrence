// @Introduce  :
// @File       : main.go
// @Author     : ryrl
// @Email      : ryrl970311@gmail.com
// @Time       : 2025/10/12 19:31
// @Description:

package main

import (
	// "embed"
	"github.com/gin-gonic/gin"
	"log"
	"net/http"
	"os"
	"path/filepath"
	// "html/template"
	"recurrence/backend/server/routers"
)

// var templateFS embed.FS //go:embed templates/*.html

// func demo() {

// 	// router := gin.Default()

// 	// tmpl := template.Must(template.ParseFS(templateFS, "templates/*.html"))
// 	// router.SetHTMLTemplate(tmpl)
// 	// router.LoadHTMLGlob("templates/*")
// 	// router.Static("static", "static")

// 	// routers.RegisterDynamicPages(router)

// 	// RegisterPage(router, "/index", "index.html")
// 	// RegisterPage(router, "/encoder", "causalencoderscm.html")

// 	// router.GET("/", func(ctx *gin.Context) {
// 	// 	ctx.Redirect(http.StatusFound, "/index")
// 	// })

// 	// router.GET("/encoder", func(ctx *gin.Context) {
// 	// 	ctx.HTML(http.StatusOK, "causalencoderscm.html", gin.H{
// 	// 		"title": "Causal Encoder SCM",
// 	// 	})
// 	// })
// }

func router() {
	router := gin.Default()

	router.LoadHTMLGlob("templates/*")
	router.Static("static", "static")

	routers.RegisterDynamicPages(router)
	router.Run()
}

func setupRouter() *gin.Engine {
	r := gin.Default()

	// 可选 CORS（开发时启用）
	if os.Getenv("ENABLE_CORS") == "1" {
		r.Use(func(c *gin.Context) {
			c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
			c.Writer.Header().Set("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
			c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type,Authorization")
			if c.Request.Method == "OPTIONS" {
				c.AbortWithStatus(http.StatusOK)
				return
			}
			c.Next()
		})
	}

	// 注册后端 API
	api := r.Group("/api")
	routers.RegisterAPIRoutes(api)

	// always expose backend static assets and templates as helpers
	r.Static("/static", "./static")  // backend/static/*
	r.Static("/html", "./templates") // allow fetch /html/<file> -> backend/templates/<file>

	distPath := "./frontend/dist"
	publicPath := "./frontend/public"

	if _, err := os.Stat(distPath); err == nil {

		r.StaticFS("/", http.Dir(distPath))
		r.NoRoute(func(c *gin.Context) {
			c.File(filepath.Join(distPath, "index.html"))
		})
		log.Println("Serving frontend from", distPath)
	} else if _, err := os.Stat(publicPath); err == nil {

		r.StaticFS("/", http.Dir(publicPath))
		r.NoRoute(func(c *gin.Context) {

			if _, err := os.Stat(filepath.Join(publicPath, "index.html")); err == nil {
				c.File(filepath.Join(publicPath, "index.html"))
				return
			}
			c.String(http.StatusOK, "Backend running. Frontend dev server: http://localhost:5173")
		})
		log.Println("Serving frontend from", publicPath)
	} else {
		if _, err := os.Stat("./templates"); err == nil {
			r.LoadHTMLGlob("templates/*")
			routers.RegisterDynamicPages(r)
			log.Println("Registered dynamic pages from templates")
		} else {
			r.GET("/", func(c *gin.Context) {
				c.String(http.StatusOK, "Backend running. No frontend files found.")
			})
		}
	}

	return r
}

func main() {
	// router()

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	r := setupRouter()

	log.Printf("Starting server on :%s\n", port)
	if err := r.Run(":" + port); err != nil {
		log.Fatalf("server exit: %v", err)
	}
}
