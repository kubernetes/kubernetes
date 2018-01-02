package main

import (
	"log"
	"net/http"

	"github.com/emicklei/go-restful"
	"github.com/emicklei/go-restful-swagger12"
)

type Book struct {
	Title  string
	Author string
}

func main() {
	ws := new(restful.WebService)
	ws.Path("/books")
	ws.Consumes(restful.MIME_JSON, restful.MIME_XML)
	ws.Produces(restful.MIME_JSON, restful.MIME_XML)
	restful.Add(ws)

	ws.Route(ws.GET("/{medium}").To(noop).
		Doc("Search all books").
		Param(ws.PathParameter("medium", "digital or paperback").DataType("string")).
		Param(ws.QueryParameter("language", "en,nl,de").DataType("string")).
		Param(ws.HeaderParameter("If-Modified-Since", "last known timestamp").DataType("datetime")).
		Do(returns200, returns500))

	ws.Route(ws.PUT("/{medium}").To(noop).
		Doc("Add a new book").
		Param(ws.PathParameter("medium", "digital or paperback").DataType("string")).
		Reads(Book{}))

	// You can install the Swagger Service which provides a nice Web UI on your REST API
	// You need to download the Swagger HTML5 assets and change the FilePath location in the config below.
	// Open http://localhost:8080/apidocs and enter http://localhost:8080/apidocs.json in the api input field.
	config := swagger.Config{
		WebServices:    restful.DefaultContainer.RegisteredWebServices(), // you control what services are visible
		WebServicesUrl: "http://localhost:8080",
		ApiPath:        "/apidocs.json",

		// Optionally, specifiy where the UI is located
		SwaggerPath:     "/apidocs/",
		SwaggerFilePath: "/Users/emicklei/xProjects/swagger-ui/dist"}
	swagger.RegisterSwaggerService(config, restful.DefaultContainer)

	log.Printf("start listening on localhost:8080")
	server := &http.Server{Addr: ":8080", Handler: restful.DefaultContainer}
	log.Fatal(server.ListenAndServe())
}

func noop(req *restful.Request, resp *restful.Response) {}

func returns200(b *restful.RouteBuilder) {
	b.Returns(http.StatusOK, "OK", Book{})
}

func returns500(b *restful.RouteBuilder) {
	b.Returns(http.StatusInternalServerError, "Bummer, something went wrong", nil)
}
