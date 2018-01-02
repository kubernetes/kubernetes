package main

import (
	"io"
	"net/http"

	"github.com/emicklei/go-restful"
)

// This example shows how to use a WebService filter that passed the Http headers to disable browser cacheing.
//
// GET http://localhost:8080/hello

func main() {
	ws := new(restful.WebService)
	ws.Filter(restful.NoBrowserCacheFilter)
	ws.Route(ws.GET("/hello").To(hello))
	restful.Add(ws)
	http.ListenAndServe(":8080", nil)
}

func hello(req *restful.Request, resp *restful.Response) {
	io.WriteString(resp, "world")
}
