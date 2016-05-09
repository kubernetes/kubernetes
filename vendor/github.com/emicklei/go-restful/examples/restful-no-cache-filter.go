package main

import (
	"io"
	"net/http"

	"github.com/emicklei/go-restful"
)

func NoBrowserCacheFilter(req *restful.Request, resp *restful.Response, chain *restful.FilterChain) {
	resp.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate") // HTTP 1.1.
	resp.Header().Set("Pragma", "no-cache")                                   // HTTP 1.0.
	resp.Header().Set("Expires", "0")                                         // Proxies.
	chain.ProcessFilter(req, resp)
}

// This example shows how to use a WebService filter that passed the Http headers to disable browser cacheing.
//
// GET http://localhost:8080/hello

func main() {
	ws := new(restful.WebService)
	ws.Filter(NoBrowserCacheFilter)
	ws.Route(ws.GET("/hello").To(hello))
	restful.Add(ws)
	http.ListenAndServe(":8080", nil)
}

func hello(req *restful.Request, resp *restful.Response) {
	io.WriteString(resp, "world")
}
