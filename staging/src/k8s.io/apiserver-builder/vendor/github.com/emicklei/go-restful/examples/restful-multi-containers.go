package main

import (
	"github.com/emicklei/go-restful"
	"io"
	"log"
	"net/http"
)

// This example shows how to have a program with 2 WebServices containers
// each having a http server listening on its own port.
//
// The first "hello" is added to the restful.DefaultContainer (and uses DefaultServeMux)
// For the second "hello", a new container and ServeMux is created
// and requires a new http.Server with the container being the Handler.
// This first server is spawn in its own go-routine such that the program proceeds to create the second.
//
// GET http://localhost:8080/hello
// GET http://localhost:8081/hello

func main() {
	ws := new(restful.WebService)
	ws.Route(ws.GET("/hello").To(hello))
	restful.Add(ws)
	go func() {
		http.ListenAndServe(":8080", nil)
	}()

	container2 := restful.NewContainer()
	ws2 := new(restful.WebService)
	ws2.Route(ws2.GET("/hello").To(hello2))
	container2.Add(ws2)
	server := &http.Server{Addr: ":8081", Handler: container2}
	log.Fatal(server.ListenAndServe())
}

func hello(req *restful.Request, resp *restful.Response) {
	io.WriteString(resp, "default world")
}

func hello2(req *restful.Request, resp *restful.Response) {
	io.WriteString(resp, "second world")
}
