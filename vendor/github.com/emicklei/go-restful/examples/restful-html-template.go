package main

import (
	"log"
	"net/http"
	"text/template"

	"github.com/emicklei/go-restful"
)

// This example shows how to serve a HTML page using the standard Go template engine.
//
// GET http://localhost:8080/

func main() {
	ws := new(restful.WebService)
	ws.Route(ws.GET("/").To(home))
	restful.Add(ws)
	print("open browser on http://localhost:8080/\n")
	http.ListenAndServe(":8080", nil)
}

type Message struct {
	Text string
}

func home(req *restful.Request, resp *restful.Response) {
	p := &Message{"restful-html-template demo"}
	// you might want to cache compiled templates
	t, err := template.ParseFiles("home.html")
	if err != nil {
		log.Fatalf("Template gave: %s", err)
	}
	t.Execute(resp.ResponseWriter, p)
}
