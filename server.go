package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
)

var port = flag.Int("port", 80, "port to serve requests on. Default: 80")
var serveDir = flag.String("serve_dir", "./", "what directory to serve, Default: current directory")

func main() {
	flag.Parse()

	http.Handle("/", http.FileServer(http.Dir(*serveDir)))
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", *port), nil))
}
