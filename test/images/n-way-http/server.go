/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// A webserver that runs n http handlers. Example invocation:
// - server -port 8080 -prefix foo -num 10 -start 0
// Will given you 10 /foo(i) endpoints that simply echo foo(i) when requested.
// - server -start 3 -num 1
// Will create just one endpoint, at /foo3
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
)

var (
	port   = flag.Int("port", 8080, "Port number for requests.")
	prefix = flag.String("prefix", "foo", "String used as path prefix")
	num    = flag.Int("num", 10, "Number of endpoints to create.")
	start  = flag.Int("start", 0, "Index to start, only makes sense with --num")
)

func main() {
	flag.Parse()
	// This container is used to test the GCE L7 controller which expects "/"
	// to return a 200 response.
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "ok")
	})
	for i := *start; i < *start+*num; i++ {
		path := fmt.Sprintf("%v%d", *prefix, i)
		http.HandleFunc(fmt.Sprintf("/%v", path), func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, path)
		})
	}
	log.Printf("server -port %d -prefix %v -num %d -start %d", *port, *prefix, *num, *start)
	http.ListenAndServe(fmt.Sprintf(":%d", *port), nil)
}
