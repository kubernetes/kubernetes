/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

// A tiny binary for testing ports.
//
// Reads env vars; for every var of the form SERVE_PORT_X, where X is a valid
// port number, porter starts an HTTP server which serves the env var's value
// in response to any query.
package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
)

const prefix = "SERVE_PORT_"

func main() {
	for _, vk := range os.Environ() {
		// Put everything before the first = sign in parts[0], and
		// everything else in parts[1] (even if there are multiple =
		// characters).
		parts := strings.SplitN(vk, "=", 2)
		key := parts[0]
		value := parts[1]
		if strings.HasPrefix(key, prefix) {
			port := strings.TrimPrefix(key, prefix)
			go servePort(port, value)
		}
	}

	select {}
}

func servePort(port, value string) {
	s := &http.Server{
		Addr: "0.0.0.0:" + port,
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprint(w, value)
		}),
	}
	log.Printf("server on port %q failed: %v", port, s.ListenAndServe())
}
