/*
Copyright 2014 The Kubernetes Authors.

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

// Package testwebserver offers a tiny web server that serves a static file.
package testwebserver

import (
	"fmt"
	"log"
	"net/http"

	"github.com/spf13/cobra"
)

// CmdTestWebserver is used by agnhost Cobra.
var CmdTestWebserver = &cobra.Command{
	Use:   "test-webserver",
	Short: "Starts a simple HTTP fileserver",
	Long:  "Starts a simple HTTP fileserver on the given --port, which serves any file specified in the URL path, if it exists.",
	Args:  cobra.MaximumNArgs(0),
	Run:   main,
}

var (
	port int
)

func init() {
	CmdTestWebserver.Flags().IntVar(&port, "port", 80, "Port number.")
}

func main(cmd *cobra.Command, args []string) {
	fs := http.StripPrefix("/", http.FileServer(http.Dir("/")))

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "private")
		// Needed for local proxy to Kubernetes API server to work.
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Credentials", "true")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "DNT,X-Mx-ReqToken,Keep-Alive,User-Agent,X-Requested-With,Cache-Control,Content-Type")
		// Disable If-Modified-Since so update-demo isn't broken by 304s
		r.Header.Del("If-Modified-Since")
		fs.ServeHTTP(w, r)
	})

	go log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", port), nil))

	select {}
}
