/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// A simple static web server for hosting the Kubernetes cluster UI.
package main

import (
	"flag"
	"fmt"
	"mime"
	"net/http"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/ui/data/dashboard"
	"github.com/golang/glog"

	assetfs "github.com/elazarl/go-bindata-assetfs"
)

var (
	port = flag.Int("port", 8080, "Port number to serve at.")
)

func main() {
	flag.Parse()

	// Send correct mime type for .svg files.  TODO: remove when
	// https://github.com/golang/go/commit/21e47d831bafb59f22b1ea8098f709677ec8ce33
	// makes it into all of our supported go versions.
	mime.AddExtensionType(".svg", "image/svg+xml")

	// Expose files in www/ on <host>
	fileServer := http.FileServer(&assetfs.AssetFS{
		Asset:    dashboard.Asset,
		AssetDir: dashboard.AssetDir,
		Prefix:   "www/app",
	})
	http.Handle("/", fileServer)

	// TODO: Add support for serving over TLS.
	glog.Fatal(http.ListenAndServe(fmt.Sprintf("0.0.0.0:%d", *port), nil))
}
