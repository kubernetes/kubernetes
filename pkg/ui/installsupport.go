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

package ui

import (
	"mime"
	"net/http"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/ui/data/swagger"

	assetfs "github.com/elazarl/go-bindata-assetfs"
)

const dashboardPath = "/api/v1/proxy/namespaces/kube-system/services/kube-ui/#/dashboard/"

type MuxInterface interface {
	Handle(pattern string, handler http.Handler)
	HandleFunc(pattern string, handler func(http.ResponseWriter, *http.Request))
}

func InstallSupport(mux MuxInterface, enableSwaggerSupport bool) {

	// Send correct mime type for .svg files.  TODO: remove when
	// https://github.com/golang/go/commit/21e47d831bafb59f22b1ea8098f709677ec8ce33
	// makes it into all of our supported go versions.
	mime.AddExtensionType(".svg", "image/svg+xml")

	// Redirect /ui to the kube-ui proxy path
	prefix := "/ui/"
	mux.HandleFunc(prefix, func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, dashboardPath, http.StatusTemporaryRedirect)
	})

	if enableSwaggerSupport {
		// Expose files in third_party/swagger-ui/ on <host>/swagger-ui
		fileServer := http.FileServer(&assetfs.AssetFS{
			Asset:    swagger.Asset,
			AssetDir: swagger.AssetDir,
			Prefix:   "third_party/swagger-ui",
		})
		prefix = "/swagger-ui/"
		mux.Handle(prefix, http.StripPrefix(prefix, fileServer))
	}
}
