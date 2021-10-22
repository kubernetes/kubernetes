// Copyright 2019 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package server

import (
	"net/http"
	"path/filepath"
)

var mimeTypes = map[string]string{
	".js":  "application/javascript",
	".css": "text/css",
	".png": "image/png",
	".jpg": "image/jpeg",
	".gif": "image/gif",
}

func StaticFileServer(root http.FileSystem) http.Handler {
	return http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			fileExt := filepath.Ext(r.URL.Path)

			if t, ok := mimeTypes[fileExt]; ok {
				w.Header().Set("Content-Type", t)
			}

			http.FileServer(root).ServeHTTP(w, r)
		},
	)
}
