// Copyright 2017, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

// Package zpages implements a collection of HTML pages that display RPC stats
// and trace data, and also functions to write that same data in plain text to
// an io.Writer.
//
// Users can also embed the HTML for stats and traces in custom status pages.
//
// zpages are currrently work-in-process and cannot display minutely and
// hourly stats correctly.
//
// Performance
//
// Installing the zpages has a performance overhead because additional traces
// and stats will be collected in-process. In most cases, we expect this
// overhead will not be significant but it depends on many factors, including
// how many spans your process creates and how richly annotated they are.
package zpages // import "go.opencensus.io/zpages"

import (
	"net/http"
	"path"
	"sync"

	"go.opencensus.io/internal"
)

// TODO(ramonza): Remove Handler to make initialization lazy.

// Handler is deprecated: Use Handle.
var Handler http.Handler

func init() {
	mux := http.NewServeMux()
	Handle(mux, "/")
	Handler = mux
}

// Handle adds the z-pages to the given ServeMux rooted at pathPrefix.
func Handle(mux *http.ServeMux, pathPrefix string) {
	enable()
	if mux == nil {
		mux = http.DefaultServeMux
	}
	mux.HandleFunc(path.Join(pathPrefix, "rpcz"), rpczHandler)
	mux.HandleFunc(path.Join(pathPrefix, "tracez"), tracezHandler)
	mux.Handle(path.Join(pathPrefix, "public/"), http.FileServer(fs))
}

var enableOnce sync.Once

func enable() {
	enableOnce.Do(func() {
		internal.LocalSpanStoreEnabled = true
		registerRPCViews()
	})
}
