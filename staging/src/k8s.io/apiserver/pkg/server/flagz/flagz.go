/*
Copyright 2024 The Kubernetes Authors.

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

package flagz

import (
	"bytes"
	"fmt"
	"net/http"

	"github.com/spf13/pflag"
	"k8s.io/apiserver/pkg/endpoints/metrics"
	cliflag "k8s.io/component-base/cli/flag"
)

// InstallHandler registers handlers for displaying flags on the path
// "/flagz" to mux. *All handlers* for mux must be specified in
// exactly one call to InstallHandler. Calling InstallHandler more
// than once for the same mux will result in a panic.
func InstallHandler(mux mux, flags ...cliflag.NamedFlagSets) {
	mux.Handle("/flagz",
		metrics.InstrumentHandlerFunc("GET",
			/* group = */ "",
			/* version = */ "",
			/* resource = */ "",
			/* subresource = */ "/flagz",
			/* scope = */ "",
			/* component = */ "",
			/* deprecated */ false,
			/* removedRelease */ "",
			handleFlags(flags)))
}

// mux is an interface describing the methods InstallHandler requires.
type mux interface {
	Handle(pattern string, handler http.Handler)
}

// handleFlags returns an http.HandlerFunc that serves the provided flags.
func handleFlags(flags []cliflag.NamedFlagSets) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var individualCheckOutput bytes.Buffer
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		for _, flagset := range flags {
			for _, fs := range flagset.FlagSets {
				fs.VisitAll(func(flag *pflag.Flag) {
					if flag.Value != nil && flag.Value.String() != "" && flag.Value.String() != "[]" {
						fmt.Fprint(w, flag.Name, "=", flag.Value, "\n")
					}
				})
			}
		}
		individualCheckOutput.WriteTo(w)
	}
}
