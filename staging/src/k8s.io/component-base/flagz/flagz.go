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
	"sync"

	"github.com/spf13/pflag"
	"k8s.io/apiserver/pkg/endpoints/metrics"

	cliflag "k8s.io/component-base/cli/flag"
)

type Registry struct {
	path           string
	lock           sync.Mutex
	flags          []cliflag.NamedFlagSets
	flagsInstalled bool
}

// Flagz installs the flagz handler
type Flagz struct{}

type mux interface {
	Handle(path string, handler http.Handler)
}

// Install adds the DefaultFlagz handler
func (f Flagz) Install(m mux, flags []cliflag.NamedFlagSets) {
	flagzRegistry := Registry{
		flags: flags,
	}
	flagzRegistry.installHandler(m)
}

func (reg *Registry) installHandler(m mux) {
	reg.lock.Lock()
	defer reg.lock.Unlock()
	reg.flagsInstalled = true
	m.Handle("/flagz",
		metrics.InstrumentHandlerFunc("GET",
			/* group = */ "",
			/* version = */ "",
			/* resource = */ "",
			/* subresource = */ "/flagz",
			/* scope = */ "",
			/* component = */ "",
			/* deprecated */ false,
			/* removedRelease */ "",
			handleFlags(reg.flags)))
}

func (reg *Registry) AddFlags(flags cliflag.NamedFlagSets) error {
	reg.lock.Lock()
	defer reg.lock.Unlock()
	if reg.flagsInstalled {
		return fmt.Errorf("unable to add because the %s endpoint has already been created", reg.path)
	}
	reg.flags = append(reg.flags, flags)
	return nil
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
