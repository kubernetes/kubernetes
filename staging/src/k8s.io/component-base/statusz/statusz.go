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

package statusz

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"time"

	"k8s.io/apiserver/pkg/endpoints/metrics"
	"k8s.io/klog/v2"
)

type Statusz struct {
	registry *statuszRegistry
}

type Options struct {
	ComponentName string
	StartTime     time.Time
}

type mux interface {
	Handle(path string, handler http.Handler)
}

func (f Statusz) Install(m mux, opts Options) {
	f.registry = Register(opts)
	f.registry.installHandler(m)
}

func (reg *statuszRegistry) installHandler(m mux) {
	reg.lock.Lock()
	defer reg.lock.Unlock()
	m.Handle("/statusz",
		metrics.InstrumentHandlerFunc("GET",
			/* group = */ "",
			/* version = */ "",
			/* resource = */ "",
			/* subresource = */ "/statusz",
			/* scope = */ "",
			/* component = */ "",
			/* deprecated */ false,
			/* removedRelease */ "",
			handleSections(reg.sections)))
}

// handleSections returns an http.HandlerFunc that serves the provided sections.
func handleSections(sections []section) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var individualCheckOutput bytes.Buffer
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		for _, section := range sections {
			err := section.Func(context.Background(), w)
			if err != nil {
				fmt.Fprintf(&individualCheckOutput, "[-]%s failed: reason withheld\n", section.Title)
				klog.V(2).Infof("%s section failed: %v", section.Title, err)
				http.Error(w, fmt.Sprintf("%s%s section failed", individualCheckOutput.String(), section.Title), http.StatusInternalServerError)
				return
			}
			fmt.Fprint(w)
		}
		individualCheckOutput.WriteTo(w)
	}
}
