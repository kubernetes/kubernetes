/*
Copyright 2020 The Kubernetes Authors.

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

package filters

import (
	"net/http"
	"sync"

	"k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/warning"
)

// WithWarningRecorder attaches a deduplicating k8s.io/apiserver/pkg/warning#WarningRecorder to the request context.
func WithWarningRecorder(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		recorder := &recorder{writer: w}
		req = req.WithContext(warning.WithWarningRecorder(req.Context(), recorder))
		handler.ServeHTTP(w, req)
	})
}

type recorder struct {
	lock     sync.Mutex
	recorded map[string]bool
	writer   http.ResponseWriter
}

func (r *recorder) AddWarning(agent, text string) {
	if len(text) == 0 {
		return
	}

	r.lock.Lock()
	defer r.lock.Unlock()

	// init if needed
	if r.recorded == nil {
		r.recorded = map[string]bool{}
	}

	// dedupe if already warned
	if r.recorded[text] {
		return
	}
	r.recorded[text] = true

	// TODO(liggitt): track total message characters written:
	// * if this takes us over 4k truncate individual messages to 256 chars and regenerate headers
	// * if we're already truncating truncate this message to 256 chars
	// * if we're still over 4k omit this message

	if header, err := net.NewWarningHeader(299, agent, text); err == nil {
		r.writer.Header().Add("Warning", header)
	}
}
