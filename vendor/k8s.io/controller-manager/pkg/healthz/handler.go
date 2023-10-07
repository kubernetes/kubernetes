/*
Copyright 2021 The Kubernetes Authors.

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

package healthz

import (
	"net/http"
	"sync"

	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/apiserver/pkg/server/mux"
)

// MutableHealthzHandler returns a http.Handler that handles "/healthz"
// following the standard healthz mechanism.
//
// This handler can register health checks after its creation, which
// is originally not allowed with standard healthz handler.
type MutableHealthzHandler struct {
	// handler is the underlying handler that will be replaced every time
	// new checks are added.
	handler http.Handler
	// mutex is a RWMutex that allows concurrent health checks (read)
	// but disallow replacing the handler at the same time (write).
	mutex  sync.RWMutex
	checks []healthz.HealthChecker
}

func (h *MutableHealthzHandler) ServeHTTP(writer http.ResponseWriter, request *http.Request) {
	h.mutex.RLock()
	defer h.mutex.RUnlock()

	h.handler.ServeHTTP(writer, request)
}

// AddHealthChecker adds health check(s) to the handler.
//
// Every time this function is called, the handler have to be re-initiated.
// It is advised to add as many checks at once as possible.
func (h *MutableHealthzHandler) AddHealthChecker(checks ...healthz.HealthChecker) {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	h.checks = append(h.checks, checks...)
	newMux := mux.NewPathRecorderMux("healthz")
	healthz.InstallHandler(newMux, h.checks...)
	h.handler = newMux
}

func NewMutableHealthzHandler(checks ...healthz.HealthChecker) *MutableHealthzHandler {
	h := &MutableHealthzHandler{}
	h.AddHealthChecker(checks...)

	return h
}
