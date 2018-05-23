/*
Copyright 2018 The Kubernetes Authors.

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

package audit

import (
	"fmt"
	"strings"
	"sync"

	auditinternal "k8s.io/apiserver/pkg/apis/audit"
)

var _ Registry = registry{}

func WithRegistry(backend Backend) Registry {
	return registry{
		staticBackend:   backend,
		dynamicBackends: map[Backend]struct{}{},
	}
}

type registry struct {
	staticBackend   Backend
	dynamicBackends map[Backend]struct{}
	lock            sync.RWMutex
}

func (r registry) ProcessEvents(events ...*auditinternal.Event) {
	if r.staticBackend != nil {
		r.staticBackend.ProcessEvents(events...)
	}
	r.lock.RLock()
	defer r.lock.RUnlock()
	for backend := range r.dynamicBackends {
		backend.ProcessEvents(events...)
	}
}

func (r registry) Run(stopCh <-chan struct{}) error {
	if r.staticBackend != nil {
		return r.staticBackend.Run(stopCh)
	}
	return nil
}

func (r registry) Shutdown() {
	if r.staticBackend != nil {
		r.staticBackend.Shutdown()
	}
	r.lock.RLock()
	defer r.lock.RUnlock()
	for backend := range r.dynamicBackends {
		backend.Shutdown()
	}
}

func (r registry) String() string {
	staticBackend := ""
	if r.staticBackend != nil {
		staticBackend = fmt.Sprint(r.staticBackend)
	}

	r.lock.RLock()
	defer r.lock.RUnlock()
	var backendStrings []string
	for backend := range r.dynamicBackends {
		backendStrings = append(backendStrings, fmt.Sprintf("%s", backend))
	}

	return fmt.Sprintf("Registry{static: %s, dynamic: %s}", staticBackend, strings.Join(backendStrings, ","))
}

func (r registry) Register(backend Backend) {
	r.lock.Lock()
	defer r.lock.Unlock()
	r.dynamicBackends[backend] = struct{}{}
}

func (r registry) UnRegister(backend Backend) {
	r.lock.Lock()
	defer r.lock.Unlock()
	delete(r.dynamicBackends, backend)
}
