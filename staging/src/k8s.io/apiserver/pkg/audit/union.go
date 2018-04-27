/*
Copyright 2017 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/util/errors"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
)

// Union returns an audit Backend which logs events to a set of backends. The returned
// Sink implementation blocks in turn for each call to ProcessEvents.
func Union(backends ...Backend) Backend {
	if len(backends) == 1 {
		return backends[0]
	}
	return union{backends}
}

type union struct {
	backends []Backend
}

func (u union) ProcessEvents(events ...*auditinternal.Event) {
	for _, backend := range u.backends {
		backend.ProcessEvents(events...)
	}
}

func (u union) Run(stopCh <-chan struct{}) error {
	var funcs []func() error
	for _, backend := range u.backends {
		funcs = append(funcs, func() error {
			return backend.Run(stopCh)
		})
	}
	return errors.AggregateGoroutines(funcs...)
}

func (u union) Shutdown() {
	for _, backend := range u.backends {
		backend.Shutdown()
	}
}

func (u union) String() string {
	var backendStrings []string
	for _, backend := range u.backends {
		backendStrings = append(backendStrings, fmt.Sprintf("%s", backend))
	}
	return fmt.Sprintf("union[%s]", strings.Join(backendStrings, ","))
}
