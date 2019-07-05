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

package fake

import (
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
)

var _ audit.Backend = &Backend{}

// Backend is a fake audit backend for testing purposes.
type Backend struct {
	OnRequest func(events []*auditinternal.Event)
}

// Run does nothing.
func (b *Backend) Run(stopCh <-chan struct{}) error {
	return nil
}

// Shutdown does nothing.
func (b *Backend) Shutdown() {
	return
}

// ProcessEvents calls a callback on a batch, if present.
func (b *Backend) ProcessEvents(ev ...*auditinternal.Event) bool {
	if b.OnRequest != nil {
		b.OnRequest(ev)
	}
	return true
}

func (b *Backend) String() string {
	return ""
}
