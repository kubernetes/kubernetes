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

package log

import (
	"fmt"
	"io"

	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
)

type backend struct {
	out io.Writer
}

var _ audit.Backend = &backend{}

func NewBackend(out io.Writer) *backend {
	return &backend{
		out: out,
	}
}

func (b *backend) ProcessEvents(events ...*auditinternal.Event) {
	for _, ev := range events {
		b.logEvent(ev)
	}
}

func (b *backend) logEvent(ev *auditinternal.Event) {
	line := audit.EventString(ev)
	if _, err := fmt.Fprintln(b.out, line); err != nil {
		audit.HandlePluginError("log", err, ev)
	}
}

func (b *backend) Run(stopCh <-chan struct{}) error {
	return nil
}
