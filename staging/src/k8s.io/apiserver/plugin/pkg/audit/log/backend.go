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
	"strings"
	"sync"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
)

const (
	// FormatLegacy saves event in 1-line text format.
	FormatLegacy = "legacy"
	// FormatJson saves event in structured json format.
	FormatJson = "json"
)

// The plugin name reported in error metrics.
const pluginName = "log"

// AllowedFormats are the formats known by log backend.
var AllowedFormats = []string{
	FormatLegacy,
	FormatJson,
}

func NewBackend(out io.Writer, format string, groupVersion schema.GroupVersion) audit.Backend {
	return &backend{
		out:          out,
		format:       format,
		groupVersion: groupVersion,
		mu:           sync.Mutex{},
	}
}

type backend struct {
	out          io.Writer
	format       string
	groupVersion schema.GroupVersion
	mu           sync.Mutex
}

var _ audit.Backend = &backend{}

func (b *backend) ProcessEvents(events ...*auditinternal.Event) {
	line := ""
	switch b.format {
	case FormatLegacy:
		for _, ev := range events {
			line += audit.EventString(ev) + "\n"
		}
	case FormatJson:
		for _, ev := range events {
			bs, err := runtime.Encode(audit.Codecs.LegacyCodec(b.groupVersion), ev)
			if err != nil {
				audit.HandlePluginError(pluginName, err, ev)
				return
			}
			line += string(bs[:])
		}
	default:
		audit.HandlePluginError(pluginName, fmt.Errorf("log format %q is not in list of known formats (%s)",
			b.format, strings.Join(AllowedFormats, ",")), events...)
		return
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	if _, err := fmt.Fprint(b.out, line); err != nil {
		audit.HandlePluginError(pluginName, err, events...)
	}
}

func (b *backend) Run(stopCh <-chan struct{}) error {
	return nil
}

func (b *backend) Shutdown() {
	// Nothing to do here.
}
