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

package slis

import (
	"net/http"
	"sync"

	"k8s.io/component-base/metrics"
)

var (
	installOnce          = sync.Once{}
	installWithResetOnce = sync.Once{}
)

type mux interface {
	Handle(path string, handler http.Handler)
}

type SLIMetrics struct{}

// Install adds the DefaultMetrics handler
func (s SLIMetrics) Install(m mux) {
	installOnce.Do(func() {
		Register(Registry)
		m.Handle("/metrics/slis", metrics.HandlerFor(Registry, metrics.HandlerOpts{}))
	})
}

type SLIMetricsWithReset struct{}

// Install adds the DefaultMetrics handler
func (s SLIMetricsWithReset) Install(m mux) {
	installWithResetOnce.Do(func() {
		Register(Registry)
		m.Handle("/metrics/slis", metrics.HandlerWithReset(Registry, metrics.HandlerOpts{}))
	})
}
