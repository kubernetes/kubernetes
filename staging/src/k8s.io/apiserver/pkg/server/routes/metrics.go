/*
Copyright 2014 The Kubernetes Authors.

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

package routes

import (
	"io"
	"net/http"

	apimetrics "k8s.io/apiserver/pkg/endpoints/metrics"
	"k8s.io/apiserver/pkg/server/mux"
	etcdmetrics "k8s.io/apiserver/pkg/storage/etcd/metrics"

	"github.com/prometheus/client_golang/prometheus"
)

// DefaultMetrics installs the default prometheus metrics handler
type DefaultMetrics struct{}

// Install adds the DefaultMetrics handler
func (m DefaultMetrics) Install(c *mux.APIContainer) {
	c.NonSwaggerRoutes.Handle("/metrics", prometheus.Handler())
}

// MetricsWithReset install the prometheus metrics handler extended with support for the DELETE method
// which resets the metrics.
type MetricsWithReset struct{}

// Install adds the MetricsWithReset handler
func (m MetricsWithReset) Install(c *mux.APIContainer) {
	defaultMetricsHandler := prometheus.Handler().ServeHTTP
	c.NonSwaggerRoutes.HandleFunc("/metrics", func(w http.ResponseWriter, req *http.Request) {
		if req.Method == "DELETE" {
			apimetrics.Reset()
			etcdmetrics.Reset()
			io.WriteString(w, "metrics reset\n")
			return
		}
		defaultMetricsHandler(w, req)
	})
}
