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

	"k8s.io/kubernetes/pkg/apiserver"
	apiservermetrics "k8s.io/kubernetes/pkg/apiserver/metrics"
	etcdmetrics "k8s.io/kubernetes/pkg/storage/etcd/metrics"

	"github.com/emicklei/go-restful"
	"github.com/prometheus/client_golang/prometheus"
)

// DefaultMetrics installs the default prometheus metrics handler
type DefaultMetrics struct{}

func (m DefaultMetrics) Install(mux *apiserver.PathRecorderMux, c *restful.Container) {
	mux.HandleFunc("/metrics", prometheus.Handler().ServeHTTP)
}

// MetricsWithReset install the prometheus metrics handler extended with support for the DELETE method
// which resets the metrics.
type MetricsWithReset struct{}

func (m MetricsWithReset) Install(mux *apiserver.PathRecorderMux, c *restful.Container) {
	defaultMetricsHandler := prometheus.Handler().ServeHTTP
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, req *http.Request) {
		if req.Method == "DELETE" {
			apiservermetrics.Reset()
			etcdmetrics.Reset()
			io.WriteString(w, "metrics reset\n")
			return
		}
		defaultMetricsHandler(w, req)
	})
}
