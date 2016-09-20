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

	apiservermetrics "k8s.io/kubernetes/pkg/apiserver/metrics"
	"k8s.io/kubernetes/pkg/genericapiserver/routes"
	etcdmetrics "k8s.io/kubernetes/pkg/storage/etcd/metrics"

	"github.com/emicklei/go-restful"
	"github.com/prometheus/client_golang/prometheus"
)

// DefaultMetrics exposes the default prometheus metrics.
func DefaultMetrics() *restful.WebService {
	handler := routes.HandlerRouteFunction(prometheus.Handler().ServeHTTP)
	ws := new(restful.WebService)
	ws.Path("/metrics")
	ws.Doc("prometheus metrics")
	ws.Route(ws.GET("/{subpath:*}").To(handler))
	ws.Route(ws.GET("/").To(handler))
	return ws
}

// MetricsWithReset install the prometheus metrics handler extended with support for the DELETE method
// which resets the metrics.
func MetricsWithReset() *restful.WebService {
	ws := DefaultMetrics()
	ws.Route(ws.DELETE("/").To(func(req *restful.Request, resp *restful.Response) {
		apiservermetrics.Reset()
		etcdmetrics.Reset()
		io.WriteString(resp.ResponseWriter, "metrics reset\n")
	}))
	return ws
}
