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
	handlersmetrics "k8s.io/apiserver/pkg/endpoints/handlers/metrics"
	apimetrics "k8s.io/apiserver/pkg/endpoints/metrics"
	"k8s.io/apiserver/pkg/server/mux"
	cachermetrics "k8s.io/apiserver/pkg/storage/cacher/metrics"
	etcd3metrics "k8s.io/apiserver/pkg/storage/etcd3/metrics"
	flowcontrolmetrics "k8s.io/apiserver/pkg/util/flowcontrol/metrics"
	peerproxymetrics "k8s.io/apiserver/pkg/util/peerproxy/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

// DefaultMetrics installs the default prometheus metrics handler
type DefaultMetrics struct{}

// Install adds the DefaultMetrics handler
func (m DefaultMetrics) Install(c *mux.PathRecorderMux) {
	register()
	c.Handle("/metrics", legacyregistry.Handler())
}

// MetricsWithReset install the prometheus metrics handler extended with support for the DELETE method
// which resets the metrics.
type MetricsWithReset struct{}

// Install adds the MetricsWithReset handler
func (m MetricsWithReset) Install(c *mux.PathRecorderMux) {
	register()
	c.Handle("/metrics", legacyregistry.HandlerWithReset())
}

// register apiserver and etcd metrics
func register() {
	apimetrics.Register()
	cachermetrics.Register()
	etcd3metrics.Register()
	flowcontrolmetrics.Register()
	peerproxymetrics.Register()
	handlersmetrics.Register()
}
