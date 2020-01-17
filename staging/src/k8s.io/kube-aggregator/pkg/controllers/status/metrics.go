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

package apiserver

import (
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

/*
 * By default, all the following metrics are defined as falling under
 * ALPHA stability level https://github.com/kubernetes/enhancements/blob/master/keps/sig-instrumentation/20190404-kubernetes-control-plane-metrics-stability.md#stability-classes)
 *
 * Promoting the stability level of the metric is a responsibility of the component owner, since it
 * involves explicitly acknowledging support for the metric across multiple releases, in accordance with
 * the metric stability policy.
 */
var (
	unavailableCounter = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name:           "aggregator_unavailable_apiservice_count",
			Help:           "Counter of APIServices which are marked as unavailable broken down by APIService name and reason.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"name", "reason"},
	)
	unavailableGauge = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Name:           "aggregator_unavailable_apiservice",
			Help:           "Gauge of APIServices which are marked as unavailable broken down by APIService name.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"name"},
	)
)

func init() {
	legacyregistry.MustRegister(unavailableCounter)
	legacyregistry.MustRegister(unavailableGauge)
}
