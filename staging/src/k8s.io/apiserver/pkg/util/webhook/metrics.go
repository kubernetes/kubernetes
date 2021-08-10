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

package webhook

import (
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

var x509MissingSANCounter = metrics.NewCounter(
	&metrics.CounterOpts{
		Subsystem: "webhooks",
		Namespace: "apiserver",
		Name:      "x509_missing_san_total",
		Help: "Counts the number of requests to servers missing SAN extension " +
			"in their serving certificate OR the number of connection failures " +
			"due to the lack of x509 certificate SAN extension missing " +
			"(either/or, based on the runtime environment)",
		StabilityLevel: metrics.ALPHA,
	},
)

func init() {
	legacyregistry.MustRegister(x509MissingSANCounter)
}
