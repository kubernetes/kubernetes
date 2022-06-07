/*
Copyright 2022 The Kubernetes Authors.

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

package openapiv3

import (
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

var (
	regenerationCounter = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name:           "apiextensions_openapi_v3_regeneration_count",
			Help:           "Counter of OpenAPI v3 spec regeneration count broken down by group, version, causing CRD and reason.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"group", "version", "crd", "reason"},
	)
)

func init() {
	legacyregistry.MustRegister(regenerationCounter)
}
