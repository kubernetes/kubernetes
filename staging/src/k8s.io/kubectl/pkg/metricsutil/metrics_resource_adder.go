/*
Copyright 2021 The Kubernetes Authors.

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

package metricsutil

import (
	corev1 "k8s.io/api/core/v1"
	metricsapi "k8s.io/metrics/pkg/apis/metrics"
)

type ResourceAdder struct {
	resources []corev1.ResourceName
	total     corev1.ResourceList
}

func NewResourceAdder(resources []corev1.ResourceName) *ResourceAdder {
	return &ResourceAdder{
		resources: resources,
		total:     make(corev1.ResourceList),
	}
}

// AddPodMetrics adds each pod metric to the total
func (adder *ResourceAdder) AddPodMetrics(m *metricsapi.PodMetrics) {
	for _, c := range m.Containers {
		for _, res := range adder.resources {
			total := adder.total[res]
			total.Add(c.Usage[res])
			adder.total[res] = total
		}
	}
}
