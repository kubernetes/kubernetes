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

package reconcilers

import (
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func makeEndpointsArray(name string, ips []string, ports []corev1.EndpointPort) []runtime.Object {
	return []runtime.Object{
		makeEndpoints(name, ips, ports),
	}
}

func makeEndpoints(name string, ips []string, ports []corev1.EndpointPort) *corev1.Endpoints {
	endpoints := &corev1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceDefault,
			Name:      name,
			Labels: map[string]string{
				discoveryv1.LabelSkipMirror: "true",
			},
		},
	}
	if len(ips) > 0 || len(ports) > 0 {
		endpoints.Subsets = []corev1.EndpointSubset{{
			Addresses: make([]corev1.EndpointAddress, len(ips)),
			Ports:     ports,
		}}
		for i := range ips {
			endpoints.Subsets[0].Addresses[i].IP = ips[i]
		}
	}
	return endpoints
}
