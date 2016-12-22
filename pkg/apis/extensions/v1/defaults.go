/*
Copyright 2015 The Kubernetes Authors.

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

package v1

import (
	"k8s.io/kubernetes/pkg/api/v1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	RegisterDefaults(scheme)
	return scheme.AddDefaultingFuncs(
		SetDefaults_NetworkPolicy,
	)
}

func SetDefaults_NetworkPolicy(obj *NetworkPolicy) {
	// Default any undefined Protocol fields to TCP.
	for _, i := range obj.Spec.Ingress {
		if i.AllowPorts == nil {
			continue
		}
		for _, p := range i.AllowPorts.Ports {
			if p.Protocol == nil {
				proto := v1.ProtocolTCP
				p.Protocol = &proto
			}
		}
	}
}
