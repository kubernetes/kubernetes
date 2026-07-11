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

package polymorphichelpers

import (
	"fmt"
	"strconv"

	appsv1 "k8s.io/api/apps/v1"
	appsv1beta1 "k8s.io/api/apps/v1beta1"
	appsv1beta2 "k8s.io/api/apps/v1beta2"
	corev1 "k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
)

func multiProtocolsForObject(object runtime.Object) (map[string][]string, error) {
	// TODO: replace with a swagger schema based approach (identify pod selector via schema introspection)
	switch t := object.(type) {
	case *corev1.ReplicationController:
		return getMultiProtocols(t.Spec.Template.Spec), nil

	case *corev1.Pod:
		return getMultiProtocols(t.Spec), nil

	case *corev1.Service:
		return getServiceMultiProtocols(t.Spec), nil

	case *extensionsv1beta1.Deployment:
		return getMultiProtocols(t.Spec.Template.Spec), nil
	case *appsv1.Deployment:
		return getMultiProtocols(t.Spec.Template.Spec), nil
	case *appsv1beta2.Deployment:
		return getMultiProtocols(t.Spec.Template.Spec), nil
	case *appsv1beta1.Deployment:
		return getMultiProtocols(t.Spec.Template.Spec), nil

	case *extensionsv1beta1.ReplicaSet:
		return getMultiProtocols(t.Spec.Template.Spec), nil
	case *appsv1.ReplicaSet:
		return getMultiProtocols(t.Spec.Template.Spec), nil
	case *appsv1beta2.ReplicaSet:
		return getMultiProtocols(t.Spec.Template.Spec), nil

	default:
		return nil, fmt.Errorf("cannot extract protocols from %T", object)
	}
}

func getMultiProtocols(spec corev1.PodSpec) map[string][]string {
	result := make(map[string][]string)
	var protocol corev1.Protocol
	for _, container := range spec.Containers {
		for _, port := range container.Ports {
			// Empty protocol must be defaulted (TCP)
			protocol = corev1.ProtocolTCP
			if len(port.Protocol) > 0 {
				protocol = port.Protocol
			}
			p := strconv.Itoa(int(port.ContainerPort))
			result[p] = append(result[p], string(protocol))
		}
	}
	return result
}

// Extracts the protocols exposed by a service from the given service spec.
func getServiceMultiProtocols(spec corev1.ServiceSpec) map[string][]string {
	result := make(map[string][]string)
	var protocol corev1.Protocol
	for _, servicePort := range spec.Ports {
		// Empty protocol must be defaulted (TCP)
		protocol = corev1.ProtocolTCP
		if len(servicePort.Protocol) > 0 {
			protocol = servicePort.Protocol
		}
		p := strconv.Itoa(int(servicePort.Port))
		result[p] = append(result[p], string(protocol))
	}
	return result
}
