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

package util

import (
	"fmt"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

// Lookup containerPort number by its named port name
func lookupContainerPortNumberByName(pod v1.Pod, name string) (int32, error) {
	for _, ctr := range pod.Spec.Containers {
		for _, ctrportspec := range ctr.Ports {
			if ctrportspec.Name == name {
				return ctrportspec.ContainerPort, nil
			}
		}
	}

	return int32(-1), fmt.Errorf("Pod '%s' does not have a named port '%s'", pod.Name, name)
}

// Lookup containerPort number from Service port number
// It implements the handling of resolving container named port, as well as ignoring targetPort when clusterIP=None
// It returns an error when a named port can't find a match (with -1 returned), or when the service does not
// declare such port (with the input port number returned).
func LookupContainerPortNumberByServicePort(svc v1.Service, pod v1.Pod, port int32) (int32, error) {
	for _, svcportspec := range svc.Spec.Ports {
		if svcportspec.Port != port {
			continue
		}
		if svc.Spec.ClusterIP == v1.ClusterIPNone {
			return port, nil
		}
		if svcportspec.TargetPort.Type == intstr.Int {
			if svcportspec.TargetPort.IntValue() == 0 {
				// targetPort is omitted, and the IntValue() would be zero
				return svcportspec.Port, nil
			} else {
				return int32(svcportspec.TargetPort.IntValue()), nil
			}
		} else {
			return lookupContainerPortNumberByName(pod, svcportspec.TargetPort.String())
		}
	}
	return port, fmt.Errorf("Service %s does not have a service port %d", svc.Name, port)
}
