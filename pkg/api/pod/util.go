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

package pod

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/intstr"
)

const (
	// TODO: to be de!eted after v1.3 is released. PodSpec has a dedicated Hostname field.
	// The annotation value is a string specifying the hostname to be used for the pod e.g 'my-webserver-1'
	PodHostnameAnnotation = "pod.beta.kubernetes.io/hostname"

	// TODO: to be de!eted after v1.3 is released. PodSpec has a dedicated Subdomain field.
	// The annotation value is a string specifying the subdomain e.g. "my-web-service"
	// If specified, on the pod itself, "<hostname>.my-web-service.<namespace>.svc.<cluster domain>" would resolve to
	// the pod's IP.
	// If there is a headless service named "my-web-service" in the same namespace as the pod, then,
	// <hostname>.my-web-service.<namespace>.svc.<cluster domain>" would be resolved by the cluster DNS Server.
	PodSubdomainAnnotation = "pod.beta.kubernetes.io/subdomain"
)

// FindPort locates the container port for the given pod and portName.  If the
// targetPort is a number, use that.  If the targetPort is a string, look that
// string up in all named ports in all containers in the target pod.  If no
// match is found, fail.
func FindPort(pod *api.Pod, svcPort *api.ServicePort) (int, error) {
	portName := svcPort.TargetPort
	switch portName.Type {
	case intstr.String:
		name := portName.StrVal
		for _, container := range pod.Spec.Containers {
			for _, port := range container.Ports {
				if port.Name == name && port.Protocol == svcPort.Protocol {
					return int(port.ContainerPort), nil
				}
			}
		}
	case intstr.Int:
		return portName.IntValue(), nil
	}

	return 0, fmt.Errorf("no suitable port for manifest: %s", pod.UID)
}
