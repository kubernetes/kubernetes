/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package testHelper

import (
	kapi "k8s.io/kubernetes/pkg/api"
)

type HostPort struct {
	Host string `json:"host"`
	Port int    `json:"port"`
}

func GetHostPort(service *kapi.Service) *HostPort {
	return &HostPort{
		Host: service.Spec.ClusterIP,
		Port: int(service.Spec.Ports[0].Port),
	}
}

func NewPod(namespace, podName, podIP string) kapi.Pod {
	pod := kapi.Pod{
		ObjectMeta: kapi.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
		},
		Status: kapi.PodStatus{
			PodIP: podIP,
		},
	}

	return pod
}

func NewService(namespace, serviceName, clusterIP, portName string, portNumber int) kapi.Service {
	service := kapi.Service{
		ObjectMeta: kapi.ObjectMeta{
			Name:      serviceName,
			Namespace: namespace,
		},
		Spec: kapi.ServiceSpec{
			ClusterIP: clusterIP,
			Ports: []kapi.ServicePort{
				{Port: int32(portNumber), Name: portName, Protocol: "TCP"},
			},
		},
	}
	return service
}
