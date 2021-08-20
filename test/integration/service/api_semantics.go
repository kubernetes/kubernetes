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

package service

import (
	"context"
	"encoding/json"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/integration/framework"
)

type servicePortsMergePatch struct {
	Spec specPortsMergePatch `json:"spec,omitempty"`
}
type specPortsMergePatch struct {
	Ports []corev1.ServicePort `json:"ports,omitempty"`
}

// tests when patching a service with additional port that has the same port number
// as existing but different proto does not override existing one.
func Test_ServiceSetPortWithPatch(t *testing.T) {
	controlPlaneConfig := framework.NewIntegrationTestControlPlaneConfig()
	_, server, closeFn := framework.RunAnAPIServer(controlPlaneConfig)
	defer closeFn()

	config := restclient.Config{Host: server.URL}
	client, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateTestingNamespace("test-service-allocate-node-ports", server, t)
	defer framework.DeleteTestingNamespace(ns, server, t)

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-123",
		},
		Spec: corev1.ServiceSpec{
			Type: corev1.ServiceTypeNodePort,
			Ports: []corev1.ServicePort{{
				Name:     "tcp",
				Port:     int32(80),
				Protocol: corev1.ProtocolTCP,
			}},
			Selector: map[string]string{
				"foo": "bar",
			},
		},
	}

	service, err = client.CoreV1().Services(ns.Name).Create(context.TODO(), service, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error creating test service: %v", err)
	}

	portsForMergePatch := servicePortsMergePatch{
		Spec: specPortsMergePatch{
			Ports: []corev1.ServicePort{
				{
					Name:     "udp",
					Port:     int32(80),
					Protocol: corev1.ProtocolUDP,
				},
			},
		},
	}
	patchBytes, err := json.Marshal(&portsForMergePatch)
	if err != nil {
		t.Fatalf("failed to json.Marshal ports: %v", err)
	}

	_, err = client.CoreV1().Services(ns.Name).Patch(context.TODO(), service.Name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{})
	if err != nil {
		t.Fatalf("unexpected error patching service using strategic merge patch. %v", err)
	}

	patched, err := client.CoreV1().Services(ns.Name).Get(context.TODO(), service.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Unexpected error to get the service %s %v", service.Name, err)
	}

	// patched should have

	if len(patched.Spec.Ports) != 2 {
		t.Fatalf("expected two ports in service:%v got: %+v", service.Name, service.Spec.Ports)
	}

	for idx, port := range patched.Spec.Ports {
		if port.Name == "tcp" {
			if port.Port != 80 {
				t.Fatalf("expected tcp port:80 got:%v at idx:%v", port.Port, idx)
			}
			if port.Protocol != corev1.ProtocolTCP {
				t.Fatalf("expected protocol:TCP got:%v at idx:%v", port.Protocol, idx)
			}
			continue
		}

		if port.Name == "udp" {
			if port.Port != 80 {
				t.Fatalf("expected udp port:80 got:%v at idx:%v", port.Port, idx)
			}
			if port.Protocol != corev1.ProtocolUDP {
				t.Fatalf("expected protocol:udp got:%v at idx:%v", port.Protocol, idx)
			}
			continue
		}

		t.Fatalf("unexpected port name:%v at:%v", port.Name, idx)
	}

}
