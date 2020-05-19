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

package network

import (
	"context"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"

	"k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/client-go/kubernetes"

	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestServicesFinalizers tests that services participate in the object
// deletion when using finalizers
func TestServicesFinalizers(t *testing.T) {
	etcd := framework.SharedEtcd()
	// cleanup the registry storage
	defer registry.CleanupStorage()
	// start a kube-apiserver
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		"--service-cluster-ip-range", "10.0.0.0/24",
		"--advertise-address", "10.0.0.1",
	}, etcd)
	defer server.TearDownFn()
	// create a client
	client, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Errorf("error creating client: %v", err)
	}

	// verify client is working
	if err := wait.PollImmediate(5*time.Second, 2*time.Minute, func() (bool, error) {
		_, err = client.CoreV1().Endpoints(metav1.NamespaceDefault).Get(context.TODO(), "kubernetes", metav1.GetOptions{})
		if err != nil {
			t.Logf("error fetching endpoints: %v", err)
			return false, nil
		}
		return true, nil
	}); err != nil {
		t.Errorf("server without enabled endpoints failed to register: %v", err)
	}

	// Create a NodePort service with one finalizer
	svcNodePort := v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "svc",
			Finalizers: []string{"foo.bar/some-finalizer"},
		},
		Spec: v1.ServiceSpec{
			ClusterIP: "10.0.0.20",
			Ports: []v1.ServicePort{{
				Port:       8443,
				NodePort:   30443,
				TargetPort: intstr.FromInt(8443),
				Protocol:   v1.ProtocolTCP,
			}},
			Type: v1.ServiceTypeNodePort,
		},
	}

	endpoint := v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name: "svc",
		},
		Subsets: []v1.EndpointSubset{
			{
				Addresses: []v1.EndpointAddress{
					{
						IP: "192.168.0.1",
					},
				},
				Ports: []v1.EndpointPort{
					{
						Name:     "port",
						Port:     8443,
						Protocol: v1.ProtocolTCP,
					},
				},
			},
		},
	}

	for i := 0; i < 2; i++ {
		// Create service
		if _, err := client.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), &svcNodePort, metav1.CreateOptions{}); err != nil {
			t.Errorf("unexpected error creating service: %v", err)
		}
		t.Logf("Created service: %s", svcNodePort.Name)

		// Create associated endpoint
		if _, err := client.CoreV1().Endpoints(metav1.NamespaceDefault).Create(context.TODO(), &endpoint, metav1.CreateOptions{}); err != nil {
			t.Errorf("unexpected error creating service: %v", err)
		}
		t.Logf("Created endpoint: %s", endpoint.Name)

		// Check the service and the endpoints were created correctly
		svc, err := client.CoreV1().Services(metav1.NamespaceDefault).Get(context.TODO(), svcNodePort.Name, metav1.GetOptions{})
		if err != nil || svc.Spec.ClusterIP != "10.0.0.20" {
			t.Errorf("created service is not correct: %v", err)
		}
		t.Logf("Service created successfully: %v", svc)

		ep, err := client.CoreV1().Endpoints(metav1.NamespaceDefault).Get(context.TODO(), svcNodePort.Name, metav1.GetOptions{})
		if err != nil {
			t.Errorf("error fetching endpoints: %v", err)
		}
		t.Logf("Endpoints created successfully: %v", ep)

		// Delete service
		if err := client.CoreV1().Services(metav1.NamespaceDefault).Delete(context.TODO(), svcNodePort.Name, metav1.DeleteOptions{}); err != nil {
			t.Errorf("unexpected error deleting service: %v", err)
		}
		t.Logf("Deleted service: %s", svcNodePort.Name)

		time.Sleep(time.Second)

		// Check that the service was not deleted and the IP is already allocated
		svc, err = client.CoreV1().Services(metav1.NamespaceDefault).Get(context.TODO(), svcNodePort.Name, metav1.GetOptions{})
		if err != nil || svc.Spec.ClusterIP != "10.0.0.20" {
			t.Errorf("created service is not correct: %v", err)
		}
		t.Logf("Service after Delete: %v", svc)

		// Check that the endpoint was deleted??
		ep, err = client.CoreV1().Endpoints(metav1.NamespaceDefault).Get(context.TODO(), svcNodePort.Name, metav1.GetOptions{})
		if err == nil {
			t.Errorf("Endpoint should be deleted???: %v", err)
		}

		// Remove the finalizer
		if _, err = client.CoreV1().Services(metav1.NamespaceDefault).Patch(context.TODO(), svcNodePort.Name, types.JSONPatchType, []byte(`[{"op":"remove","path":"/metadata/finalizers"}]`), metav1.PatchOptions{}); err != nil {
			t.Errorf("unexpected error removing finalizer: %v", err)
		}
		t.Logf("Removed service finalizer: %s", svcNodePort.Name)

		// Check thath the service was deleted
		svc, err = client.CoreV1().Services(metav1.NamespaceDefault).Get(context.TODO(), svcNodePort.Name, metav1.GetOptions{})
		if err == nil {
			t.Errorf("service was not delete: %v", err)
		}

	}

}
