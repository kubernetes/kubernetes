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
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/controlplane"

	"k8s.io/kubernetes/test/integration/framework"
)

// TestServicesFinalizersRepairLoop tests that Services participate in the object
// deletion when using finalizers, and that the Services Repair controller doesn't,
// mistakenly, repair the ClusterIP assigned to the Service that is being deleted.
// https://issues.k8s.io/87603
func TestServicesFinalizersRepairLoop(t *testing.T) {
	serviceCIDR := "10.0.0.0/16"
	clusterIP := "10.0.0.20"
	interval := 5 * time.Second

	client, _, tearDownFn := framework.StartTestServer(t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.ServiceClusterIPRanges = serviceCIDR
		},
		ModifyServerConfig: func(cfg *controlplane.Config) {
			cfg.ExtraConfig.RepairServicesInterval = interval
		},
	})
	defer tearDownFn()

	// verify client is working
	if err := wait.PollImmediate(5*time.Second, 2*time.Minute, func() (bool, error) {
		_, err := client.CoreV1().Endpoints(metav1.NamespaceDefault).Get(context.TODO(), "kubernetes", metav1.GetOptions{})
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
			ClusterIP: clusterIP,
			Ports: []v1.ServicePort{{
				Port:       8443,
				NodePort:   30443,
				TargetPort: intstr.FromInt(8443),
				Protocol:   v1.ProtocolTCP,
			}},
			Type: v1.ServiceTypeNodePort,
		},
	}

	// Create service
	if _, err := client.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), &svcNodePort, metav1.CreateOptions{}); err != nil {
		t.Errorf("unexpected error creating service: %v", err)
	}
	t.Logf("Created service: %s", svcNodePort.Name)

	// Check the service has been created correctly
	svc, err := client.CoreV1().Services(metav1.NamespaceDefault).Get(context.TODO(), svcNodePort.Name, metav1.GetOptions{})
	if err != nil || svc.Spec.ClusterIP != clusterIP {
		t.Errorf("created service is not correct: %v", err)
	}
	t.Logf("Service created successfully: %v", svc)

	// Delete service
	if err := client.CoreV1().Services(metav1.NamespaceDefault).Delete(context.TODO(), svcNodePort.Name, metav1.DeleteOptions{}); err != nil {
		t.Errorf("unexpected error deleting service: %v", err)
	}
	t.Logf("Deleted service: %s", svcNodePort.Name)

	// wait for the repair loop to recover the deleted resources
	time.Sleep(interval + 1)

	// Check that the service was not deleted and the IP is already allocated
	svc, err = client.CoreV1().Services(metav1.NamespaceDefault).Get(context.TODO(), svcNodePort.Name, metav1.GetOptions{})
	if err != nil || svc.Spec.ClusterIP != clusterIP {
		t.Errorf("created service is not correct: %v", err)
	}
	t.Logf("Service after Delete: %v", svc)

	// Remove the finalizer
	if _, err = client.CoreV1().Services(metav1.NamespaceDefault).Patch(context.TODO(), svcNodePort.Name, types.JSONPatchType, []byte(`[{"op":"remove","path":"/metadata/finalizers"}]`), metav1.PatchOptions{}); err != nil {
		t.Errorf("unexpected error removing finalizer: %v", err)
	}
	t.Logf("Removed service finalizer: %s", svcNodePort.Name)

	// Check that the service was deleted
	_, err = client.CoreV1().Services(metav1.NamespaceDefault).Get(context.TODO(), svcNodePort.Name, metav1.GetOptions{})
	if err == nil {
		t.Errorf("service was not delete: %v", err)
	}

	// Try to create service again
	if _, err := client.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), &svcNodePort, metav1.CreateOptions{}); err != nil {
		t.Errorf("unexpected error creating service: %v", err)
	}
	t.Logf("Created service: %s", svcNodePort.Name)
}
