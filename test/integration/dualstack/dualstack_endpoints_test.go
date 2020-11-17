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

package dualstack

import (
	"context"
	"fmt"
	"net"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/controller/endpoint"
	"k8s.io/kubernetes/pkg/controller/endpointslice"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestDualStackEndpoints(t *testing.T) {
	// Create an IPv4IPv6 dual stack control-plane
	serviceCIDR := "10.0.0.0/16"
	secondaryServiceCIDR := "2001:db8:1::/48"
	labelMap := func() map[string]string {
		return map[string]string{"foo": "bar"}
	}

	dualStack := true
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, dualStack)()

	cfg := framework.NewIntegrationTestMasterConfig()
	_, cidr, err := net.ParseCIDR(serviceCIDR)
	if err != nil {
		t.Fatalf("Bad cidr: %v", err)
	}
	cfg.ExtraConfig.ServiceIPRange = *cidr

	_, secCidr, err := net.ParseCIDR(secondaryServiceCIDR)
	if err != nil {
		t.Fatalf("Bad cidr: %v", err)
	}
	cfg.ExtraConfig.SecondaryServiceIPRange = *secCidr

	_, s, closeFn := framework.RunAMaster(cfg)
	defer closeFn()

	client := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL})

	// Wait until the default "kubernetes" service is created.
	if err = wait.Poll(250*time.Millisecond, time.Minute, func() (bool, error) {
		_, err := client.CoreV1().Services(metav1.NamespaceDefault).Get(context.TODO(), "kubernetes", metav1.GetOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			return false, err
		}
		return !apierrors.IsNotFound(err), nil
	}); err != nil {
		t.Fatalf("Creating kubernetes service timed out")
	}

	resyncPeriod := 0 * time.Hour
	informers := informers.NewSharedInformerFactory(client, resyncPeriod)

	// Create fake node
	testNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "fakenode",
		},
		Spec: v1.NodeSpec{Unschedulable: false},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{
				{
					Type:              v1.NodeReady,
					Status:            v1.ConditionTrue,
					Reason:            fmt.Sprintf("schedulable condition"),
					LastHeartbeatTime: metav1.Time{Time: time.Now()},
				},
			},
		},
	}
	if _, err := client.CoreV1().Nodes().Create(context.TODO(), testNode, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create Node %q: %v", testNode.Name, err)
	}

	epController := endpoint.NewEndpointController(
		informers.Core().V1().Pods(),
		informers.Core().V1().Services(),
		informers.Core().V1().Endpoints(),
		client,
		1*time.Second)

	epsController := endpointslice.NewController(
		informers.Core().V1().Pods(),
		informers.Core().V1().Services(),
		informers.Core().V1().Nodes(),
		informers.Discovery().V1beta1().EndpointSlices(),
		int32(100),
		client,
		1*time.Second)

	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	// use only one worker to serialize the updates
	go epController.Run(1, stopCh)
	go epsController.Run(1, stopCh)

	var testcases = []struct {
		name           string
		serviceType    v1.ServiceType
		ipFamilies     []v1.IPFamily
		ipFamilyPolicy v1.IPFamilyPolicyType
	}{
		{
			name:           "Service IPv4 Only",
			serviceType:    v1.ServiceTypeClusterIP,
			ipFamilies:     []v1.IPFamily{v1.IPv4Protocol},
			ipFamilyPolicy: v1.IPFamilyPolicySingleStack,
		},
		{
			name:           "Service IPv6 Only",
			serviceType:    v1.ServiceTypeClusterIP,
			ipFamilies:     []v1.IPFamily{v1.IPv6Protocol},
			ipFamilyPolicy: v1.IPFamilyPolicySingleStack,
		},
		{
			name:           "Service IPv6 IPv4 Dual Stack",
			serviceType:    v1.ServiceTypeClusterIP,
			ipFamilies:     []v1.IPFamily{v1.IPv6Protocol, v1.IPv4Protocol},
			ipFamilyPolicy: v1.IPFamilyPolicyRequireDualStack,
		},
		{
			name:           "Service IPv4 IPv6 Dual Stack",
			serviceType:    v1.ServiceTypeClusterIP,
			ipFamilies:     []v1.IPFamily{v1.IPv4Protocol, v1.IPv6Protocol},
			ipFamilyPolicy: v1.IPFamilyPolicyRequireDualStack,
		},
	}

	for i, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			ns := framework.CreateTestingNamespace(fmt.Sprintf("test-endpointslice-dualstack-%d", i), s, t)
			defer framework.DeleteTestingNamespace(ns, s, t)

			// Create a pod with labels
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: ns.Name,
					Labels:    labelMap(),
				},
				Spec: v1.PodSpec{
					NodeName: "fakenode",
					Containers: []v1.Container{
						{
							Name:  "fake-name",
							Image: "fakeimage",
						},
					},
				},
			}

			createdPod, err := client.CoreV1().Pods(ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Failed to create pod %s: %v", pod.Name, err)
			}

			// Set pod IPs
			podIPbyFamily := map[v1.IPFamily]string{v1.IPv4Protocol: "1.1.1.1", v1.IPv6Protocol: "2001:db2::65"}
			createdPod.Status = v1.PodStatus{
				Phase:  v1.PodRunning,
				PodIPs: []v1.PodIP{{IP: podIPbyFamily[v1.IPv4Protocol]}, {IP: podIPbyFamily[v1.IPv6Protocol]}},
			}
			_, err = client.CoreV1().Pods(ns.Name).UpdateStatus(context.TODO(), createdPod, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("Failed to update status of pod %s: %v", pod.Name, err)
			}

			svc := &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("svc-test-%d", i), // use different services for each test
					Namespace: ns.Name,
					Labels:    labelMap(),
				},
				Spec: v1.ServiceSpec{
					Type:           v1.ServiceTypeClusterIP,
					IPFamilies:     tc.ipFamilies,
					IPFamilyPolicy: &tc.ipFamilyPolicy,
					Selector:       labelMap(),
					Ports: []v1.ServicePort{
						{
							Name:       fmt.Sprintf("port-test-%d", i),
							Port:       443,
							TargetPort: intstr.IntOrString{IntVal: 443},
							Protocol:   "TCP",
						},
					},
				},
			}

			// create a service
			_, err = client.CoreV1().Services(ns.Name).Create(context.TODO(), svc, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Error creating service: %v", err)
			}

			// wait until endpoints are created
			// legacy endpoints are not dual stack
			// and use the address of the first IP family
			if err := wait.PollImmediate(1*time.Second, wait.ForeverTestTimeout, func() (bool, error) {
				e, err := client.CoreV1().Endpoints(ns.Name).Get(context.TODO(), svc.Name, metav1.GetOptions{})
				if err != nil {
					t.Logf("Error fetching endpoints: %v", err)
					return false, nil
				}
				// check if the endpoint addresses match the pod IP of the first IPFamily of the service
				// since this is an integration test PodIPs are not "ready"
				if len(e.Subsets) > 0 && len(e.Subsets[0].NotReadyAddresses) > 0 {
					if e.Subsets[0].NotReadyAddresses[0].IP == podIPbyFamily[tc.ipFamilies[0]] {
						return true, nil
					}
					t.Logf("Endpoint address %s does not match PodIP %s ", e.Subsets[0].Addresses[0].IP, podIPbyFamily[tc.ipFamilies[0]])
				}
				t.Logf("Endpoint does not contain addresses: %s", e.Name)
				return false, nil
			}); err != nil {
				t.Fatalf("Endpoints not found: %v", err)
			}

			// wait until the endpoint slices are created
			err = wait.PollImmediate(1*time.Second, wait.ForeverTestTimeout, func() (bool, error) {
				lSelector := discovery.LabelServiceName + "=" + svc.Name
				esList, err := client.DiscoveryV1beta1().EndpointSlices(ns.Name).List(context.TODO(), metav1.ListOptions{LabelSelector: lSelector})
				if err != nil {
					t.Logf("Error listing EndpointSlices: %v", err)
					return false, nil
				}
				// there must be an endpoint slice per ipFamily
				if len(esList.Items) != len(tc.ipFamilies) {
					t.Logf("Waiting for EndpointSlice to be created %v", esList)
					return false, nil
				}
				// there must be an endpoint address per each IP family
				for _, ipFamily := range tc.ipFamilies {
					found := false
					for _, slice := range esList.Items {
						// check if the endpoint addresses match the pod IPs
						if len(slice.Endpoints) > 0 && len(slice.Endpoints[0].Addresses) > 0 {
							if string(ipFamily) == string(slice.AddressType) &&
								slice.Endpoints[0].Addresses[0] == podIPbyFamily[ipFamily] {
								found = true
								break
							}
						}
						t.Logf("Waiting endpoint slice to contain addresses")
					}
					if !found {
						t.Logf("Endpoint slices does not contain PodIP %s", podIPbyFamily[ipFamily])
						return false, nil
					}
				}
				return true, nil
			})
			if err != nil {
				t.Fatalf("Error waiting for endpoint slices: %v", err)
			}
		})
	}
}
