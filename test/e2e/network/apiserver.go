/*
Copyright 2019 The Kubernetes Authors.

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

	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/network/common"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var _ = common.SIGDescribe("API Server", func() {
	f := framework.NewDefaultFramework("apiserver")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var cs clientset.Interface

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
	})

	/*
		Release: v1.9
		Testname: Kubernetes Service
		Description: By default when a kubernetes cluster is running there MUST be a 'kubernetes' service running in the cluster.
	*/
	framework.ConformanceIt("should provide secure master service", func(ctx context.Context) {
		_, err := cs.CoreV1().Services(metav1.NamespaceDefault).Get(ctx, "kubernetes", metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to fetch the service object for the service named kubernetes")
	})

	/*
		Release: v1.21
		Testname: kubernetes.default Endpoints and EndpointSlices
		Description: The "kubernetes.default" service MUST have Endpoints and EndpointSlices pointing to each API server instance.
	*/
	framework.ConformanceIt("should have Endpoints and EndpointSlices pointing to API Server", func(ctx context.Context) {
		namespace := "default"
		name := "kubernetes"
		// verify "kubernetes.default" service exist
		_, err := cs.CoreV1().Services(namespace).Get(ctx, name, metav1.GetOptions{})
		framework.ExpectNoError(err, "error obtaining API server \"kubernetes\" Service resource on \"default\" namespace")

		// verify Endpoints for the API servers exist
		endpoints, err := cs.CoreV1().Endpoints(namespace).Get(ctx, name, metav1.GetOptions{})
		framework.ExpectNoError(err, "error obtaining API server \"kubernetes\" Endpoint resource on \"default\" namespace")
		if len(endpoints.Subsets) == 0 {
			framework.Failf("Expected at least 1 subset in endpoints, got %d: %#v", len(endpoints.Subsets), endpoints.Subsets)
		}

		// get the addresses and ports from the Endpoints
		epAddresses := sets.New[string]()
		epPorts := sets.New[int32]()
		for _, subset := range endpoints.Subsets {
			for _, addr := range subset.Addresses {
				epAddresses.Insert(addr.IP)
			}
			for _, addr := range subset.NotReadyAddresses {
				epAddresses.Insert(addr.IP)
			}
			for _, port := range subset.Ports {
				epPorts.Insert(port.Port)
			}
		}
		framework.Logf("Endpoints addresses: %v , ports: %v", sets.List(epAddresses), sets.List(epPorts))

		// verify EndpointSlices for the API servers exist
		endpointSliceList, err := cs.DiscoveryV1().EndpointSlices(namespace).List(ctx, metav1.ListOptions{
			LabelSelector: "kubernetes.io/service-name=" + name,
		})
		framework.ExpectNoError(err, "error obtaining API server \"kubernetes\" EndpointSlice resource on \"default\" namespace")
		if len(endpointSliceList.Items) == 0 {
			framework.Failf("Expected at least 1 EndpointSlice, got %d: %#v", len(endpoints.Subsets), endpoints.Subsets)
		}

		// get the addresses and ports from the EndpointSlice(s). Since the
		// Endpoints are single stack, and match the cluster's primary IP family,
		// we ignore EndpointSlice(s) of the other IP family.
		var addrType discoveryv1.AddressType
		if framework.TestContext.ClusterIsIPv6() {
			addrType = discoveryv1.AddressTypeIPv6
		} else {
			addrType = discoveryv1.AddressTypeIPv4
		}

		sliceAddresses := sets.New[string]()
		slicePorts := sets.New[int32]()
		for _, slice := range endpointSliceList.Items {
			if slice.AddressType != addrType {
				framework.Logf("Skipping slice %s: wanted %s family, got %s", slice.Name, addrType, slice.AddressType)
				continue
			}
			for _, s := range slice.Endpoints {
				sliceAddresses.Insert(s.Addresses...)
			}
			for _, port := range slice.Ports {
				if port.Port != nil {
					slicePorts.Insert(*port.Port)
				}
			}
		}

		framework.Logf("EndpointSlices addresses: %v , ports: %v", sets.List(sliceAddresses), sets.List(slicePorts))
		if !sliceAddresses.Equal(epAddresses) {
			framework.Failf("EndpointSlice addresses do not match Endpoints addresses")
		}
		if !slicePorts.Equal(epPorts) {
			framework.Failf("EndpointSlice ports do not match Endpoints ports")
		}
	})
})
