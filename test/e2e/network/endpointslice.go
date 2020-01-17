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
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	discoveryv1alpha1 "k8s.io/api/discovery/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

var _ = SIGDescribe("EndpointSlice [Feature:EndpointSlice]", func() {
	version := "v1"
	ginkgo.Context("version "+version, func() {
		f := framework.NewDefaultFramework("endpointslice")

		var cs clientset.Interface
		var podClient *framework.PodClient

		ginkgo.BeforeEach(func() {
			cs = f.ClientSet
			podClient = f.PodClient()
		})

		labelPod1 := "pod1"
		labelPod2 := "pod2"
		labelPod3 := "pod3"
		labelShared12 := "shared12"
		labelValue := "on"

		ginkgo.It("should create Endpoints and EndpointSlices for Pods matching a Service", func() {
			pod1 := podClient.Create(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod1",
					Labels: map[string]string{
						labelPod1:     labelValue,
						labelShared12: labelValue,
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "container1",
							Image: imageutils.GetE2EImage(imageutils.Nginx),
							Ports: []v1.ContainerPort{{
								Name:          "example-name",
								ContainerPort: int32(3000),
							}},
						},
					},
				},
			})

			pod2 := podClient.Create(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod2",
					Labels: map[string]string{
						labelPod2:     labelValue,
						labelShared12: labelValue,
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "container1",
							Image: imageutils.GetE2EImage(imageutils.Nginx),
							Ports: []v1.ContainerPort{{
								Name:          "example-name",
								ContainerPort: int32(3001),
							}, {
								Name:          "other-port",
								ContainerPort: int32(3002),
							}},
						},
					},
				},
			})

			svc1 := createServiceReportErr(cs, f.Namespace.Name, &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "example-int-port",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{labelPod1: labelValue},
					Ports: []v1.ServicePort{{
						Name:       "example",
						Port:       80,
						TargetPort: intstr.FromInt(3000),
						Protocol:   v1.ProtocolTCP,
					}},
				},
			})

			svc2 := createServiceReportErr(cs, f.Namespace.Name, &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "example-named-port",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{labelShared12: labelValue},
					Ports: []v1.ServicePort{{
						Name:       "http",
						Port:       80,
						TargetPort: intstr.FromString("example-name"),
						Protocol:   v1.ProtocolTCP,
					}},
				},
			})

			svc3 := createServiceReportErr(cs, f.Namespace.Name, &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "example-no-match",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{labelPod3: labelValue},
					Ports: []v1.ServicePort{{
						Name:       "example-no-match",
						Port:       80,
						TargetPort: intstr.FromInt(8080),
						Protocol:   v1.ProtocolTCP,
					}},
				},
			})

			err := wait.Poll(5*time.Second, 3*time.Minute, func() (bool, error) {
				if !podClient.PodIsReady(pod1.Name) {
					framework.Logf("Pod 1 not ready yet")
					return false, nil
				}

				if !podClient.PodIsReady(pod2.Name) {
					framework.Logf("Pod 2 not ready yet")
					return false, nil
				}

				var err error
				pod1, err = podClient.Get(pod1.Name, metav1.GetOptions{})
				if err != nil {
					return false, err
				}

				pod2, err = podClient.Get(pod2.Name, metav1.GetOptions{})
				if err != nil {
					return false, err
				}

				return true, nil
			})
			framework.ExpectNoError(err)

			ginkgo.By("referencing a single matching pod")
			expectEndpointsAndSlices(cs, f.Namespace.Name, svc1, []*v1.Pod{pod1}, 1, 1, false)

			ginkgo.By("referencing matching pods with named port")
			expectEndpointsAndSlices(cs, f.Namespace.Name, svc2, []*v1.Pod{pod1, pod2}, 2, 2, true)

			ginkgo.By("creating empty endpoints and endpointslices for no matching pods")
			expectEndpointsAndSlices(cs, f.Namespace.Name, svc3, []*v1.Pod{}, 0, 1, false)

		})
	})
})

// expectEndpointsAndSlices verifies that Endpoints and EndpointSlices exist for
// a given Service and Namespace with the appropriate attributes set. This is a
// relatively complex function as the order of attributes or resources is not
// necessarily consistent. It is used as a helper function for the tests above
// and takes some shortcuts with the assumption that those test cases will be
// the only caller of this function.
func expectEndpointsAndSlices(cs clientset.Interface, ns string, svc *v1.Service, pods []*v1.Pod, numSubsets, numSlices int, namedPort bool) {
	endpointSlices := []discoveryv1alpha1.EndpointSlice{}
	endpoints := &v1.Endpoints{}

	err := wait.Poll(5*time.Second, 1*time.Minute, func() (bool, error) {
		endpointSlicesFound, matchingSlices := hasMatchingEndpointSlices(cs, ns, svc.Name, len(pods), numSlices)
		if !matchingSlices {
			framework.Logf("Matching EndpointSlices not found")
			return false, nil
		}

		endpointsFound, matchingEndpoints := hasMatchingEndpoints(cs, ns, svc.Name, len(pods), numSubsets)
		if !matchingEndpoints {
			framework.Logf("Matching EndpointSlices not found")
			return false, nil
		}

		endpointSlices = endpointSlicesFound
		endpoints = endpointsFound

		return true, nil
	})
	framework.ExpectNoError(err)

	podsByIP := map[string]*v1.Pod{}
	for _, pod := range pods {
		podsByIP[pod.Status.PodIP] = pod
		if len(pod.Spec.Containers) != 1 {
			framework.Failf("Expected pod to have 1 container, got %d", len(pod.Spec.Containers))
		}
	}

	if endpoints.Name != svc.Name {
		framework.Failf("Expected Endpoints name to be %s, got %s", svc.Name, endpoints.Name)
	}

	totalEndpointAddresses := 0
	for _, subset := range endpoints.Subsets {
		addresses := append(subset.Addresses, subset.NotReadyAddresses...)
		totalEndpointAddresses += len(addresses)

		if len(subset.Ports) != len(svc.Spec.Ports) {
			framework.Failf("Expected subset to have %d ports, got %d", len(svc.Spec.Ports), len(subset.Ports))
		}

		// If not a named port, the subset ports should directly correspond with
		// the Service ports.
		if !namedPort {
			for i, subsetPort := range subset.Ports {
				svcPort := svc.Spec.Ports[i]
				if subsetPort.Name != svcPort.Name {
					framework.Failf("Expected port name to be %s, got %s", svcPort.Name, subsetPort.Name)
				}
				if subsetPort.Protocol != svcPort.Protocol {
					framework.Failf("Expected protocol to be %s, got %s", svcPort.Protocol, subsetPort.Protocol)
				}
				if subsetPort.Port != svcPort.TargetPort.IntVal {
					framework.Failf("Expected port to be %d, got %d", svcPort.TargetPort.IntVal, subsetPort.Port)
				}
			}
		}

		for _, address := range addresses {
			pod, ok := podsByIP[address.IP]
			if !ok {
				framework.Failf("Unexpected address with IP: %s", address.IP)
			}

			ensurePodTargetRef(pod, address.TargetRef)

			// If a named port, the subset ports should directly correspond with
			// each individual pod.
			if namedPort {
				container := pod.Spec.Containers[0]
				for _, port := range container.Ports {
					if port.Name == svc.Spec.Ports[0].TargetPort.String() {
						subsetPort := subset.Ports[0]
						if subsetPort.Port != port.ContainerPort {
							framework.Failf("Expected subset port to be %d, got %d", port.ContainerPort, subsetPort.Port)
						}
						if subsetPort.Name != svc.Spec.Ports[0].Name {
							framework.Failf("Expected subset port name to be %s, got %s", svc.Spec.Ports[0].Name, subsetPort.Name)
						}
					}
				}
			}
		}
	}

	if len(pods) != totalEndpointAddresses {
		framework.Failf("Expected %d addresses, got %d", len(pods), totalEndpointAddresses)
	}

	if len(pods) == 0 && len(endpointSlices) != 1 {
		framework.Failf("Expected 1 EndpointSlice, got %d", len(endpointSlices))
	}

	totalEndpointSliceAddresses := 0
	for _, endpointSlice := range endpointSlices {
		totalEndpointSliceAddresses += len(endpointSlice.Endpoints)
		if len(pods) == 0 && len(endpointSlice.Ports) != 0 {
			framework.Failf("Expected EndpointSlice to have 0 ports, got %d", len(endpointSlice.Ports))
		}
		if len(pods) > 0 && len(endpointSlice.Ports) != len(svc.Spec.Ports) {
			framework.Failf("Expected EndpointSlice to have %d ports, got %d", len(svc.Spec.Ports), len(endpointSlice.Ports))
		}

		// If not a named port, the EndpointSlice ports should directly
		// correspond with the Service ports.
		if !namedPort {
			for i, esPort := range endpointSlice.Ports {
				svcPort := svc.Spec.Ports[i]
				if *esPort.Name != svcPort.Name {
					framework.Failf("Expected port name to be %s, got %s", svcPort.Name, *esPort.Name)
				}
				if *esPort.Protocol != svcPort.Protocol {
					framework.Failf("Expected protocol to be %s, got %s", svcPort.Protocol, *esPort.Protocol)
				}
				if *esPort.Port != svcPort.TargetPort.IntVal {
					framework.Failf("Expected port to be %d, got %d", svcPort.TargetPort.IntVal, *esPort.Port)
				}
			}
		}

		for _, endpoint := range endpointSlice.Endpoints {
			if len(endpoint.Addresses) == 0 {
				framework.Failf("Expected EndpointSlice endpoint to have at least 1 address")
			}
			pod, ok := podsByIP[endpoint.Addresses[0]]
			if !ok {
				framework.Failf("Unexpected address with IP: %s", endpoint.Addresses[0])
			}

			ensurePodTargetRef(pod, endpoint.TargetRef)

			// If a named port, the EndpointSlice ports should directly
			// correspond with each individual pod.
			if namedPort {
				container := pod.Spec.Containers[0]
				for _, port := range container.Ports {
					if port.Name == svc.Spec.Ports[0].TargetPort.String() {
						esPort := endpointSlice.Ports[0]
						if *esPort.Port != port.ContainerPort {
							framework.Failf("Expected EndpointSlice port to be %d, got %d", port.ContainerPort, *esPort.Port)
						}
						if *esPort.Name != svc.Spec.Ports[0].Name {
							framework.Failf("Expected EndpointSlice port name to be %s, got %s", svc.Spec.Ports[0].Name, *esPort.Name)
						}
					}
				}
			}
		}
	}

	if len(pods) != totalEndpointSliceAddresses {
		framework.Failf("Expected %d addresses, got %d", len(pods), totalEndpointSliceAddresses)
	}
}

// hasMatchingEndpointSlices returns any EndpointSlices that match the
// conditions along with a boolean indicating if all the conditions have been
// met.
func hasMatchingEndpointSlices(cs clientset.Interface, ns, svcName string, numEndpoints, numSlices int) ([]discoveryv1alpha1.EndpointSlice, bool) {
	listOptions := metav1.ListOptions{LabelSelector: fmt.Sprintf("%s=%s", discoveryv1alpha1.LabelServiceName, svcName)}
	esList, err := cs.DiscoveryV1alpha1().EndpointSlices(ns).List(listOptions)
	framework.ExpectNoError(err, "Error fetching EndpointSlice for %s/%s Service", ns, svcName)

	if len(esList.Items) == 0 {
		framework.Logf("EndpointSlice for %s/%s Service not found", ns, svcName)
		return []discoveryv1alpha1.EndpointSlice{}, false
	}
	if len(esList.Items) != numSlices {
		framework.Logf("Expected %d EndpointSlices for %s/%s Service, got %d", numSlices, ns, svcName, len(esList.Items))
		return []discoveryv1alpha1.EndpointSlice{}, false
	}

	actualNumEndpoints := 0
	for _, endpointSlice := range esList.Items {
		actualNumEndpoints += len(endpointSlice.Endpoints)
	}
	if actualNumEndpoints != numEndpoints {
		framework.Logf("EndpointSlices for %s/%s Service have %d/%d endpoints", ns, svcName, actualNumEndpoints, numEndpoints)
		return []discoveryv1alpha1.EndpointSlice{}, false
	}

	return esList.Items, true
}

// hasMatchingEndpoints returns any Endpoints that match the conditions along
// with a boolean indicating if all the conditions have been met.
func hasMatchingEndpoints(cs clientset.Interface, ns, svcName string, numIPs, numSubsets int) (*v1.Endpoints, bool) {
	endpoints, err := cs.CoreV1().Endpoints(ns).Get(svcName, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			framework.Logf("Endpoints for %s/%s Service not found", ns, svcName)
			return nil, false
		}
		framework.ExpectNoError(err, "Error fetching Endpoints for %s/%s Service", ns, svcName)
	}
	if len(endpoints.Subsets) != numSubsets {
		framework.Logf("Endpoints for %s/%s Service with %d/%d Subsets", ns, svcName, len(endpoints.Subsets), numSubsets)
		return nil, false
	}

	actualNumIPs := 0
	for _, endpointSubset := range endpoints.Subsets {
		actualNumIPs += len(endpointSubset.Addresses) + len(endpointSubset.NotReadyAddresses)
	}
	if actualNumIPs != numIPs {
		framework.Logf("Endpoints for %s/%s Service with %d/%d IPs", ns, svcName, actualNumIPs, numIPs)
		return nil, false
	}

	return endpoints, true
}

// ensurePodTargetRef ensures that a Pod matches the provided target reference.
func ensurePodTargetRef(pod *v1.Pod, targetRef *v1.ObjectReference) {
	if targetRef == nil {
		framework.Failf("Expected TargetRef to not be nil")
	}
	if targetRef.Kind != "Pod" {
		framework.Failf("Expected TargetRef.Kind to be Pod, got %s", targetRef.Kind)
	}
	if targetRef.Namespace != pod.Namespace {
		framework.Failf("Expected TargetRef.Namespace to be %s, got %s", pod.Namespace, targetRef.Namespace)
	}
	if targetRef.Name != pod.Name {
		framework.Failf("Expected TargetRef.Name to be %s, got %s", pod.Name, targetRef.Name)
	}
	if targetRef.UID != pod.UID {
		framework.Failf("Expected TargetRef.UID to be %s, got %s", pod.UID, targetRef.UID)
	}
}

// createServiceReportErr creates a Service and reports any associated error.
func createServiceReportErr(cs clientset.Interface, ns string, service *v1.Service) *v1.Service {
	svc, err := cs.CoreV1().Services(ns).Create(service)
	framework.ExpectNoError(err)
	return svc
}
