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
	"encoding/json"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	"k8s.io/apimachinery/pkg/util/wait"
	watch "k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	apimachineryutils "k8s.io/kubernetes/test/e2e/common/apimachinery"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/network/common"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = common.SIGDescribe("Endpoints", func() {
	f := framework.NewDefaultFramework("endpoints")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	/*
	   Release: v1.19
	   Testname: Endpoint resource lifecycle
	   Description: Create an endpoint, the endpoint MUST exist.
	   The endpoint is updated with a new label, a check after the update MUST find the changes.
	   The endpoint is then patched with a new IPv4 address and port, a check after the patch MUST the changes.
	   The endpoint is deleted by its label, a watch listens for the deleted watch event.
	*/
	framework.ConformanceIt("should test the lifecycle of an Endpoint", func(ctx context.Context) {
		testNamespaceName := f.Namespace.Name
		testEndpointName := "testservice"
		testEndpoints := v1.Endpoints{
			ObjectMeta: metav1.ObjectMeta{
				Name: testEndpointName,
				Labels: map[string]string{
					"test-endpoint-static": "true",
				},
			},
			Subsets: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{
					IP: "10.0.0.24",
				}},
				Ports: []v1.EndpointPort{{
					Name:     "http",
					Port:     80,
					Protocol: v1.ProtocolTCP,
				}},
			}},
		}
		w := &cache.ListWatch{
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				options.LabelSelector = "test-endpoint-static=true"
				return f.ClientSet.CoreV1().Endpoints(testNamespaceName).Watch(ctx, options)
			},
		}
		endpointsList, err := f.ClientSet.CoreV1().Endpoints("").List(ctx, metav1.ListOptions{LabelSelector: "test-endpoint-static=true"})
		framework.ExpectNoError(err, "failed to list Endpoints")

		ginkgo.By("creating an Endpoint")
		createdEP, err := f.ClientSet.CoreV1().Endpoints(testNamespaceName).Create(ctx, &testEndpoints, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create Endpoint")
		gomega.Expect(createdEP).To(apimachineryutils.HaveValidResourceVersion())
		ginkgo.By("waiting for available Endpoint")
		ctxUntil, cancel := context.WithTimeout(ctx, 30*time.Second)
		defer cancel()
		_, err = watchtools.Until(ctxUntil, endpointsList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			switch event.Type {
			case watch.Added:
				if endpoints, ok := event.Object.(*v1.Endpoints); ok {
					found := endpoints.ObjectMeta.Name == endpoints.Name &&
						endpoints.Labels["test-endpoint-static"] == "true"
					return found, nil
				}
			default:
				framework.Logf("observed event type %v", event.Type)
			}
			return false, nil
		})
		framework.ExpectNoError(err, "failed to see %v event", watch.Added)

		ginkgo.By("listing all Endpoints")
		endpointsList, err = f.ClientSet.CoreV1().Endpoints("").List(ctx, metav1.ListOptions{LabelSelector: "test-endpoint-static=true"})
		framework.ExpectNoError(err, "failed to list Endpoints")
		eventFound := false
		var foundEndpoint v1.Endpoints
		for _, endpoint := range endpointsList.Items {
			if endpoint.ObjectMeta.Name == testEndpointName && endpoint.ObjectMeta.Namespace == testNamespaceName {
				eventFound = true
				foundEndpoint = endpoint
				break
			}
		}
		if !eventFound {
			framework.Fail("unable to find Endpoint Service in list of Endpoints")
		}

		ginkgo.By("updating the Endpoint")
		foundEndpoint.ObjectMeta.Labels["test-service"] = "updated"
		_, err = f.ClientSet.CoreV1().Endpoints(testNamespaceName).Update(ctx, &foundEndpoint, metav1.UpdateOptions{})
		framework.ExpectNoError(err, "failed to update Endpoint with new label")

		ctxUntil, cancel = context.WithTimeout(ctx, 30*time.Second)
		defer cancel()
		_, err = watchtools.Until(ctxUntil, endpointsList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			switch event.Type {
			case watch.Modified:
				if endpoints, ok := event.Object.(*v1.Endpoints); ok {
					found := endpoints.ObjectMeta.Name == endpoints.Name &&
						endpoints.Labels["test-endpoint-static"] == "true"
					return found, nil
				}
			default:
				framework.Logf("observed event type %v", event.Type)
			}
			return false, nil
		})
		framework.ExpectNoError(err, "failed to see %v event", watch.Modified)

		ginkgo.By("fetching the Endpoint")
		endpoints, err := f.ClientSet.CoreV1().Endpoints(testNamespaceName).Get(ctx, testEndpointName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to fetch Endpoint")
		gomega.Expect(foundEndpoint.ObjectMeta.Labels).To(gomega.HaveKeyWithValue("test-service", "updated"), "failed to update Endpoint %v in namespace %v label not updated", testEndpointName, testNamespaceName)

		endpointPatch, err := json.Marshal(map[string]interface{}{
			"metadata": map[string]interface{}{
				"labels": map[string]string{
					"test-service": "patched",
				},
			},
			"subsets": []map[string]interface{}{
				{
					"addresses": []map[string]string{
						{
							"ip": "10.0.0.25",
						},
					},
					"ports": []map[string]interface{}{
						{
							"name": "http-test",
							"port": int32(8080),
						},
					},
				},
			},
		})
		framework.ExpectNoError(err, "failed to marshal JSON for WatchEvent patch")
		ginkgo.By("patching the Endpoint")
		patchedEP, err := f.ClientSet.CoreV1().Endpoints(testNamespaceName).Patch(ctx, testEndpointName, types.StrategicMergePatchType, []byte(endpointPatch), metav1.PatchOptions{})
		framework.ExpectNoError(err, "failed to patch Endpoint")
		gomega.Expect(resourceversion.CompareResourceVersion(createdEP.ResourceVersion, patchedEP.ResourceVersion)).To(gomega.BeNumerically("==", -1), "patched object should have a larger resource version")

		ctxUntil, cancel = context.WithTimeout(ctx, 30*time.Second)
		defer cancel()
		_, err = watchtools.Until(ctxUntil, endpoints.ResourceVersion, w, func(event watch.Event) (bool, error) {
			switch event.Type {
			case watch.Modified:
				if endpoints, ok := event.Object.(*v1.Endpoints); ok {
					found := endpoints.ObjectMeta.Name == endpoints.Name &&
						endpoints.Labels["test-endpoint-static"] == "true"
					return found, nil
				}
			default:
				framework.Logf("observed event type %v", event.Type)
			}
			return false, nil
		})
		framework.ExpectNoError(err, "failed to see %v event", watch.Modified)

		ginkgo.By("fetching the Endpoint")
		endpoints, err = f.ClientSet.CoreV1().Endpoints(testNamespaceName).Get(ctx, testEndpointName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to fetch Endpoint")
		gomega.Expect(endpoints.ObjectMeta.Labels).To(gomega.HaveKeyWithValue("test-service", "patched"), "failed to patch Endpoint with Label")
		endpointSubsetOne := endpoints.Subsets[0]
		endpointSubsetOneAddresses := endpointSubsetOne.Addresses[0]
		endpointSubsetOnePorts := endpointSubsetOne.Ports[0]
		gomega.Expect(endpointSubsetOneAddresses.IP).To(gomega.Equal("10.0.0.25"), "failed to patch Endpoint")
		gomega.Expect(endpointSubsetOnePorts.Name).To(gomega.Equal("http-test"), "failed to patch Endpoint")
		gomega.Expect(endpointSubsetOnePorts.Port).To(gomega.Equal(int32(8080)), "failed to patch Endpoint")

		ginkgo.By("deleting the Endpoint by Collection")
		err = f.ClientSet.CoreV1().Endpoints(testNamespaceName).DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: "test-endpoint-static=true"})
		framework.ExpectNoError(err, "failed to delete Endpoint by Collection")

		ginkgo.By("waiting for Endpoint deletion")
		ctxUntil, cancel = context.WithTimeout(ctx, 30*time.Second)
		defer cancel()
		_, err = watchtools.Until(ctxUntil, endpoints.ResourceVersion, w, func(event watch.Event) (bool, error) {
			switch event.Type {
			case watch.Deleted:
				if endpoints, ok := event.Object.(*v1.Endpoints); ok {
					found := endpoints.ObjectMeta.Name == endpoints.Name &&
						endpoints.Labels["test-endpoint-static"] == "true"
					return found, nil
				}
			default:
				framework.Logf("observed event type %v", event.Type)
			}
			return false, nil
		})
		framework.ExpectNoError(err, "failed to see %v event", watch.Deleted)

		ginkgo.By("fetching the Endpoint")
		_, err = f.ClientSet.CoreV1().Endpoints(testNamespaceName).Get(ctx, testEndpointName, metav1.GetOptions{})
		gomega.Expect(err).To(gomega.HaveOccurred(), "should not be able to fetch Endpoint")
	})
})

var _ = common.SIGDescribe("EndpointsController", func() {
	f := framework.NewDefaultFramework("endpoints")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	var cs clientset.Interface
	var podClient *e2epod.PodClient

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
		podClient = e2epod.NewPodClient(f)
	})

	/*
		Release: v1.21, v1.35
		Testname: Endpoints, "empty" Service
		Description: The Endpoints controller should create and delete an empty Endpoints for a Service that matches no pods.
	*/
	framework.ConformanceIt("should create and delete Endpoints for a Service with a selector that matches no pods", func(ctx context.Context) {
		svc := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: "example-empty-selector",
			},
			Spec: v1.ServiceSpec{
				Selector: map[string]string{
					"does-not-match-anything": "endpoints-and-endpoint-slices-should-still-be-created",
				},
				Ports: []v1.ServicePort{{
					Name:     "example",
					Port:     80,
					Protocol: v1.ProtocolTCP,
				}},
			},
		}
		svc, err := cs.CoreV1().Services(f.Namespace.Name).Create(ctx, svc, metav1.CreateOptions{})
		framework.ExpectNoError(err, "error creating Service")

		// Expect Endpoints resource to be created.
		if err := wait.PollUntilContextTimeout(ctx, 2*time.Second, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
			_, err := cs.CoreV1().Endpoints(svc.Namespace).Get(ctx, svc.Name, metav1.GetOptions{})
			if err != nil {
				return false, nil
			}
			return true, nil
		}); err != nil {
			framework.Failf("No Endpoints found for Service %s/%s: %s", svc.Namespace, svc.Name, err)
		}

		err = cs.CoreV1().Services(svc.Namespace).Delete(ctx, svc.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "error deleting Service")

		// Expect Endpoints resource to be deleted when Service is.
		if err := wait.PollUntilContextTimeout(ctx, 2*time.Second, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
			_, err := cs.CoreV1().Endpoints(svc.Namespace).Get(ctx, svc.Name, metav1.GetOptions{})
			if err != nil {
				if apierrors.IsNotFound(err) {
					return true, nil
				}
				return false, err
			}
			return false, nil
		}); err != nil {
			framework.Failf("Endpoints resource not deleted after Service %s/%s was deleted: %s", svc.Namespace, svc.Name, err)
		}
	})

	/*
		Release: v1.21, v1.35
		Testname: Endpoints, creation/deletion
		Description: The Endpoints controller must create an Endpoints for Pods matching a Service.
	*/
	framework.ConformanceIt("should create Endpoints for Pods matching a Service", func(ctx context.Context) {
		labelPod1 := "pod1"
		labelPod2 := "pod2"
		labelShared12 := "shared12"
		labelValue := "on"

		pod1 := podClient.Create(ctx, &v1.Pod{
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

		pod2 := podClient.Create(ctx, &v1.Pod{
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

		svc1 := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: "example-int-port",
			},
			Spec: v1.ServiceSpec{
				Selector:                 map[string]string{labelPod1: labelValue},
				PublishNotReadyAddresses: true,
				Ports: []v1.ServicePort{{
					Name:       "example",
					Port:       80,
					TargetPort: intstr.FromInt32(3000),
					Protocol:   v1.ProtocolTCP,
				}},
			},
		}
		svc1, err := cs.CoreV1().Services(f.Namespace.Name).Create(ctx, svc1, metav1.CreateOptions{})
		framework.ExpectNoError(err, "error creating Service")

		svc2 := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: "example-named-port",
			},
			Spec: v1.ServiceSpec{
				Selector:                 map[string]string{labelShared12: labelValue},
				PublishNotReadyAddresses: true,
				Ports: []v1.ServicePort{{
					Name:       "http",
					Port:       80,
					TargetPort: intstr.FromString("example-name"),
					Protocol:   v1.ProtocolTCP,
				}},
			},
		}
		svc2, err = cs.CoreV1().Services(f.Namespace.Name).Create(ctx, svc2, metav1.CreateOptions{})
		framework.ExpectNoError(err, "error creating Service")

		err = wait.PollUntilContextTimeout(ctx, 2*time.Second, 3*time.Minute, true, func(ctx context.Context) (bool, error) {
			var err error
			pod1, err = podClient.Get(ctx, pod1.Name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			if len(pod1.Status.PodIPs) == 0 {
				return false, nil
			}

			pod2, err = podClient.Get(ctx, pod2.Name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			if len(pod2.Status.PodIPs) == 0 {
				return false, nil
			}

			return true, nil
		})
		framework.ExpectNoError(err, "timed out waiting for Pods to have IPs assigned")

		ginkgo.By("referencing a single matching pod")
		expectEndpoints(ctx, cs, f.Namespace.Name, svc1, []*v1.Pod{pod1}, 1, false)

		ginkgo.By("referencing matching pods with named port")
		expectEndpoints(ctx, cs, f.Namespace.Name, svc2, []*v1.Pod{pod1, pod2}, 2, true)
	})
})

// expectEndpoints verifies that Endpoints exist for a given Service and Namespace with
// the appropriate attributes set. This is a relatively complex function as the order of
// attributes or resources is not necessarily consistent. It is used as a helper function
// for the tests above and takes some shortcuts with the assumption that those test cases
// will be the only caller of this function.
func expectEndpoints(ctx context.Context, cs clientset.Interface, ns string, svc *v1.Service, pods []*v1.Pod, numSubsets int, namedPort bool) {
	endpoints := &v1.Endpoints{}
	if err := wait.PollUntilContextTimeout(ctx, 2*time.Second, 2*time.Minute, true, func(ctx context.Context) (bool, error) {
		endpointsFound, hasMatchingEndpoints := hasMatchingEndpoints(ctx, cs, ns, svc.Name, len(pods), numSubsets)
		if !hasMatchingEndpoints {
			framework.Logf("Matching Endpoints not found")
			return false, nil
		}
		endpoints = endpointsFound
		return true, nil
	}); err != nil {
		framework.Failf("Timed out waiting for Endpoints to match expectations: %v", err)
	}

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
		addresses := []v1.EndpointAddress{}
		addresses = append(addresses, subset.Addresses...)
		addresses = append(addresses, subset.NotReadyAddresses...)
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
}

// hasMatchingEndpoints returns any Endpoints that match the conditions along
// with a boolean indicating if all the conditions have been met.
func hasMatchingEndpoints(ctx context.Context, cs clientset.Interface, ns, svcName string, numIPs, numSubsets int) (*v1.Endpoints, bool) {
	endpoints, err := cs.CoreV1().Endpoints(ns).Get(ctx, svcName, metav1.GetOptions{})
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
