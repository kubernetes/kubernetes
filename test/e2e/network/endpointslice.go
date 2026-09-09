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
	"fmt"
	"net"
	"time"

	v1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
	apimachineryutils "k8s.io/kubernetes/test/e2e/common/apimachinery"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/network/common"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = common.SIGDescribe("EndpointSlice", func() {
	f := framework.NewDefaultFramework("endpointslice")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var cs clientset.Interface
	var podClient *e2epod.PodClient

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
		podClient = e2epod.NewPodClient(f)
	})

	/*
		Release: v1.21, v1.35
		Testname: EndpointSlice, "empty" Service
		Description: The EndpointSlice controller should create and delete empty EndpointSlices for a Service that matches no pods.
	*/
	framework.ConformanceIt("should create and delete EndpointSlices for a Service with a selector that matches no pods", func(ctx context.Context) {
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

		// Expect EndpointSlice resource to be created.
		var endpointSlice discoveryv1.EndpointSlice
		if err := wait.PollUntilContextTimeout(ctx, 2*time.Second, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
			endpointSliceList, err := cs.DiscoveryV1().EndpointSlices(svc.Namespace).List(ctx, metav1.ListOptions{
				LabelSelector: "kubernetes.io/service-name=" + svc.Name,
			})
			if err != nil {
				return false, err
			}
			if len(endpointSliceList.Items) == 0 {
				return false, nil
			}
			endpointSlice = endpointSliceList.Items[0]
			return true, nil
		}); err != nil {
			framework.Failf("No EndpointSlice found for Service %s/%s: %s", svc.Namespace, svc.Name, err)
		}

		// Ensure EndpointSlice has expected values.
		managedBy, ok := endpointSlice.Labels[discoveryv1.LabelManagedBy]
		expectedManagedBy := "endpointslice-controller.k8s.io"
		if !ok {
			framework.Failf("Expected EndpointSlice to have %s label, got %#v", discoveryv1.LabelManagedBy, endpointSlice.Labels)
		} else if managedBy != expectedManagedBy {
			framework.Failf("Expected EndpointSlice to have %s label with %s value, got %s", discoveryv1.LabelManagedBy, expectedManagedBy, managedBy)
		}
		if len(endpointSlice.Endpoints) != 0 {
			framework.Failf("Expected EndpointSlice to have 0 endpoints, got %d: %#v", len(endpointSlice.Endpoints), endpointSlice.Endpoints)
		}

		err = cs.CoreV1().Services(svc.Namespace).Delete(ctx, svc.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "error deleting Service")

		// Expect EndpointSlice resource to be deleted when Service is. Wait for
		// up to 90 seconds since garbage collector only polls every 30 seconds
		// and may need to retry informer resync at some point during an e2e
		// run.
		if err := wait.PollUntilContextTimeout(ctx, 2*time.Second, 90*time.Second, true, func(ctx context.Context) (bool, error) {
			endpointSliceList, err := cs.DiscoveryV1().EndpointSlices(svc.Namespace).List(ctx, metav1.ListOptions{
				LabelSelector: "kubernetes.io/service-name=" + svc.Name,
			})
			if err != nil {
				return false, err
			}
			if len(endpointSliceList.Items) == 0 {
				return true, nil
			}
			return false, nil
		}); err != nil {
			framework.Failf("EndpointSlice resource not deleted after Service %s/%s was deleted: %s", svc.Namespace, svc.Name, err)
		}
	})

	/*
		Release: v1.21, v1.35
		Testname: EndpointSlice, creation/deletion
		Description: The endpointslice controller must create and delete EndpointSlices for Pods matching a Service.
	*/
	framework.ConformanceIt("should create Endpoints and EndpointSlices for Pods matching a Service", func(ctx context.Context) {
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
		expectEndpointSlices(ctx, cs, f.Namespace.Name, svc1, []*v1.Pod{pod1}, 1, false)

		ginkgo.By("referencing matching pods with named port")
		expectEndpointSlices(ctx, cs, f.Namespace.Name, svc2, []*v1.Pod{pod1, pod2}, 2, true)

		ginkgo.By("recreating EndpointSlices after they've been deleted")
		deleteEndpointSlices(ctx, cs, f.Namespace.Name, svc2)
		expectEndpointSlices(ctx, cs, f.Namespace.Name, svc2, []*v1.Pod{pod1, pod2}, 2, true)
	})

	/*
		Release: v1.21
		Testname: EndpointSlice API
		Description: The discovery.k8s.io API group MUST exist in the /apis discovery document.
		The discovery.k8s.io/v1 API group/version MUST exist in the /apis/discovery.k8s.io discovery document.
		The endpointslices resource MUST exist in the /apis/discovery.k8s.io/v1 discovery document.
		The endpointslices resource must support create, get, list, watch, update, patch, delete, and deletecollection.
	*/
	framework.ConformanceIt("should support creating EndpointSlice API operations", func(ctx context.Context) {
		// Setup
		ns := f.Namespace.Name
		epsVersion := "v1"
		epsClient := f.ClientSet.DiscoveryV1().EndpointSlices(ns)

		epsTemplate := &discoveryv1.EndpointSlice{
			ObjectMeta: metav1.ObjectMeta{GenerateName: "e2e-example-ing",
				Labels: map[string]string{
					"special-label": f.UniqueName,
				}},
			AddressType: discoveryv1.AddressTypeIPv4,
			Endpoints: []discoveryv1.Endpoint{
				{Addresses: []string{"1.2.3.4", "5.6.7.8"}},
				{Addresses: []string{"2.2.3.4", "6.6.7.8"}},
			},
		}
		// Discovery
		ginkgo.By("getting /apis")
		{
			discoveryGroups, err := f.ClientSet.Discovery().ServerGroups()
			framework.ExpectNoError(err)
			found := false
			for _, group := range discoveryGroups.Groups {
				if group.Name == discoveryv1.GroupName {
					for _, version := range group.Versions {
						if version.Version == epsVersion {
							found = true
							break
						}
					}
				}
			}
			if !found {
				framework.Failf("expected discovery API group/version, got %#v", discoveryGroups.Groups)
			}
		}

		ginkgo.By("getting /apis/discovery.k8s.io")
		{
			group := &metav1.APIGroup{}
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/discovery.k8s.io").Do(ctx).Into(group)
			framework.ExpectNoError(err)
			found := false
			for _, version := range group.Versions {
				if version.Version == epsVersion {
					found = true
					break
				}
			}
			if !found {
				framework.Failf("expected discovery API version, got %#v", group.Versions)
			}
		}

		ginkgo.By("getting /apis/discovery.k8s.io" + epsVersion)
		{
			resources, err := f.ClientSet.Discovery().ServerResourcesForGroupVersion(discoveryv1.SchemeGroupVersion.String())
			framework.ExpectNoError(err)
			foundEPS := false
			for _, resource := range resources.APIResources {
				switch resource.Name {
				case "endpointslices":
					foundEPS = true
				}
			}
			if !foundEPS {
				framework.Failf("expected endpointslices, got %#v", resources.APIResources)
			}
		}

		// EndpointSlice resource create/read/update/watch verbs
		ginkgo.By("creating")
		_, err := epsClient.Create(ctx, epsTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		_, err = epsClient.Create(ctx, epsTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		createdEPS, err := epsClient.Create(ctx, epsTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("getting")
		queriedEPS, err := epsClient.Get(ctx, createdEPS.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(queriedEPS.UID).To(gomega.Equal(createdEPS.UID))
		gomega.Expect(queriedEPS).To(apimachineryutils.HaveValidResourceVersion())

		ginkgo.By("listing")
		epsList, err := epsClient.List(ctx, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		gomega.Expect(epsList.Items).To(gomega.HaveLen(3), "filtered list should have 3 items")

		ginkgo.By("watching")
		framework.Logf("starting watch")
		epsWatch, err := epsClient.Watch(ctx, metav1.ListOptions{ResourceVersion: epsList.ResourceVersion, LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)

		// Test cluster-wide list and watch
		clusterEPSClient := f.ClientSet.DiscoveryV1().EndpointSlices("")
		ginkgo.By("cluster-wide listing")
		clusterEPSList, err := clusterEPSClient.List(ctx, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		gomega.Expect(clusterEPSList.Items).To(gomega.HaveLen(3), "filtered list should have 3 items")

		ginkgo.By("cluster-wide watching")
		framework.Logf("starting watch")
		_, err = clusterEPSClient.Watch(ctx, metav1.ListOptions{ResourceVersion: epsList.ResourceVersion, LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)

		ginkgo.By("patching")
		patchedEPS, err := epsClient.Patch(ctx, createdEPS.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"patched":"true"}}}`), metav1.PatchOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(patchedEPS.Annotations).To(gomega.HaveKeyWithValue("patched", "true"), "patched object should have the applied annotation")
		gomega.Expect(resourceversion.CompareResourceVersion(createdEPS.ResourceVersion, patchedEPS.ResourceVersion)).To(gomega.BeNumerically("==", -1), "patched object should have a larger resource version")

		ginkgo.By("updating")
		var epsToUpdate, updatedEPS *discoveryv1.EndpointSlice
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			epsToUpdate, err = epsClient.Get(ctx, createdEPS.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			epsToUpdate.Annotations["updated"] = "true"
			updatedEPS, err = epsClient.Update(ctx, epsToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err)
		gomega.Expect(updatedEPS.Annotations).To(gomega.HaveKeyWithValue("updated", "true"), "updated object should have the applied annotation")

		framework.Logf("waiting for watch events with expected annotations")
		for sawAnnotations := false; !sawAnnotations; {
			select {
			case evt, ok := <-epsWatch.ResultChan():
				if !ok {
					framework.Fail("watch channel should not close")
				}
				gomega.Expect(evt.Type).To(gomega.Equal(watch.Modified))
				watchedEPS, isEPS := evt.Object.(*discoveryv1.EndpointSlice)
				if !isEPS {
					framework.Failf("expected EndpointSlice, got %T", evt.Object)
				}
				if watchedEPS.Annotations["patched"] == "true" {
					framework.Logf("saw patched and updated annotations")
					sawAnnotations = true
					epsWatch.Stop()
				} else {
					framework.Logf("missing expected annotations, waiting: %#v", watchedEPS.Annotations)
				}
			case <-time.After(wait.ForeverTestTimeout):
				framework.Fail("timed out waiting for watch event")
			}
		}

		ginkgo.By("deleting")

		err = epsClient.Delete(ctx, createdEPS.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		_, err = epsClient.Get(ctx, createdEPS.Name, metav1.GetOptions{})
		if !apierrors.IsNotFound(err) {
			framework.Failf("expected 404, got %v", err)
		}
		epsList, err = epsClient.List(ctx, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		gomega.Expect(epsList.Items).To(gomega.HaveLen(2), "filtered list should have 2 items")
		for _, eps := range epsList.Items {
			if eps.Namespace == createdEPS.Namespace && eps.Name == createdEPS.Name {
				framework.Fail("listing after deleting createdEPS")
			}
		}

		ginkgo.By("deleting a collection")
		err = epsClient.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		epsList, err = epsClient.List(ctx, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		gomega.Expect(epsList.Items).To(gomega.BeEmpty(), "filtered list should have 0 items")
	})

	/*
		Release: v1.34
		Testname: EndpointSlice, single IP, multiple ports
		Description: Given a selector-less Service with multiple manually-created
		EndpointSlices (and no Endpoints) where the endpoints have the same IP
		but different Ports, the service proxy MUST allow connections to both ports.
	*/
	framework.ConformanceIt("should support a Service with multiple ports specified in multiple EndpointSlices", func(ctx context.Context) {
		ns := f.Namespace.Name
		svc := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: "example-custom-endpoints",
			},
			Spec: v1.ServiceSpec{
				Ports: []v1.ServicePort{
					{
						Name:     "port80",
						Port:     80,
						Protocol: v1.ProtocolTCP,
					},
					{
						Name:     "port81",
						Port:     81,
						Protocol: v1.ProtocolTCP,
					},
				},
			},
		}
		svc, err := cs.CoreV1().Services(ns).Create(ctx, svc, metav1.CreateOptions{})
		framework.ExpectNoError(err, "error creating Service")

		// Add a backend pod to the service in the other node
		port8090 := []v1.ContainerPort{
			{
				ContainerPort: 8090,
				Protocol:      v1.ProtocolTCP,
			},
		}
		port9090 := []v1.ContainerPort{
			{
				ContainerPort: 9090,
				Protocol:      v1.ProtocolTCP,
			},
		}

		serverPod := e2epod.NewAgnhostPodFromContainers(
			"", "pod-handle-http-request", nil,
			e2epod.NewAgnhostContainer("container-handle-8090-request", nil, port8090, "netexec", "--http-port", "8090", "--udp-port", "-1"),
			e2epod.NewAgnhostContainer("container-handle-9090-request", nil, port9090, "netexec", "--http-port", "9090", "--udp-port", "-1"),
		)

		pod := e2epod.NewPodClient(f).CreateSync(ctx, serverPod)

		if pod.Status.PodIP == "" {
			framework.Failf("PodIP not assigned for pod %s", pod.Name)
		}

		addressType := discoveryv1.AddressTypeIPv4
		if framework.TestContext.ClusterIsIPv6() {
			addressType = discoveryv1.AddressTypeIPv6
		}

		// create custom endpoint slices
		tcpProtocol := v1.ProtocolTCP
		readyCondTrue := true
		epsTemplate := &discoveryv1.EndpointSlice{
			ObjectMeta: metav1.ObjectMeta{GenerateName: "e2e-custom-slice",
				Labels: map[string]string{
					discoveryv1.LabelServiceName: svc.Name,
					discoveryv1.LabelManagedBy:   "e2e-test" + ns,
				}},
			AddressType: addressType,
			Endpoints: []discoveryv1.Endpoint{
				{
					Addresses:  []string{pod.Status.PodIP},
					Conditions: discoveryv1.EndpointConditions{Ready: &readyCondTrue},
				},
			},
		}

		ginkgo.By("creating")
		eps1 := epsTemplate.DeepCopy()
		eps1.Ports = []discoveryv1.EndpointPort{{
			Name:     ptr.To("port80"),
			Port:     ptr.To[int32](8090),
			Protocol: &tcpProtocol,
		}}

		_, err = f.ClientSet.DiscoveryV1().EndpointSlices(ns).Create(ctx, eps1, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		eps2 := epsTemplate.DeepCopy()
		eps2.Ports = []discoveryv1.EndpointPort{{
			Name:     ptr.To("port81"),
			Port:     ptr.To[int32](9090),
			Protocol: &tcpProtocol,
		}}

		_, err = f.ClientSet.DiscoveryV1().EndpointSlices(ns).Create(ctx, eps2, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		// connect to the service must work
		ginkgo.By("Creating a pause pods that will try to connect to the webserver")
		pausePod0 := e2epod.NewAgnhostPod(ns, "pause-pod-0", nil, nil, nil)
		e2epod.NewPodClient(f).CreateSync(ctx, pausePod0)

		dest1 := net.JoinHostPort(svc.Spec.ClusterIP, "80")
		dest2 := net.JoinHostPort(svc.Spec.ClusterIP, "81")
		execHostnameTest(*pausePod0, dest1, serverPod.Name)
		execHostnameTest(*pausePod0, dest2, serverPod.Name)

	})

	/*
		Release: v1.34
		Testname: EndpointSlice, multiple IPs, multiple ports
		Description: Given a selector-less Service with multiple manually-created
		EndpointSlices (and no Endpoints) where the endpoints have different IPs
		and different Ports, the service proxy MUST allow connections to both ports.
	*/
	framework.ConformanceIt("should support a Service with multiple endpoint IPs specified in multiple EndpointSlices", func(ctx context.Context) {
		ns := f.Namespace.Name
		svc := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: "example-custom-endpoints",
			},
			Spec: v1.ServiceSpec{
				Ports: []v1.ServicePort{
					{
						Name:     "port80",
						Port:     80,
						Protocol: v1.ProtocolTCP,
					},
					{
						Name:     "port81",
						Port:     81,
						Protocol: v1.ProtocolTCP,
					},
				},
			},
		}
		svc, err := cs.CoreV1().Services(ns).Create(ctx, svc, metav1.CreateOptions{})
		framework.ExpectNoError(err, "error creating Service")

		// Add a backend pod to the service in the other node
		port8090 := []v1.ContainerPort{
			{
				ContainerPort: 8090,
				Protocol:      v1.ProtocolTCP,
			},
		}

		serverPod1 := e2epod.NewAgnhostPodFromContainers(
			"", "pod1-handle-http-request", nil,
			e2epod.NewAgnhostContainer("container-handle-8090-request", nil, port8090, "netexec", "--http-port", "8090", "--udp-port", "-1"),
		)
		pod1 := e2epod.NewPodClient(f).CreateSync(ctx, serverPod1)

		if pod1.Status.PodIP == "" {
			framework.Failf("PodIP not assigned for pod %s", pod1.Name)
		}

		serverPod2 := e2epod.NewAgnhostPodFromContainers(
			"", "pod2-handle-http-request", nil,
			e2epod.NewAgnhostContainer("container-handle-8090-request", nil, port8090, "netexec", "--http-port", "8090", "--udp-port", "-1"),
		)
		pod2 := e2epod.NewPodClient(f).CreateSync(ctx, serverPod2)

		if pod2.Status.PodIP == "" {
			framework.Failf("PodIP not assigned for pod %s", pod2.Name)
		}

		addressType := discoveryv1.AddressTypeIPv4
		if framework.TestContext.ClusterIsIPv6() {
			addressType = discoveryv1.AddressTypeIPv6
		}

		// create custom endpoint slices
		tcpProtocol := v1.ProtocolTCP
		readyCondTrue := true
		epsTemplate := &discoveryv1.EndpointSlice{
			ObjectMeta: metav1.ObjectMeta{GenerateName: "e2e-custom-slice",
				Labels: map[string]string{
					discoveryv1.LabelServiceName: svc.Name,
					discoveryv1.LabelManagedBy:   "e2e-test" + ns,
				}},
			AddressType: addressType,
		}

		ginkgo.By("creating")
		eps1 := epsTemplate.DeepCopy()
		eps1.Endpoints = []discoveryv1.Endpoint{
			{
				Addresses:  []string{pod1.Status.PodIP},
				Conditions: discoveryv1.EndpointConditions{Ready: &readyCondTrue},
			},
		}
		eps1.Ports = []discoveryv1.EndpointPort{{
			Name:     ptr.To("port80"),
			Port:     ptr.To[int32](8090),
			Protocol: &tcpProtocol,
		}}

		_, err = f.ClientSet.DiscoveryV1().EndpointSlices(ns).Create(context.TODO(), eps1, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		eps2 := epsTemplate.DeepCopy()
		eps2.Endpoints = []discoveryv1.Endpoint{
			{
				Addresses:  []string{pod2.Status.PodIP},
				Conditions: discoveryv1.EndpointConditions{Ready: &readyCondTrue},
			},
		}
		eps2.Ports = []discoveryv1.EndpointPort{{
			Name:     ptr.To("port81"),
			Port:     ptr.To[int32](8090),
			Protocol: &tcpProtocol,
		}}
		_, err = f.ClientSet.DiscoveryV1().EndpointSlices(ns).Create(context.TODO(), eps2, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		// connect to the service must work
		ginkgo.By("Creating a pause pods that will try to connect to the webserver")
		pausePod0 := e2epod.NewAgnhostPod(ns, "pause-pod-0", nil, nil, nil)
		e2epod.NewPodClient(f).CreateSync(ctx, pausePod0)

		dest1 := net.JoinHostPort(svc.Spec.ClusterIP, "80")
		dest2 := net.JoinHostPort(svc.Spec.ClusterIP, "81")
		execHostnameTest(*pausePod0, dest1, serverPod1.Name)
		execHostnameTest(*pausePod0, dest2, serverPod2.Name)

	})

})

// expectEndpointSlices verifies that EndpointSlices exist for a given Service and
// Namespace with the appropriate attributes set. This is a relatively complex function as
// the order of attributes or resources is not necessarily consistent. It is used as a
// helper function for the tests above and takes some shortcuts with the assumption that
// those test cases will be the only caller of this function.
func expectEndpointSlices(ctx context.Context, cs clientset.Interface, ns string, svc *v1.Service, pods []*v1.Pod, numSlices int, namedPort bool) {
	endpointSlices := []discoveryv1.EndpointSlice{}
	if err := wait.PollUntilContextTimeout(ctx, 2*time.Second, 2*time.Minute, true, func(ctx context.Context) (bool, error) {
		endpointSlicesFound, hasMatchingSlices := hasMatchingEndpointSlices(ctx, cs, ns, svc.Name, len(pods), numSlices)
		if !hasMatchingSlices {
			return false, nil
		}
		endpointSlices = endpointSlicesFound
		return true, nil
	}); err != nil {
		framework.Failf("Timed out waiting for EndpointSlices to match expectations: %v", err)
	}

	podsByIP := map[string]*v1.Pod{}
	for _, pod := range pods {
		podsByIP[pod.Status.PodIP] = pod
		if len(pod.Spec.Containers) != 1 {
			framework.Failf("Expected pod to have 1 container, got %d", len(pod.Spec.Containers))
		}
	}

	if len(pods) == 0 && len(endpointSlices) != 1 {
		framework.Failf("Expected 1 EndpointSlice, got %d", len(endpointSlices))
	}

	// Use a set for deduping values. Duplicate addresses are technically valid
	// here although rare.
	esAddresses := sets.NewString()
	for _, endpointSlice := range endpointSlices {
		for _, endpoint := range endpointSlice.Endpoints {
			esAddresses.Insert(endpoint.Addresses[0])
		}
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

	if len(pods) != esAddresses.Len() {
		framework.Failf("Expected %d addresses, got %d", len(pods), esAddresses.Len())
	}
}

// deleteEndpointSlices deletes EndpointSlices for the specified Service.
func deleteEndpointSlices(ctx context.Context, cs clientset.Interface, ns string, svc *v1.Service) {
	listOptions := metav1.ListOptions{LabelSelector: fmt.Sprintf("%s=%s", discoveryv1.LabelServiceName, svc.Name)}
	esList, err := cs.DiscoveryV1().EndpointSlices(ns).List(ctx, listOptions)
	framework.ExpectNoError(err, "Error fetching EndpointSlices for %s/%s Service", ns, svc.Name)

	for _, endpointSlice := range esList.Items {
		err := cs.DiscoveryV1().EndpointSlices(ns).Delete(ctx, endpointSlice.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "Error deleting %s/%s EndpointSlice", ns, endpointSlice.Name)
	}
}

// hasMatchingEndpointSlices returns any EndpointSlices that match the
// conditions along with a boolean indicating if all the conditions have been
// met.
func hasMatchingEndpointSlices(ctx context.Context, cs clientset.Interface, ns, svcName string, numEndpoints, numSlices int) ([]discoveryv1.EndpointSlice, bool) {
	listOptions := metav1.ListOptions{LabelSelector: fmt.Sprintf("%s=%s", discoveryv1.LabelServiceName, svcName)}
	esList, err := cs.DiscoveryV1().EndpointSlices(ns).List(ctx, listOptions)
	framework.ExpectNoError(err, "Error fetching EndpointSlice for Service %s/%s", ns, svcName)

	if len(esList.Items) == 0 {
		framework.Logf("EndpointSlice for Service %s/%s not found", ns, svcName)
		return []discoveryv1.EndpointSlice{}, false
	}
	// In some cases the EndpointSlice controller will create more
	// EndpointSlices than necessary resulting in some duplication. This is
	// valid and tests should only fail here if less EndpointSlices than
	// expected are added.
	if len(esList.Items) < numSlices {
		framework.Logf("Expected at least %d EndpointSlices for Service %s/%s, got %d", numSlices, ns, svcName, len(esList.Items))
		for i, epSlice := range esList.Items {
			epsData, err := json.Marshal(epSlice)
			if err != nil {
				framework.Logf("Error marshaling JSON for EndpointSlice: %v", err)
			} else {
				framework.Logf("%d - %v", i, string(epsData))
			}
		}
		return esList.Items, false
	}

	actualNumEndpoints := 0
	for _, endpointSlice := range esList.Items {
		actualNumEndpoints += len(endpointSlice.Endpoints)
	}
	// In some cases the EndpointSlice controller will create more
	// EndpointSlices than necessary resulting in some duplication. This is
	// valid and tests should only fail here if less EndpointSlices than
	// expected are added.
	if actualNumEndpoints < numEndpoints {
		framework.Logf("EndpointSlices for %s/%s Service have %d/%d endpoints", ns, svcName, actualNumEndpoints, numEndpoints)
		return esList.Items, false
	}

	return esList.Items, true
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
