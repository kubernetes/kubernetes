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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	watch "k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	apimachineryutils "k8s.io/kubernetes/test/e2e/common/apimachinery"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/network/common"
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
