/*
Copyright The Kubernetes Authors.

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

package lifecycle

import (
	"context"

	v1 "k8s.io/api/core/v1"
	lifecycleapi "k8s.io/api/lifecycle/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/kubernetes/test/e2e/framework"
	e2econformance "k8s.io/kubernetes/test/e2e/framework/conformance"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("EvictionRequest API", func() {
	f := framework.NewDefaultFramework("evictionrequest")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	/*
	   Release: v1.37
	   Testname: EvictionRequest API operations
	   Description:
	   The lifecycle.k8s.io API group MUST exist in the /apis discovery document.
	   The lifecycle.k8s.io/v1alpha1 API group/version MUST exist
	     in the /apis/lifecycle.k8s.io discovery document.
	   The evictionrequests and evictionrequests/status resources MUST exist
	     in the /apis/lifecycle.k8s.io/v1alpha1 discovery document.
	   The evictionrequests resource must support create, get, list, watch,
	     update, patch, delete, and deletecollection.
	*/
	framework.It("lifecycle.k8s.io/v1alpha1 EvictionRequest", func(ctx context.Context) {
		erVersion := "v1alpha1"

		ginkgo.By("getting /apis")
		{
			discoveryGroups, err := f.ClientSet.Discovery().ServerGroups()
			framework.ExpectNoError(err)
			found := false
			for _, group := range discoveryGroups.Groups {
				if group.Name == lifecycleapi.GroupName {
					for _, version := range group.Versions {
						if version.Version == erVersion {
							found = true
							break
						}
					}
				}
			}
			if !found {
				framework.Failf("expected lifecycle API group/version, got %#v", discoveryGroups.Groups)
			}
		}

		ginkgo.By("getting /apis/lifecycle.k8s.io")
		{
			group := &metav1.APIGroup{}
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/lifecycle.k8s.io").Do(ctx).Into(group)
			framework.ExpectNoError(err)
			found := false
			for _, version := range group.Versions {
				if version.Version == erVersion {
					found = true
					break
				}
			}
			if !found {
				framework.Failf("expected lifecycle API version, got %#v", group.Versions)
			}
		}

		ginkgo.By("getting /apis/lifecycle.k8s.io/" + erVersion)
		{
			resources, err := f.ClientSet.Discovery().ServerResourcesForGroupVersion(lifecycleapi.SchemeGroupVersion.String())
			framework.ExpectNoError(err)
			foundER, foundERStatus := false, false
			for _, resource := range resources.APIResources {
				switch resource.Name {
				case "evictionrequests":
					foundER = true
				case "evictionrequests/status":
					foundERStatus = true
				}
			}
			if !foundER {
				framework.Failf("expected evictionrequests, got %#v", resources.APIResources)
			}
			if !foundERStatus {
				framework.Failf("expected evictionrequests/status, got %#v", resources.APIResources)
			}
		}

		ginkgo.By("creating a target pod for EvictionRequest")
		podClient := f.ClientSet.CoreV1().Pods(f.Namespace.Name)
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "e2e-er-target-" + utilrand.String(5),
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "agnhost",
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
					},
				},
			},
		}
		pod, err := podClient.Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		e2econformance.TestResource(ctx, f,
			&e2econformance.ResourceTestcase[*lifecycleapi.EvictionRequest]{
				GVR:        lifecycleapi.SchemeGroupVersion.WithResource("evictionrequests"),
				Namespaced: ptr.To(true),
				InitialSpec: &lifecycleapi.EvictionRequest{
					Spec: lifecycleapi.EvictionRequestSpec{
						Target: lifecycleapi.EvictionRequestTarget{
							Pod: &lifecycleapi.EvictionRequestPodReference{
								Name: pod.Name,
								UID:  pod.UID,
							},
						},
						Requester: "e2e-test.example.com/evictionrequest",
						Intent:    lifecycleapi.EvictionRequestIntentEviction,
					},
				},
				UpdateSpec: func(obj *lifecycleapi.EvictionRequest) *lifecycleapi.EvictionRequest {
					obj.Labels["foo"] = "bar"
					return obj
				},
				UpdateStatus: func(obj *lifecycleapi.EvictionRequest) *lifecycleapi.EvictionRequest {
					obj.Status.ObservedGeneration = ptr.To(obj.Generation)
					return obj
				},
				StrategicMergePatchSpec: `{"metadata": {"labels": {"baz": "qux"}}}`,
			},
		)
	})
})