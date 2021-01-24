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

package netpol

import (
	"context"
	"fmt"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"

	"github.com/onsi/ginkgo"
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = SIGDescribeCopy("Netpol API", func() {
	f := framework.NewDefaultFramework("netpol")
	/*
		Release: v1.20
		Testname: NetworkPolicies API
		Description:
		- The networking.k8s.io API group MUST exist in the /apis discovery document.
		- The networking.k8s.io/v1 API group/version MUST exist in the /apis/networking.k8s.io discovery document.
		- The NetworkPolicies resources MUST exist in the /apis/networking.k8s.io/v1 discovery document.
		- The NetworkPolicies resource must support create, get, list, watch, update, patch, delete, and deletecollection.
	*/

	ginkgo.It("should support creating NetworkPolicy API operations", func() {
		// Setup
		ns := f.Namespace.Name
		npVersion := "v1"
		npClient := f.ClientSet.NetworkingV1().NetworkPolicies(ns)
		npTemplate := &networkingv1.NetworkPolicy{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "e2e-example-netpol",
				Labels: map[string]string{
					"special-label": f.UniqueName,
				},
			},
			Spec: networkingv1.NetworkPolicySpec{
				// Apply this policy to the Server
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{
						"pod-name": "test-pod",
					},
				},
				// Allow traffic only from client-a in namespace-b
				Ingress: []networkingv1.NetworkPolicyIngressRule{{
					From: []networkingv1.NetworkPolicyPeer{{
						NamespaceSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{
								"ns-name": "pod-b",
							},
						},
						PodSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{
								"pod-name": "client-a",
							},
						},
					}},
				}},
			},
		}
		// Discovery
		ginkgo.By("getting /apis")
		{
			discoveryGroups, err := f.ClientSet.Discovery().ServerGroups()
			framework.ExpectNoError(err)
			found := false
			for _, group := range discoveryGroups.Groups {
				if group.Name == networkingv1.GroupName {
					for _, version := range group.Versions {
						if version.Version == npVersion {
							found = true
							break
						}
					}
				}
			}
			framework.ExpectEqual(found, true, fmt.Sprintf("expected networking API group/version, got %#v", discoveryGroups.Groups))
		}
		ginkgo.By("getting /apis/networking.k8s.io")
		{
			group := &metav1.APIGroup{}
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/networking.k8s.io").Do(context.TODO()).Into(group)
			framework.ExpectNoError(err)
			found := false
			for _, version := range group.Versions {
				if version.Version == npVersion {
					found = true
					break
				}
			}
			framework.ExpectEqual(found, true, fmt.Sprintf("expected networking API version, got %#v", group.Versions))
		}
		ginkgo.By("getting /apis/networking.k8s.io" + npVersion)
		{
			resources, err := f.ClientSet.Discovery().ServerResourcesForGroupVersion(networkingv1.SchemeGroupVersion.String())
			framework.ExpectNoError(err)
			foundNetPol := false
			for _, resource := range resources.APIResources {
				switch resource.Name {
				case "networkpolicies":
					foundNetPol = true
				}
			}
			framework.ExpectEqual(foundNetPol, true, fmt.Sprintf("expected networkpolicies, got %#v", resources.APIResources))
		}
		// NetPol resource create/read/update/watch verbs
		ginkgo.By("creating")
		_, err := npClient.Create(context.TODO(), npTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		_, err = npClient.Create(context.TODO(), npTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		createdNetPol, err := npClient.Create(context.TODO(), npTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("getting")
		gottenNetPol, err := npClient.Get(context.TODO(), createdNetPol.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(gottenNetPol.UID, createdNetPol.UID)

		ginkgo.By("listing")
		nps, err := npClient.List(context.TODO(), metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		framework.ExpectEqual(len(nps.Items), 3, "filtered list should have 3 items")

		ginkgo.By("watching")
		framework.Logf("starting watch")
		npWatch, err := npClient.Watch(context.TODO(), metav1.ListOptions{ResourceVersion: nps.ResourceVersion, LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		// Test cluster-wide list and watch
		clusterNPClient := f.ClientSet.NetworkingV1().NetworkPolicies("")
		ginkgo.By("cluster-wide listing")
		clusterNPs, err := clusterNPClient.List(context.TODO(), metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		framework.ExpectEqual(len(clusterNPs.Items), 3, "filtered list should have 3 items")

		ginkgo.By("cluster-wide watching")
		framework.Logf("starting watch")
		_, err = clusterNPClient.Watch(context.TODO(), metav1.ListOptions{ResourceVersion: nps.ResourceVersion, LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)

		ginkgo.By("patching")
		patchedNetPols, err := npClient.Patch(context.TODO(), createdNetPol.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"patched":"true"}}}`), metav1.PatchOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(patchedNetPols.Annotations["patched"], "true", "patched object should have the applied annotation")

		ginkgo.By("updating")
		npToUpdate := patchedNetPols.DeepCopy()
		npToUpdate.Annotations["updated"] = "true"
		updatedNetPols, err := npClient.Update(context.TODO(), npToUpdate, metav1.UpdateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(updatedNetPols.Annotations["updated"], "true", "updated object should have the applied annotation")

		framework.Logf("waiting for watch events with expected annotations")
		for sawAnnotations := false; !sawAnnotations; {
			select {
			case evt, ok := <-npWatch.ResultChan():
				framework.ExpectEqual(ok, true, "watch channel should not close")
				framework.ExpectEqual(evt.Type, watch.Modified)
				watchedNetPol, isNetPol := evt.Object.(*networkingv1.NetworkPolicy)
				framework.ExpectEqual(isNetPol, true, fmt.Sprintf("expected NetworkPolicy, got %T", evt.Object))
				if watchedNetPol.Annotations["patched"] == "true" && watchedNetPol.Annotations["updated"] == "true" {
					framework.Logf("saw patched and updated annotations")
					sawAnnotations = true
					npWatch.Stop()
				} else {
					framework.Logf("missing expected annotations, waiting: %#v", watchedNetPol.Annotations)
				}
			case <-time.After(wait.ForeverTestTimeout):
				framework.Fail("timed out waiting for watch event")
			}
		}
		// NetPol resource delete operations
		ginkgo.By("deleting")
		err = npClient.Delete(context.TODO(), createdNetPol.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		_, err = npClient.Get(context.TODO(), createdNetPol.Name, metav1.GetOptions{})
		framework.ExpectEqual(apierrors.IsNotFound(err), true, fmt.Sprintf("expected 404, got %#v", err))
		nps, err = npClient.List(context.TODO(), metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		framework.ExpectEqual(len(nps.Items), 2, "filtered list should have 2 items")

		ginkgo.By("deleting a collection")
		err = npClient.DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		nps, err = npClient.List(context.TODO(), metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		framework.ExpectEqual(len(nps.Items), 0, "filtered list should have 0 items")
	})
})
