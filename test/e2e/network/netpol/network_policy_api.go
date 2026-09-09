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
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/network/common"
)

var _ = common.SIGDescribe("Netpol API", func() {
	f := framework.NewDefaultFramework("netpol")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	/*
		Release: v1.20
		Testname: NetworkPolicies API
		Description:
		- The networking.k8s.io API group MUST exist in the /apis discovery document.
		- The networking.k8s.io/v1 API group/version MUST exist in the /apis/networking.k8s.io discovery document.
		- The NetworkPolicies resources MUST exist in the /apis/networking.k8s.io/v1 discovery document.
		- The NetworkPolicies resource must support create, get, list, watch, update, patch, delete, and deletecollection.
	*/

	ginkgo.It("should support creating NetworkPolicy API operations", func(ctx context.Context) {
		// Setup
		ns := f.Namespace.Name
		npVersion := "v1"
		npClient := f.ClientSet.NetworkingV1().NetworkPolicies(ns)

		namespaceSelector := &metav1.LabelSelector{
			MatchLabels: map[string]string{
				"ns-name": "pod-b",
			},
		}
		podSelector := &metav1.LabelSelector{
			MatchLabels: map[string]string{
				"pod-name": "client-a",
			},
		}
		ingressRule := networkingv1.NetworkPolicyIngressRule{}
		ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{PodSelector: podSelector, NamespaceSelector: namespaceSelector})
		npTemplate := GenNetworkPolicy(SetGenerateName("e2e-example-netpol"),
			SetObjectMetaLabel(map[string]string{"special-label": f.UniqueName}),
			SetSpecPodSelectorMatchLabels(map[string]string{"pod-name": "test-pod"}),
			SetSpecIngressRules(ingressRule))

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
			if !found {
				framework.Failf("expected networking API group/version, got %#v", discoveryGroups.Groups)
			}
		}
		ginkgo.By("getting /apis/networking.k8s.io")
		{
			group := &metav1.APIGroup{}
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/networking.k8s.io").Do(ctx).Into(group)
			framework.ExpectNoError(err)
			found := false
			for _, version := range group.Versions {
				if version.Version == npVersion {
					found = true
					break
				}
			}
			if !found {
				framework.Failf("expected networking API version, got %#v", group.Versions)
			}
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
			if !foundNetPol {
				framework.Failf("expected networkpolicies, got %#v", resources.APIResources)
			}
		}
		// NetPol resource create/read/update/watch verbs
		ginkgo.By("creating")
		_, err := npClient.Create(ctx, npTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		_, err = npClient.Create(ctx, npTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		createdNetPol, err := npClient.Create(ctx, npTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("getting")
		gottenNetPol, err := npClient.Get(ctx, createdNetPol.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(gottenNetPol.UID).To(gomega.Equal(createdNetPol.UID))

		ginkgo.By("listing")
		nps, err := npClient.List(ctx, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		gomega.Expect(nps.Items).To(gomega.HaveLen(3), "filtered list should have 3 items")

		ginkgo.By("watching")
		framework.Logf("starting watch")
		npWatch, err := npClient.Watch(ctx, metav1.ListOptions{ResourceVersion: nps.ResourceVersion, LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		// Test cluster-wide list and watch
		clusterNPClient := f.ClientSet.NetworkingV1().NetworkPolicies("")
		ginkgo.By("cluster-wide listing")
		clusterNPs, err := clusterNPClient.List(ctx, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		gomega.Expect(clusterNPs.Items).To(gomega.HaveLen(3), "filtered list should have 3 items")

		ginkgo.By("cluster-wide watching")
		framework.Logf("starting watch")
		_, err = clusterNPClient.Watch(ctx, metav1.ListOptions{ResourceVersion: nps.ResourceVersion, LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)

		ginkgo.By("patching")
		patchedNetPols, err := npClient.Patch(ctx, createdNetPol.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"patched":"true"}}}`), metav1.PatchOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(patchedNetPols.Annotations).To(gomega.HaveKeyWithValue("patched", "true"), "patched object should have the applied annotation")

		ginkgo.By("updating")
		npToUpdate := patchedNetPols.DeepCopy()
		npToUpdate.Annotations["updated"] = "true"
		updatedNetPols, err := npClient.Update(ctx, npToUpdate, metav1.UpdateOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(updatedNetPols.Annotations).To(gomega.HaveKeyWithValue("updated", "true"), "updated object should have the applied annotation")

		framework.Logf("waiting for watch events with expected annotations")
		for sawAnnotations := false; !sawAnnotations; {
			select {
			case evt, ok := <-npWatch.ResultChan():
				if !ok {
					framework.Fail("watch channel should not close")
				}
				gomega.Expect(evt.Type).To(gomega.Equal(watch.Modified))
				watchedNetPol, isNetPol := evt.Object.(*networkingv1.NetworkPolicy)
				if !isNetPol {
					framework.Failf("expected NetworkPolicy, got %T", evt.Object)
				}
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
		err = npClient.Delete(ctx, createdNetPol.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		err = wait.PollUntilContextTimeout(ctx, 2*time.Second, 1*time.Minute, false, func(ctx context.Context) (done bool, err error) {
			_, err = npClient.Get(ctx, createdNetPol.Name, metav1.GetOptions{})
			if !apierrors.IsNotFound(err) {
				framework.Logf("expected 404, got %#v", err)
				return false, nil
			}
			return true, nil
		})
		if err != nil {
			framework.Failf("unexpected error deleting existing network policy: %v", err)
		}
		nps, err = npClient.List(ctx, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		gomega.Expect(nps.Items).To(gomega.HaveLen(2), "filtered list should have 2 items")

		ginkgo.By("deleting a collection")
		err = npClient.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		err = wait.PollUntilContextTimeout(ctx, 2*time.Second, 1*time.Minute, false, func(ctx context.Context) (done bool, err error) {
			nps, err = npClient.List(ctx, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
			if err != nil {
				return false, err
			}
			if len(nps.Items) > 0 {
				framework.Logf("still %d network policies present, retrying ...", len(nps.Items))
				return false, nil
			}
			return true, nil
		})
		if err != nil {
			framework.Failf("unexpected error deleting existing network policies: %v", err)
		}
	})

	/*
		Release: v1.21
		Testname: NetworkPolicy support EndPort Field
		Description:
		- EndPort field cannot be defined if the Port field is not defined
		- EndPort field cannot be defined if the Port field is defined as a named (string) port.
		- EndPort field must be equal or greater than port.
	*/
	ginkgo.It("should support creating NetworkPolicy API with endport field", func(ctx context.Context) {
		ns := f.Namespace.Name
		npClient := f.ClientSet.NetworkingV1().NetworkPolicies(ns)

		ginkgo.By("EndPort field cannot be defined if the Port field is not defined.")
		var endport int32 = 20000
		egressRule := networkingv1.NetworkPolicyEgressRule{}
		egressRule.Ports = append(egressRule.Ports, networkingv1.NetworkPolicyPort{EndPort: &endport})
		npTemplate := GenNetworkPolicy(SetGenerateName("e2e-example-netpol-endport-validate"),
			SetObjectMetaLabel(map[string]string{"special-label": f.UniqueName}),
			SetSpecPodSelectorMatchLabels(map[string]string{"pod-name": "test-pod"}),
			SetSpecEgressRules(egressRule))
		_, err := npClient.Create(ctx, npTemplate, metav1.CreateOptions{})
		gomega.Expect(err).To(gomega.HaveOccurred(), "request template:%v", npTemplate)

		ginkgo.By("EndPort field cannot be defined if the Port field is defined as a named (string) port.")
		egressRule = networkingv1.NetworkPolicyEgressRule{}
		egressRule.Ports = append(egressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{Type: intstr.String, StrVal: "serve-80"}, EndPort: &endport})
		npTemplate.Spec.Egress = []networkingv1.NetworkPolicyEgressRule{egressRule}
		_, err = npClient.Create(ctx, npTemplate, metav1.CreateOptions{})
		gomega.Expect(err).To(gomega.HaveOccurred(), "request template:%v", npTemplate)

		ginkgo.By("EndPort field must be equal or greater than port.")
		ginkgo.By("When EndPort field is smaller than port, it will failed")
		egressRule = networkingv1.NetworkPolicyEgressRule{}
		egressRule.Ports = append(egressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{Type: intstr.Int, IntVal: 30000}, EndPort: &endport})
		npTemplate.Spec.Egress = []networkingv1.NetworkPolicyEgressRule{egressRule}
		_, err = npClient.Create(ctx, npTemplate, metav1.CreateOptions{})
		gomega.Expect(err).To(gomega.HaveOccurred(), "request template:%v", npTemplate)

		ginkgo.By("EndPort field is equal with port.")
		egressRule.Ports[0].Port = &intstr.IntOrString{Type: intstr.Int, IntVal: 20000}
		npTemplate.Spec.Egress = []networkingv1.NetworkPolicyEgressRule{egressRule}
		_, err = npClient.Create(ctx, npTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err, "request template:%v", npTemplate)

		ginkgo.By("EndPort field is greater than port.")
		egressRule = networkingv1.NetworkPolicyEgressRule{}
		egressRule.Ports = append(egressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{Type: intstr.Int, IntVal: 10000}, EndPort: &endport})
		npTemplate.Spec.Egress = []networkingv1.NetworkPolicyEgressRule{egressRule}
		_, err = npClient.Create(ctx, npTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err, "request template:%v", npTemplate)

		ginkgo.By("deleting all test collection")
		err = npClient.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		err = wait.PollUntilContextTimeout(ctx, 2*time.Second, 1*time.Minute, false, func(ctx context.Context) (done bool, err error) {
			nps, err := npClient.List(ctx, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
			if err != nil {
				return false, err
			}
			if len(nps.Items) > 0 {
				framework.Logf("still %d network policies present, retrying ...", len(nps.Items))
				return false, nil
			}
			return true, nil
		})
		if err != nil {
			framework.Failf("unexpected error deleting existing network policies: %v", err)
		}
	})
})
