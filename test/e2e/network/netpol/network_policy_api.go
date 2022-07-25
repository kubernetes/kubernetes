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
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/network/common"
)

var _ = common.SIGDescribe("Netpol API", func() {
	f := framework.NewDefaultFramework("netpol")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
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
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/networking.k8s.io").Do(context.TODO()).Into(group)
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
				if !ok {
					framework.Fail("watch channel should not close")
				}
				framework.ExpectEqual(evt.Type, watch.Modified)
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
		err = npClient.Delete(context.TODO(), createdNetPol.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		_, err = npClient.Get(context.TODO(), createdNetPol.Name, metav1.GetOptions{})
		if !apierrors.IsNotFound(err) {
			framework.Failf("expected 404, got %#v", err)
		}
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

	/*
		Release: v1.21
		Testname: NetworkPolicy support EndPort Field
		Description:
		- EndPort field cannot be defined if the Port field is not defined
		- EndPort field cannot be defined if the Port field is defined as a named (string) port.
		- EndPort field must be equal or greater than port.
	*/
	ginkgo.It("should support creating NetworkPolicy API with endport field", func() {
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
		_, err := npClient.Create(context.TODO(), npTemplate, metav1.CreateOptions{})
		framework.ExpectError(err, "request template:%v", npTemplate)

		ginkgo.By("EndPort field cannot be defined if the Port field is defined as a named (string) port.")
		egressRule = networkingv1.NetworkPolicyEgressRule{}
		egressRule.Ports = append(egressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{Type: intstr.String, StrVal: "serve-80"}, EndPort: &endport})
		npTemplate.Spec.Egress = []networkingv1.NetworkPolicyEgressRule{egressRule}
		_, err = npClient.Create(context.TODO(), npTemplate, metav1.CreateOptions{})
		framework.ExpectError(err, "request template:%v", npTemplate)

		ginkgo.By("EndPort field must be equal or greater than port.")
		ginkgo.By("When EndPort field is smaller than port, it will failed")
		egressRule = networkingv1.NetworkPolicyEgressRule{}
		egressRule.Ports = append(egressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{Type: intstr.Int, IntVal: 30000}, EndPort: &endport})
		npTemplate.Spec.Egress = []networkingv1.NetworkPolicyEgressRule{egressRule}
		_, err = npClient.Create(context.TODO(), npTemplate, metav1.CreateOptions{})
		framework.ExpectError(err, "request template:%v", npTemplate)

		ginkgo.By("EndPort field is equal with port.")
		egressRule.Ports[0].Port = &intstr.IntOrString{Type: intstr.Int, IntVal: 20000}
		npTemplate.Spec.Egress = []networkingv1.NetworkPolicyEgressRule{egressRule}
		_, err = npClient.Create(context.TODO(), npTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err, "request template:%v", npTemplate)

		ginkgo.By("EndPort field is greater than port.")
		egressRule = networkingv1.NetworkPolicyEgressRule{}
		egressRule.Ports = append(egressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{Type: intstr.Int, IntVal: 10000}, EndPort: &endport})
		npTemplate.Spec.Egress = []networkingv1.NetworkPolicyEgressRule{egressRule}
		_, err = npClient.Create(context.TODO(), npTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err, "request template:%v", npTemplate)

		ginkgo.By("deleting all test collection")
		err = npClient.DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		nps, err := npClient.List(context.TODO(), metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		framework.ExpectEqual(len(nps.Items), 0, "filtered list should be 0 items")
	})

	/*
			Release: v1.24
			Testname: NetworkPolicy support status subresource
			Description:
		    - Status condition without a Reason cannot exist
		    - Status should support conditions
			- Two conditions with the same type cannot exist.
	*/
	ginkgo.It("should support creating NetworkPolicy with Status subresource [Feature:NetworkPolicyStatus]", func() {
		ns := f.Namespace.Name
		npClient := f.ClientSet.NetworkingV1().NetworkPolicies(ns)

		ginkgo.By("NetworkPolicy should deny invalid status condition without Reason field")

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

		npTemplate := GenNetworkPolicy(SetGenerateName("e2e-example-netpol-status-validate"),
			SetObjectMetaLabel(map[string]string{"special-label": f.UniqueName}),
			SetSpecPodSelectorMatchLabels(map[string]string{"pod-name": "test-pod"}),
			SetSpecIngressRules(ingressRule))
		newNetPol, err := npClient.Create(context.TODO(), npTemplate, metav1.CreateOptions{})

		framework.ExpectNoError(err, "request template:%v", npTemplate)

		condition := metav1.Condition{
			Type:               string(networkingv1.NetworkPolicyConditionStatusAccepted),
			Status:             metav1.ConditionTrue,
			Reason:             "RuleApplied",
			LastTransitionTime: metav1.Time{Time: time.Now().Add(-5 * time.Minute)},
			Message:            "rule was successfully applied",
			ObservedGeneration: 2,
		}

		status := networkingv1.NetworkPolicyStatus{
			Conditions: []metav1.Condition{
				condition,
			},
		}

		ginkgo.By("NetworkPolicy should support valid status condition")
		newNetPol.Status = status

		_, err = npClient.UpdateStatus(context.TODO(), newNetPol, metav1.UpdateOptions{})
		framework.ExpectNoError(err, "request template:%v", newNetPol)

		ginkgo.By("NetworkPolicy should not support status condition without reason field")
		newNetPol.Status.Conditions[0].Reason = ""
		_, err = npClient.UpdateStatus(context.TODO(), newNetPol, metav1.UpdateOptions{})
		framework.ExpectError(err, "request template:%v", newNetPol)

		ginkgo.By("NetworkPolicy should not support status condition with duplicated types")
		newNetPol.Status.Conditions = []metav1.Condition{condition, condition}
		newNetPol.Status.Conditions[1].Status = metav1.ConditionFalse
		_, err = npClient.UpdateStatus(context.TODO(), newNetPol, metav1.UpdateOptions{})
		framework.ExpectError(err, "request template:%v", newNetPol)

		ginkgo.By("deleting all test collection")
		err = npClient.DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		nps, err := npClient.List(context.TODO(), metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		framework.ExpectEqual(len(nps.Items), 0, "filtered list should be 0 items")
	})
})
