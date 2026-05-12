/*
Copyright 2016 The Kubernetes Authors.

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
	"net"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/util/intstr"

	v1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"

	"github.com/onsi/ginkgo/v2"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/network/common"
	admissionapi "k8s.io/pod-security-admission/api"
	utilnet "k8s.io/utils/net"
)

const (
	isVerbose = true

	// See https://github.com/kubernetes/kubernetes/issues/95879
	// The semantics of the effect of network policies on loopback calls may be undefined: should
	//   they always be ALLOWED; how do Services affect this?
	//   Calico, Cillium, Antrea seem to do different things.
	// Since different CNIs have different results, that causes tests including loopback to fail
	//   on some CNIs.  So let's just ignore loopback calls for the purposes of deciding test pass/fail.
	ignoreLoopback    = true
	namespaceLabelKey = "kubernetes.io/metadata.name"
)

var (
	protocolTCP  = v1.ProtocolTCP
	protocolUDP  = v1.ProtocolUDP
	protocolSCTP = v1.ProtocolSCTP
)

/*
You might be wondering, why are there multiple namespaces used for each test case?

These tests are based on "truth tables" that compare the expected and actual connectivity of each pair of pods.
Since network policies live in namespaces, and peers can be selected by namespace,
showing the connectivity of pods in other namespaces is key information to show whether a network policy is working as intended or not.

Each test specifies the exact pod topology it needs (for example: x/a, x/b, y/a, z/a).
*/

var _ = common.SIGDescribe("Netpol", func() {
	f := framework.NewDefaultFramework("netpol")
	f.SkipNamespaceCreation = true // we create our own 3 test namespaces, we don't need the default one
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.Context("NetworkPolicy between server and client", func() {
		var k8s *kubeManager

		f.It("should support a 'default-deny-ingress' policy", feature.NetworkPolicy, func(ctx context.Context) {

			// Only poll TCP
			protocols := []v1.Protocol{protocolTCP}

			// Only testing port 80
			ports := []int32{80}

			// Namespace X has a default-deny-ingress policy, so we need 2 pods in X to
			// verify X-to-X ingress is blocked, and 2 pods in Y to verify Y-to-X is
			// also blocked while X-to-Y and Y-to-Y traffic remain allowed.
			// (In later tests, we just assume that a policy in X will not affect Y-to-Y traffic.)
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a", "y/b")

			// Only going to make a policy in namespace X
			nsX, _, _ := getK8sNamespaces(k8s)
			policy := GenNetworkPolicyWithNameAndPodSelector("deny-ingress", metav1.LabelSelector{}, SetSpecIngressRules())

			// Create the policy
			CreatePolicy(ctx, k8s, policy, nsX)

			// Make a truth table of connectivity for all pods in ns x y z
			reachability := NewReachability(k8s.AllPodStrings(), true)
			// Set the nsX as false, since it has a policy that blocks traffic
			reachability.ExpectPeer(&Peer{}, &Peer{Namespace: nsX}, false)

			// Confirm that the real world connectivity matches our matrix
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should support a 'default-deny-all' policy", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// Namespace X has a default-deny for both ingress and egress, so we need 2
			// pods in X to verify X-to-X traffic is blocked, and a pod in Y to verify
			// X<->Y traffic is blocked in both directions.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a")
			nsX, _, _ := getK8sNamespaces(k8s)

			policy := GenNetworkPolicyWithNameAndPodSelector("deny-all", metav1.LabelSelector{}, SetSpecIngressRules(), SetSpecEgressRules())
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&Peer{}, &Peer{Namespace: nsX}, false)
			reachability.ExpectPeer(&Peer{Namespace: nsX}, &Peer{}, false)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy to allow traffic from pods within server namespace based on PodSelector", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// Policy isolates x/a and only allows ingress from x/b, so we need x/b as the
			// allowed same-namespace peer, x/c as a same-namespace non-matching pod, and
			// y/a as a cross-namespace peer that must not be able to reach x/a.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "x/c", "y/a")
			nsX, _, _ := getK8sNamespaces(k8s)

			allowedPods := metav1.LabelSelector{
				MatchLabels: map[string]string{
					"pod": "b",
				},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{PodSelector: &allowedPods})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("x-a-allows-x-b", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(NewPodString(nsX, "a"), false)
			reachability.Expect(NewPodString(nsX, "b"), NewPodString(nsX, "a"), true)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy to allow ingress traffic for a target", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// Namespace X is default-deny-ingress, but we then allow ingress to x/a.
			// We need both x/a and x/b to show the target-specific behavior, and y/a to
			// exercise cross-namespace ingress.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a")
			nsX, _, _ := getK8sNamespaces(k8s)

			ginkgo.By("having a deny all ingress policy")
			// Deny all Ingress traffic policy to pods on namespace nsX
			policy := GenNetworkPolicyWithNameAndPodSelector("deny-all", metav1.LabelSelector{}, SetSpecIngressRules())
			CreatePolicy(ctx, k8s, policy, nsX)

			// Allow Ingress traffic only to pod x/a from any pod
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{PodSelector: &metav1.LabelSelector{}, NamespaceSelector: &metav1.LabelSelector{}})
			allowPolicy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-all-to-a", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, allowPolicy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(NewPodString(nsX, "a"), true)
			reachability.ExpectAllIngress(NewPodString(nsX, "b"), false)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy to allow ingress traffic from pods in all namespaces", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// Policy on x/a should allow ingress from any namespace, including its own.
			// We include x/b as a same-namespace client, plus one pod in Y and Z as
			// cross-namespace clients.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a", "z/a")
			nsX, _, _ := getK8sNamespaces(k8s)

			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: &metav1.LabelSelector{MatchLabels: map[string]string{}}})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-from-another-ns", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy to allow traffic only from a different namespace, based on NamespaceSelector", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// Policy on x/a allows ingress only from namespace Y. We need x/b to show
			// same-namespace ingress is denied, y/a as the allowed source, and z/a as an
			// unrelated namespace that must be denied.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a", "z/a")
			nsX, nsY, nsZ := getK8sNamespaces(k8s)

			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: &metav1.LabelSelector{MatchLabels: map[string]string{namespaceLabelKey: nsY}}})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-ns-selector", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			// disallow all traffic from the x or z namespaces
			reachability.ExpectPeer(&Peer{Namespace: nsX}, &Peer{Namespace: nsX, Pod: "a"}, false)
			reachability.ExpectPeer(&Peer{Namespace: nsZ}, &Peer{Namespace: nsX, Pod: "a"}, false)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy based on PodSelector with MatchExpressions", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// Same as the basic PodSelector test, but using MatchExpressions: x/b must be
			// able to reach x/a, while y/a (other namespace) must not be able to reach x/a.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a")
			nsX, _, _ := getK8sNamespaces(k8s)

			allowedPods := metav1.LabelSelector{
				MatchExpressions: []metav1.LabelSelectorRequirement{{
					Key:      "pod",
					Operator: metav1.LabelSelectorOpIn,
					Values:   []string{"b"},
				}},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{PodSelector: &allowedPods})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("x-a-allows-x-b", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(NewPodString(nsX, "a"), false)
			reachability.Expect(NewPodString(nsX, "b"), NewPodString(nsX, "a"), true)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy based on NamespaceSelector with MatchExpressions", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// Same as the basic NamespaceSelector test, but using MatchExpressions: y/a is
			// the allowed source namespace, while x/b (same namespace) and z/a must be
			// denied when connecting to x/a.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a", "z/a")
			nsX, nsY, nsZ := getK8sNamespaces(k8s)

			allowedNamespaces := &metav1.LabelSelector{
				MatchExpressions: []metav1.LabelSelectorRequirement{{
					Key:      namespaceLabelKey,
					Operator: metav1.LabelSelectorOpIn,
					Values:   []string{nsY},
				}},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedNamespaces})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-ns-y-match-selector", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			// disallow all traffic from the x or z namespaces
			reachability.ExpectPeer(&Peer{Namespace: nsX}, &Peer{Namespace: nsX, Pod: "a"}, false)
			reachability.ExpectPeer(&Peer{Namespace: nsZ}, &Peer{Namespace: nsX, Pod: "a"}, false)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy based on PodSelector or NamespaceSelector", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// Policy on x/a allows ingress from either (a) any namespace other than X, or
			// (b) pod x/b. We include x/b as the PodSelector-allowed source, x/c as a
			// non-matching pod in X that must be denied, and y/a as the NamespaceSelector-
			// allowed source.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "x/c", "y/a")
			nsX, _, _ := getK8sNamespaces(k8s)

			allowedNamespaces := &metav1.LabelSelector{
				MatchExpressions: []metav1.LabelSelectorRequirement{{
					Key:      namespaceLabelKey,
					Operator: metav1.LabelSelectorOpNotIn,
					Values:   []string{nsX},
				}},
			}
			podBAllowlisting := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"pod": "b",
				},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedNamespaces}, networkingv1.NetworkPolicyPeer{PodSelector: podBAllowlisting})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-ns-y-match-selector", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.Expect(NewPodString(nsX, "a"), NewPodString(nsX, "a"), false)
			reachability.Expect(NewPodString(nsX, "c"), NewPodString(nsX, "a"), false)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy based on PodSelector and NamespaceSelector", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// Policy on x/a requires BOTH a namespace match and a pod match. We use y/b
			// and z/b as the sources that match both selectors, x/b as the negative case
			// (pod matches but namespace does not), and y/a as the negative case (namespace
			// matches but pod does not).
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a", "y/b", "z/b")
			nsX, nsY, nsZ := getK8sNamespaces(k8s)

			allowedNamespaces := &metav1.LabelSelector{
				MatchExpressions: []metav1.LabelSelectorRequirement{{
					Key:      namespaceLabelKey,
					Operator: metav1.LabelSelectorOpNotIn,
					Values:   []string{nsX},
				}},
			}
			allowedPod := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"pod": "b",
				},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedNamespaces, PodSelector: allowedPod})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-ns-y-podselector-and-nsselector", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(NewPodString(nsX, "a"), false)
			reachability.Expect(NewPodString(nsY, "b"), NewPodString(nsX, "a"), true)
			reachability.Expect(NewPodString(nsZ, "b"), NewPodString(nsX, "a"), true)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy based on Multiple PodSelectors and NamespaceSelectors", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// Policy on x/a allows ingress only from pods {b,c} in namespaces other than X.
			// We need x/b to prove same-namespace traffic is denied, y/b and y/c as allowed
			// sources, and y/a as a non-matching pod in an allowed namespace.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a", "y/b", "y/c")
			nsX, nsY, _ := getK8sNamespaces(k8s)

			allowedNamespaces := &metav1.LabelSelector{
				MatchExpressions: []metav1.LabelSelectorRequirement{{
					Key:      namespaceLabelKey,
					Operator: metav1.LabelSelectorOpNotIn,
					Values:   []string{nsX},
				}},
			}
			allowedPod := &metav1.LabelSelector{
				MatchExpressions: []metav1.LabelSelectorRequirement{{
					Key:      "pod",
					Operator: metav1.LabelSelectorOpIn,
					Values:   []string{"b", "c"},
				}},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedNamespaces, PodSelector: allowedPod})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-ns-y-z-pod-b-c", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&Peer{Namespace: nsX}, &Peer{Namespace: nsX, Pod: "a"}, false)
			reachability.Expect(NewPodString(nsY, "a"), NewPodString(nsX, "a"), false)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy based on any PodSelectors", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// Policy on x/a uses multiple "from" entries (pod=b OR pod=c). We need both
			// x/b and x/c as allowed sources and x/a as the isolated target pod.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "x/c")
			nsX, _, _ := getK8sNamespaces(k8s)

			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			for _, label := range []map[string]string{{"pod": "b"}, {"pod": "c"}} {
				ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{PodSelector: &metav1.LabelSelector{MatchLabels: label}})
			}
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-ns-x-pod-b-c", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(NewPodString(nsX, "a"), false)

			// Connect Pods b and c to pod a from namespace nsX
			reachability.Expect(NewPodString(nsX, "b"), NewPodString(nsX, "a"), true)
			reachability.Expect(NewPodString(nsX, "c"), NewPodString(nsX, "a"), true)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy to allow traffic only from a pod in a different namespace based on PodSelector and NamespaceSelector", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// Policy on x/a allows ingress only from y/a (namespace+pod selectors). We keep
			// x/b as a same-namespace/non-matching pod, plus y/b and z/a as non-matching
			// cross-namespace pods, to ensure they cannot reach x/a.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a", "y/b", "z/a")
			nsX, nsY, _ := getK8sNamespaces(k8s)

			allowedNamespaces := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					namespaceLabelKey: nsY,
				},
			}
			allowedPods := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"pod": "a",
				},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedNamespaces, PodSelector: allowedPods})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-ns-y-pod-a-via-namespace-pod-selector", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(NewPodString(nsX, "a"), false)
			reachability.Expect(NewPodString(nsY, "a"), NewPodString(nsX, "a"), true)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy based on Ports", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{81}
			// This test is port-specific: namespace X should allow ingress to x/a on
			// port 81 from namespace Y only. We include x/b as a same-namespace source
			// (denied), y/a,y/b as namespace-Y sources (allowed), and z/a as a denied
			// third-namespace source.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a", "y/b", "z/a")
			nsX, nsY, nsZ := getK8sNamespaces(k8s)

			ginkgo.By("Creating a network allowPort81Policy which only allows allow listed namespaces (y) to connect on exactly one port (81)")
			allowedLabels := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					namespaceLabelKey: nsY,
				},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedLabels})
			ingressRule.Ports = append(ingressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{IntVal: 81}, Protocol: &protocolTCP})
			allowPort81Policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-ns-selector", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, allowPort81Policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&Peer{Namespace: nsX}, &Peer{Namespace: nsX, Pod: "a"}, false)
			reachability.ExpectPeer(&Peer{Namespace: nsY}, &Peer{Namespace: nsX, Pod: "a"}, true)
			reachability.ExpectPeer(&Peer{Namespace: nsZ}, &Peer{Namespace: nsX, Pod: "a"}, false)

			ValidateOrFail(k8s, &TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce multiple, stacked policies with overlapping podSelectors", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80, 81}
			// We stack multiple policies to verify per-port behavior and policy precedence.
			// We need x/a as the target, x/b as a same-namespace source, y/a,y/b as the
			// allowed namespace, and z/a as a namespace that must remain denied.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a", "y/b", "z/a")
			nsX, nsY, nsZ := getK8sNamespaces(k8s)

			ginkgo.By("Creating a network allowPort81Policy which only allows allow listed namespaces (y) to connect on exactly one port (81)")
			allowedLabels := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					namespaceLabelKey: nsY,
				},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedLabels})
			ingressRule.Ports = append(ingressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{IntVal: 81}, Protocol: &protocolTCP})
			allowPort81Policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-ns-selector", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, allowPort81Policy, nsX)

			reachabilityALLOW := NewReachability(k8s.AllPodStrings(), true)
			reachabilityALLOW.ExpectPeer(&Peer{Namespace: nsX}, &Peer{Namespace: nsX, Pod: "a"}, false)
			reachabilityALLOW.ExpectPeer(&Peer{Namespace: nsY}, &Peer{Namespace: nsX, Pod: "a"}, true)
			reachabilityALLOW.ExpectPeer(&Peer{Namespace: nsZ}, &Peer{Namespace: nsX, Pod: "a"}, false)

			ginkgo.By("Verifying traffic on port 81.")
			ValidateOrFail(k8s, &TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachabilityALLOW})

			reachabilityDENY := NewReachability(k8s.AllPodStrings(), true)
			reachabilityDENY.ExpectAllIngress(NewPodString(nsX, "a"), false)

			ginkgo.By("Verifying traffic on port 80.")
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityDENY})

			ingressRule = networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedLabels})
			ingressRule.Ports = append(ingressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{IntVal: 80}, Protocol: &protocolTCP})
			allowPort80Policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-ns-selector-80", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, allowPort80Policy, nsX)

			ginkgo.By("Verifying that we can add a policy to unblock port 80")
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityALLOW})
		})

		f.It("should support allow-all policy", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80, 81}
			// Allow-all should not restrict anything. We include two pods in X and one in Y
			// so we cover both intra-namespace (x/a<->x/b) and cross-namespace traffic.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a")
			nsX, _, _ := getK8sNamespaces(k8s)

			ginkgo.By("Creating a network policy which allows all traffic.")
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-all", map[string]string{}, SetSpecIngressRules(networkingv1.NetworkPolicyIngressRule{}))
			CreatePolicy(ctx, k8s, policy, nsX)

			ginkgo.By("Testing pods can connect to both ports when an 'allow-all' policy is present.")
			reachability := NewReachability(k8s.AllPodStrings(), true)
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
			ValidateOrFail(k8s, &TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should allow ingress access on one named port", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80, 81}
			// Policy applies to all pods in X and only allows the named port serve-81-tcp.
			// We need two pods in X to validate X-to-X traffic is blocked on port 80, and a
			// pod in Y to validate cross-namespace ingress is also restricted as expected.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a")
			nsX, _, _ := getK8sNamespaces(k8s)

			ginkgo.By("Blocking all ports other then 81 in the entire namespace")
			IngressRules := networkingv1.NetworkPolicyIngressRule{}
			IngressRules.Ports = append(IngressRules.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{Type: intstr.String, StrVal: "serve-81-tcp"}})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-all", map[string]string{}, SetSpecIngressRules(IngressRules))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachabilityPort81 := NewReachability(k8s.AllPodStrings(), true)
			ValidateOrFail(k8s, &TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachabilityPort81})

			// disallow all traffic to the x namespace
			reachabilityPort80 := NewReachability(k8s.AllPodStrings(), true)
			reachabilityPort80.ExpectPeer(&Peer{}, &Peer{Namespace: nsX}, false)
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityPort80})
		})

		f.It("should allow ingress access from namespace on one named port", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80, 81}
			// Only namespace Y should be able to reach x/a on the named port serve-80-tcp,
			// and port 81 should be blocked for everyone. We include x/b as a
			// same-namespace peer (denied), y/a as the allowed namespace, and z/a as a
			// denied namespace.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a", "z/a")
			nsX, nsY, nsZ := getK8sNamespaces(k8s)

			allowedLabels := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					namespaceLabelKey: nsY,
				},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedLabels})
			ingressRule.Ports = append(ingressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{Type: intstr.String, StrVal: "serve-80-tcp"}, Protocol: &protocolTCP})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-ns-selector-80", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			// disallow all traffic from the x or z namespaces
			reachability.ExpectPeer(&Peer{Namespace: nsX}, &Peer{Namespace: nsX, Pod: "a"}, false)
			reachability.ExpectPeer(&Peer{Namespace: nsZ}, &Peer{Namespace: nsX, Pod: "a"}, false)

			ginkgo.By("Verify that port 80 is allowed for namespace y")
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})

			ginkgo.By("Verify that port 81 is blocked for all namespaces including y")
			reachabilityFAIL := NewReachability(k8s.AllPodStrings(), true)
			reachabilityFAIL.ExpectAllIngress(NewPodString(nsX, "a"), false)
			ValidateOrFail(k8s, &TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachabilityFAIL})
		})

		f.It("should allow egress access on one named port", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80, 81}
			// Policy restricts egress from x/a by named port: port 80 allowed, port 81
			// denied. One pod in X and one pod in Y is sufficient to validate egress
			// behavior.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "y/a")
			nsX, _, _ := getK8sNamespaces(k8s)

			egressRule := networkingv1.NetworkPolicyEgressRule{}
			egressRule.Ports = append(egressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{Type: intstr.String, StrVal: "serve-80-tcp"}})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-egress", map[string]string{}, SetSpecEgressRules(egressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachabilityPort80 := NewReachability(k8s.AllPodStrings(), true)
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityPort80})

			// meanwhile no traffic over 81 should work, since our egress policy is on 80
			reachabilityPort81 := NewReachability(k8s.AllPodStrings(), true)
			reachabilityPort81.ExpectPeer(&Peer{Namespace: nsX}, &Peer{}, false)
			ValidateOrFail(k8s, &TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachabilityPort81})
		})

		f.It("should enforce updated policy", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{81}
			// This test mutates a policy in namespace X and validates connectivity changes
			// from allowed -> denied. One pod in X and one in Y is enough to observe the
			// before/after behavior.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "y/a")
			nsX, _, _ := getK8sNamespaces(k8s)

			ginkgo.By("Using the simplest possible mutation: start with allow all, then switch to deny all")
			// part 1) allow all
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-all-mutate-to-deny-all", map[string]string{}, SetSpecIngressRules(networkingv1.NetworkPolicyIngressRule{}))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			ValidateOrFail(k8s, &TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachability})

			// part 2) update the policy to deny all
			policy.Spec.Ingress = []networkingv1.NetworkPolicyIngressRule{}
			UpdatePolicy(ctx, k8s, policy, nsX)

			reachabilityDeny := NewReachability(k8s.AllPodStrings(), true)
			reachabilityDeny.ExpectPeer(&Peer{}, &Peer{Namespace: nsX}, false)
			ValidateOrFail(k8s, &TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachabilityDeny})
		})

		f.It("should allow ingress access from updated namespace", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// Namespace label changes are what we are testing here. We need x/a as the
			// target and y/a as the external client whose namespace labels are updated.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "y/a")
			nsX, nsY, _ := getK8sNamespaces(k8s)
			ginkgo.DeferCleanup(DeleteNamespaceLabel, k8s, nsY, "ns2")

			allowedLabels := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"ns2": "updated",
				},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedLabels})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-ns-selector", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(NewPodString(nsX, "a"), false)
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})

			// add a new label
			AddNamespaceLabel(ctx, k8s, nsY, "ns2", "updated")

			// anything from namespace 'y' should be able to get to x/a
			reachabilityWithLabel := NewReachability(k8s.AllPodStrings(), true)
			reachabilityWithLabel.ExpectAllIngress(NewPodString(nsX, "a"), false)
			reachabilityWithLabel.ExpectPeer(&Peer{Namespace: nsY}, &Peer{}, true)
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityWithLabel})
		})

		f.It("should allow ingress access from updated pod", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// Pod label changes are what we are testing here. We need x/a as the target and
			// x/b as the client whose labels are updated to match the PodSelector.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b")
			nsX, _, _ := getK8sNamespaces(k8s)
			ginkgo.DeferCleanup(ResetPodLabels, k8s, nsX, "b")

			// add a new label
			matchLabels := map[string]string{"pod": "b", "pod2": "updated"}
			allowedLabels := &metav1.LabelSelector{MatchLabels: matchLabels}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{PodSelector: allowedLabels})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-pod-selector", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(NewPodString(nsX, "a"), false)
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})

			AddPodLabels(ctx, k8s, nsX, "b", matchLabels)

			ginkgo.By("x/b is able to reach x/a when label is updated")

			reachabilityWithLabel := NewReachability(k8s.AllPodStrings(), true)
			reachabilityWithLabel.ExpectAllIngress(NewPodString(nsX, "a"), false)
			reachabilityWithLabel.Expect(NewPodString(nsX, "b"), NewPodString(nsX, "a"), true)
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityWithLabel})
		})

		f.It("should deny ingress from pods on other namespaces", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// Ingress policy in X should only allow same-namespace pods. We need two pods
			// in X to verify X-to-X is allowed, plus one pod in each of Y and Z to verify
			// that cross-namespace ingress into X is denied.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a", "z/a")
			nsX, nsY, nsZ := getK8sNamespaces(k8s)

			IngressRules := networkingv1.NetworkPolicyIngressRule{}
			IngressRules.From = append(IngressRules.From, networkingv1.NetworkPolicyPeer{PodSelector: &metav1.LabelSelector{MatchLabels: map[string]string{}}})
			policy := GenNetworkPolicyWithNameAndPodSelector("deny-empty-policy", metav1.LabelSelector{}, SetSpecIngressRules(IngressRules))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&Peer{Namespace: nsY}, &Peer{Namespace: nsX}, false)
			reachability.ExpectPeer(&Peer{Namespace: nsZ}, &Peer{Namespace: nsX}, false)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should deny ingress access to updated pod", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// This test verifies that x/a becomes isolated when its labels are updated to
			// match a deny-ingress policy. We need x/a as the target and y/a as an external
			// source.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "y/a")
			nsX, _, _ := getK8sNamespaces(k8s)
			ginkgo.DeferCleanup(ResetPodLabels, k8s, nsX, "a")

			policy := GenNetworkPolicyWithNameAndPodSelector("deny-ingress-via-label-selector",
				metav1.LabelSelector{MatchLabels: map[string]string{"target": "isolated"}}, SetSpecIngressRules())
			CreatePolicy(ctx, k8s, policy, nsX)

			ginkgo.By("Verify that everything can reach x/a")
			reachability := NewReachability(k8s.AllPodStrings(), true)
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})

			AddPodLabels(ctx, k8s, nsX, "a", map[string]string{"target": "isolated"})

			reachabilityIsolated := NewReachability(k8s.AllPodStrings(), true)
			reachabilityIsolated.ExpectAllIngress(NewPodString(nsX, "a"), false)
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityIsolated})
		})

		f.It("should deny egress from pods based on PodSelector", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// Egress policy selects x/a and denies all egress. We include x/b to verify
			// non-selected pods are unaffected, plus y/a as a cross-namespace destination.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a")
			nsX, _, _ := getK8sNamespaces(k8s)

			policy := GenNetworkPolicyWithNameAndPodSelector("deny-egress-pod-a", metav1.LabelSelector{MatchLabels: map[string]string{"pod": "a"}}, SetSpecEgressRules())
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllEgress(NewPodString(nsX, "a"), false)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should deny egress from all pods in a namespace", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// Namespace-wide egress deny in X: we include x/b to verify x/a->x/b is denied,
			// and y/a to verify cross-namespace egress from X is denied as well.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a")
			nsX, _, _ := getK8sNamespaces(k8s)

			policy := GenNetworkPolicyWithNameAndPodSelector("deny-egress-ns-x", metav1.LabelSelector{}, SetSpecEgressRules())
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&Peer{Namespace: nsX}, &Peer{}, false)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should work with Ingress, Egress specified together", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80, 81}
			// Policy on x/a allows ingress only from x/a and x/b, and allows egress only to
			// port 80. We include x/c as a same-namespace disallowed ingress source and y/a
			// as a cross-namespace destination for egress checks.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "x/c", "y/a")
			nsX, _, _ := getK8sNamespaces(k8s)

			allowedPodLabels := &metav1.LabelSelector{MatchLabels: map[string]string{"pod": "b"}}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{PodSelector: allowedPodLabels})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-pod-selector", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))

			// add an egress rule on to it...
			policy.Spec.Egress = []networkingv1.NetworkPolicyEgressRule{
				{
					Ports: []networkingv1.NetworkPolicyPort{
						{
							// don't use named ports
							Port: &intstr.IntOrString{Type: intstr.Int, IntVal: 80},
						},
					},
				},
			}
			policy.Spec.PolicyTypes = []networkingv1.PolicyType{networkingv1.PolicyTypeEgress, networkingv1.PolicyTypeIngress}
			CreatePolicy(ctx, k8s, policy, nsX)

			reachabilityPort80 := NewReachability(k8s.AllPodStrings(), true)
			reachabilityPort80.ExpectAllIngress(NewPodString(nsX, "a"), false)
			reachabilityPort80.Expect(NewPodString(nsX, "b"), NewPodString(nsX, "a"), true)
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityPort80})

			ginkgo.By("validating that port 81 doesn't work")
			// meanwhile no egress traffic on 81 should work, since our egress policy is on 80
			reachabilityPort81 := NewReachability(k8s.AllPodStrings(), true)
			reachabilityPort81.ExpectAllIngress(NewPodString(nsX, "a"), false)
			reachabilityPort81.ExpectAllEgress(NewPodString(nsX, "a"), false)
			reachabilityPort81.Expect(NewPodString(nsX, "b"), NewPodString(nsX, "a"), true)
			ValidateOrFail(k8s, &TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachabilityPort81})
		})

		f.It("should support denying of egress traffic on the client side (even if the server explicitly allows this traffic)", feature.NetworkPolicy, func(ctx context.Context) {
			// x/a --> y/a and y/b
			// Egress allowed to y/a only. Egress to y/b should be blocked
			// Ingress on y/a and y/b allow traffic from x/a
			// Expectation: traffic from x/a to y/a allowed only, traffic from x/a to y/b denied by egress policy

			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// We need x/a as the client selected by the egress policy, and two pods in Y so
			// we can show y/a is the allowed destination while y/b is denied by egress even
			// though ingress on the server side allows both.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "y/a", "y/b")
			nsX, nsY, _ := getK8sNamespaces(k8s)

			// Building egress policy for x/a to y/a only
			allowedEgressNamespaces := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					namespaceLabelKey: nsY,
				},
			}
			allowedEgressPods := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"pod": "a",
				},
			}
			egressRule1 := networkingv1.NetworkPolicyEgressRule{}
			egressRule1.To = append(egressRule1.To, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedEgressNamespaces, PodSelector: allowedEgressPods})
			egressPolicy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-to-ns-y-pod-a", map[string]string{"pod": "a"}, SetSpecEgressRules(egressRule1))
			CreatePolicy(ctx, k8s, egressPolicy, nsX)

			// Creating ingress policy to allow from x/a to y/a and y/b
			allowedIngressNamespaces := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					namespaceLabelKey: nsX,
				},
			}
			allowedIngressPods := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"pod": "a",
				},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedIngressNamespaces, PodSelector: allowedIngressPods})
			allowIngressPolicyPodA := GenNetworkPolicyWithNameAndPodMatchLabel("allow-from-xa-on-ya-match-selector", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			allowIngressPolicyPodB := GenNetworkPolicyWithNameAndPodMatchLabel("allow-from-xa-on-yb-match-selector", map[string]string{"pod": "b"}, SetSpecIngressRules(ingressRule))

			CreatePolicy(ctx, k8s, allowIngressPolicyPodA, nsY)
			CreatePolicy(ctx, k8s, allowIngressPolicyPodB, nsY)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&Peer{Namespace: nsX, Pod: "a"}, &Peer{}, false)
			reachability.ExpectPeer(&Peer{}, &Peer{Namespace: nsY, Pod: "a"}, false)
			reachability.ExpectPeer(&Peer{Namespace: nsX, Pod: "a"}, &Peer{Namespace: nsY, Pod: "a"}, true)
			reachability.ExpectPeer(&Peer{}, &Peer{Namespace: nsY, Pod: "b"}, false)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce egress policy allowing traffic to a server in a different namespace based on PodSelector and NamespaceSelector", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// Egress policy on x/a only allows traffic to y/a (namespace+pod selectors).
			// We include y/b as a non-matching destination to ensure it is denied.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "y/a", "y/b")
			nsX, nsY, _ := getK8sNamespaces(k8s)

			allowedNamespaces := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					namespaceLabelKey: nsY,
				},
			}
			allowedPods := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"pod": "a",
				},
			}
			egressRule1 := networkingv1.NetworkPolicyEgressRule{}
			egressRule1.To = append(egressRule1.To, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedNamespaces, PodSelector: allowedPods})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-to-ns-y-pod-a", map[string]string{"pod": "a"}, SetSpecEgressRules(egressRule1))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllEgress(NewPodString(nsX, "a"), false)
			reachability.Expect(NewPodString(nsX, "a"), NewPodString(nsY, "a"), true)
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce ingress policy allowing any port traffic to a server on a specific protocol", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP, protocolUDP}
			ports := []int32{80}
			// Protocol-only ingress allowlist: one server (x/a) and one client (y/a) is
			// enough to validate TCP is allowed while UDP is denied.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "y/a")
			nsX, _, _ := getK8sNamespaces(k8s)

			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.Ports = append(ingressRule.Ports, networkingv1.NetworkPolicyPort{Protocol: &protocolTCP})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-ingress-by-proto", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachabilityTCP := NewReachability(k8s.AllPodStrings(), true)
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityTCP})

			reachabilityUDP := NewReachability(k8s.AllPodStrings(), true)
			reachabilityUDP.ExpectPeer(&Peer{}, &Peer{Namespace: nsX, Pod: "a"}, false)
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolUDP, Reachability: reachabilityUDP})
		})

		f.It("should enforce multiple ingress policies with ingress allow-all policy taking precedence", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{81}
			// We only need x/a as the policy target and y/a as the client to observe that
			// an allow-all ingress policy overrides a more restrictive ingress policy.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "y/a")
			nsX, _, _ := getK8sNamespaces(k8s)

			IngressRules := networkingv1.NetworkPolicyIngressRule{}
			IngressRules.Ports = append(IngressRules.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{Type: intstr.Int, IntVal: 80}})
			policyAllowOnlyPort80 := GenNetworkPolicyWithNameAndPodMatchLabel("allow-ingress-port-80", map[string]string{}, SetSpecIngressRules(IngressRules))
			CreatePolicy(ctx, k8s, policyAllowOnlyPort80, nsX)

			ginkgo.By("The policy targets port 80 -- so let's make sure traffic on port 81 is blocked")

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&Peer{}, &Peer{Namespace: nsX}, false)
			ValidateOrFail(k8s, &TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachability})

			ginkgo.By("Allowing all ports")

			policyAllowAll := GenNetworkPolicyWithNameAndPodMatchLabel("allow-ingress", map[string]string{}, SetSpecIngressRules(networkingv1.NetworkPolicyIngressRule{}))
			CreatePolicy(ctx, k8s, policyAllowAll, nsX)

			reachabilityAll := NewReachability(k8s.AllPodStrings(), true)
			ValidateOrFail(k8s, &TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachabilityAll})
		})

		f.It("should enforce multiple egress policies with egress allow-all policy taking precedence", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{81}
			// We only need x/a as the policy target and y/a as the destination to observe
			// that an allow-all egress policy overrides a more restrictive egress policy.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "y/a")
			nsX, _, _ := getK8sNamespaces(k8s)

			egressRule := networkingv1.NetworkPolicyEgressRule{}
			egressRule.Ports = append(egressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{Type: intstr.Int, IntVal: 80}})
			policyAllowPort80 := GenNetworkPolicyWithNameAndPodMatchLabel("allow-egress-port-80", map[string]string{}, SetSpecEgressRules(egressRule))
			CreatePolicy(ctx, k8s, policyAllowPort80, nsX)

			ginkgo.By("Making sure ingress doesn't work other than port 80")

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&Peer{Namespace: nsX}, &Peer{}, false)
			ValidateOrFail(k8s, &TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachability})

			ginkgo.By("Allowing all ports")

			policyAllowAll := GenNetworkPolicyWithNameAndPodMatchLabel("allow-egress", map[string]string{}, SetSpecEgressRules(networkingv1.NetworkPolicyEgressRule{}))
			CreatePolicy(ctx, k8s, policyAllowAll, nsX)

			reachabilityAll := NewReachability(k8s.AllPodStrings(), true)
			ValidateOrFail(k8s, &TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachabilityAll})
		})

		f.It("should stop enforcing policies after they are deleted", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// A single client/server pair is sufficient to validate connectivity is denied
			// while the policy exists and returns to allowed after the policy is deleted.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "y/a")
			nsX, _, _ := getK8sNamespaces(k8s)

			ginkgo.By("Creating a network policy for the server which denies all traffic.")
			// Deny all traffic into and out of "x".
			policy := GenNetworkPolicyWithNameAndPodSelector("deny-all", metav1.LabelSelector{}, SetSpecIngressRules(), SetSpecEgressRules())
			CreatePolicy(ctx, k8s, policy, nsX)
			reachability := NewReachability(k8s.AllPodStrings(), true)

			// Expect all traffic into, and out of "x" to be False.
			reachability.ExpectPeer(&Peer{Namespace: nsX}, &Peer{}, false)
			reachability.ExpectPeer(&Peer{}, &Peer{Namespace: nsX}, false)
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})

			err := k8s.cleanNetworkPolicies(ctx)
			time.Sleep(3 * time.Second) // TODO we can remove this eventually, its just a hack to keep CI stable.
			framework.ExpectNoError(err, "unable to clean network policies")

			// Now the policy is deleted, we expect all connectivity to work again.
			reachabilityAll := NewReachability(k8s.AllPodStrings(), true)
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityAll})
		})

		// TODO, figure out how the next 3 tests should work with dual stack : do we need a different abstraction then just "podIP"?

		f.It("should allow egress access to server in CIDR block", feature.NetworkPolicy, func(ctx context.Context) {
			// Getting podServer's status to get podServer's IP, to create the CIDR
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// This test needs the IP of a real pod to build the IPBlock CIDR. We create y/b
			// as the server (to source the IP), y/a as a non-matching destination, and x/a
			// as the client whose egress is restricted.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "y/a", "y/b")
			nsX, nsY, _ := getK8sNamespaces(k8s)

			podList, err := f.ClientSet.CoreV1().Pods(nsY).List(ctx, metav1.ListOptions{LabelSelector: "pod=b"})
			framework.ExpectNoError(err, "Failing to list pods in namespace y")
			pod := podList.Items[0]

			hostMask := 32
			if utilnet.IsIPv6String(pod.Status.PodIP) {
				hostMask = 128
			}
			podServerCIDR := fmt.Sprintf("%s/%d", pod.Status.PodIP, hostMask)
			egressRule1 := networkingv1.NetworkPolicyEgressRule{}
			egressRule1.To = append(egressRule1.To, networkingv1.NetworkPolicyPeer{IPBlock: &networkingv1.IPBlock{CIDR: podServerCIDR}})
			policyAllowCIDR := GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-cidr-egress-rule",
				map[string]string{"pod": "a"}, SetSpecEgressRules(egressRule1))
			CreatePolicy(ctx, k8s, policyAllowCIDR, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllEgress(NewPodString(nsX, "a"), false)
			reachability.Expect(NewPodString(nsX, "a"), NewPodString(nsY, "b"), true)
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce except clause while egress access to server in CIDR block", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// We need x/b to supply the IP used in the IPBlock (and as the destination),
			// x/a as the client, and x/c to verify non-excepted traffic still works.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "x/c")
			nsX, _, _ := getK8sNamespaces(k8s)

			// Getting podServer's status to get podServer's IP, to create the CIDR with except clause
			podList, err := f.ClientSet.CoreV1().Pods(nsX).List(ctx, metav1.ListOptions{LabelSelector: "pod=b"})
			framework.ExpectNoError(err, "Failing to find pod x/b")
			podB := podList.Items[0]

			// Create a rule that allows egress to a large set of IPs around
			// podB, but not podB itself.

			podServerAllowCIDR := makeLargeCIDRForIP(podB.Status.PodIP)
			hostMask := 32
			if utilnet.IsIPv6String(podB.Status.PodIP) {
				hostMask = 128
			}
			podServerExceptList := []string{fmt.Sprintf("%s/%d", podB.Status.PodIP, hostMask)}

			egressRule1 := networkingv1.NetworkPolicyEgressRule{}
			egressRule1.To = append(egressRule1.To, networkingv1.NetworkPolicyPeer{IPBlock: &networkingv1.IPBlock{CIDR: podServerAllowCIDR, Except: podServerExceptList}})
			policyAllowCIDR := GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-cidr-egress-rule", map[string]string{"pod": "a"}, SetSpecEgressRules(egressRule1))

			CreatePolicy(ctx, k8s, policyAllowCIDR, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.Expect(NewPodString(nsX, "a"), NewPodString(nsX, "b"), false)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should ensure an IP overlapping both IPBlock.CIDR and IPBlock.Except is allowed", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// We need x/b to supply the IP used in the IPBlock (and as the destination),
			// x/a as the client, and x/c to verify non-excepted traffic still works for
			// CIDR/Except overlap behavior.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "x/c")
			nsX, _, _ := getK8sNamespaces(k8s)

			// Getting podServer's status to get podServer's IP, to create the CIDR with except clause
			podList, err := f.ClientSet.CoreV1().Pods(nsX).List(ctx, metav1.ListOptions{LabelSelector: "pod=b"})
			framework.ExpectNoError(err, "Failing to find pod x/b")
			podB := podList.Items[0]

			// Create a rule that allows egress to a large set of IPs around
			// podB, but not podB itself.

			podServerAllowCIDR := makeLargeCIDRForIP(podB.Status.PodIP)
			hostMask := 32
			if utilnet.IsIPv6String(podB.Status.PodIP) {
				hostMask = 128
			}
			podServerExceptList := []string{fmt.Sprintf("%s/%d", podB.Status.PodIP, hostMask)}
			egressRule1 := networkingv1.NetworkPolicyEgressRule{}
			egressRule1.To = append(egressRule1.To, networkingv1.NetworkPolicyPeer{IPBlock: &networkingv1.IPBlock{CIDR: podServerAllowCIDR, Except: podServerExceptList}})
			policyAllowCIDR := GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-cidr-egress-rule",
				map[string]string{"pod": "a"}, SetSpecEgressRules(egressRule1))
			CreatePolicy(ctx, k8s, policyAllowCIDR, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.Expect(NewPodString(nsX, "a"), NewPodString(nsX, "b"), false)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})

			// Create a second NetworkPolicy which allows access to podB
			podBIP := fmt.Sprintf("%s/%d", podB.Status.PodIP, hostMask)
			egressRule3 := networkingv1.NetworkPolicyEgressRule{}
			egressRule3.To = append(egressRule3.To, networkingv1.NetworkPolicyPeer{IPBlock: &networkingv1.IPBlock{CIDR: podBIP}})
			allowPolicy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-cidr-egress-rule",
				map[string]string{"pod": "a"}, SetSpecEgressRules(egressRule3))
			// SHOULD THIS BE UPDATE OR CREATE JAY TESTING 10/31
			UpdatePolicy(ctx, k8s, allowPolicy, nsX)

			reachabilityAllow := NewReachability(k8s.AllPodStrings(), true)
			reachabilityAllow.ExpectAllEgress(NewPodString(nsX, "a"), false)
			reachabilityAllow.Expect(NewPodString(nsX, "a"), NewPodString(nsX, "b"), true)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityAllow})
		})

		f.It("should enforce policies to check ingress and egress policies can be controlled independently based on PodSelector", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// Policies target x/a. We need x/b to verify a->b allowed while b->a can be
			// denied, and y/a as an external pod so the initial "all traffic allowed" step
			// covers cross-namespace connectivity as well.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a")
			nsX, _, _ := getK8sNamespaces(k8s)

			/*
					Test steps:
					1. Verify every pod in every namespace can talk to each other
				       - including a -> b and b -> a
					2. Create a policy to allow egress a -> b (target = a)
				    3. Create a policy to *deny* ingress b -> a (target = a)
					4. Verify a -> b allowed; b -> a blocked
			*/
			targetLabels := map[string]string{"pod": "a"}

			ginkgo.By("Creating a network policy for pod-a which allows Egress traffic to pod-b.")

			allowEgressPolicy := GenNetworkPolicyWithNameAndPodSelector("allow-egress-for-target",
				metav1.LabelSelector{MatchLabels: targetLabels}, SetSpecEgressRules(networkingv1.NetworkPolicyEgressRule{}))
			CreatePolicy(ctx, k8s, allowEgressPolicy, nsX)

			allowEgressReachability := NewReachability(k8s.AllPodStrings(), true)
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: allowEgressReachability})

			ginkgo.By("Creating a network policy for pod-a that denies traffic from pod-b.")

			denyAllIngressPolicy := GenNetworkPolicyWithNameAndPodSelector("deny-ingress-via-label-selector", metav1.LabelSelector{MatchLabels: targetLabels}, SetSpecIngressRules())
			CreatePolicy(ctx, k8s, denyAllIngressPolicy, nsX)

			denyIngressToXReachability := NewReachability(k8s.AllPodStrings(), true)
			denyIngressToXReachability.ExpectAllIngress(NewPodString(nsX, "a"), false)
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: denyIngressToXReachability})
		})

		// This test *does* apply to plugins that do not implement SCTP. It is a
		// security hole if you fail this test, because you are allowing TCP
		// traffic that is supposed to be blocked.
		f.It("should not mistakenly treat 'protocol: SCTP' as 'protocol: TCP', even if the plugin doesn't support SCTP", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{81}
			// One server (x/a) and one client (y/a) is sufficient: we create an SCTP-only
			// allow policy and verify that TCP to the same port remains blocked.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "y/a")
			nsX, _, _ := getK8sNamespaces(k8s)

			ginkgo.By("Creating a default-deny ingress policy.")
			// Empty podSelector blocks the entire namespace
			policy := GenNetworkPolicyWithNameAndPodSelector("deny-ingress", metav1.LabelSelector{}, SetSpecIngressRules())
			CreatePolicy(ctx, k8s, policy, nsX)

			ginkgo.By("Creating a network policy for the server which allows traffic only via SCTP on port 81.")
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.Ports = append(ingressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{IntVal: 81}, Protocol: &protocolSCTP})
			policy = GenNetworkPolicyWithNameAndPodMatchLabel("allow-only-sctp-ingress-on-port-81", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			ginkgo.By("Trying to connect to TCP port 81, which should be blocked by the deny-ingress policy.")
			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&Peer{}, &Peer{Namespace: nsX}, false)
			ValidateOrFail(k8s, &TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		// This test *does* apply to plugins that do not implement SCTP. It is a
		// security hole if you fail this test, because you are allowing TCP
		// traffic that is supposed to be blocked.
		f.It("should properly isolate pods that are selected by a policy allowing SCTP, even if the plugin doesn't support SCTP", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{81}
			// One server (x/a) and one client (y/a) is sufficient: an SCTP-only allow rule
			// should still isolate the pod for TCP traffic on other ports.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "y/a")
			nsX, _, _ := getK8sNamespaces(k8s)

			ginkgo.By("Creating a network policy for the server which allows traffic only via SCTP on port 80.")
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.Ports = append(ingressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{IntVal: 80}, Protocol: &protocolSCTP})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-only-sctp-ingress-on-port-80", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			ginkgo.By("Trying to connect to TCP port 81, which should be blocked by implicit isolation.")
			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(NewPodString(nsX, "a"), false)
			ValidateOrFail(k8s, &TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should not allow access by TCP when a policy specifies only UDP", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{81}
			// One server (x/a) and one client (y/a) is sufficient: a UDP-only allow rule
			// must not accidentally permit TCP connectivity.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "y/a")
			nsX, _, _ := getK8sNamespaces(k8s)

			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.Ports = append(ingressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{IntVal: 81}, Protocol: &protocolUDP})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-only-udp-ingress-on-port-81", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			ginkgo.By("Creating a network policy for the server which allows traffic only via UDP on port 81.")

			// Probing with TCP, so all traffic should be dropped.
			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(NewPodString(nsX, "a"), false)
			ValidateOrFail(k8s, &TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		// Note that this default ns functionality is maintained by the APIMachinery group, but we test it here anyways because its an important feature.
		f.It("should enforce policy to allow traffic based on NamespaceSelector with MatchLabels using default ns label", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// Policy on x/a allows ingress only from namespace Y (using the default ns name
			// label). We need x/b (same namespace) and z/a (other namespace) as negative
			// cases.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a", "z/a")
			nsX, nsY, nsZ := getK8sNamespaces(k8s)

			allowedLabels := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					v1.LabelMetadataName: nsY,
				},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedLabels})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-ns-selector-for-immutable-ns-label", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&Peer{Namespace: nsX}, &Peer{Namespace: nsX, Pod: "a"}, false)
			reachability.ExpectPeer(&Peer{Namespace: nsZ}, &Peer{Namespace: nsX, Pod: "a"}, false)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		// Note that this default ns functionality is maintained by the APIMachinery group, but we test it here anyways because its an important feature.
		f.It("should enforce policy based on NamespaceSelector with MatchExpressions using default ns label", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			// This uses MatchExpressions on the default namespace-name label for an egress
			// policy selecting x/a. The selector is NotIn{Y}, so x/a -> Y must be denied
			// while x/a -> X and x/a -> Z remain allowed.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a", "z/a")
			nsX, nsY, _ := getK8sNamespaces(k8s)

			allowedNamespaces := &metav1.LabelSelector{
				MatchExpressions: []metav1.LabelSelectorRequirement{{
					Key:      v1.LabelMetadataName,
					Operator: metav1.LabelSelectorOpNotIn,
					Values:   []string{nsY},
				}},
			}
			egressRule := networkingv1.NetworkPolicyEgressRule{}
			egressRule.To = append(egressRule.To, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedNamespaces})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-ns-y-match-selector-for-immutable-ns-label", map[string]string{"pod": "a"}, SetSpecEgressRules(egressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&Peer{Namespace: nsX, Pod: "a"}, &Peer{Namespace: nsY}, false)
			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})
	})
})

var _ = common.SIGDescribe("Netpol [LinuxOnly]", func() {
	f := framework.NewDefaultFramework("udp-network-policy")
	f.SkipNamespaceCreation = true
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	var k8s *kubeManager
	ginkgo.BeforeEach(func() {
		// Windows does not support UDP testing via agnhost.
		e2eskipper.SkipIfNodeOSDistroIs("windows")
	})

	ginkgo.Context("NetworkPolicy between server and client using UDP", func() {

		f.It("should support a 'default-deny-ingress' policy", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolUDP}
			ports := []int32{80}
			// UDP: namespace X has a default-deny-ingress policy, so we need 2 pods in X to
			// verify X-to-X ingress is blocked, and 2 pods in Y to verify Y-to-X is also
			// blocked while X-to-Y and Y-to-Y traffic remain allowed.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a", "y/b")
			nsX, _, _ := getK8sNamespaces(k8s)

			policy := GenNetworkPolicyWithNameAndPodSelector("deny-all", metav1.LabelSelector{}, SetSpecIngressRules())
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&Peer{}, &Peer{Namespace: nsX}, false)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolUDP, Reachability: reachability})
		})

		f.It("should enforce policy based on Ports", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolUDP}
			ports := []int32{81}
			// UDP port test: namespace X should allow ingress to x/a on port 81 from
			// namespace Y only. We include x/b as a same-namespace source (denied) and z/a
			// as an unrelated namespace (denied). One pod in Y is enough because the rule
			// is namespace-based.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a", "z/a")
			nsX, nsY, nsZ := getK8sNamespaces(k8s)

			ginkgo.By("Creating a network policy allowPort81Policy which only allows allow listed namespaces (y) to connect on exactly one port (81)")
			allowedLabels := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					namespaceLabelKey: nsY,
				},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedLabels})
			ingressRule.Ports = append(ingressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{IntVal: 81}, Protocol: &protocolUDP})
			allowPort81Policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-ingress-on-port-81-ns-x", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, allowPort81Policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&Peer{Namespace: nsX}, &Peer{Namespace: nsX, Pod: "a"}, false)
			reachability.ExpectPeer(&Peer{Namespace: nsZ}, &Peer{Namespace: nsX, Pod: "a"}, false)

			ValidateOrFail(k8s, &TestCase{ToPort: 81, Protocol: v1.ProtocolUDP, Reachability: reachability})
		})

		f.It("should enforce policy to allow traffic only from a pod in a different namespace based on PodSelector and NamespaceSelector", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolUDP}
			ports := []int32{80}
			// UDP: policy on x/a allows ingress only from y/a (namespace+pod selectors).
			// We keep x/b as a same-namespace/non-matching pod to ensure it cannot reach
			// x/a.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a")
			nsX, nsY, _ := getK8sNamespaces(k8s)

			allowedNamespaces := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					namespaceLabelKey: nsY,
				},
			}
			allowedPods := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"pod": "a",
				},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedNamespaces, PodSelector: allowedPods})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-ns-y-pod-a-via-namespace-pod-selector", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(NewPodString(nsX, "a"), false)
			reachability.Expect(NewPodString(nsY, "a"), NewPodString(nsX, "a"), true)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolUDP, Reachability: reachability})
		})
	})
})

var _ = common.SIGDescribe("Netpol", feature.SCTPConnectivity, "[LinuxOnly]", func() {
	f := framework.NewDefaultFramework("sctp-network-policy")
	f.SkipNamespaceCreation = true
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	var k8s *kubeManager
	ginkgo.BeforeEach(func() {
		// Windows does not support SCTP
		e2eskipper.SkipIfNodeOSDistroIs("windows")
	})

	ginkgo.Context("NetworkPolicy between server and client using SCTP", func() {

		f.It("should support a 'default-deny-ingress' policy", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolSCTP}
			ports := []int32{80}
			// SCTP: namespace X has a default-deny-ingress policy, so we need 2 pods in X
			// to verify X-to-X ingress is blocked, and 2 pods in Y to verify Y-to-X is also
			// blocked while X-to-Y and Y-to-Y traffic remain allowed.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a", "y/b")
			nsX, _, _ := getK8sNamespaces(k8s)

			policy := GenNetworkPolicyWithNameAndPodSelector("deny-all", metav1.LabelSelector{}, SetSpecIngressRules())
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&Peer{}, &Peer{Namespace: nsX}, false)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolSCTP, Reachability: reachability})
		})

		f.It("should enforce policy based on Ports", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolSCTP}
			ports := []int32{81}
			// SCTP port test: namespace X should allow ingress to x/a on port 81 from
			// namespace Y only. We include x/b as a same-namespace source (denied) and z/a
			// as an unrelated namespace (denied). One pod in Y is enough because the rule
			// is namespace-based.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a", "z/a")
			nsX, nsY, nsZ := getK8sNamespaces(k8s)

			ginkgo.By("Creating a network allowPort81Policy which only allows allow listed namespaces (y) to connect on exactly one port (81)")
			allowedLabels := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					namespaceLabelKey: nsY,
				},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedLabels})
			ingressRule.Ports = append(ingressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{IntVal: 81}, Protocol: &protocolSCTP})
			allowPort81Policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-ingress-on-port-81-ns-x", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, allowPort81Policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&Peer{Namespace: nsX}, &Peer{Namespace: nsX, Pod: "a"}, false)
			reachability.ExpectPeer(&Peer{Namespace: nsZ}, &Peer{Namespace: nsX, Pod: "a"}, false)

			ValidateOrFail(k8s, &TestCase{ToPort: 81, Protocol: v1.ProtocolSCTP, Reachability: reachability})
		})

		f.It("should enforce policy to allow traffic only from a pod in a different namespace based on PodSelector and NamespaceSelector", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolSCTP}
			ports := []int32{80}
			// SCTP: policy on x/a allows ingress only from y/a (namespace+pod selectors).
			// We keep x/b as a same-namespace/non-matching pod to ensure it cannot reach
			// x/a.
			k8s = initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "y/a")
			nsX, nsY, _ := getK8sNamespaces(k8s)

			allowedNamespaces := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					namespaceLabelKey: nsY,
				},
			}
			allowedPods := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"pod": "a",
				},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedNamespaces, PodSelector: allowedPods})
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-ns-y-pod-a-via-namespace-pod-selector", map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(NewPodString(nsX, "a"), false)
			reachability.Expect(NewPodString(nsY, "a"), NewPodString(nsX, "a"), true)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolSCTP, Reachability: reachability})
		})
	})
})

// getNamespaceNames returns the set of base namespace names used by this test, taking a root ns as input.
// The framework will also append a unique suffix when creating the namespaces.
// This allows tests to run in parallel.
func getNamespaceNames(rootNs string) []string {
	if rootNs != "" {
		rootNs += "-"
	}
	nsX := fmt.Sprintf("%sx", rootNs)
	nsY := fmt.Sprintf("%sy", rootNs)
	nsZ := fmt.Sprintf("%sz", rootNs)
	return []string{nsX, nsY, nsZ}
}

// modelFromPodStrings converts pod references like "x/a" into a model that uses the
// actual generated namespace names.
func modelFromPodStrings(namespaces []string, modelPods []string, protocols []v1.Protocol, ports []int32) (*Model, error) {
	if len(modelPods) == 0 {
		return nil, fmt.Errorf("model pod list must not be empty")
	}

	namespaceBySuffix := map[string]string{
		"x": namespaces[0],
		"y": namespaces[1],
		"z": namespaces[2],
	}
	podNamesByNamespace := map[string]sets.Set[string]{
		namespaces[0]: sets.New[string](),
		namespaces[1]: sets.New[string](),
		namespaces[2]: sets.New[string](),
	}

	for _, modelPod := range modelPods {
		namespaceSuffix, podName, found := strings.Cut(modelPod, "/")
		if !found || namespaceSuffix == "" || podName == "" {
			return nil, fmt.Errorf("invalid model pod %q; expected <namespace>/<pod> like x/a", modelPod)
		}

		namespaceName, found := namespaceBySuffix[namespaceSuffix]
		if !found {
			return nil, fmt.Errorf("invalid model pod %q; namespace suffix %q must be one of x, y, z", modelPod, namespaceSuffix)
		}

		podNamesByNamespace[namespaceName].Insert(podName)
	}

	return newModelWithPerNamespacePodNames(namespaces, podNamesByNamespace, ports, protocols), nil
}

// getK8sNamespaces returns the 3 actual namespace names.
func getK8sNamespaces(k8s *kubeManager) (string, string, string) {
	ns := k8s.NamespaceNames()
	return ns[0], ns[1], ns[2]
}

func initializeCluster(ctx context.Context, f *framework.Framework, protocols []v1.Protocol, ports []int32, modelPods ...string) (*kubeManager, error) {
	dnsDomain := framework.TestContext.ClusterDNSDomain
	framework.Logf("dns domain: %s", dnsDomain)

	k8s := newKubeManager(f, dnsDomain)
	rootNs := f.BaseName
	namespaceNames := getNamespaceNames(rootNs)

	model, err := modelFromPodStrings(namespaceNames, modelPods, protocols, ports)
	if err != nil {
		return nil, err
	}

	framework.Logf("initializing cluster: ensuring namespaces, pods and services exist and are ready")

	if err := k8s.initializeClusterFromModel(ctx, model); err != nil {
		return nil, err
	}

	framework.Logf("finished initializing cluster state")

	if err := waitForHTTPServers(k8s, model); err != nil {
		return nil, err
	}

	return k8s, nil
}

// initializeResources uses the e2e framework to create all necessary namespace resources, based on the network policy
// model derived from the framework.  It then waits for the resources described by the model to be up and running
// (i.e. all pods are ready and running in their namespaces).
func initializeResources(ctx context.Context, f *framework.Framework, protocols []v1.Protocol, ports []int32, modelPods ...string) *kubeManager {
	k8s, err := initializeCluster(ctx, f, protocols, ports, modelPods...)
	framework.ExpectNoError(err, "unable to initialize resources")
	return k8s
}

// makeLargeCIDRForIP returns a CIDR that matches the given IP and many many many other
// IPs. (Specifically, it returns the /4 that contains the IP.)
func makeLargeCIDRForIP(ip string) string {
	podIP := utilnet.ParseIPSloppy(ip)
	if ip4 := podIP.To4(); ip4 != nil {
		podIP = ip4
	}
	cidrBase := podIP.Mask(net.CIDRMask(4, 8*len(podIP)))
	return fmt.Sprintf("%s/4", cidrBase.String())
}
