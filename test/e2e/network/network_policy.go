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

package network

import (
	"context"
	"errors"
	"fmt"
	"net"
	"time"

	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"

	v1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"

	"github.com/onsi/ginkgo/v2"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/network/common"
	netpol "k8s.io/kubernetes/test/e2e/network/netpol"
	admissionapi "k8s.io/pod-security-admission/api"
	utilnet "k8s.io/utils/net"
)

const (
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
howing the connectivity of pods in other namespaces is key information to show whether a network policy is working as intended or not.

We use 3 namespaces each with 3 pods, and probe all combinations ( 9 pods x 9 pods = 81 data points ) -- including cross-namespace calls.
rm -rf ~/.cache/golangci-lint
./hack/verify-golangci-lint.sh


		Oct 19 10:34:16.907: INFO: expected:

		-	x/a	x/b	x/c	y/a	y/b	y/c	z/a	z/b	z/c
		x/a	X	.	.	.	.	.	.	.	.
		x/b	X	.	.	.	.	.	.	.	.
		x/c	X	.	.	.	.	.	.	.	.
		y/a	.	.	.	.	.	.	.	.	.
		y/b	.	.	.	.	.	.	.	.	.
		y/c	.	.	.	.	.	.	.	.	.
		z/a	X	.	.	.	.	.	.	.	.
		z/b	X	.	.	.	.	.	.	.	.
		z/c	X	.	.	.	.	.	.	.	.

		Oct 19 10:34:16.907: INFO: observed:

		-	x/a	x/b	x/c	y/a	y/b	y/c	z/a	z/b	z/c
		x/a	X	.	.	.	.	.	.	.	.
		x/b	X	.	.	.	.	.	.	.	.
		x/c	X	.	.	.	.	.	.	.	.
		y/a	.	.	.	.	.	.	.	.	.
		y/b	.	.	.	.	.	.	.	.	.
		y/c	.	.	.	.	.	.	.	.	.
		z/a	X	.	.	.	.	.	.	.	.
		z/b	X	.	.	.	.	.	.	.	.
		z/c	X	.	.	.	.	.	.	.	.

		Oct 19 10:34:16.907: INFO: comparison:

		-	x/a	x/b	x/c	y/a	y/b	y/c	z/a	z/b	z/c
		x/a	.	.	.	.	.	.	.	.	.
		x/b	.	.	.	.	.	.	.	.	.
		x/c	.	.	.	.	.	.	.	.	.
		y/a	.	.	.	.	.	.	.	.	.
		y/b	.	.	.	.	.	.	.	.	.
		y/c	.	.	.	.	.	.	.	.	.
		z/a	.	.	.	.	.	.	.	.	.
		z/b	.	.	.	.	.	.	.	.	.
		z/c	.	.	.	.	.	.	.	.	.
*/

var _ = common.SIGDescribe("Netpol", func() {
	f := framework.NewDefaultFramework("netpol")
	f.SkipNamespaceCreation = true // we create our own 3 test namespaces, we don't need the default one
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.Context("NetworkPolicy between server and client", func() {
		var k8s *netpol.KubeManager

		f.It("should support a 'default-deny-ingress' policy", feature.NetworkPolicy, func(ctx context.Context) {

			// Only poll TCP
			protocols := []v1.Protocol{protocolTCP}

			// Only testing port 80
			ports := []int32{80}

			// Create pods and namespaces for this test
			k8s = initializeResources(ctx, f, protocols, ports)

			// Only going to make a policy in namespace X
			nsX, _, _ := getK8sNamespaces(k8s)
			policy := netpol.GenNetworkPolicyWithNameAndPodSelector("deny-ingress", metav1.LabelSelector{}, netpol.SetSpecIngressRules())

			// Create the policy
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			// Make a truth table of connectivity for all pods in ns x y z
			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			// Set the nsX as false, since it has a policy that blocks traffic
			reachability.ExpectPeer(&netpol.Peer{}, &netpol.Peer{Namespace: nsX}, false)

			// Confirm that the real world connectivity matches our matrix
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should support a 'default-deny-all' policy", feature.NetworkPolicy, func(ctx context.Context) {
			policy := netpol.GenNetworkPolicyWithNameAndPodSelector("deny-all", metav1.LabelSelector{}, netpol.SetSpecIngressRules(), netpol.SetSpecEgressRules())
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&netpol.Peer{}, &netpol.Peer{Namespace: nsX}, false)
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsX}, &netpol.Peer{}, false)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy to allow traffic from pods within server namespace based on PodSelector", feature.NetworkPolicy, func(ctx context.Context) {
			allowedPods := metav1.LabelSelector{
				MatchLabels: map[string]string{
					"pod": "b",
				},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{PodSelector: &allowedPods})
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("x-a-allows-x-b", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))

			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(netpol.NewPodString(nsX, "a"), false)
			reachability.Expect(netpol.NewPodString(nsX, "b"), netpol.NewPodString(nsX, "a"), true)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy to allow ingress traffic for a target", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			ginkgo.By("having a deny all ingress policy", func() {
				// Deny all Ingress traffic policy to pods on namespace nsX
				policy := netpol.GenNetworkPolicyWithNameAndPodSelector("deny-all", metav1.LabelSelector{}, netpol.SetSpecIngressRules())
				netpol.CreatePolicy(ctx, k8s, policy, nsX)
			})

			// Allow Ingress traffic only to pod x/a from any pod
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{PodSelector: &metav1.LabelSelector{}, NamespaceSelector: &metav1.LabelSelector{}})
			allowPolicy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-all-to-a", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, allowPolicy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(netpol.NewPodString(nsX, "a"), true)
			reachability.ExpectAllIngress(netpol.NewPodString(nsX, "b"), false)
			reachability.ExpectAllIngress(netpol.NewPodString(nsX, "c"), false)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy to allow ingress traffic from pods in all namespaces", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: &metav1.LabelSelector{MatchLabels: map[string]string{}}})
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-from-another-ns", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy to allow traffic only from a different namespace, based on NamespaceSelector", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, nsY, nsZ := getK8sNamespaces(k8s)
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: &metav1.LabelSelector{MatchLabels: map[string]string{namespaceLabelKey: nsY}}})
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-ns-selector", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			// disallow all traffic from the x or z namespaces
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsX}, &netpol.Peer{Namespace: nsX, Pod: "a"}, false)
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsZ}, &netpol.Peer{Namespace: nsX, Pod: "a"}, false)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy based on PodSelector with MatchExpressions", feature.NetworkPolicy, func(ctx context.Context) {
			allowedPods := metav1.LabelSelector{
				MatchExpressions: []metav1.LabelSelectorRequirement{{
					Key:      "pod",
					Operator: metav1.LabelSelectorOpIn,
					Values:   []string{"b"},
				}},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{PodSelector: &allowedPods})
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("x-a-allows-x-b", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))

			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(netpol.NewPodString(nsX, "a"), false)
			reachability.Expect(netpol.NewPodString(nsX, "b"), netpol.NewPodString(nsX, "a"), true)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy based on NamespaceSelector with MatchExpressions", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
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
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-ns-y-match-selector", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			// disallow all traffic from the x or z namespaces
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsX}, &netpol.Peer{Namespace: nsX, Pod: "a"}, false)
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsZ}, &netpol.Peer{Namespace: nsX, Pod: "a"}, false)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy based on PodSelector or NamespaceSelector", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
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
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-ns-y-match-selector", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.Expect(netpol.NewPodString(nsX, "a"), netpol.NewPodString(nsX, "a"), false)
			reachability.Expect(netpol.NewPodString(nsX, "c"), netpol.NewPodString(nsX, "a"), false)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy based on PodSelector and NamespaceSelector", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
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
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-ns-y-podselector-and-nsselector", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(netpol.NewPodString(nsX, "a"), false)
			reachability.Expect(netpol.NewPodString(nsY, "b"), netpol.NewPodString(nsX, "a"), true)
			reachability.Expect(netpol.NewPodString(nsZ, "b"), netpol.NewPodString(nsX, "a"), true)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy based on Multiple PodSelectors and NamespaceSelectors", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, nsY, nsZ := getK8sNamespaces(k8s)
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
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-ns-y-z-pod-b-c", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsX}, &netpol.Peer{Namespace: nsX, Pod: "a"}, false)
			reachability.Expect(netpol.NewPodString(nsY, "a"), netpol.NewPodString(nsX, "a"), false)
			reachability.Expect(netpol.NewPodString(nsZ, "a"), netpol.NewPodString(nsX, "a"), false)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy based on any PodSelectors", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			for _, label := range []map[string]string{{"pod": "b"}, {"pod": "c"}} {
				ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{PodSelector: &metav1.LabelSelector{MatchLabels: label}})
			}
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-ns-x-pod-b-c", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(netpol.NewPodString(nsX, "a"), false)

			// Connect Pods b and c to pod a from namespace nsX
			reachability.Expect(netpol.NewPodString(nsX, "b"), netpol.NewPodString(nsX, "a"), true)
			reachability.Expect(netpol.NewPodString(nsX, "c"), netpol.NewPodString(nsX, "a"), true)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy to allow traffic only from a pod in a different namespace based on PodSelector and NamespaceSelector", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
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
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-ns-y-pod-a-via-namespace-pod-selector", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(netpol.NewPodString(nsX, "a"), false)
			reachability.Expect(netpol.NewPodString(nsY, "a"), netpol.NewPodString(nsX, "a"), true)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce policy based on Ports", feature.NetworkPolicy, func(ctx context.Context) {
			ginkgo.By("Creating a network allowPort81Policy which only allows allow listed namespaces (y) to connect on exactly one port (81)")
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{81}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, nsY, nsZ := getK8sNamespaces(k8s)
			allowedLabels := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					namespaceLabelKey: nsY,
				},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedLabels})
			ingressRule.Ports = append(ingressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{IntVal: 81}, Protocol: &protocolTCP})
			allowPort81Policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-ns-selector", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, allowPort81Policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsX}, &netpol.Peer{Namespace: nsX, Pod: "a"}, false)
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsY}, &netpol.Peer{Namespace: nsX, Pod: "a"}, true)
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsZ}, &netpol.Peer{Namespace: nsX, Pod: "a"}, false)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce multiple, stacked policies with overlapping podSelectors", feature.NetworkPolicy, func(ctx context.Context) {
			ginkgo.By("Creating a network allowPort81Policy which only allows allow listed namespaces (y) to connect on exactly one port (81)")
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80, 81}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, nsY, nsZ := getK8sNamespaces(k8s)
			allowedLabels := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					namespaceLabelKey: nsY,
				},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedLabels})
			ingressRule.Ports = append(ingressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{IntVal: 81}, Protocol: &protocolTCP})
			allowPort81Policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-ns-selector", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, allowPort81Policy, nsX)

			reachabilityALLOW := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachabilityALLOW.ExpectPeer(&netpol.Peer{Namespace: nsX}, &netpol.Peer{Namespace: nsX, Pod: "a"}, false)
			reachabilityALLOW.ExpectPeer(&netpol.Peer{Namespace: nsY}, &netpol.Peer{Namespace: nsX, Pod: "a"}, true)
			reachabilityALLOW.ExpectPeer(&netpol.Peer{Namespace: nsZ}, &netpol.Peer{Namespace: nsX, Pod: "a"}, false)

			ginkgo.By("Verifying traffic on port 81.")
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachabilityALLOW})

			reachabilityDENY := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachabilityDENY.ExpectAllIngress(netpol.NewPodString(nsX, "a"), false)

			ginkgo.By("Verifying traffic on port 80.")
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityDENY})

			ingressRule = networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedLabels})
			ingressRule.Ports = append(ingressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{IntVal: 80}, Protocol: &protocolTCP})

			allowPort80Policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-ns-selector-80", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, allowPort80Policy, nsX)

			ginkgo.By("Verifying that we can add a policy to unblock port 80")
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityALLOW})
		})

		f.It("should support allow-all policy", feature.NetworkPolicy, func(ctx context.Context) {
			ginkgo.By("Creating a network policy which allows all traffic.")
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-all", map[string]string{}, netpol.SetSpecIngressRules(networkingv1.NetworkPolicyIngressRule{}))
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80, 81}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			ginkgo.By("Testing pods can connect to both ports when an 'allow-all' policy is present.")
			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should allow ingress access on one named port", feature.NetworkPolicy, func(ctx context.Context) {
			IngressRules := networkingv1.NetworkPolicyIngressRule{}
			IngressRules.Ports = append(IngressRules.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{Type: intstr.String, StrVal: "serve-81-tcp"}})
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-all", map[string]string{}, netpol.SetSpecIngressRules(IngressRules))
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80, 81}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			ginkgo.By("Blocking all ports other then 81 in the entire namespace")

			reachabilityPort81 := netpol.NewReachability(k8s.AllPodStrings(), true)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachabilityPort81})

			// disallow all traffic to the x namespace
			reachabilityPort80 := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachabilityPort80.ExpectPeer(&netpol.Peer{}, &netpol.Peer{Namespace: nsX}, false)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityPort80})
		})

		f.It("should allow ingress access from namespace on one named port", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80, 81}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, nsY, nsZ := getK8sNamespaces(k8s)
			allowedLabels := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					namespaceLabelKey: nsY,
				},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedLabels})
			ingressRule.Ports = append(ingressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{Type: intstr.String, StrVal: "serve-80-tcp"}, Protocol: &protocolTCP})
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-ns-selector-80", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			// disallow all traffic from the x or z namespaces
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsX}, &netpol.Peer{Namespace: nsX, Pod: "a"}, false)
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsZ}, &netpol.Peer{Namespace: nsX, Pod: "a"}, false)

			ginkgo.By("Verify that port 80 is allowed for namespace y")
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})

			ginkgo.By("Verify that port 81 is blocked for all namespaces including y")
			reachabilityFAIL := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachabilityFAIL.ExpectAllIngress(netpol.NewPodString(nsX, "a"), false)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachabilityFAIL})
		})

		f.It("should allow egress access on one named port", feature.NetworkPolicy, func(ctx context.Context) {
			ginkgo.By("validating egress from port 81 to port 80")
			egressRule := networkingv1.NetworkPolicyEgressRule{}
			egressRule.Ports = append(egressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{Type: intstr.String, StrVal: "serve-80-tcp"}})
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-egress", map[string]string{}, netpol.SetSpecEgressRules(egressRule))

			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80, 81}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachabilityPort80 := netpol.NewReachability(k8s.AllPodStrings(), true)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityPort80})

			// meanwhile no traffic over 81 should work, since our egress policy is on 80
			reachabilityPort81 := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachabilityPort81.ExpectPeer(&netpol.Peer{Namespace: nsX}, &netpol.Peer{}, false)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachabilityPort81})
		})

		f.It("should enforce updated policy", feature.NetworkPolicy, func(ctx context.Context) {
			ginkgo.By("Using the simplest possible mutation: start with allow all, then switch to deny all")
			// part 1) allow all
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-all-mutate-to-deny-all", map[string]string{}, netpol.SetSpecIngressRules(networkingv1.NetworkPolicyIngressRule{}))
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{81}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachability})

			// part 2) update the policy to deny all
			policy.Spec.Ingress = []networkingv1.NetworkPolicyIngressRule{}
			netpol.UpdatePolicy(ctx, k8s, policy, nsX)

			reachabilityDeny := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachabilityDeny.ExpectPeer(&netpol.Peer{}, &netpol.Peer{Namespace: nsX}, false)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachabilityDeny})
		})

		f.It("should allow ingress access from updated namespace", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, nsY, _ := getK8sNamespaces(k8s)
			ginkgo.DeferCleanup(netpol.DeleteNamespaceLabel, k8s, nsY, "ns2")

			allowedLabels := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"ns2": "updated",
				},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedLabels})
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-ns-selector", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(netpol.NewPodString(nsX, "a"), false)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})

			// add a new label
			netpol.AddNamespaceLabel(ctx, k8s, nsY, "ns2", "updated")

			// anything from namespace 'y' should be able to get to x/a
			reachabilityWithLabel := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachabilityWithLabel.ExpectAllIngress(netpol.NewPodString(nsX, "a"), false)
			reachabilityWithLabel.ExpectPeer(&netpol.Peer{Namespace: nsY}, &netpol.Peer{}, true)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityWithLabel})
		})

		f.It("should allow ingress access from updated pod", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			ginkgo.DeferCleanup(netpol.ResetPodLabels, k8s, nsX, "b")

			// add a new label
			matchLabels := map[string]string{"pod": "b", "pod2": "updated"}
			allowedLabels := &metav1.LabelSelector{MatchLabels: matchLabels}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{PodSelector: allowedLabels})
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-pod-selector", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(netpol.NewPodString(nsX, "a"), false)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})

			netpol.AddPodLabels(ctx, k8s, nsX, "b", matchLabels)

			ginkgo.By("x/b is able to reach x/a when label is updated")

			reachabilityWithLabel := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachabilityWithLabel.ExpectAllIngress(netpol.NewPodString(nsX, "a"), false)
			reachabilityWithLabel.Expect(netpol.NewPodString(nsX, "b"), netpol.NewPodString(nsX, "a"), true)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityWithLabel})
		})

		f.It("should deny ingress from pods on other namespaces", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, nsY, nsZ := getK8sNamespaces(k8s)
			IngressRules := networkingv1.NetworkPolicyIngressRule{}
			IngressRules.From = append(IngressRules.From, networkingv1.NetworkPolicyPeer{PodSelector: &metav1.LabelSelector{MatchLabels: map[string]string{}}})
			policy := netpol.GenNetworkPolicyWithNameAndPodSelector("deny-empty-policy", metav1.LabelSelector{}, netpol.SetSpecIngressRules(IngressRules))
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsY}, &netpol.Peer{Namespace: nsX}, false)
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsZ}, &netpol.Peer{Namespace: nsX}, false)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should deny ingress access to updated pod", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			ginkgo.DeferCleanup(netpol.ResetPodLabels, k8s, nsX, "a")

			policy := netpol.GenNetworkPolicyWithNameAndPodSelector("deny-ingress-via-label-selector",
				metav1.LabelSelector{MatchLabels: map[string]string{"target": "isolated"}}, netpol.SetSpecIngressRules())
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			ginkgo.By("Verify that everything can reach x/a")
			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})

			netpol.AddPodLabels(ctx, k8s, nsX, "a", map[string]string{"target": "isolated"})

			reachabilityIsolated := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachabilityIsolated.ExpectAllIngress(netpol.NewPodString(nsX, "a"), false)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityIsolated})
		})

		f.It("should deny egress from pods based on PodSelector", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			policy := netpol.GenNetworkPolicyWithNameAndPodSelector("deny-egress-pod-a", metav1.LabelSelector{MatchLabels: map[string]string{"pod": "a"}}, netpol.SetSpecEgressRules())
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllEgress(netpol.NewPodString(nsX, "a"), false)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should deny egress from all pods in a namespace", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			policy := netpol.GenNetworkPolicyWithNameAndPodSelector("deny-egress-ns-x", metav1.LabelSelector{}, netpol.SetSpecEgressRules())
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsX}, &netpol.Peer{}, false)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should work with Ingress, Egress specified together", feature.NetworkPolicy, func(ctx context.Context) {
			allowedPodLabels := &metav1.LabelSelector{MatchLabels: map[string]string{"pod": "b"}}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{PodSelector: allowedPodLabels})
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-pod-selector", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))

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
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80, 81}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachabilityPort80 := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachabilityPort80.ExpectAllIngress(netpol.NewPodString(nsX, "a"), false)
			reachabilityPort80.Expect(netpol.NewPodString(nsX, "b"), netpol.NewPodString(nsX, "a"), true)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityPort80})

			ginkgo.By("validating that port 81 doesn't work")
			// meanwhile no egress traffic on 81 should work, since our egress policy is on 80
			reachabilityPort81 := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachabilityPort81.ExpectAllIngress(netpol.NewPodString(nsX, "a"), false)
			reachabilityPort81.ExpectAllEgress(netpol.NewPodString(nsX, "a"), false)
			reachabilityPort81.Expect(netpol.NewPodString(nsX, "b"), netpol.NewPodString(nsX, "a"), true)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachabilityPort81})
		})

		f.It("should support denying of egress traffic on the client side (even if the server explicitly allows this traffic)", feature.NetworkPolicy, func(ctx context.Context) {
			// x/a --> y/a and y/b
			// Egress allowed to y/a only. Egress to y/b should be blocked
			// Ingress on y/a and y/b allow traffic from x/a
			// Expectation: traffic from x/a to y/a allowed only, traffic from x/a to y/b denied by egress policy

			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
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
			egressPolicy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-to-ns-y-pod-a", map[string]string{"pod": "a"}, netpol.SetSpecEgressRules(egressRule1))
			netpol.CreatePolicy(ctx, k8s, egressPolicy, nsX)

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
			allowIngressPolicyPodA := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-from-xa-on-ya-match-selector", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			allowIngressPolicyPodB := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-from-xa-on-yb-match-selector", map[string]string{"pod": "b"}, netpol.SetSpecIngressRules(ingressRule))

			netpol.CreatePolicy(ctx, k8s, allowIngressPolicyPodA, nsY)
			netpol.CreatePolicy(ctx, k8s, allowIngressPolicyPodB, nsY)

			// While applying the policies, traffic needs to be allowed by both egress and ingress rules.
			// Egress rules only
			// 	xa	xb	xc	ya	yb	yc	za	zb	zc
			// xa	X	X	X	.	*X*	X	X	X	X
			// xb	.	.	.	.	.	.	.	.	.
			// xc	.	.	.	.	.	.	.	.	.
			// ya	.	.	.	.	.	.	.	.	.
			// yb	.	.	.	.	.	.	.	.	.
			// yc	.	.	.	.	.	.	.	.	.
			// za	.	.	.	.	.	.	.	.	.
			// zb	.	.	.	.	.	.	.	.	.
			// zc	.	.	.	.	.	.	.	.	.
			// Ingress rules only
			// 	xa	xb	xc	ya	yb	yc	za	zb	zc
			// xa	.	.	.	*.*	.	.	.	.	.
			// xb	.	.	X	X	.	.	.	.	.
			// xc	.	.	X	X	.	.	.	.	.
			// ya	.	.	X	X	.	.	.	.	.
			// yb	.	.	X	X	.	.	.	.	.
			// yc	.	.	X	X	.	.	.	.	.
			// za	.	.	X	X	.	.	.	.	.
			// zb	.	.	X	X	.	.	.	.	.
			// zc	.	.	X	X	.	.	.	.	.
			// In the resulting truth table, connections from x/a should only be allowed to y/a. x/a to y/b should be blocked by the egress on x/a.
			// Expected results
			// 	xa	xb	xc	ya	yb	yc	za	zb	zc
			// xa	X	X	X	.	*X*	X	X	X	X
			// xb	.	.	.	X	X	.	.	.	.
			// xc	.	.	.	X	X	.	.	.	.
			// ya	.	.	.	X	X	.	.	.	.
			// yb	.	.	.	X	X	.	.	.	.
			// yc	.	.	.	X	X	.	.	.	.
			// za	.	.	.	X	X	.	.	.	.
			// zb	.	.	.	X	X	.	.	.	.
			// zc	.	.	.	X	X	.	.	.	.

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			// Default all traffic flows.
			// Exception: x/a can only egress to y/a, others are false
			// Exception: y/a can only allow ingress from x/a, others are false
			// Exception: y/b has no allowed traffic (due to limit on x/a egress)

			reachability.ExpectPeer(&netpol.Peer{Namespace: nsX, Pod: "a"}, &netpol.Peer{}, false)
			reachability.ExpectPeer(&netpol.Peer{}, &netpol.Peer{Namespace: nsY, Pod: "a"}, false)
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsX, Pod: "a"}, &netpol.Peer{Namespace: nsY, Pod: "a"}, true)
			reachability.ExpectPeer(&netpol.Peer{}, &netpol.Peer{Namespace: nsY, Pod: "b"}, false)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce egress policy allowing traffic to a server in a different namespace based on PodSelector and NamespaceSelector", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
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
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-to-ns-y-pod-a", map[string]string{"pod": "a"}, netpol.SetSpecEgressRules(egressRule1))
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllEgress(netpol.NewPodString(nsX, "a"), false)
			reachability.Expect(netpol.NewPodString(nsX, "a"), netpol.NewPodString(nsY, "a"), true)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce ingress policy allowing any port traffic to a server on a specific protocol", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP, protocolUDP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.Ports = append(ingressRule.Ports, networkingv1.NetworkPolicyPort{Protocol: &protocolTCP})
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-ingress-by-proto", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachabilityTCP := netpol.NewReachability(k8s.AllPodStrings(), true)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityTCP})

			reachabilityUDP := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachabilityUDP.ExpectPeer(&netpol.Peer{}, &netpol.Peer{Namespace: nsX, Pod: "a"}, false)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolUDP, Reachability: reachabilityUDP})
		})

		f.It("should enforce multiple ingress policies with ingress allow-all policy taking precedence", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{81}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			IngressRules := networkingv1.NetworkPolicyIngressRule{}
			IngressRules.Ports = append(IngressRules.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{Type: intstr.Int, IntVal: 80}})
			policyAllowOnlyPort80 := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-ingress-port-80", map[string]string{}, netpol.SetSpecIngressRules(IngressRules))
			netpol.CreatePolicy(ctx, k8s, policyAllowOnlyPort80, nsX)

			ginkgo.By("The policy targets port 80 -- so let's make sure traffic on port 81 is blocked")

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&netpol.Peer{}, &netpol.Peer{Namespace: nsX}, false)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachability})

			ginkgo.By("Allowing all ports")

			policyAllowAll := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-ingress", map[string]string{}, netpol.SetSpecIngressRules(networkingv1.NetworkPolicyIngressRule{}))
			netpol.CreatePolicy(ctx, k8s, policyAllowAll, nsX)

			reachabilityAll := netpol.NewReachability(k8s.AllPodStrings(), true)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachabilityAll})
		})

		f.It("should enforce multiple egress policies with egress allow-all policy taking precedence", feature.NetworkPolicy, func(ctx context.Context) {
			egressRule := networkingv1.NetworkPolicyEgressRule{}
			egressRule.Ports = append(egressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{Type: intstr.Int, IntVal: 80}})
			policyAllowPort80 := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-egress-port-80", map[string]string{}, netpol.SetSpecEgressRules(egressRule))
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{81}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			netpol.CreatePolicy(ctx, k8s, policyAllowPort80, nsX)

			ginkgo.By("Making sure ingress doesn't work other than port 80")

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsX}, &netpol.Peer{}, false)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachability})

			ginkgo.By("Allowing all ports")

			policyAllowAll := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-egress", map[string]string{}, netpol.SetSpecEgressRules(networkingv1.NetworkPolicyEgressRule{}))
			netpol.CreatePolicy(ctx, k8s, policyAllowAll, nsX)

			reachabilityAll := netpol.NewReachability(k8s.AllPodStrings(), true)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachabilityAll})
		})

		f.It("should stop enforcing policies after they are deleted", feature.NetworkPolicy, func(ctx context.Context) {
			ginkgo.By("Creating a network policy for the server which denies all traffic.")

			// Deny all traffic into and out of "x".
			policy := netpol.GenNetworkPolicyWithNameAndPodSelector("deny-all", metav1.LabelSelector{}, netpol.SetSpecIngressRules(), netpol.SetSpecEgressRules())
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			netpol.CreatePolicy(ctx, k8s, policy, nsX)
			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)

			// Expect all traffic into, and out of "x" to be False.
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsX}, &netpol.Peer{}, false)
			reachability.ExpectPeer(&netpol.Peer{}, &netpol.Peer{Namespace: nsX}, false)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})

			err := k8s.CleanNetworkPolicies(ctx)
			time.Sleep(3 * time.Second) // TODO we can remove this eventually, its just a hack to keep CI stable.
			framework.ExpectNoError(err, "unable to clean network policies")

			// Now the policy is deleted, we expect all connectivity to work again.
			reachabilityAll := netpol.NewReachability(k8s.AllPodStrings(), true)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityAll})
		})

		// TODO, figure out how the next 3 tests should work with dual stack : do we need a different abstraction then just "podIP"?

		f.It("should allow egress access to server in CIDR block", feature.NetworkPolicy, func(ctx context.Context) {
			// Getting podServer's status to get podServer's IP, to create the CIDR
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
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
			policyAllowCIDR := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-cidr-egress-rule",
				map[string]string{"pod": "a"}, netpol.SetSpecEgressRules(egressRule1))
			netpol.CreatePolicy(ctx, k8s, policyAllowCIDR, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllEgress(netpol.NewPodString(nsX, "a"), false)
			reachability.Expect(netpol.NewPodString(nsX, "a"), netpol.NewPodString(nsY, "b"), true)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce except clause while egress access to server in CIDR block", feature.NetworkPolicy, func(ctx context.Context) {
			// Getting podServer's status to get podServer's IP, to create the CIDR with except clause
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)

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
			policyAllowCIDR := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-cidr-egress-rule", map[string]string{"pod": "a"}, netpol.SetSpecEgressRules(egressRule1))

			netpol.CreatePolicy(ctx, k8s, policyAllowCIDR, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.Expect(netpol.NewPodString(nsX, "a"), netpol.NewPodString(nsX, "b"), false)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should ensure an IP overlapping both IPBlock.CIDR and IPBlock.Except is allowed", feature.NetworkPolicy, func(ctx context.Context) {
			// Getting podServer's status to get podServer's IP, to create the CIDR with except clause
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)

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
			policyAllowCIDR := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-cidr-egress-rule",
				map[string]string{"pod": "a"}, netpol.SetSpecEgressRules(egressRule1))
			netpol.CreatePolicy(ctx, k8s, policyAllowCIDR, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.Expect(netpol.NewPodString(nsX, "a"), netpol.NewPodString(nsX, "b"), false)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})

			// Create a second NetworkPolicy which allows access to podB
			podBIP := fmt.Sprintf("%s/%d", podB.Status.PodIP, hostMask)
			egressRule3 := networkingv1.NetworkPolicyEgressRule{}
			egressRule3.To = append(egressRule3.To, networkingv1.NetworkPolicyPeer{IPBlock: &networkingv1.IPBlock{CIDR: podBIP}})
			allowPolicy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-cidr-egress-rule",
				map[string]string{"pod": "a"}, netpol.SetSpecEgressRules(egressRule3))
			// SHOULD THIS BE UPDATE OR CREATE JAY TESTING 10/31
			netpol.UpdatePolicy(ctx, k8s, allowPolicy, nsX)

			reachabilityAllow := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachabilityAllow.ExpectAllEgress(netpol.NewPodString(nsX, "a"), false)
			reachabilityAllow.Expect(netpol.NewPodString(nsX, "a"), netpol.NewPodString(nsX, "b"), true)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityAllow})
		})

		f.It("should enforce policies to check ingress and egress policies can be controlled independently based on PodSelector", feature.NetworkPolicy, func(ctx context.Context) {
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

			allowEgressPolicy := netpol.GenNetworkPolicyWithNameAndPodSelector("allow-egress-for-target",
				metav1.LabelSelector{MatchLabels: targetLabels}, netpol.SetSpecEgressRules(networkingv1.NetworkPolicyEgressRule{}))
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			netpol.CreatePolicy(ctx, k8s, allowEgressPolicy, nsX)

			allowEgressReachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: allowEgressReachability})

			ginkgo.By("Creating a network policy for pod-a that denies traffic from pod-b.")

			denyAllIngressPolicy := netpol.GenNetworkPolicyWithNameAndPodSelector("deny-ingress-via-label-selector", metav1.LabelSelector{MatchLabels: targetLabels}, netpol.SetSpecIngressRules())
			netpol.CreatePolicy(ctx, k8s, denyAllIngressPolicy, nsX)

			denyIngressToXReachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			denyIngressToXReachability.ExpectAllIngress(netpol.NewPodString(nsX, "a"), false)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: denyIngressToXReachability})
		})

		// This test *does* apply to plugins that do not implement SCTP. It is a
		// security hole if you fail this test, because you are allowing TCP
		// traffic that is supposed to be blocked.
		f.It("should not mistakenly treat 'protocol: SCTP' as 'protocol: TCP', even if the plugin doesn't support SCTP", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{81}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)

			ginkgo.By("Creating a default-deny ingress policy.")
			// Empty podSelector blocks the entire namespace
			policy := netpol.GenNetworkPolicyWithNameAndPodSelector("deny-ingress", metav1.LabelSelector{}, netpol.SetSpecIngressRules())
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			ginkgo.By("Creating a network policy for the server which allows traffic only via SCTP on port 81.")
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.Ports = append(ingressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{IntVal: 81}, Protocol: &protocolSCTP})
			policy = netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-only-sctp-ingress-on-port-81", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			ginkgo.By("Trying to connect to TCP port 81, which should be blocked by the deny-ingress policy.")
			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&netpol.Peer{}, &netpol.Peer{Namespace: nsX}, false)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		// This test *does* apply to plugins that do not implement SCTP. It is a
		// security hole if you fail this test, because you are allowing TCP
		// traffic that is supposed to be blocked.
		f.It("should properly isolate pods that are selected by a policy allowing SCTP, even if the plugin doesn't support SCTP", feature.NetworkPolicy, func(ctx context.Context) {
			ginkgo.By("Creating a network policy for the server which allows traffic only via SCTP on port 80.")
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.Ports = append(ingressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{IntVal: 80}, Protocol: &protocolSCTP})
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-only-sctp-ingress-on-port-80", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{81}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			ginkgo.By("Trying to connect to TCP port 81, which should be blocked by implicit isolation.")
			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(netpol.NewPodString(nsX, "a"), false)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should not allow access by TCP when a policy specifies only UDP", feature.NetworkPolicy, func(ctx context.Context) {
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.Ports = append(ingressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{IntVal: 81}, Protocol: &protocolUDP})
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-only-udp-ingress-on-port-81", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{81}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			ginkgo.By("Creating a network policy for the server which allows traffic only via UDP on port 81.")

			// Probing with TCP, so all traffic should be dropped.
			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(netpol.NewPodString(nsX, "a"), false)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 81, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		// Note that this default ns functionality is maintained by the APIMachinery group, but we test it here anyways because its an important feature.
		f.It("should enforce policy to allow traffic based on NamespaceSelector with MatchLabels using default ns label", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, nsY, nsZ := getK8sNamespaces(k8s)
			allowedLabels := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					v1.LabelMetadataName: nsY,
				},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedLabels})
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-ns-selector-for-immutable-ns-label", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsX}, &netpol.Peer{Namespace: nsX, Pod: "a"}, false)
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsZ}, &netpol.Peer{Namespace: nsX, Pod: "a"}, false)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		// Note that this default ns functionality is maintained by the APIMachinery group, but we test it here anyways because its an important feature.
		f.It("should enforce policy based on NamespaceSelector with MatchExpressions using default ns label", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
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
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-ns-y-match-selector-for-immutable-ns-label", map[string]string{"pod": "a"}, netpol.SetSpecEgressRules(egressRule))
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsX, Pod: "a"}, &netpol.Peer{Namespace: nsY}, false)
			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		f.It("should enforce default-deny ingress policy against LoadBalancer traffic with ExternalTrafficPolicy: Local", feature.NetworkPolicy, feature.LoadBalancer, func(ctx context.Context) {
			// Create pods and namespaces to utilize netpol helper functions
			k8s := initializeResources(ctx, f, []v1.Protocol{v1.ProtocolTCP}, []int32{80})

			// Make and apply deny-all-ingress policy in namespace x
			nsX, _, _ := getK8sNamespaces(k8s)
			policy := netpol.GenNetworkPolicyWithNameAndPodSelector(
				"deny-ingress",
				metav1.LabelSelector{},
				netpol.SetSpecIngressRules(),
			)
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			// Create a “Local” LoadBalancer Service in namespace x with one netexec pod
			cs := f.ClientSet
			timeout := e2eservice.GetServiceLoadBalancerCreationTimeout(ctx, cs)
			jig := e2eservice.NewTestJig(cs, nsX, "deny-lb")
			svc, err := jig.CreateOnlyLocalLoadBalancerService(ctx, timeout, true, nil)
			framework.ExpectNoError(err)

			// FIXME: figure out the actual expected semantics for
			// "ExternalTrafficPolicy: Local" + "IPMode: Proxy".
			// https://issues.k8s.io/123714    ing := &svc.Status.LoadBalancer.Ingress[0]
			ingress := &svc.Status.LoadBalancer.Ingress[0]
			if ingress.IP == "" || (ingress.IPMode != nil && *ingress.IPMode == v1.LoadBalancerIPModeProxy) {
				e2eskipper.Skipf("LoadBalancer uses 'Proxy' IPMode")
			}

			ingressIP := e2eservice.GetIngressPoint(ingress)
			svcPort := int(svc.Spec.Ports[0].Port)

			// before poll, give kube-proxy time to converge
			time.Sleep(e2eservice.KubeProxyLagTimeout)

			// Attempt HTTP request from the e2e “host” to LB_IP:svcPort
			shortTry := 2 * time.Second
			err = wait.PollUntilContextTimeout(
				ctx,
				shortTry,   // interval
				5*shortTry, // timeout window
				true,       // check immediately
				func(ctx context.Context) (bool, error) {
					_, e := GetHTTPContent(ingressIP, svcPort, shortTry, "/clientip")
					// Return true if it ever becomes reachable (failure for this test).
					return e == nil, nil
				},
			)
			if err == nil {
				framework.Failf("LoadBalancer traffic became reachable within the poll window, but default-deny NetworkPolicy should have blocked it")
			} else if errors.Is(err, context.DeadlineExceeded) {
				// Timed out without ever becoming reachable — this is the expected outcome.
			} else {
				framework.ExpectNoError(err)
			}

			// Verify there is at least one local endpoint and pick a node that hosts it
			ep, err := cs.CoreV1().Endpoints(nsX).Get(ctx, svc.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			if ep == nil || len(ep.Subsets) == 0 {
				e2eskipper.Skipf("Service has no endpoints")
			}

			var nodeName string
			found := false
			for _, subset := range ep.Subsets {
				for _, addr := range subset.Addresses {
					if addr.NodeName != nil && *addr.NodeName != "" {
						nodeName = *addr.NodeName
						found = true
						break
					}
				}
				if found {
					break
				}
			}
			if !found {
				e2eskipper.Skipf("Could not find an endpoint node for Service %q", svc.Name)
			}

			// Resolve a reachable node IP for that node
			node, err := cs.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			var nodeIP string
			for _, a := range node.Status.Addresses {
				if a.Type == v1.NodeExternalIP && a.Address != "" {
					nodeIP = a.Address
					break
				}
			}
			if nodeIP == "" {
				for _, a := range node.Status.Addresses {
					if a.Type == v1.NodeInternalIP && a.Address != "" {
						nodeIP = a.Address
						break
					}
				}
			}
			if nodeIP == "" {
				e2eskipper.Skipf("No usable IP found for node %q", nodeName)
			}

			// Confirm the NodePort path (on a node with a local endpoint) remains unreachable.
			nodePort := int(svc.Spec.Ports[0].NodePort)
			err = wait.PollUntilContextTimeout(
				ctx,
				shortTry,
				5*shortTry,
				true,
				func(ctx context.Context) (bool, error) {
					_, e := GetHTTPContent(nodeIP, nodePort, shortTry, "/clientip")
					return e == nil, nil
				},
			)
			if err == nil {
				framework.Failf("NodePort traffic on node %q became reachable within the poll window, but default-deny NetworkPolicy should have blocked it", nodeName)
			} else if errors.Is(err, context.DeadlineExceeded) {
				// Expected: stayed unreachable the whole time.
			} else {
				framework.ExpectNoError(err)
			}
		})
	})
})

var _ = common.SIGDescribe("Netpol [LinuxOnly]", func() {
	f := framework.NewDefaultFramework("udp-network-policy")
	f.SkipNamespaceCreation = true
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	var k8s *netpol.KubeManager
	ginkgo.BeforeEach(func() {
		// Windows does not support UDP testing via agnhost.
		e2eskipper.SkipIfNodeOSDistroIs("windows")
	})

	ginkgo.Context("NetworkPolicy between server and client using UDP", func() {

		f.It("should support a 'default-deny-ingress' policy", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolUDP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			policy := netpol.GenNetworkPolicyWithNameAndPodSelector("deny-all", metav1.LabelSelector{}, netpol.SetSpecIngressRules())
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&netpol.Peer{}, &netpol.Peer{Namespace: nsX}, false)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolUDP, Reachability: reachability})
		})

		f.It("should enforce policy based on Ports", feature.NetworkPolicy, func(ctx context.Context) {
			ginkgo.By("Creating a network policy allowPort81Policy which only allows allow listed namespaces (y) to connect on exactly one port (81)")
			protocols := []v1.Protocol{protocolUDP}
			ports := []int32{81}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, nsY, nsZ := getK8sNamespaces(k8s)
			allowedLabels := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					namespaceLabelKey: nsY,
				},
			}

			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedLabels})
			ingressRule.Ports = append(ingressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{IntVal: 81}, Protocol: &protocolUDP})
			allowPort81Policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-ingress-on-port-81-ns-x", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, allowPort81Policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsX}, &netpol.Peer{Namespace: nsX, Pod: "a"}, false)
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsZ}, &netpol.Peer{Namespace: nsX, Pod: "a"}, false)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 81, Protocol: v1.ProtocolUDP, Reachability: reachability})
		})

		f.It("should enforce policy to allow traffic only from a pod in a different namespace based on PodSelector and NamespaceSelector", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolUDP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
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
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-ns-y-pod-a-via-namespace-pod-selector", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(netpol.NewPodString(nsX, "a"), false)
			reachability.Expect(netpol.NewPodString(nsY, "a"), netpol.NewPodString(nsX, "a"), true)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolUDP, Reachability: reachability})
		})
	})
})

var _ = common.SIGDescribe("Netpol", feature.SCTPConnectivity, "[LinuxOnly]", func() {
	f := framework.NewDefaultFramework("sctp-network-policy")
	f.SkipNamespaceCreation = true
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	var k8s *netpol.KubeManager
	ginkgo.BeforeEach(func() {
		// Windows does not support network policies.
		e2eskipper.SkipIfNodeOSDistroIs("windows")
	})

	ginkgo.Context("NetworkPolicy between server and client using SCTP", func() {

		f.It("should support a 'default-deny-ingress' policy", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolSCTP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, _, _ := getK8sNamespaces(k8s)
			policy := netpol.GenNetworkPolicyWithNameAndPodSelector("deny-all", metav1.LabelSelector{}, netpol.SetSpecIngressRules())
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&netpol.Peer{}, &netpol.Peer{Namespace: nsX}, false)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolSCTP, Reachability: reachability})
		})

		f.It("should enforce policy based on Ports", feature.NetworkPolicy, func(ctx context.Context) {
			ginkgo.By("Creating a network allowPort81Policy which only allows allow listed namespaces (y) to connect on exactly one port (81)")
			protocols := []v1.Protocol{protocolSCTP}
			ports := []int32{81}
			k8s = initializeResources(ctx, f, protocols, ports)
			nsX, nsY, nsZ := getK8sNamespaces(k8s)
			allowedLabels := &metav1.LabelSelector{
				MatchLabels: map[string]string{
					namespaceLabelKey: nsY,
				},
			}
			ingressRule := networkingv1.NetworkPolicyIngressRule{}
			ingressRule.From = append(ingressRule.From, networkingv1.NetworkPolicyPeer{NamespaceSelector: allowedLabels})
			ingressRule.Ports = append(ingressRule.Ports, networkingv1.NetworkPolicyPort{Port: &intstr.IntOrString{IntVal: 81}, Protocol: &protocolSCTP})
			allowPort81Policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-ingress-on-port-81-ns-x", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, allowPort81Policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsX}, &netpol.Peer{Namespace: nsX, Pod: "a"}, false)
			reachability.ExpectPeer(&netpol.Peer{Namespace: nsZ}, &netpol.Peer{Namespace: nsX, Pod: "a"}, false)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 81, Protocol: v1.ProtocolSCTP, Reachability: reachability})
		})

		f.It("should enforce policy to allow traffic only from a pod in a different namespace based on PodSelector and NamespaceSelector", feature.NetworkPolicy, func(ctx context.Context) {
			protocols := []v1.Protocol{protocolSCTP}
			ports := []int32{80}
			k8s = initializeResources(ctx, f, protocols, ports)
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
			policy := netpol.GenNetworkPolicyWithNameAndPodMatchLabel("allow-ns-y-pod-a-via-namespace-pod-selector", map[string]string{"pod": "a"}, netpol.SetSpecIngressRules(ingressRule))
			netpol.CreatePolicy(ctx, k8s, policy, nsX)

			reachability := netpol.NewReachability(k8s.AllPodStrings(), true)
			reachability.ExpectAllIngress(netpol.NewPodString(nsX, "a"), false)
			reachability.Expect(netpol.NewPodString(nsY, "a"), netpol.NewPodString(nsX, "a"), true)

			netpol.ValidateOrFail(k8s, &netpol.TestCase{ToPort: 80, Protocol: v1.ProtocolSCTP, Reachability: reachability})
		})
	})
})

// getNamespaceBaseNames returns the set of base namespace names used by this test, taking a root ns as input.
// The framework will also append a unique suffix when creating the namespaces.
// This allows tests to run in parallel.
func getNamespaceBaseNames(rootNs string) []string {
	if rootNs != "" {
		rootNs += "-"
	}
	nsX := fmt.Sprintf("%sx", rootNs)
	nsY := fmt.Sprintf("%sy", rootNs)
	nsZ := fmt.Sprintf("%sz", rootNs)
	return []string{nsX, nsY, nsZ}
}

// defaultModel creates a new "model" pod system under namespaces (x,y,z) which has pods a, b, and c.  Thus resulting in the
// truth table matrix that is identical for all tests, comprising 81 total connections between 9 pods (x/a, x/b, x/c, ..., z/c).
func defaultModel(namespaces []string, protocols []v1.Protocol, ports []int32) *netpol.Model {
	if framework.NodeOSDistroIs("windows") {
		return netpol.NewWindowsModel(namespaces, []string{"a", "b", "c"}, ports)
	}
	return netpol.NewModel(namespaces, []string{"a", "b", "c"}, ports, protocols)
}

// getK8sNamespaces returns the 3 actual namespace names.
func getK8sNamespaces(k8s *netpol.KubeManager) (string, string, string) {
	ns := k8s.NamespaceNames()
	return ns[0], ns[1], ns[2]
}

func initializeCluster(ctx context.Context, f *framework.Framework, protocols []v1.Protocol, ports []int32) (*netpol.KubeManager, error) {
	dnsDomain := framework.TestContext.ClusterDNSDomain
	framework.Logf("dns domain: %s", dnsDomain)

	k8s := netpol.NewKubeManager(f, dnsDomain)
	rootNs := f.BaseName
	namespaceBaseNames := getNamespaceBaseNames(rootNs)

	model := defaultModel(namespaceBaseNames, protocols, ports)

	framework.Logf("initializing cluster: ensuring namespaces, pods and services exist and are ready")

	if err := k8s.InitializeClusterFromModel(ctx, model); err != nil {
		return nil, err
	}

	framework.Logf("finished initializing cluster state")

	if err := netpol.WaitForHTTPServers(k8s, model); err != nil {
		return nil, err
	}

	return k8s, nil
}

// initializeResources uses the e2e framework to create all necessary namespace resources, based on the network policy
// model derived from the framework.  It then waits for the resources described by the model to be up and running
// (i.e. all pods are ready and running in their namespaces).
func initializeResources(ctx context.Context, f *framework.Framework, protocols []v1.Protocol, ports []int32) *netpol.KubeManager {
	k8s, err := initializeCluster(ctx, f, protocols, ports)
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
