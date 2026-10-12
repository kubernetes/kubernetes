/*
Copyright 2026 The Kubernetes Authors.

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

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/network/common"
	admissionapi "k8s.io/pod-security-admission/api"
	utilnet "k8s.io/utils/net"
)

// These tests complement the ipBlock coverage in network_policy.go, which only
// exercises ipBlock on egress and builds its CIDR from a single pod.Status.PodIP
// (see the "figure out how the next 3 tests should work with dual stack" TODO).
// The cases below add:
//   - ingress ipBlock (there is none above), single and multiple CIDRs, on all families
//   - dual-stack egress ipBlock and ipBlock-with-except, probing each family
//     independently so a per-family regression (one family blocked while the
//     other leaks) cannot be masked.
var _ = common.SIGDescribe("Netpol", func() {
	f := framework.NewDefaultFramework("netpol-ipblock")
	f.SkipNamespaceCreation = true // we create our own x/y/z test namespaces
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.Context("NetworkPolicy ipBlock", func() {
		const port int32 = 80

		// Upstream covers ipBlock only on egress, so this adds the ingress
		// direction across all supported IP families.
		f.It("should enforce an ingress ipBlock policy allowing traffic only from the specified CIDR", feature.NetworkPolicy, func(ctx context.Context) {
			k8s := initializeResources(ctx, f, []v1.Protocol{protocolTCP}, []int32{port}, "x/a", "x/b", "x/c")
			nsX, _, _ := getK8sNamespaces(k8s)

			// Allow ingress to the server (x/a) only from the allowed client's CIDRs.
			allowed := podByLabel(ctx, f, nsX, "b")
			ingressRule := networkingv1.NetworkPolicyIngressRule{From: ipBlockPeersFromCIDRs(hostCIDRsForPod(allowed))}
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-ingress-from-cidr",
				map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			server := podByLabel(ctx, f, nsX, "a")

			ginkgo.By("verifying the allowed client can reach the server on every IP family")
			probePodIPs(k8s, nsX, "b", server, int(port), true)
			ginkgo.By("verifying the blocked client cannot reach the server on any IP family")
			probePodIPs(k8s, nsX, "c", server, int(port), false)
		})

		// Multiple CIDRs in a single ipBlock, ensuring every CIDR in the list is
		// honored (not just the first/last). Runs on all families.
		f.It("should enforce an ingress ipBlock policy honoring all CIDRs in the list", feature.NetworkPolicy, func(ctx context.Context) {
			k8s := initializeResources(ctx, f, []v1.Protocol{protocolTCP}, []int32{port}, "x/a", "x/b", "x/c", "y/a")
			nsX, nsY, _ := getK8sNamespaces(k8s)

			// Allow ingress to the server (x/a) from both x/b and x/c CIDRs.
			allowedB := podByLabel(ctx, f, nsX, "b")
			allowedC := podByLabel(ctx, f, nsX, "c")
			cidrs := append(hostCIDRsForPod(allowedB), hostCIDRsForPod(allowedC)...)
			ingressRule := networkingv1.NetworkPolicyIngressRule{From: ipBlockPeersFromCIDRs(cidrs)}
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-ingress-from-multiple-cidrs",
				map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			server := podByLabel(ctx, f, nsX, "a")

			ginkgo.By("verifying every listed client CIDR can reach the server on every IP family")
			probePodIPs(k8s, nsX, "b", server, int(port), true)
			probePodIPs(k8s, nsX, "c", server, int(port), true)
			ginkgo.By("verifying a client outside the listed CIDRs cannot reach the server on any IP family")
			probePodIPs(k8s, nsY, "a", server, int(port), false)
		})

		// Dual-stack egress: a single policy with both an IPv4 and IPv6 CIDR,
		// verifying each family is enforced independently.
		f.It("should enforce a dual-stack egress ipBlock policy allowing traffic only to the specified CIDRs", feature.NetworkPolicy, func(ctx context.Context) {
			k8s := initializeResources(ctx, f, []v1.Protocol{protocolTCP}, []int32{port}, "x/a", "x/b", "x/c")
			nsX, _, _ := getK8sNamespaces(k8s)

			server := podByLabel(ctx, f, nsX, "b")
			skipUnlessDualStack(server)

			// Allow egress from x/a only to the server's dual-stack CIDRs.
			egressRule := networkingv1.NetworkPolicyEgressRule{To: ipBlockPeersFromCIDRs(hostCIDRsForPod(server))}
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-egress-to-cidrs",
				map[string]string{"pod": "a"}, SetSpecEgressRules(egressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			other := podByLabel(ctx, f, nsX, "c")

			ginkgo.By("verifying the client can reach the allowed server on both IP families")
			probePodIPs(k8s, nsX, "a", server, int(port), true)
			ginkgo.By("verifying the client cannot reach a non-allowed pod on either IP family")
			probePodIPs(k8s, nsX, "a", other, int(port), false)
		})

		// Dual-stack egress with a per-family except clause: allow a large CIDR
		// for each family but carve out the server, verifying the except is
		// enforced on both families while non-excepted peers remain reachable.
		f.It("should enforce a dual-stack egress ipBlock policy with an except clause", feature.NetworkPolicy, func(ctx context.Context) {
			k8s := initializeResources(ctx, f, []v1.Protocol{protocolTCP}, []int32{port}, "x/a", "x/b", "x/c")
			nsX, _, _ := getK8sNamespaces(k8s)

			server := podByLabel(ctx, f, nsX, "b")
			skipUnlessDualStack(server)

			// For each of the server's families, allow a large CIDR but except
			// the server itself.
			var peers []networkingv1.NetworkPolicyPeer
			for _, podIP := range server.Status.PodIPs {
				peers = append(peers, networkingv1.NetworkPolicyPeer{IPBlock: &networkingv1.IPBlock{
					CIDR:   makeLargeCIDRForIP(podIP.IP),
					Except: []string{hostCIDRForIP(podIP.IP)},
				}})
			}
			egressRule := networkingv1.NetworkPolicyEgressRule{To: peers}
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-egress-except-server",
				map[string]string{"pod": "a"}, SetSpecEgressRules(egressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			other := podByLabel(ctx, f, nsX, "c")

			ginkgo.By("verifying the excepted server is unreachable on both IP families")
			probePodIPs(k8s, nsX, "a", server, int(port), false)
			ginkgo.By("verifying a non-excepted pod in the allowed CIDR is reachable on both IP families")
			probePodIPs(k8s, nsX, "a", other, int(port), true)
		})
	})
})

// podByLabel returns the single pod labeled pod=<podName> in namespace ns.
func podByLabel(ctx context.Context, f *framework.Framework, ns, podName string) *v1.Pod {
	podList, err := f.ClientSet.CoreV1().Pods(ns).List(ctx, metav1.ListOptions{LabelSelector: "pod=" + podName})
	framework.ExpectNoError(err, "listing pod %s/%s", ns, podName)
	gomega.Expect(podList.Items).NotTo(gomega.BeEmpty(), "expected a pod %q in namespace %q", podName, ns)
	return &podList.Items[0]
}

// podIPs returns the pod's IPs, falling back to the single PodIP when PodIPs is
// not populated.
func podIPs(pod *v1.Pod) []string {
	if len(pod.Status.PodIPs) == 0 {
		return []string{pod.Status.PodIP}
	}
	ips := make([]string, 0, len(pod.Status.PodIPs))
	for _, podIP := range pod.Status.PodIPs {
		ips = append(ips, podIP.IP)
	}
	return ips
}

// hostCIDRForIP returns a single-host CIDR (/32 or /128) for the given IP.
func hostCIDRForIP(ip string) string {
	mask := 32
	if utilnet.IsIPv6String(ip) {
		mask = 128
	}
	return fmt.Sprintf("%s/%d", ip, mask)
}

// hostCIDRsForPod returns a single-host CIDR for each of the pod's IPs, one per
// IP family on a dual-stack cluster.
func hostCIDRsForPod(pod *v1.Pod) []string {
	ips := podIPs(pod)
	cidrs := make([]string, 0, len(ips))
	for _, ip := range ips {
		cidrs = append(cidrs, hostCIDRForIP(ip))
	}
	return cidrs
}

// ipBlockPeersFromCIDRs builds one NetworkPolicyPeer per CIDR, each an ipBlock.
func ipBlockPeersFromCIDRs(cidrs []string) []networkingv1.NetworkPolicyPeer {
	peers := make([]networkingv1.NetworkPolicyPeer, 0, len(cidrs))
	for _, cidr := range cidrs {
		peers = append(peers, networkingv1.NetworkPolicyPeer{IPBlock: &networkingv1.IPBlock{CIDR: cidr}})
	}
	return peers
}

// skipUnlessDualStack skips the current test unless the given pod has both an
// IPv4 and an IPv6 address.
func skipUnlessDualStack(pod *v1.Pod) {
	var v4, v6 bool
	for _, ip := range podIPs(pod) {
		if utilnet.IsIPv6String(ip) {
			v6 = true
		} else {
			v4 = true
		}
	}
	if !v4 || !v6 {
		ginkgo.Skip("requires a dual-stack cluster")
	}
}

// probePodIPs probes connectivity from podFrom to each of dstPod's IPs
// independently and asserts every probe matches expectConnected. Probing each
// family separately (rather than aggregating) preserves per-family coverage, so
// a policy that enforces only one family cannot pass unnoticed.
func probePodIPs(k8s *kubeManager, nsFrom, podFrom string, dstPod *v1.Pod, toPort int, expectConnected bool) {
	for _, ip := range podIPs(dstPod) {
		family := "IPv4"
		if utilnet.IsIPv6String(ip) {
			family = "IPv6"
		}
		connected, cmd, err := k8s.probeConnectivity(&probeConnectivityArgs{
			nsFrom:              nsFrom,
			podFrom:             podFrom,
			addrTo:              ip,
			protocol:            protocolTCP,
			toPort:              toPort,
			expectConnectivity:  expectConnected,
			timeoutSeconds:      getProbeTimeoutSeconds(),
			pollIntervalSeconds: getPollIntervalSeconds(),
			pollTimeoutSeconds:  getPollTimeoutSeconds(),
		})
		framework.ExpectNoError(err, "probing %s/%s -> %s", nsFrom, podFrom, ip)
		gomega.Expect(connected).To(gomega.Equal(expectConnected),
			"%s connectivity %s/%s -> %s: expected connected=%v; probe: %s",
			family, nsFrom, podFrom, ip, expectConnected, cmd)
	}
}
