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

package netpol

import (
	"context"
	"fmt"
	"net"

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

var _ = common.SIGDescribe("Netpol", func() {
	f := framework.NewDefaultFramework("netpol-ipblock")
	f.SkipNamespaceCreation = true // we create our own x/y/z test namespaces
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.Context("NetworkPolicy ipBlock", func() {
		const port int32 = 80

		f.It("should allow egress access to server in CIDR block", feature.NetworkPolicy, func(ctx context.Context) {
			// Getting podServer's status to get podServer's IP, to create the CIDR
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s := initializeResources(ctx, f, protocols, ports, "x/a", "y/a", "y/b")
			nsX, nsY, _ := getK8sNamespaces(k8s)

			podList, err := f.ClientSet.CoreV1().Pods(nsY).List(ctx, metav1.ListOptions{LabelSelector: "pod=b"})
			framework.ExpectNoError(err, "Failing to list pods in namespace y")
			pod := podList.Items[0]

			podServerCIDR := cidrForIP(pod.Status.PodIP)
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
			k8s := initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "x/c")
			nsX, _, _ := getK8sNamespaces(k8s)

			// Getting podServer's status to get podServer's IP, to create the CIDR with except clause
			podList, err := f.ClientSet.CoreV1().Pods(nsX).List(ctx, metav1.ListOptions{LabelSelector: "pod=b"})
			framework.ExpectNoError(err, "Failing to find pod x/b")
			podB := podList.Items[0]

			// Create a rule that allows egress to a large set of IPs around
			// podB, but not podB itself.
			podServerAllowCIDR := makeLargeCIDRForIP(podB.Status.PodIP)
			podServerExceptList := []string{cidrForIP(podB.Status.PodIP)}

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
			k8s := initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "x/c")
			nsX, _, _ := getK8sNamespaces(k8s)

			// Getting podServer's status to get podServer's IP, to create the CIDR with except clause
			podList, err := f.ClientSet.CoreV1().Pods(nsX).List(ctx, metav1.ListOptions{LabelSelector: "pod=b"})
			framework.ExpectNoError(err, "Failing to find pod x/b")
			podB := podList.Items[0]

			// Create a rule that allows egress to a large set of IPs around
			// podB, but not podB itself.
			podServerAllowCIDR := makeLargeCIDRForIP(podB.Status.PodIP)
			podServerExceptList := []string{cidrForIP(podB.Status.PodIP)}
			egressRule1 := networkingv1.NetworkPolicyEgressRule{}
			egressRule1.To = append(egressRule1.To, networkingv1.NetworkPolicyPeer{IPBlock: &networkingv1.IPBlock{CIDR: podServerAllowCIDR, Except: podServerExceptList}})
			policyAllowCIDR := GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-cidr-egress-rule",
				map[string]string{"pod": "a"}, SetSpecEgressRules(egressRule1))
			CreatePolicy(ctx, k8s, policyAllowCIDR, nsX)

			reachability := NewReachability(k8s.AllPodStrings(), true)
			reachability.Expect(NewPodString(nsX, "a"), NewPodString(nsX, "b"), false)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})

			// Create a second NetworkPolicy which allows access to podB
			podBIP := cidrForIP(podB.Status.PodIP)
			egressRule3 := networkingv1.NetworkPolicyEgressRule{}
			egressRule3.To = append(egressRule3.To, networkingv1.NetworkPolicyPeer{IPBlock: &networkingv1.IPBlock{CIDR: podBIP}})
			allowPolicy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-client-a-via-cidr-egress-rule",
				map[string]string{"pod": "a"}, SetSpecEgressRules(egressRule3))
			UpdatePolicy(ctx, k8s, allowPolicy, nsX)

			reachabilityAllow := NewReachability(k8s.AllPodStrings(), true)
			reachabilityAllow.ExpectAllEgress(NewPodString(nsX, "a"), false)
			reachabilityAllow.Expect(NewPodString(nsX, "a"), NewPodString(nsX, "b"), true)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityAllow})
		})

		// Verify that every entry in a multi-entry ipBlock ingress rule is
		// honored, not just the first.
		f.It("should enforce an ingress ipBlock policy honoring all IPs in the list", feature.NetworkPolicy, func(ctx context.Context) {
			k8s := initializeResources(ctx, f, []v1.Protocol{protocolTCP}, []int32{port}, "x/a", "x/b", "x/c", "y/a")
			nsX, nsY, _ := getK8sNamespaces(k8s)

			// Allow ingress to the server (x/a) from both x/b's and x/c's IPs.
			allowedB := podByLabel(ctx, f, nsX, "b")
			allowedC := podByLabel(ctx, f, nsX, "c")
			ingressRule := networkingv1.NetworkPolicyIngressRule{
				From: []networkingv1.NetworkPolicyPeer{
					{IPBlock: &networkingv1.IPBlock{CIDR: cidrForIP(allowedB.Status.PodIP)}},
					{IPBlock: &networkingv1.IPBlock{CIDR: cidrForIP(allowedC.Status.PodIP)}},
				},
			}
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-ingress-from-listed-ips",
				map[string]string{"pod": "a"}, SetSpecIngressRules(ingressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			// The Reachability matrix doesn't support asserting on a specific
			// peer IP (or per-IP-family), so probe the server address directly.
			server := podByLabel(ctx, f, nsX, "a")
			ginkgo.By("verifying every listed client can reach the server")
			expectProbe(k8s, nsX, "b", server.Status.PodIP, int(port), true)
			expectProbe(k8s, nsX, "c", server.Status.PodIP, int(port), true)
			ginkgo.By("verifying a client outside the listed IPs cannot reach the server")
			expectProbe(k8s, nsY, "a", server.Status.PodIP, int(port), false)
		})

		// Dual-stack egress: a single policy with both an IPv4 and IPv6 CIDR,
		// verifying each family is enforced independently.
		f.It("should enforce a dual-stack egress ipBlock policy allowing traffic only to the specified CIDRs", feature.NetworkPolicy, feature.IPv6DualStack, func(ctx context.Context) {
			k8s := initializeResources(ctx, f, []v1.Protocol{protocolTCP}, []int32{port}, "x/a", "x/b", "x/c")
			nsX, _, _ := getK8sNamespaces(k8s)

			server := podByLabel(ctx, f, nsX, "b")
			requireDualStack(server)

			// Allow egress from x/a only to the server's IPv4 and IPv6 addresses.
			egressRule := networkingv1.NetworkPolicyEgressRule{To: []networkingv1.NetworkPolicyPeer{
				{IPBlock: &networkingv1.IPBlock{CIDR: cidrForIP(server.Status.PodIPs[0].IP)}},
				{IPBlock: &networkingv1.IPBlock{CIDR: cidrForIP(server.Status.PodIPs[1].IP)}},
			}}
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-egress-to-server",
				map[string]string{"pod": "a"}, SetSpecEgressRules(egressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			other := podByLabel(ctx, f, nsX, "c")
			ginkgo.By("verifying the client can reach the allowed server on both IP families")
			for _, podIP := range server.Status.PodIPs {
				expectProbe(k8s, nsX, "a", podIP.IP, int(port), true)
			}
			ginkgo.By("verifying the client cannot reach a non-allowed pod on either IP family")
			for _, podIP := range other.Status.PodIPs {
				expectProbe(k8s, nsX, "a", podIP.IP, int(port), false)
			}
		})

		// Dual-stack egress with a per-family except clause: allow a large CIDR
		// for each family but carve out the server.
		f.It("should enforce a dual-stack egress ipBlock policy with an except clause", feature.NetworkPolicy, feature.IPv6DualStack, func(ctx context.Context) {
			k8s := initializeResources(ctx, f, []v1.Protocol{protocolTCP}, []int32{port}, "x/a", "x/b", "x/c")
			nsX, _, _ := getK8sNamespaces(k8s)

			server := podByLabel(ctx, f, nsX, "b")
			requireDualStack(server)

			egressRule := networkingv1.NetworkPolicyEgressRule{To: []networkingv1.NetworkPolicyPeer{
				{IPBlock: &networkingv1.IPBlock{CIDR: makeLargeCIDRForIP(server.Status.PodIPs[0].IP), Except: []string{cidrForIP(server.Status.PodIPs[0].IP)}}},
				{IPBlock: &networkingv1.IPBlock{CIDR: makeLargeCIDRForIP(server.Status.PodIPs[1].IP), Except: []string{cidrForIP(server.Status.PodIPs[1].IP)}}},
			}}
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-egress-except-server",
				map[string]string{"pod": "a"}, SetSpecEgressRules(egressRule))
			CreatePolicy(ctx, k8s, policy, nsX)

			other := podByLabel(ctx, f, nsX, "c")
			ginkgo.By("verifying the excepted server is unreachable on both IP families")
			for _, podIP := range server.Status.PodIPs {
				expectProbe(k8s, nsX, "a", podIP.IP, int(port), false)
			}
			ginkgo.By("verifying a non-excepted pod in the allowed CIDR is reachable on both IP families")
			for _, podIP := range other.Status.PodIPs {
				expectProbe(k8s, nsX, "a", podIP.IP, int(port), true)
			}
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

// cidrForIP returns a single-address CIDR (/32 or /128) for the given IP.
func cidrForIP(ip string) string {
	mask := 32
	if utilnet.IsIPv6String(ip) {
		mask = 128
	}
	return fmt.Sprintf("%s/%d", ip, mask)
}

// requireDualStack fails the test unless the pod has both an IPv4 and IPv6 address.
func requireDualStack(pod *v1.Pod) {
	gomega.Expect(pod.Status.PodIPs).To(gomega.HaveLen(2),
		"test requires a dual-stack cluster, but pod %s/%s has IPs %v", pod.Namespace, pod.Name, pod.Status.PodIPs)
}

// expectProbe probes connectivity from podFrom to addrTo and asserts the result
// matches expectConnected. It probes a specific address rather than letting the
// Reachability matrix aggregate across families, so a policy that enforces only
// one IP family cannot pass unnoticed.
func expectProbe(k8s *kubeManager, nsFrom, podFrom, addrTo string, toPort int, expectConnected bool) {
	connected, cmd, err := k8s.probeConnectivity(&probeConnectivityArgs{
		nsFrom:              nsFrom,
		podFrom:             podFrom,
		addrTo:              addrTo,
		protocol:            protocolTCP,
		toPort:              toPort,
		expectConnectivity:  expectConnected,
		timeoutSeconds:      getProbeTimeoutSeconds(),
		pollIntervalSeconds: getPollIntervalSeconds(),
		pollTimeoutSeconds:  getPollTimeoutSeconds(),
	})
	framework.ExpectNoError(err, "probing %s/%s -> %s", nsFrom, podFrom, addrTo)
	gomega.Expect(connected).To(gomega.Equal(expectConnected),
		"connectivity %s/%s -> %s: expected connected=%v; probe: %s", nsFrom, podFrom, addrTo, expectConnected, cmd)
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
