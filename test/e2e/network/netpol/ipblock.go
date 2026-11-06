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
		// TODO, figure out how the next 3 tests should work with dual stack : do we need a different abstraction then just "podIP"?

		f.It("should allow egress access to server in CIDR block", feature.NetworkPolicy, func(ctx context.Context) {
			// Getting podServer's status to get podServer's IP, to create the CIDR
			protocols := []v1.Protocol{protocolTCP}
			ports := []int32{80}
			k8s := initializeResources(ctx, f, protocols, ports, "x/a", "y/a", "y/b")
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
			k8s := initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "x/c")
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
			k8s := initializeResources(ctx, f, protocols, ports, "x/a", "x/b", "x/c")
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
			UpdatePolicy(ctx, k8s, allowPolicy, nsX)

			reachabilityAllow := NewReachability(k8s.AllPodStrings(), true)
			reachabilityAllow.ExpectAllEgress(NewPodString(nsX, "a"), false)
			reachabilityAllow.Expect(NewPodString(nsX, "a"), NewPodString(nsX, "b"), true)

			ValidateOrFail(k8s, &TestCase{ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachabilityAllow})
		})
	})
})

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
