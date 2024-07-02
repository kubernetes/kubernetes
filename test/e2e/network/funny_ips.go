/*
Copyright 2022 The Kubernetes Authors.

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
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"

	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/network/common"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	admissionapi "k8s.io/pod-security-admission/api"
	netutils "k8s.io/utils/net"
)

// What are funny IPs:
// The adjective is because of the curl blog that explains the history and the problem of liberal
// parsing of IP addresses and the consequences and security risks caused the lack of normalization,
// mainly due to the use of different notations to abuse parsers misalignment to bypass filters.
// xref: https://daniel.haxx.se/blog/2021/04/19/curl-those-funny-ipv4-addresses/
//
// Since golang 1.17, IPv4 addresses with leading zeros are rejected by the standard library.
// xref: https://github.com/golang/go/issues/30999
//
// Because this change on the parsers can cause that previous valid data become invalid, Kubernetes
// forked the old parsers allowing leading zeros on IPv4 address to not break the compatibility.
//
// Kubernetes interprets leading zeros on IPv4 addresses as decimal, users must not rely on parser
// alignment to not being impacted by the associated security advisory: CVE-2021-29923 golang
// standard library "net" - Improper Input Validation of octal literals in golang 1.16.2 and below
// standard library "net" results in indeterminate SSRF & RFI vulnerabilities. xref:
// https://nvd.nist.gov/vuln/detail/CVE-2021-29923
//
// Kubernetes
// ip := net.ParseIPSloppy("192.168.0.011")
// ip.String() yields 192.168.0.11
//
// BSD stacks
// ping 192.168.0.011
// PING 192.168.0.011 (192.168.0.9) 56(84) bytes of data.

var _ = common.SIGDescribe("CVE-2021-29923", func() {
	var (
		ns string
		cs clientset.Interface
	)

	f := framework.NewDefaultFramework("funny-ips")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.BeforeEach(func() {
		if framework.TestContext.ClusterIsIPv6() {
			e2eskipper.Skipf("The test doesn't apply to IPv6 addresses, only IPv4 addresses are affected by CVE-2021-29923")
		}
		ns = f.Namespace.Name
		cs = f.ClientSet
	})

	/*
		Create a ClusterIP service with a ClusterIP with leading zeros and check that it is reachable in the IP interpreted as decimal.
		If the Service is not reachable only FAIL if it is reachable in the IP interpreted as binary, because it will be exposed to CVE-2021-29923.
		IMPORTANT: CoreDNS since version 1.8.5 discard IPs with leading zeros so Services are not resolvable, and is probably that
		most of the ecosystem has done the same, however, Kubernetes doesn't impose any restriction, users should migrate their IPs.
	*/
	ginkgo.It("IPv4 Service Type ClusterIP with leading zeros should work interpreted as decimal", func(ctx context.Context) {
		serviceName := "funny-ip"
		// Use a very uncommon port to reduce the risk of conflicts with other tests that create services.
		servicePort := 7180
		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		clusterIPZero, clusterIPOctal := getServiceIPWithLeadingZeros(ctx, cs)
		if clusterIPZero == "" {
			e2eskipper.Skipf("Couldn't find a free ClusterIP")
		}

		ginkgo.By("creating service " + serviceName + " with type=ClusterIP and ip " + clusterIPZero + " in namespace " + ns)
		_, err := jig.CreateTCPService(ctx, func(svc *v1.Service) {
			svc.Spec.ClusterIP = clusterIPZero // IP with a leading zero
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.Ports = []v1.ServicePort{
				{Port: int32(servicePort), Name: "http", Protocol: v1.ProtocolTCP, TargetPort: intstr.FromInt32(9376)},
			}
		})
		framework.ExpectNoError(err)

		err = jig.CreateServicePods(ctx, 1)
		framework.ExpectNoError(err)

		execPod := e2epod.CreateExecPodOrFail(ctx, cs, ns, "execpod", nil)
		ip := netutils.ParseIPSloppy(clusterIPZero)
		cmd := fmt.Sprintf("echo hostName | nc -v -t -w 2 %s %v", ip.String(), servicePort)
		err = wait.PollUntilContextTimeout(ctx, 1*time.Second, e2eservice.ServiceReachabilityShortPollTimeout, true, func(ctx context.Context) (bool, error) {
			stdout, err := e2eoutput.RunHostCmd(execPod.Namespace, execPod.Name, cmd)
			if err != nil {
				framework.Logf("Service reachability failing with error: %v\nRetrying...", err)
				return false, nil
			}
			trimmed := strings.TrimSpace(stdout)
			if trimmed != "" {
				return true, nil
			}
			return false, nil
		})
		// Service is working on the expected IP.
		if err == nil {
			return
		}
		// It may happen that the component implementing Services discard the IP.
		// We have to check that the Service is not reachable in the address interpreted as decimal.
		cmd = fmt.Sprintf("echo hostName | nc -v -t -w 2 %s %v", clusterIPOctal, servicePort)
		err = wait.PollUntilContextTimeout(ctx, 1*time.Second, e2eservice.ServiceReachabilityShortPollTimeout, true, func(ctx context.Context) (bool, error) {
			stdout, err := e2eoutput.RunHostCmd(execPod.Namespace, execPod.Name, cmd)
			if err != nil {
				framework.Logf("Service reachability failing with error: %v\nRetrying...", err)
				return false, nil
			}
			trimmed := strings.TrimSpace(stdout)
			if trimmed != "" {
				return true, nil
			}
			return false, nil
		})
		// Ouch, Service has worked on IP interpreted as octal.
		if err == nil {
			framework.Failf("WARNING: Your Cluster interprets Service ClusterIP %s as %s, please see https://nvd.nist.gov/vuln/detail/CVE-2021-29923", clusterIPZero, clusterIPOctal)
		}
		framework.Logf("Service reachability failing for Service against ClusterIP %s and %s, most probably leading zeros on IPs are not supported by the cluster proxy", clusterIPZero, clusterIPOctal)
	})

})

// Try to get a free IP that has different decimal and binary interpretation with leading zeros.
// Return both IPs, the one interpretad as binary and the one interpreted as decimal.
// Return empty if not IPs are found.
func getServiceIPWithLeadingZeros(ctx context.Context, cs clientset.Interface) (string, string) {
	clusterIPMap := map[string]struct{}{}
	var clusterIPPrefix string
	// Dump all the IPs and look for the ones we want.
	list, err := cs.CoreV1().Services(metav1.NamespaceAll).List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err)
	for _, svc := range list.Items {
		if len(svc.Spec.ClusterIP) == 0 || svc.Spec.ClusterIP == v1.ClusterIPNone {
			continue
		}
		// Get the clusterIP prefix, needed later for building the IPs with octal format.
		if clusterIPPrefix == "" {
			// split the IP by "."
			// get the first 3 values
			// join them again using dots, dropping the last field
			clusterIPPrefix = strings.Join(strings.Split(svc.Spec.ClusterIP, ".")[:3], ".")
		}
		// Canonicalize IP.
		ip := netutils.ParseIPSloppy(svc.Spec.ClusterIP)
		clusterIPMap[ip.String()] = struct{}{}
	}

	// Try to get a free IP between x.x.x.11 and x.x.x.31, this assumes a /27 Service IP range that should be safe.
	for i := 11; i < 31; i++ {
		ip := fmt.Sprintf("%s.%d", clusterIPPrefix, i)
		// Check the IP in decimal format is free.
		if _, ok := clusterIPMap[ip]; !ok {
			o, err := strconv.ParseInt(fmt.Sprintf("%d", i), 8, 64)
			// Omit ips without binary representation i.e. x.x.x.018, x.x.x.019 ...
			if err != nil {
				continue
			}
			ipOctal := fmt.Sprintf("%s.%d", clusterIPPrefix, o)
			// Check the same IP in octal format is free.
			if _, ok := clusterIPMap[ipOctal]; !ok {
				// Add a leading zero to the decimal format.
				return fmt.Sprintf("%s.0%d", clusterIPPrefix, i), ipOctal
			}
		}
	}
	return "", ""
}
