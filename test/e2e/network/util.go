/*
Copyright 2014 The Kubernetes Authors.

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
	"bytes"
	"context"
	"fmt"
	"net"
	"regexp"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2enetwork "k8s.io/kubernetes/test/e2e/framework/network"
	e2eoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

// secondNodePortSvcName is the name of the secondary node port service
const secondNodePortSvcName = "second-node-port-service"

// GetHTTPContent returns the content of the given url by HTTP.
func GetHTTPContent(ctx context.Context, host string, port int, timeout time.Duration, url string) (string, error) {
	var body bytes.Buffer
	pollErr := wait.PollUntilContextTimeout(ctx, framework.Poll, timeout, true, func(ctx context.Context) (bool, error) {
		result := e2enetwork.PokeHTTP(host, port, url, nil)
		if result.Status == e2enetwork.HTTPSuccess {
			body.Write(result.Body)
			return true, nil
		}
		return false, nil
	})
	if pollErr != nil {
		framework.Logf("Could not reach HTTP service through %v:%v%v after %v: %v", host, port, url, timeout, pollErr)
	}
	return body.String(), pollErr
}

// GetHTTPContentFromTestContainer returns the content of the given url by HTTP via a test container.
func GetHTTPContentFromTestContainer(ctx context.Context, config *e2enetwork.NetworkingTestConfig, host string, port int, timeout time.Duration, dialCmd string) (string, error) {
	var body string
	pollFn := func(ctx context.Context) (bool, error) {
		resp, err := config.GetResponseFromTestContainer(ctx, "http", dialCmd, host, port)
		if err != nil || len(resp.Errors) > 0 || len(resp.Responses) == 0 {
			return false, nil
		}
		body = resp.Responses[0]
		return true, nil
	}
	if pollErr := wait.PollUntilContextTimeout(ctx, framework.Poll, timeout, true, pollFn); pollErr != nil {
		return "", pollErr
	}
	return body, nil
}

// DescribeSvc logs the output of kubectl describe svc for the given namespace
func DescribeSvc(ns string) {
	framework.Logf("\nOutput of kubectl describe svc:\n")
	desc, _ := e2ekubectl.RunKubectl(
		ns, "describe", "svc", fmt.Sprintf("--namespace=%v", ns))
	framework.Logf(desc)
}

// CheckSCTPModuleLoadedOnNodes checks whether any node on the list has the
// sctp.ko module loaded
// For security reasons, and also to allow clusters to use userspace SCTP implementations,
// we require that just creating an SCTP Pod/Service/NetworkPolicy must not do anything
// that would cause the sctp kernel module to be loaded.
func CheckSCTPModuleLoadedOnNodes(ctx context.Context, f *framework.Framework, nodes *v1.NodeList) bool {
	hostExec := utils.NewHostExec(f)
	ginkgo.DeferCleanup(hostExec.Cleanup)
	re := regexp.MustCompile(`^\s*sctp\s+`)
	cmd := "lsmod | grep sctp"
	for _, node := range nodes.Items {
		framework.Logf("Executing cmd %q on node %v", cmd, node.Name)
		result, err := hostExec.IssueCommandWithResult(ctx, cmd, &node)
		if err != nil {
			framework.Logf("sctp module is not loaded or error occurred while executing command %s on node: %v", cmd, err)
		}
		for _, line := range strings.Split(result, "\n") {
			if found := re.Find([]byte(line)); found != nil {
				framework.Logf("the sctp module is loaded on node: %v", node.Name)
				return true
			}
		}
		framework.Logf("the sctp module is not loaded on node: %v", node.Name)
	}
	return false
}

// execSourceIPTest executes curl to access "/clientip" endpoint on target address
// from given Pod to check if source ip is preserved.
func execSourceIPTest(sourcePod v1.Pod, targetAddr string) (string, string) {
	var (
		err     error
		stdout  string
		timeout = 2 * time.Minute
	)

	framework.Logf("Waiting up to %v to get response from %s", timeout, targetAddr)
	cmd := fmt.Sprintf(`curl -q -s --connect-timeout 30 %s/clientip`, targetAddr)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(2 * time.Second) {
		stdout, err = e2eoutput.RunHostCmd(sourcePod.Namespace, sourcePod.Name, cmd)
		if err != nil {
			framework.Logf("got err: %v, retry until timeout", err)
			continue
		}
		// Need to check output because it might omit in case of error.
		if strings.TrimSpace(stdout) == "" {
			framework.Logf("got empty stdout, retry until timeout")
			continue
		}
		break
	}

	framework.ExpectNoError(err)

	// The stdout return from RunHostCmd is in this format: x.x.x.x:port or [xx:xx:xx::x]:port
	host, _, err := net.SplitHostPort(stdout)
	if err != nil {
		// ginkgo.Fail the test if output format is unexpected.
		framework.Failf("exec pod returned unexpected stdout: [%v]\n", stdout)
	}
	return sourcePod.Status.PodIP, host
}

// execHostnameTest executes curl to access "/hostname" endpoint on target address
// from given Pod to check the hostname of the target destination.
// It also converts FQDNs to hostnames, so if an FQDN is passed as
// targetHostname only the hostname part will be considered for comparison.
func execHostnameTest(sourcePod v1.Pod, targetAddr, targetHostname string) {
	var (
		err     error
		stdout  string
		timeout = 2 * time.Minute
	)

	framework.Logf("Waiting up to %v to get response from %s", timeout, targetAddr)
	cmd := fmt.Sprintf(`curl -q -s --max-time 30 %s/hostname`, targetAddr)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(2 * time.Second) {
		stdout, err = e2eoutput.RunHostCmd(sourcePod.Namespace, sourcePod.Name, cmd)
		if err != nil {
			framework.Logf("got err: %v, retry until timeout", err)
			continue
		}
		// Need to check output because it might omit in case of error.
		if strings.TrimSpace(stdout) == "" {
			framework.Logf("got empty stdout, retry until timeout")
			continue
		}
		break
	}

	// Ensure we're comparing hostnames and not FQDNs
	targetHostname = strings.Split(targetHostname, ".")[0]
	hostname := strings.TrimSpace(strings.Split(stdout, ".")[0])

	framework.ExpectNoError(err)
	gomega.Expect(hostname).To(gomega.Equal(targetHostname))
}

// createSecondNodePortService creates a service with the same selector as config.NodePortService and same HTTP Port
func createSecondNodePortService(ctx context.Context, f *framework.Framework, config *e2enetwork.NetworkingTestConfig) (*v1.Service, int) {
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: secondNodePortSvcName,
		},
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeNodePort,
			Ports: []v1.ServicePort{
				{
					Port:       e2enetwork.ClusterHTTPPort,
					Name:       "http",
					Protocol:   v1.ProtocolTCP,
					TargetPort: intstr.FromInt32(e2enetwork.EndpointHTTPPort),
				},
			},
			Selector: config.NodePortService.Spec.Selector,
		},
	}

	createdService := config.CreateService(ctx, svc)

	err := framework.WaitForServiceEndpointsNum(ctx, f.ClientSet, config.Namespace, secondNodePortSvcName, len(config.EndpointPods), time.Second, wait.ForeverTestTimeout)
	framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", secondNodePortSvcName, config.Namespace)

	var httpPort int
	for _, p := range createdService.Spec.Ports {
		switch p.Protocol {
		case v1.ProtocolTCP:
			httpPort = int(p.NodePort)
		default:
			continue
		}
	}

	return createdService, httpPort
}

// testEndpointReachability tests reachability to endpoints (i.e. IP, ServiceName) and ports. Test request is initiated from specified execPod.
// TCP and UDP protocol based service are supported at this moment
func testEndpointReachability(ctx context.Context, endpoint string, port int32, protocol v1.Protocol, execPod *v1.Pod, timeout time.Duration) error {
	cmd := ""
	switch protocol {
	case v1.ProtocolTCP:
		cmd = fmt.Sprintf("echo hostName | nc -v -t -w 2 %s %v", endpoint, port)
	case v1.ProtocolUDP:
		cmd = fmt.Sprintf("echo hostName | nc -v -u -w 2 %s %v", endpoint, port)
	default:
		return fmt.Errorf("service reachability check is not supported for %v", protocol)
	}

	err := wait.PollUntilContextTimeout(ctx, framework.Poll, timeout, true, func(ctx context.Context) (bool, error) {
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
	if err != nil {
		return fmt.Errorf("service is not reachable within %v timeout on endpoint %s %d over %s protocol", timeout, endpoint, port, protocol)
	}
	return nil
}
