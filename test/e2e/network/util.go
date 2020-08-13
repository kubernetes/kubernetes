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
	"fmt"
	"net"
	"regexp"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enetwork "k8s.io/kubernetes/test/e2e/framework/network"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

var (
	// agnHostImage is the image URI of AgnHost
	agnHostImage = imageutils.GetE2EImage(imageutils.Agnhost)
)

// GetHTTPContent returns the content of the given url by HTTP.
func GetHTTPContent(host string, port int, timeout time.Duration, url string) bytes.Buffer {
	var body bytes.Buffer
	if pollErr := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		result := e2enetwork.PokeHTTP(host, port, url, nil)
		if result.Status == e2enetwork.HTTPSuccess {
			body.Write(result.Body)
			return true, nil
		}
		return false, nil
	}); pollErr != nil {
		framework.Failf("Could not reach HTTP service through %v:%v%v after %v: %v", host, port, url, timeout, pollErr)
	}
	return body
}

// GetHTTPContentFromTestContainer returns the content of the given url by HTTP via a test container.
func GetHTTPContentFromTestContainer(config *e2enetwork.NetworkingTestConfig, host string, port int, timeout time.Duration, dialCmd string) (string, error) {
	var body string
	pollFn := func() (bool, error) {
		resp, err := config.GetResponseFromTestContainer("http", dialCmd, host, port)
		if err != nil || len(resp.Errors) > 0 || len(resp.Responses) == 0 {
			return false, nil
		}
		body = resp.Responses[0]
		return true, nil
	}
	if pollErr := wait.PollImmediate(framework.Poll, timeout, pollFn); pollErr != nil {
		return "", pollErr
	}
	return body, nil
}

// DescribeSvc logs the output of kubectl describe svc for the given namespace
func DescribeSvc(ns string) {
	framework.Logf("\nOutput of kubectl describe svc:\n")
	desc, _ := framework.RunKubectl(
		ns, "describe", "svc", fmt.Sprintf("--namespace=%v", ns))
	framework.Logf(desc)
}

// newAgnhostPod returns a pod that uses the agnhost image. The image's binary supports various subcommands
// that behave the same, no matter the underlying OS.
func newAgnhostPod(name string, args ...string) *v1.Pod {
	zero := int64(0)
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			TerminationGracePeriodSeconds: &zero,
			Containers: []v1.Container{
				{
					Name:  "agnhost",
					Image: agnHostImage,
					Args:  args,
				},
			},
		},
	}
}

// CheckSCTPModuleLoadedOnNodes checks whether any node on the list has the
// sctp.ko module loaded
// For security reasons, and also to allow clusters to use userspace SCTP implementations,
// we require that just creating an SCTP Pod/Service/NetworkPolicy must not do anything
// that would cause the sctp kernel module to be loaded.
func CheckSCTPModuleLoadedOnNodes(f *framework.Framework, nodes *v1.NodeList) bool {
	hostExec := utils.NewHostExec(f)
	defer hostExec.Cleanup()
	re := regexp.MustCompile(`^\s*sctp\s+`)
	cmd := "lsmod | grep sctp"
	for _, node := range nodes.Items {
		framework.Logf("Executing cmd %q on node %v", cmd, node.Name)
		result, err := hostExec.IssueCommandWithResult(cmd, &node)
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
		stdout, err = framework.RunHostCmd(sourcePod.Namespace, sourcePod.Name, cmd)
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
