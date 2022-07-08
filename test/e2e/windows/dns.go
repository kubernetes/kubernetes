/*
Copyright 2019 The Kubernetes Authors.

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

package windows

import (
	"context"
	"strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("[Feature:Windows] DNS", func() {

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessNodeOSDistroIs("windows")
	})

	f := framework.NewDefaultFramework("dns")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	ginkgo.It("should support configurable pod DNS servers", func() {

		ginkgo.By("Getting the IP address of the internal Kubernetes service")

		svc, err := f.ClientSet.CoreV1().Services("kube-system").Get(context.TODO(), "kube-dns", metav1.GetOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Preparing a test DNS service with injected DNS names...")
		// the default service IP will vary from cluster to cluster, but will always be present and is a good DNS test target
		testInjectedIP := svc.Spec.ClusterIP
		testSearchPath := "default.svc.cluster.local"

		ginkgo.By("Creating a windows pod with dnsPolicy=None and customized dnsConfig...")
		testPod := e2epod.NewAgnhostPod(f.Namespace.Name, "e2e-dns-utils", nil, nil, nil)
		testPod.Spec.DNSPolicy = v1.DNSNone
		testPod.Spec.DNSConfig = &v1.PodDNSConfig{
			Nameservers: []string{testInjectedIP, "1.1.1.1"},
			Searches:    []string{testSearchPath},
		}
		testPod.Spec.NodeSelector = map[string]string{
			"kubernetes.io/os": "windows",
		}
		testPod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), testPod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("confirming that the pod has a windows label")
		framework.ExpectEqual(testPod.Spec.NodeSelector["kubernetes.io/os"], "windows")
		framework.Logf("Created pod %v", testPod)
		defer func() {
			framework.Logf("Deleting pod %s...", testPod.Name)
			if err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), testPod.Name, *metav1.NewDeleteOptions(0)); err != nil {
				framework.Failf("Failed to delete pod %s: %v", testPod.Name, err)
			}
		}()
		framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(f.ClientSet, testPod.Name, f.Namespace.Name), "failed to wait for pod %s to be running", testPod.Name)

		// This isn't the best 'test' but it is a great diagnostic, see later test for the 'real' test.
		ginkgo.By("Calling ipconfig to get debugging info for this pod's DNS and confirm that a dns server 1.1.1.1 can be injected, along with ")
		cmd := []string{"ipconfig", "/all"}
		stdout, _, err := f.ExecWithOptions(framework.ExecOptions{
			Command:       cmd,
			Namespace:     f.Namespace.Name,
			PodName:       testPod.Name,
			ContainerName: "agnhost-container",
			CaptureStdout: true,
			CaptureStderr: true,
		})
		framework.ExpectNoError(err)
		framework.Logf("ipconfig /all:\n%s", stdout)

		if !strings.Contains(stdout, "1.1.1.1") {
			framework.Failf("One of the custom DNS options 1.1.1.1, not found in ipconfig /all")
		}

		// We've now verified that the DNS stuff is injected...  now lets make sure that curl'ing 'wrong' endpoints fails, i.e.
		// a negative control, to run before we run our final test...

		ginkgo.By("Verifying that curl queries FAIL for wrong URLs")

		// the below tests use curl because nslookup doesn't seem to use ndots properly
		// ideally we'd use the powershell native ResolveDns but, that is not a part of agnhost images (as of k8s 1.20)
		// TODO @jayunit100 add ResolveHost to agn images

		cmd = []string{"curl.exe", "-k", "https://kubernetezzzzzzzz:443"}
		stdout, _, err = f.ExecWithOptions(framework.ExecOptions{
			Command:       cmd,
			Namespace:     f.Namespace.Name,
			PodName:       testPod.Name,
			ContainerName: "agnhost-container",
			CaptureStdout: true,
			CaptureStderr: true,
		})
		if err == nil {
			framework.Logf("Warning: Somehow the curl command succeeded... The output was \n %v", stdout)
			framework.Failf("Expected a bogus URL query to fail - something is wrong with this test harness, cannot proceed.")
		}

		ginkgo.By("Verifying that injected dns records for 'kubernetes' resolve to the valid ip address")
		cmd = []string{"curl.exe", "-k", "https://kubernetes:443"}
		stdout, _, err = f.ExecWithOptions(framework.ExecOptions{
			Command:       cmd,
			Namespace:     f.Namespace.Name,
			PodName:       testPod.Name,
			ContainerName: "agnhost-container",
			CaptureStdout: true,
			CaptureStderr: true,
		})
		framework.Logf("Result of curling the kubernetes service... (Failure ok, only testing for the sake of DNS resolution) %v ... error = %v", stdout, err)

		// curl returns an error if the host isnt resolved, otherwise, it will return a passing result.
		if err != nil {
			framework.ExpectNoError(err)
		}

		// TODO: Add more test cases for other DNSPolicies.
	})
})
