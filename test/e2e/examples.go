/*
Copyright 2015 The Kubernetes Authors.

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

package e2e

import (
	"fmt"
	"path/filepath"
	"sync"
	"time"

	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	clientset "k8s.io/client-go/kubernetes"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/testfiles"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	serverStartTimeout = framework.PodStartTimeout + 3*time.Minute
)

var _ = framework.KubeDescribe("[Feature:Example]", func() {
	f := framework.NewDefaultFramework("examples")

	var c clientset.Interface
	var ns string
	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name

		// this test wants powerful permissions.  Since the namespace names are unique, we can leave this
		// lying around so we don't have to race any caches
		framework.BindClusterRoleInNamespace(c.RbacV1beta1(), "edit", f.Namespace.Name,
			rbacv1beta1.Subject{Kind: rbacv1beta1.ServiceAccountKind, Namespace: f.Namespace.Name, Name: "default"})

		err := framework.WaitForAuthorizationUpdate(c.AuthorizationV1beta1(),
			serviceaccount.MakeUsername(f.Namespace.Name, "default"),
			f.Namespace.Name, "create", schema.GroupResource{Resource: "pods"}, true)
		framework.ExpectNoError(err)
	})

	framework.KubeDescribe("Liveness", func() {
		It("liveness pods should be automatically restarted", func() {
			test := "test/fixtures/doc-yaml/user-guide/liveness"
			execYaml := readFile(test, "exec-liveness.yaml")
			httpYaml := readFile(test, "http-liveness.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			framework.RunKubectlOrDieInput(execYaml, "create", "-f", "-", nsFlag)
			framework.RunKubectlOrDieInput(httpYaml, "create", "-f", "-", nsFlag)

			// Since both containers start rapidly, we can easily run this test in parallel.
			var wg sync.WaitGroup
			passed := true
			checkRestart := func(podName string, timeout time.Duration) {
				err := framework.WaitForPodNameRunningInNamespace(c, podName, ns)
				Expect(err).NotTo(HaveOccurred())
				for t := time.Now(); time.Since(t) < timeout; time.Sleep(framework.Poll) {
					pod, err := c.CoreV1().Pods(ns).Get(podName, metav1.GetOptions{})
					framework.ExpectNoError(err, fmt.Sprintf("getting pod %s", podName))
					stat := podutil.GetExistingContainerStatus(pod.Status.ContainerStatuses, podName)
					framework.Logf("Pod: %s, restart count:%d", stat.Name, stat.RestartCount)
					if stat.RestartCount > 0 {
						framework.Logf("Saw %v restart, succeeded...", podName)
						wg.Done()
						return
					}
				}
				framework.Logf("Failed waiting for %v restart! ", podName)
				passed = false
				wg.Done()
			}

			By("Check restarts")

			// Start the "actual test", and wait for both pods to complete.
			// If 2 fail: Something is broken with the test (or maybe even with liveness).
			// If 1 fails: Its probably just an error in the examples/ files themselves.
			wg.Add(2)
			for _, c := range []string{"liveness-http", "liveness-exec"} {
				go checkRestart(c, 2*time.Minute)
			}
			wg.Wait()
			if !passed {
				framework.Failf("At least one liveness example failed.  See the logs above.")
			}
		})
	})

	framework.KubeDescribe("Secret", func() {
		It("should create a pod that reads a secret", func() {
			test := "test/fixtures/doc-yaml/user-guide/secrets"
			secretYaml := readFile(test, "secret.yaml")
			podYaml := readFile(test, "secret-pod.yaml")

			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			podName := "secret-test-pod"

			By("creating secret and pod")
			framework.RunKubectlOrDieInput(secretYaml, "create", "-f", "-", nsFlag)
			framework.RunKubectlOrDieInput(podYaml, "create", "-f", "-", nsFlag)
			err := framework.WaitForPodNoLongerRunningInNamespace(c, podName, ns)
			Expect(err).NotTo(HaveOccurred())

			By("checking if secret was read correctly")
			_, err = framework.LookForStringInLog(ns, "secret-test-pod", "test-container", "value-1", serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	framework.KubeDescribe("Downward API", func() {
		It("should create a pod that prints his name and namespace", func() {
			test := "test/fixtures/doc-yaml/user-guide/downward-api"
			podYaml := readFile(test, "dapi-pod.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			podName := "dapi-test-pod"

			By("creating the pod")
			framework.RunKubectlOrDieInput(podYaml, "create", "-f", "-", nsFlag)
			err := framework.WaitForPodNoLongerRunningInNamespace(c, podName, ns)
			Expect(err).NotTo(HaveOccurred())

			By("checking if name and namespace were passed correctly")
			_, err = framework.LookForStringInLog(ns, podName, "test-container", fmt.Sprintf("MY_POD_NAMESPACE=%v", ns), serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())
			_, err = framework.LookForStringInLog(ns, podName, "test-container", fmt.Sprintf("MY_POD_NAME=%v", podName), serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())
		})
	})
})

func readFile(test, file string) string {
	from := filepath.Join(test, file)
	return string(testfiles.ReadOrDie(from, Fail))
}
