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

package node

import (
	"context"
	"fmt"
	"path/filepath"
	"sync"
	"time"

	e2eutils "k8s.io/kubernetes/test/e2e/framework/utils"

	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	clientset "k8s.io/client-go/kubernetes"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	commonutils "k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eauth "k8s.io/kubernetes/test/e2e/framework/auth"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo"
)

const (
	serverStartTimeout = e2eutils.PodStartTimeout + 3*time.Minute
)

var _ = SIGDescribe("[Feature:Example]", func() {
	f := framework.NewDefaultFramework("examples")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline

	var c clientset.Interface
	var ns string
	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name

		// this test wants powerful permissions.  Since the namespace names are unique, we can leave this
		// lying around so we don't have to race any caches
		err := e2eauth.BindClusterRoleInNamespace(c.RbacV1(), "edit", f.Namespace.Name,
			rbacv1.Subject{Kind: rbacv1.ServiceAccountKind, Namespace: f.Namespace.Name, Name: "default"})
		e2eutils.ExpectNoError(err)

		err = e2eauth.WaitForAuthorizationUpdate(c.AuthorizationV1(),
			serviceaccount.MakeUsername(f.Namespace.Name, "default"),
			f.Namespace.Name, "create", schema.GroupResource{Resource: "pods"}, true)
		e2eutils.ExpectNoError(err)
	})

	ginkgo.Describe("Liveness", func() {
		ginkgo.It("liveness pods should be automatically restarted", func() {
			test := "test/fixtures/doc-yaml/user-guide/liveness"
			execYaml := readFile(test, "exec-liveness.yaml.in")
			httpYaml := readFile(test, "http-liveness.yaml.in")

			e2eutils.RunKubectlOrDieInput(ns, execYaml, "create", "-f", "-")
			e2eutils.RunKubectlOrDieInput(ns, httpYaml, "create", "-f", "-")

			// Since both containers start rapidly, we can easily run this test in parallel.
			var wg sync.WaitGroup
			passed := true
			checkRestart := func(podName string, timeout time.Duration) {
				err := e2epod.WaitForPodNameRunningInNamespace(c, podName, ns)
				e2eutils.ExpectNoError(err)
				for t := time.Now(); time.Since(t) < timeout; time.Sleep(e2eutils.Poll) {
					pod, err := c.CoreV1().Pods(ns).Get(context.TODO(), podName, metav1.GetOptions{})
					e2eutils.ExpectNoError(err, fmt.Sprintf("getting pod %s", podName))
					stat := podutil.GetExistingContainerStatus(pod.Status.ContainerStatuses, podName)
					e2eutils.Logf("Pod: %s, restart count:%d", stat.Name, stat.RestartCount)
					if stat.RestartCount > 0 {
						e2eutils.Logf("Saw %v restart, succeeded...", podName)
						wg.Done()
						return
					}
				}
				e2eutils.Logf("Failed waiting for %v restart! ", podName)
				passed = false
				wg.Done()
			}

			ginkgo.By("Check restarts")

			// Start the "actual test", and wait for both pods to complete.
			// If 2 fail: Something is broken with the test (or maybe even with liveness).
			// If 1 fails: Its probably just an error in the examples/ files themselves.
			wg.Add(2)
			for _, c := range []string{"liveness-http", "liveness-exec"} {
				go checkRestart(c, 2*time.Minute)
			}
			wg.Wait()
			if !passed {
				e2eutils.Failf("At least one liveness example failed.  See the logs above.")
			}
		})
	})

	ginkgo.Describe("Secret", func() {
		ginkgo.It("should create a pod that reads a secret", func() {
			test := "test/fixtures/doc-yaml/user-guide/secrets"
			secretYaml := readFile(test, "secret.yaml")
			podYaml := readFile(test, "secret-pod.yaml.in")

			podName := "secret-test-pod"

			ginkgo.By("creating secret and pod")
			e2eutils.RunKubectlOrDieInput(ns, secretYaml, "create", "-f", "-")
			e2eutils.RunKubectlOrDieInput(ns, podYaml, "create", "-f", "-")
			err := e2epod.WaitForPodNoLongerRunningInNamespace(c, podName, ns)
			e2eutils.ExpectNoError(err)

			ginkgo.By("checking if secret was read correctly")
			_, err = e2eutils.LookForStringInLog(ns, "secret-test-pod", "test-container", "value-1", serverStartTimeout)
			e2eutils.ExpectNoError(err)
		})
	})

	ginkgo.Describe("Downward API", func() {
		ginkgo.It("should create a pod that prints his name and namespace", func() {
			test := "test/fixtures/doc-yaml/user-guide/downward-api"
			podYaml := readFile(test, "dapi-pod.yaml.in")
			podName := "dapi-test-pod"

			ginkgo.By("creating the pod")
			e2eutils.RunKubectlOrDieInput(ns, podYaml, "create", "-f", "-")
			err := e2epod.WaitForPodNoLongerRunningInNamespace(c, podName, ns)
			e2eutils.ExpectNoError(err)

			ginkgo.By("checking if name and namespace were passed correctly")
			_, err = e2eutils.LookForStringInLog(ns, podName, "test-container", fmt.Sprintf("MY_POD_NAMESPACE=%v", ns), serverStartTimeout)
			e2eutils.ExpectNoError(err)
			_, err = e2eutils.LookForStringInLog(ns, podName, "test-container", fmt.Sprintf("MY_POD_NAME=%v", podName), serverStartTimeout)
			e2eutils.ExpectNoError(err)
		})
	})
})

func readFile(test, file string) string {
	from := filepath.Join(test, file)
	data, err := e2etestfiles.Read(from)
	if err != nil {
		e2eutils.Fail(err.Error())
	}
	return commonutils.SubstituteImageName(string(data))
}
