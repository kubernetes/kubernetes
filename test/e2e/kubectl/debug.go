/*
Copyright 2024 The Kubernetes Authors.

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

// OWNER = sig/cli

package kubectl

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("kubectl debug", func() {
	defer ginkgo.GinkgoRecover()
	f := framework.NewDefaultFramework("kubectl-debug")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	var ns string
	var c clientset.Interface
	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
	})

	ginkgo.Describe("custom profile", func() {
		ginkgo.It("should be applied on static profiles on ephemeral container", func(ctx context.Context) {
			debugPodName := "e2e-test-debug-custom-profile-pod-ephemeral"
			customProfileImage := imageutils.GetE2EImage(imageutils.BusyBox)
			ginkgo.By("running the image " + customProfileImage)
			e2ekubectl.RunKubectlOrDie(ns,
				"run", debugPodName,
				"--image="+customProfileImage,
				podRunningTimeoutArg,
				"--labels=run="+debugPodName,
				"--", "sh", "-c",
				"sleep 3600")

			ginkgo.By("verifying the pod " + debugPodName + " is running")
			label := labels.SelectorFromSet(map[string]string{"run": debugPodName})
			err := testutils.WaitForPodsWithLabelRunning(c, ns, label)
			if err != nil {
				framework.Failf("Failed getting pod %s: %v", debugPodName, err)
			}

			tmpDir, err := os.MkdirTemp("", "test-custom-profile-debug")
			customProfileFile := "custom.yaml"
			framework.ExpectNoError(err)
			defer os.Remove(tmpDir) //nolint:errcheck
			framework.ExpectNoError(os.WriteFile(filepath.Join(tmpDir, customProfileFile), []byte(`
securityContext:
  capabilities:
    add:
      - NET_ADMIN
env:
  - name: REQUIRED_ENV_VAR
    value: value2
`), os.FileMode(0755)))
			ginkgo.By("verifying ephemeral container has correct fields")
			e2ekubectl.RunKubectlOrDie(ns,
				"debug",
				debugPodName,
				"--image="+imageutils.GetE2EImage(imageutils.Nginx),
				"--container=debugger",
				"--profile=general", // general profile sets capability to SYS_PTRACE and custom should overwrite it to NET_ADMIN
				"--custom="+filepath.Join(tmpDir, customProfileFile),
				"--", "sh", "-c",
				"sleep 3600")

			ginkgo.By("verifying the container debugger is running")
			err = wait.PollUntilContextTimeout(ctx, 3*time.Second, 2*time.Minute, false, func(ctx context.Context) (bool, error) {
				running := e2ekubectl.RunKubectlOrDie(ns, "get", "pod", debugPodName, "-o", "template", `--template={{if (exists . "status" "ephemeralContainerStatuses")}}{{range .status.ephemeralContainerStatuses}}{{if (and (eq .name "debugger") (exists . "state" "running"))}}true{{end}}{{end}}{{end}}`)
				if running != "true" {
					return false, nil
				}
				return true, nil
			})
			framework.ExpectNoError(err)
			output := e2ekubectl.RunKubectlOrDie(ns, "get", "pod", debugPodName, "-o", "jsonpath={.spec.ephemeralContainers[*]}")
			ginkgo.By("verifying NET_ADMIN is added")
			if !strings.Contains(output, "NET_ADMIN") {
				framework.Failf("failed to find the expected NET_ADMIN capability")
			}
			ginkgo.By("verifying SYS_PTRACE is overridden")
			if strings.Contains(output, "SYS_PTRACE") {
				framework.Failf("unexpected SYS_PTRACE capability")
			}
			ginkgo.By("verifying REQUIRED_ENV_VAR is added")
			if !strings.Contains(output, "REQUIRED_ENV_VAR") {
				framework.Failf("failed to find the expected REQUIRED_ENV_VAR environment variable")
			}
		})

		ginkgo.It("should be applied on static profiles while copying from pod", func(ctx context.Context) {
			debugPodName := "e2e-test-debug-custom-profile-pod"
			customProfileImage := imageutils.GetE2EImage(imageutils.BusyBox)
			ginkgo.By("running the image " + customProfileImage)
			e2ekubectl.RunKubectlOrDie(ns,
				"run", debugPodName,
				"--image="+customProfileImage,
				podRunningTimeoutArg,
				"--labels=run="+debugPodName,
				"--", "sh", "-c",
				"sleep 3600")

			ginkgo.By("verifying the pod " + debugPodName + " is running")
			label := labels.SelectorFromSet(map[string]string{"run": debugPodName})
			err := testutils.WaitForPodsWithLabelRunning(c, ns, label)
			if err != nil {
				framework.Failf("Failed getting pod %s: %v", debugPodName, err)
			}

			tmpDir, err := os.MkdirTemp("", "test-custom-profile-debug")
			customProfileFile := "custom.yaml"
			framework.ExpectNoError(err)
			defer os.Remove(tmpDir) //nolint:errcheck
			framework.ExpectNoError(os.WriteFile(filepath.Join(tmpDir, customProfileFile), []byte(`
securityContext:
  capabilities:
    add:
      - NET_ADMIN
env:
  - name: REQUIRED_ENV_VAR
    value: value2
`), os.FileMode(0755)))
			ginkgo.By("verifying copied pod has correct fields")
			e2ekubectl.RunKubectlOrDie(ns,
				"debug",
				debugPodName,
				"--image="+imageutils.GetE2EImage(imageutils.Nginx),
				"--copy-to=my-debugger",
				"--profile=general", // general profile sets capability to SYS_PTRACE and custom should overwrite it to NET_ADMIN
				"--custom="+filepath.Join(tmpDir, customProfileFile),
				"--", "sh", "-c",
				"sleep 3600")

			ginkgo.By("verifying the container in pod my-debugger is running")
			err = wait.PollUntilContextTimeout(ctx, 3*time.Second, 2*time.Minute, false, func(ctx context.Context) (bool, error) {
				running := e2ekubectl.RunKubectlOrDie(ns, "get", "pod", "my-debugger", "-o", "template", fmt.Sprintf(`--template={{if (exists . "status" "containerStatuses")}}{{range .status.containerStatuses}}{{if (and (ne .name "%s") (exists . "state" "running"))}}true{{end}}{{end}}{{end}}`, debugPodName))
				if running != "true" {
					return false, nil
				}
				return true, nil
			})
			framework.ExpectNoError(err)
			output := e2ekubectl.RunKubectlOrDie(ns, "get", "pod", "my-debugger", "-o", "jsonpath={.spec.containers[*]}")
			ginkgo.By("verifying NET_ADMIN is added")
			if !strings.Contains(output, "NET_ADMIN") {
				framework.Failf("failed to find the expected NET_ADMIN capability")
			}
			ginkgo.By("verifying SYS_PTRACE is overridden")
			if strings.Contains(output, "SYS_PTRACE") {
				framework.Failf("unexpected SYS_PTRACE capability")
			}
			ginkgo.By("verifying REQUIRED_ENV_VAR is added")
			if !strings.Contains(output, "REQUIRED_ENV_VAR") {
				framework.Failf("failed to find the expected REQUIRED_ENV_VAR environment variable")
			}
		})
	})
})
