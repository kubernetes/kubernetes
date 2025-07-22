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
	"os"
	"path/filepath"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	"k8s.io/apimachinery/pkg/labels"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("kubectl debug", func() {
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
			_, err := e2epod.WaitForPodsWithLabelRunningReady(ctx, c, ns, label, 1, framework.PodStartShortTimeout)
			if err != nil {
				framework.Failf("Failed getting pod %s: %v", debugPodName, err)
			}

			tmpDir, err := os.MkdirTemp("", "test-custom-profile-debug")
			framework.ExpectNoError(err)
			defer os.Remove(tmpDir) //nolint:errcheck
			customProfileFile := "custom.yaml"
			framework.ExpectNoError(os.WriteFile(filepath.Join(tmpDir, customProfileFile), []byte(`
securityContext:
  capabilities:
    add:
      - NET_ADMIN
env:
  - name: REQUIRED_ENV_VAR
    value: value2
`), os.FileMode(0755)), "creating a custom.yaml in temp directory")
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
			framework.ExpectNoError(e2epod.WaitForContainerRunning(ctx, c, ns, debugPodName, "debugger", framework.PodStartShortTimeout))
			output := e2ekubectl.RunKubectlOrDie(ns, "get", "pod", debugPodName, "-o", "jsonpath={.spec.ephemeralContainers[*]}")
			ginkgo.By("verifying NET_ADMIN is added")
			gomega.Expect(output).To(gomega.ContainSubstring("NET_ADMIN"))
			ginkgo.By("verifying SYS_PTRACE is overridden")
			gomega.Expect(output).NotTo(gomega.ContainSubstring("SYS_PTRACE"))
			ginkgo.By("verifying REQUIRED_ENV_VAR is added")
			gomega.Expect(output).To(gomega.ContainSubstring("REQUIRED_ENV_VAR"))
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
			framework.ExpectNoError(err)
			defer os.Remove(tmpDir) //nolint:errcheck
			customProfileFile := "custom.yaml"
			framework.ExpectNoError(os.WriteFile(filepath.Join(tmpDir, customProfileFile), []byte(`
securityContext:
  capabilities:
    add:
      - NET_ADMIN
env:
  - name: REQUIRED_ENV_VAR
    value: value2
`), os.FileMode(0755)), "creating a custom.yaml in temp directory")
			ginkgo.By("verifying copied pod has correct fields")
			e2ekubectl.RunKubectlOrDie(ns,
				"debug",
				debugPodName,
				"--image="+imageutils.GetE2EImage(imageutils.Nginx),
				"--copy-to=my-debugger",
				"--container=debugger",
				"--profile=general", // general profile sets capability to SYS_PTRACE and custom should overwrite it to NET_ADMIN
				"--custom="+filepath.Join(tmpDir, customProfileFile),
				"--", "sh", "-c",
				"sleep 3600")

			ginkgo.By("verifying the container in pod my-debugger is running")
			framework.ExpectNoError(e2epod.WaitForContainerRunning(ctx, c, ns, "my-debugger", "debugger", framework.PodStartShortTimeout))

			output := e2ekubectl.RunKubectlOrDie(ns, "get", "pod", "my-debugger", "-o", "jsonpath={.spec.containers[*]}")
			ginkgo.By("verifying NET_ADMIN is added")
			gomega.Expect(output).To(gomega.ContainSubstring("NET_ADMIN"))
			ginkgo.By("verifying SYS_PTRACE is overridden")
			gomega.Expect(output).NotTo(gomega.ContainSubstring("SYS_PTRACE"))
			ginkgo.By("verifying REQUIRED_ENV_VAR is added")
			gomega.Expect(output).To(gomega.ContainSubstring("REQUIRED_ENV_VAR"))
		})
	})
})
