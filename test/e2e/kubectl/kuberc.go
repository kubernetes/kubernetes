/*
Copyright 2025 The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	"sigs.k8s.io/yaml"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var kuberc = `
apiVersion: kubectl.config.k8s.io/v1beta1
kind: Preference
aliases:
- name: getn
  command: get
  prependArgs:
   - namespace
  options:
   - name: output
     default: json
- name: runx
  command: run
  options:
  - name: image
    default: %s
  - name: labels
    default: app=test,env=test
  - name: env
    default: DNS_DOMAIN=test
  - name: namespace
    default: %s
  appendArgs:
  - test-pod-2
  - --
  - custom-arg1
  - custom-arg2
defaults:
- command: apply
  options:
  - name: server-side
    default: "true"
  - name: dry-run
    default: "server"
  - name: validate
    default: "strict"
- command: delete
  options:
  - name: interactive
    default: "true"
- command: get
  options:
  - name: namespace
    default: "%s"
  - name: output
    default: "json"
`

var _ = SIGDescribe("kubectl kuberc", func() {
	f := framework.NewDefaultFramework("kubectl-kuberc")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var ns string
	ginkgo.BeforeEach(func() {
		ns = f.Namespace.Name
	})

	ginkgo.Describe("given preferences", func() {
		kubercContent := fmt.Sprintf(kuberc, imageutils.GetE2EImage(imageutils.BusyBox), ns, ns)
		tmpDir, err := os.MkdirTemp("", "test-kuberc")
		framework.ExpectNoError(err)
		defer os.Remove(tmpDir) //nolint:errcheck
		kubercFile := filepath.Join(tmpDir, "kuberc.yaml")
		framework.ExpectNoError(os.WriteFile(kubercFile, []byte(kubercContent), os.FileMode(0755)), "creating a kuberc.yaml in temp directory")

		ginkgo.It("should be applied", func(ctx context.Context) {
			ginkgo.By("verifying that json formatter output of the alias is working")
			args := []string{"getn", fmt.Sprintf("--kuberc=%s", kubercFile), ns}
			output := e2ekubectl.NewKubectlCommand(ns, args...).ExecOrDie(ns)
			var namespace v1.Namespace
			err = json.Unmarshal([]byte(output), &namespace)
			framework.ExpectNoError(err)
			gomega.Expect(namespace.Name).To(gomega.Equal(ns))

			ginkgo.By("verifying that alias for run is working")
			args = []string{"runx", fmt.Sprintf("--kuberc=%s", kubercFile)}
			e2ekubectl.NewKubectlCommand(ns, args...).ExecOrDie(ns)
			ginkgo.By("verifying that overridden flags in get command is used")
			args = []string{"get", "pod/test-pod-2", fmt.Sprintf("--kuberc=%s", kubercFile)}
			podOutput := e2ekubectl.NewKubectlCommand(ns, args...).ExecOrDie(ns)
			var pod v1.Pod
			err = json.Unmarshal([]byte(podOutput), &pod)
			framework.ExpectNoError(err)
			gomega.Expect(pod.Spec.Containers[0].Image).To(gomega.Equal(imageutils.GetE2EImage(imageutils.BusyBox)))
			gomega.Expect(pod.Labels).To(gomega.HaveKeyWithValue("app", "test"))
			gomega.Expect(pod.Labels).To(gomega.HaveKeyWithValue("env", "test"))
			gomega.Expect(strings.Join(pod.Spec.Containers[0].Args, ",")).To(gomega.ContainSubstring("custom-arg1"))
			gomega.Expect(strings.Join(pod.Spec.Containers[0].Args, ",")).To(gomega.ContainSubstring("custom-arg2"))

			ginkgo.By("verifying that interactive flag that is enabled by default in kuberc is used")
			args = []string{"delete", "pod/test-pod-2", "--kuberc", kubercFile}
			output = e2ekubectl.NewKubectlCommand(ns, args...).ExecOrDie(ns)
			gomega.Expect(output).To(gomega.ContainSubstring("You are about to delete the following 1 resource(s)"))
		})

		ginkgo.It("should be ignored when flags are explicitly passed", func(ctx context.Context) {
			ginkgo.By("verifying that yaml formatter output surpasses the flag in kuberc alias")
			args := []string{"getn", fmt.Sprintf("--kuberc=%s", kubercFile), ns, "-oyaml"}
			output := e2ekubectl.NewKubectlCommand(ns, args...).ExecOrDie(ns)
			var namespace v1.Namespace
			err = yaml.Unmarshal([]byte(output), &namespace)
			framework.ExpectNoError(err)
			gomega.Expect(namespace.Name).To(gomega.Equal(ns))

			ginkgo.By("verifying that explicitly passed flag surpasses the flag in kuberc alias")
			args = []string{"runx", fmt.Sprintf("--kuberc=%s", kubercFile), fmt.Sprintf("--image=%s", imageutils.GetE2EImage(imageutils.Nginx))}
			e2ekubectl.NewKubectlCommand(ns, args...).ExecOrDie(ns)
			args = []string{"get", "pod/test-pod-2", fmt.Sprintf("--kuberc=%s", kubercFile), "-oyaml"}
			podOutput := e2ekubectl.NewKubectlCommand(ns, args...).ExecOrDie(ns)
			var pod v1.Pod
			err = yaml.Unmarshal([]byte(podOutput), &pod)
			framework.ExpectNoError(err)
			gomega.Expect(pod.Spec.Containers[0].Image).To(gomega.Equal(imageutils.GetE2EImage(imageutils.Nginx)))
			gomega.Expect(pod.Labels).To(gomega.HaveKeyWithValue("app", "test"))
			gomega.Expect(pod.Labels).To(gomega.HaveKeyWithValue("env", "test"))
			gomega.Expect(strings.Join(pod.Spec.Containers[0].Args, ",")).To(gomega.ContainSubstring("custom-arg1"))
			gomega.Expect(strings.Join(pod.Spec.Containers[0].Args, ",")).To(gomega.ContainSubstring("custom-arg2"))

			ginkgo.By("verifying that interactive flag that is enabled by default in kuberc is surpassed by explicit")
			args = []string{"delete", "pod/test-pod-2", "--interactive=false", "--kuberc", kubercFile}
			output = e2ekubectl.NewKubectlCommand(ns, args...).ExecOrDie(ns)
			gomega.Expect(output).NotTo(gomega.ContainSubstring("You are about to delete the following 1 resource(s)"))
		})
	})
})
