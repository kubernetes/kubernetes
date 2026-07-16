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
	"k8s.io/kubectl/pkg/config/v1beta1"

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
	ginkgo.Describe("commands", func() {
		var tmpDir string
		var kubercFile string

		ginkgo.BeforeEach(func() {
			var err error
			tmpDir, err = os.MkdirTemp("", "test-kuberc-cmd")
			framework.ExpectNoError(err)
			kubercFile = filepath.Join(tmpDir, "kuberc.yaml")
			minimalKuberc := `apiVersion: kubectl.config.k8s.io/v1beta1
kind: Preference
`
			framework.ExpectNoError(os.WriteFile(kubercFile, []byte(minimalKuberc), os.FileMode(0644)))
		})

		ginkgo.AfterEach(func() {
			if tmpDir != "" {
				os.RemoveAll(tmpDir) //nolint:errcheck
			}
		})

		ginkgo.It("view should display kuberc file", func(ctx context.Context) {
			ginkgo.By("creating a kuberc file with defaults and aliases")
			kubercContent := fmt.Sprintf(kuberc, imageutils.GetE2EImage(imageutils.BusyBox), ns, ns)
			framework.ExpectNoError(os.WriteFile(kubercFile, []byte(kubercContent), os.FileMode(0755)))

			ginkgo.By("viewing the kuberc file and parsing as Preference")
			output := e2ekubectl.RunKubectlOrDie(ns, "kuberc", "view", fmt.Sprintf("--kuberc=%s", kubercFile))
			var pref v1beta1.Preference
			err := yaml.Unmarshal([]byte(output), &pref)
			framework.ExpectNoError(err, "failed to unmarshal kuberc view output")

			ginkgo.By("verifying structure")
			if pref.APIVersion != "kubectl.config.k8s.io/v1beta1" {
				framework.Failf("expected apiVersion kubectl.config.k8s.io/v1beta1, got: %s", pref.APIVersion)
			}
			if pref.Kind != "Preference" {
				framework.Failf("expected kind Preference, got: %s", pref.Kind)
			}
			if len(pref.Aliases) == 0 {
				framework.Failf("expected aliases to be present")
			}
			if len(pref.Defaults) == 0 {
				framework.Failf("expected defaults to be present")
			}
		})

		ginkgo.It("set should create defaults and apply them to kubectl commands", func(ctx context.Context) {
			ginkgo.By("setting default output format to yaml for get command")
			e2ekubectl.RunKubectlOrDie(ns, "kuberc", "set", fmt.Sprintf("--kuberc=%s", kubercFile), "--section", "defaults", "--command", "get", "--option", "output=yaml")

			ginkgo.By("running kubectl get namespace with the kuberc file")
			output := e2ekubectl.RunKubectlOrDie(ns, "get", "namespace", ns, fmt.Sprintf("--kuberc=%s", kubercFile))

			ginkgo.By("verifying output is in YAML format due to kuberc default")
			var namespace v1.Namespace
			err := yaml.Unmarshal([]byte(output), &namespace)
			framework.ExpectNoError(err, "expected output to be valid YAML")
			if namespace.Name != ns {
				framework.Failf("expected namespace name %s, got: %s", ns, namespace.Name)
			}

			ginkgo.By("explicitly passing -ojson should override the kuberc default")
			output = e2ekubectl.RunKubectlOrDie(ns, "get", "namespace", ns, fmt.Sprintf("--kuberc=%s", kubercFile), "-ojson")

			ginkgo.By("verifying output is in JSON format when explicitly requested")
			var namespaceJSON v1.Namespace
			err = json.Unmarshal([]byte(output), &namespaceJSON)
			framework.ExpectNoError(err, "expected output to be valid JSON when explicitly passed -ojson")
			if namespaceJSON.Name != ns {
				framework.Failf("expected namespace name %s, got: %s", ns, namespaceJSON.Name)
			}
		})

		ginkgo.It("set should create aliases with prependArgs and execute them", func(ctx context.Context) {
			ginkgo.By("creating an alias 'getns' for 'get namespace' with yaml output")
			e2ekubectl.RunKubectlOrDie(ns, "kuberc", "set", fmt.Sprintf("--kuberc=%s", kubercFile),
				"--section", "aliases",
				"--name", "getns",
				"--command", "get",
				"--prependarg", "namespace",
				"--option", "output=yaml")

			ginkgo.By("using the alias to get namespace")
			output := e2ekubectl.RunKubectlOrDie(ns, "getns", ns, fmt.Sprintf("--kuberc=%s", kubercFile))

			ginkgo.By("verifying the alias worked and returned yaml output")
			var namespace v1.Namespace
			err := yaml.Unmarshal([]byte(output), &namespace)
			framework.ExpectNoError(err, "expected alias output to be valid YAML")
			if namespace.Name != ns {
				framework.Failf("expected namespace name %s, got: %s", ns, namespace.Name)
			}
		})

		ginkgo.It("set should create aliases with appendArgs and execute them", func(ctx context.Context) {
			ginkgo.By("creating an alias with appendArgs for run command")
			e2ekubectl.RunKubectlOrDie(ns, "kuberc", "set", fmt.Sprintf("--kuberc=%s", kubercFile),
				"--section", "aliases",
				"--name", "runtestpod",
				"--command", "run",
				"--option", "image="+imageutils.GetE2EImage(imageutils.BusyBox),
				"--appendarg", "--",
				"--appendarg", "sleep",
				"--appendarg", "3600")

			ginkgo.By("using the alias to create a pod")
			e2ekubectl.RunKubectlOrDie(ns, "runtestpod", "test-pod", fmt.Sprintf("--kuberc=%s", kubercFile), "-n", ns)

			ginkgo.By("verifying the pod was created with appended args")
			output := e2ekubectl.RunKubectlOrDie(ns, "get", "pod", "test-pod", "-n", ns, "-oyaml")
			var pod v1.Pod
			err := yaml.Unmarshal([]byte(output), &pod)
			framework.ExpectNoError(err)

			if len(pod.Spec.Containers) == 0 {
				framework.Failf("expected pod to have at least one container")
			}
			if len(pod.Spec.Containers[0].Args) < 2 {
				framework.Failf("expected pod to have appended args, got: %v", pod.Spec.Containers[0].Args)
			}
			foundSleep := false
			for _, arg := range pod.Spec.Containers[0].Args {
				if arg == "sleep" {
					foundSleep = true
					break
				}
			}
			if !foundSleep {
				framework.Failf("expected pod args to contain 'sleep' from appendArgs, got: %v", pod.Spec.Containers[0].Args)
			}
		})

		ginkgo.It("set should require --overwrite to replace existing entries", func(ctx context.Context) {
			ginkgo.By("creating initial default")
			e2ekubectl.RunKubectlOrDie(ns, "kuberc", "set", fmt.Sprintf("--kuberc=%s", kubercFile), "--section", "defaults", "--command", "get", "--option", "output=yaml")

			ginkgo.By("attempting to overwrite without --overwrite flag should fail")
			_, err := e2ekubectl.RunKubectl(ns, "kuberc", "set", fmt.Sprintf("--kuberc=%s", kubercFile), "--section", "defaults", "--command", "get", "--option", "output=json")
			if err == nil {
				framework.Failf("expected command to fail without --overwrite flag")
			}

			ginkgo.By("overwriting with --overwrite flag should succeed")
			e2ekubectl.RunKubectlOrDie(ns, "kuberc", "set", fmt.Sprintf("--kuberc=%s", kubercFile), "--section", "defaults", "--command", "get", "--option", "output=json", "--overwrite")

			ginkgo.By("verifying the updated default is applied")
			output := e2ekubectl.RunKubectlOrDie(ns, "get", "namespace", ns, fmt.Sprintf("--kuberc=%s", kubercFile))
			var namespace v1.Namespace
			err = json.Unmarshal([]byte(output), &namespace)
			framework.ExpectNoError(err, "expected output to be JSON after overwrite")
			if namespace.Name != ns {
				framework.Failf("expected namespace name %s, got: %s", ns, namespace.Name)
			}
		})

		ginkgo.It("set should validate inputs", func(ctx context.Context) {
			ginkgo.By("attempting to create alias without --name should fail")
			_, err := e2ekubectl.RunKubectl(ns, "kuberc", "set", fmt.Sprintf("--kuberc=%s", kubercFile), "--section", "aliases", "--command", "get")
			if err == nil {
				framework.Failf("expected command to fail when --name is missing for aliases")
			}

			ginkgo.By("attempting to use invalid section should fail")
			_, err = e2ekubectl.RunKubectl(ns, "kuberc", "set", fmt.Sprintf("--kuberc=%s", kubercFile), "--section", "invalid", "--command", "get")
			if err == nil {
				framework.Failf("expected command to fail with invalid section")
			}
		})
	})
})
