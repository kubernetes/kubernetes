/*
Copyright 2016 The Kubernetes Authors.

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

package e2e_node

import (
	"regexp"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// TODO: consolidate with security context test.
func scTestPod() *api.Pod {
	podName := "seccomp-" + string(uuid.NewUUID())
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:        podName,
			Labels:      map[string]string{"name": podName},
			Annotations: map[string]string{},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "test-container",
					Image: "gcr.io/google_containers/busybox:1.24",
				},
			},
			RestartPolicy: api.RestartPolicyNever,
		},
	}

	return pod
}

var _ = framework.KubeDescribe("Seccomp [Feature:Seccomp]", func() {
	f := framework.NewDefaultFramework("seccomp")
	// TODO: port to SecurityContext as soon as seccomp is out of alpha
	BeforeEach(func() {
		skipIfSeccompIsNotSupported(f)
	})

	It("should support seccomp alpha unconfined annotation on the container", func() {
		pod := scTestPod()
		pod.Annotations[api.SeccompContainerAnnotationKeyPrefix+"test-container"] = "unconfined"
		pod.Annotations[api.SeccompPodAnnotationKey] = "docker/default"
		pod.Spec.Containers[0].Command = []string{"grep", "ecc", "/proc/self/status"}
		f.TestContainerOutput(api.SeccompPodAnnotationKey, pod, 0, []string{"0"}) // seccomp disabled
	})

	It("should support seccomp alpha unconfined annotation on the pod", func() {
		pod := scTestPod()
		pod.Annotations[api.SeccompPodAnnotationKey] = "unconfined"
		pod.Spec.Containers[0].Command = []string{"grep", "ecc", "/proc/self/status"}
		f.TestContainerOutput(api.SeccompPodAnnotationKey, pod, 0, []string{"0"}) // seccomp disabled
	})

	It("should support seccomp alpha docker/default annotation", func() {
		pod := scTestPod()
		pod.Annotations[api.SeccompContainerAnnotationKeyPrefix+"test-container"] = "docker/default"
		pod.Spec.Containers[0].Command = []string{"grep", "ecc", "/proc/self/status"}
		f.TestContainerOutput(api.SeccompPodAnnotationKey, pod, 0, []string{"2"}) // seccomp filtered
	})

	It("should support seccomp default which is unconfined", func() {
		pod := scTestPod()
		pod.Spec.Containers[0].Command = []string{"grep", "ecc", "/proc/self/status"}
		f.TestContainerOutput(api.SeccompPodAnnotationKey, pod, 0, []string{"0"}) // seccomp disabled
	})
})

// TODO(random-liu): Consider whether we should also skip based on kernel configuration or enforce corresponding
// kernel configurations in system verification.
func skipIfSeccompIsNotSupported(f *framework.Framework) {
	nodeName := framework.TestContext.NodeName
	node, err := f.Client.Nodes().Get(nodeName)
	Expect(err).NotTo(HaveOccurred(), "get status of node %q", node)

	const dockerVersionPrefix = "docker://"
	const os = "Google Container-VM Image"                    // Only run seccomp test on gci
	dockerVersionRegex := regexp.MustCompile(`1\.\d{2,}\..*`) // >= docker 1.10
	if node.Status.NodeInfo.OSImage != os {
		framework.Skipf("seccomp test should not run on os %q", node.Status.NodeInfo.OSImage)
	}
	runtimeVersion := node.Status.NodeInfo.ContainerRuntimeVersion
	if !strings.HasPrefix(runtimeVersion, dockerVersionPrefix) {
		framework.Skipf("seccomp is not supported by runtime: %q", runtimeVersion)
	}
	if !dockerVersionRegex.MatchString(strings.Trim(runtimeVersion, dockerVersionPrefix)) {
		framework.Skipf("seccomp is not supported by docker version: %q", runtimeVersion)
	}
}
