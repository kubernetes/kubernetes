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

package common

import (
	"fmt"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
)

type PrivilegedPodTestConfig struct {
	f *framework.Framework

	privilegedPod          string
	privilegedContainer    string
	notPrivilegedContainer string

	pod *v1.Pod
}

var _ = framework.KubeDescribe("PrivilegedPod", func() {
	config := &PrivilegedPodTestConfig{
		f:                      framework.NewDefaultFramework("e2e-privileged-pod"),
		privilegedPod:          "privileged-pod",
		privilegedContainer:    "privileged-container",
		notPrivilegedContainer: "not-privileged-container",
	}

	It("should enable privileged commands", func() {
		By("Creating a pod with a privileged container")
		config.createPods()

		By("Executing in the privileged container")
		config.run(config.privilegedContainer, true)

		By("Executing in the non-privileged container")
		config.run(config.notPrivilegedContainer, false)
	})
})

func (c *PrivilegedPodTestConfig) run(containerName string, expectSuccess bool) {
	cmd := []string{"ip", "link", "add", "dummy1", "type", "dummy"}
	reverseCmd := []string{"ip", "link", "del", "dummy1"}

	stdout, stderr, err := c.f.ExecCommandInContainerWithFullOutput(
		c.privilegedPod, containerName, cmd...)
	msg := fmt.Sprintf("cmd %v, stdout %q, stderr %q", cmd, stdout, stderr)

	if expectSuccess {
		Expect(err).NotTo(HaveOccurred(), msg)
		// We need to clean up the dummy link that was created, as it
		// leaks out into the node level -- yuck.
		_, _, err := c.f.ExecCommandInContainerWithFullOutput(
			c.privilegedPod, containerName, reverseCmd...)
		Expect(err).NotTo(HaveOccurred(),
			fmt.Sprintf("could not remove dummy1 link: %v", err))
	} else {
		Expect(err).To(HaveOccurred(), msg)
	}
}

func (c *PrivilegedPodTestConfig) createPodsSpec() *v1.Pod {
	isPrivileged := true
	notPrivileged := false

	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      c.privilegedPod,
			Namespace: c.f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            c.privilegedContainer,
					Image:           busyboxImage,
					ImagePullPolicy: v1.PullIfNotPresent,
					SecurityContext: &v1.SecurityContext{Privileged: &isPrivileged},
					Command:         []string{"/bin/sleep", "10000"},
				},
				{
					Name:            c.notPrivilegedContainer,
					Image:           busyboxImage,
					ImagePullPolicy: v1.PullIfNotPresent,
					SecurityContext: &v1.SecurityContext{Privileged: &notPrivileged},
					Command:         []string{"/bin/sleep", "10000"},
				},
			},
		},
	}
}

func (c *PrivilegedPodTestConfig) createPods() {
	podSpec := c.createPodsSpec()
	c.pod = c.f.PodClient().CreateSync(podSpec)
}
