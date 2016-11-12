/*
Copyright 2017 The Kubernetes Authors.

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

package extension

import (
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = framework.KubeDescribe("Initializers", func() {
	f := framework.NewDefaultFramework("initializers")

	// TODO: Add failure traps once we have JustAfterEach
	// See https://github.com/onsi/ginkgo/issues/303

	It("should be invisible to controllers by default", func() {
		ns := f.Namespace.Name
		c := f.ClientSet

		podName := "uninitialized-pod"
		framework.Logf("Creating pod %s", podName)

		ch := make(chan struct{})
		go func() {
			_, err := c.Core().Pods(ns).Create(newUninitializedPod(podName))
			Expect(err).NotTo(HaveOccurred())
			close(ch)
		}()

		// wait to ensure the scheduler does not act on an uninitialized pod
		err := wait.PollImmediate(2*time.Second, 15*time.Second, func() (bool, error) {
			p, err := c.Core().Pods(ns).Get(podName, metav1.GetOptions{})
			if err != nil {
				if errors.IsNotFound(err) {
					return false, nil
				}
				return false, err
			}
			return len(p.Spec.NodeName) > 0, nil
		})
		Expect(err).To(Equal(wait.ErrWaitTimeout))

		// verify that we can update an initializing pod
		pod, err := c.Core().Pods(ns).Get(podName, metav1.GetOptions{})
		pod.Annotations = map[string]string{"update-1": "test"}
		pod, err = c.Core().Pods(ns).Update(pod)
		Expect(err).NotTo(HaveOccurred())

		// clear initializers
		pod.Initializers = nil
		pod, err = c.Core().Pods(ns).Update(pod)
		Expect(err).NotTo(HaveOccurred())

		// pod should now start running
		err = framework.WaitForPodRunningInNamespace(c, pod)
		Expect(err).NotTo(HaveOccurred())

		// ensure create call returns
		<-ch

		// verify that we cannot start the pod initializing again
		pod, err = c.Core().Pods(ns).Get(podName, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		pod.Initializers = &metav1.Initializers{
			Pending: []metav1.Initializer{{Name: "Other"}},
		}
		_, err = c.Core().Pods(ns).Update(pod)
		if !errors.IsInvalid(err) || !strings.Contains(err.Error(), "immutable") {
			Fail(fmt.Sprintf("expected invalid error: %v", err))
		}
	})

})

func newUninitializedPod(podName string) *v1.Pod {
	containerName := fmt.Sprintf("%s-container", podName)
	port := 8080
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
			Initializers: &metav1.Initializers{
				Pending: []metav1.Initializer{{Name: "Test"}},
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  containerName,
					Image: "gcr.io/google_containers/porter:4524579c0eb935c056c8e75563b4e1eda31587e0",
					Env:   []v1.EnvVar{{Name: fmt.Sprintf("SERVE_PORT_%d", port), Value: "foo"}},
					Ports: []v1.ContainerPort{{ContainerPort: int32(port)}},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
	return pod
}
