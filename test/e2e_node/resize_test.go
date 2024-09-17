/*
Copyright 2023 The Kubernetes Authors.

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

package e2enode

import (
	"context"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("iholder Resize", func() {
	f := framework.NewDefaultFramework("resize-test")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline

	ginkgo.It("resize test", func() {
		initialResources := v1.ResourceRequirements{
			Requests: map[v1.ResourceName]resource.Quantity{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("100Mi"),
			},
		}

		resourcesAfterResize := v1.ResourceRequirements{
			Requests: map[v1.ResourceName]resource.Quantity{
				v1.ResourceCPU:    resource.MustParse("150m"),
				v1.ResourceMemory: resource.MustParse("150Mi"),
			},
		}

		ginkgo.By("Creating a pod")
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-resize-" + rand.String(5),
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyAlways,
				Containers: []v1.Container{
					{
						Name:      "busybox-container",
						Image:     busyboxImage,
						Command:   []string{"sleep", "600"},
						Resources: initialResources,
					},
				},
			},
		}

		pod = runPodAndWaitUntilScheduled(f, pod)

		resize := &v1.Resize{
			Spec: v1.ResizeOptionsSpec{
				Resize: map[string]v1.ResourceRequirements{
					"busybox-container": resourcesAfterResize,
				},
			},
		}
		//resize, err := f.ClientSet.CoreV1().Pods(pod.Namespace).UpdateResize(context.Background(), pod.Name, resize, metav1.UpdateOptions{})
		resize, err := f.ClientSet.CoreV1().Pods(pod.Namespace).GetResize(context.Background(), pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		gomega.Expect(resize).To(gomega.BeNil())
	})
})
