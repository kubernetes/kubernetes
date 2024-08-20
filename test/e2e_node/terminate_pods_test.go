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
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	admissionapi "k8s.io/pod-security-admission/api"

	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
)

var _ = SIGDescribe("Terminate Pods", func() {
	f := framework.NewDefaultFramework("terminate-pods")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.It("should not hang when terminating pods mounting non-existent volumes", func(ctx context.Context) {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod1",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "container1",
						Image: busyboxImage,
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "vol1",
								MountPath: "/mnt/vol1",
							},
						},
					},
				},
				Volumes: []v1.Volume{
					{
						Name: "vol1",
						VolumeSource: v1.VolumeSource{
							Secret: &v1.SecretVolumeSource{
								SecretName: "non-existent-" + string(uuid.NewUUID()),
							},
						},
					},
				},
			},
		}
		client := e2epod.NewPodClient(f)
		pod = client.Create(context.TODO(), pod)
		gomega.Expect(pod.Spec.NodeName).ToNot(gomega.BeEmpty())

		gomega.Eventually(ctx, func() bool {
			pod, _ = client.Get(context.TODO(), pod.Name, metav1.GetOptions{})
			for _, c := range pod.Status.Conditions {
				if c.Type == v1.ContainersReady && c.Status == v1.ConditionFalse {
					return true
				}
			}
			return false
		}, 20*time.Second, 1*time.Second).Should(gomega.BeTrueBecause("expected container to be ready"))

		err := client.Delete(context.Background(), pod.Name, metav1.DeleteOptions{})

		// Wait for the pod to disappear from the API server up to 10 seconds, this shouldn't hang for minutes due to
		// non-existent secret being mounted.
		gomega.Eventually(ctx, func() bool {
			_, err := client.Get(context.TODO(), pod.Name, metav1.GetOptions{})
			return apierrors.IsNotFound(err)
		}, 10*time.Second, time.Second).Should(gomega.BeTrueBecause("expected pod to disappear from API server within 10 seconds"))

		framework.ExpectNoError(err)
	})
})
