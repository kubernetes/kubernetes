/*
Copyright The Kubernetes Authors.

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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Dynamic Node-Local Ephemeral Volumes", framework.WithSerial(), framework.WithDisruptive(), framework.WithFeatureGate(features.DynamicNodeLocalEphemeralVolumes), framework.WithFeatureGate(features.DynamicContainers), func() {
	f := framework.NewDefaultFramework("dyn-eph-vols-restart")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.Context("across kubelet restart", func() {
		ginkgo.It("should preserve actuated dynamic volume state across kubelet restart without spurious container restarts", func(ctx context.Context) {
			cm := &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "dyn-restart-cm-" + string(uuid.NewUUID()),
					Namespace: f.Namespace.Name,
				},
				Data: map[string]string{"k": "v"},
			}
			_, err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(ctx, cm, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			pod := e2epod.NewPodClient(f).CreateSync(ctx, &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "dyn-vol-pod-" + string(uuid.NewUUID()),
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    "c1",
							Image:   busyboxImage,
							Command: []string{"sh", "-c", "sleep 3600"},
						},
					},
				},
			})

			err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
				currentPod, getErr := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
				if getErr != nil {
					return getErr
				}
				currentPod.Spec.Volumes = append(currentPod.Spec.Volumes,
					v1.Volume{Name: "dyn-empty", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}},
					v1.Volume{Name: "dyn-cm", VolumeSource: v1.VolumeSource{ConfigMap: &v1.ConfigMapVolumeSource{LocalObjectReference: v1.LocalObjectReference{Name: cm.Name}}}},
				)
				currentPod.Spec.Containers = append(currentPod.Spec.Containers, v1.Container{
					Name:    "c2",
					Image:   busyboxImage,
					Command: []string{"sh", "-c", "sleep 3600"},
					VolumeMounts: []v1.VolumeMount{
						{Name: "dyn-empty", MountPath: "/mnt/dyn-empty"},
						{Name: "dyn-cm", MountPath: "/mnt/dyn-cm"},
					},
				})
				_, updateErr := f.ClientSet.CoreV1().Pods(f.Namespace.Name).UpdateDynamic(ctx, currentPod.Name, currentPod, metav1.UpdateOptions{})
				return updateErr
			})
			framework.ExpectNoError(err)

			gomega.Eventually(ctx, func(g gomega.Gomega) {
				p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
				g.Expect(err).To(gomega.Succeed())
				g.Expect(p.Status.ContainerStatuses).To(gomega.HaveLen(2))
				for _, cs := range p.Status.ContainerStatuses {
					g.Expect(cs.RestartCount).To(gomega.Equal(int32(0)))
					g.Expect(cs.State.Running).NotTo(gomega.BeNil())
				}
				g.Expect(p.Status.VolumeStatuses).To(gomega.ContainElements(
					v1.PodVolumeStatus{Name: "dyn-empty", HostPhase: v1.PodVolumeHostMounted},
					v1.PodVolumeStatus{Name: "dyn-cm", HostPhase: v1.PodVolumeHostMounted},
				))
			}, 60*time.Second, 2*time.Second).Should(gomega.Succeed())

			e2epod.ExecCommandInContainer(f, pod.Name, "c2", "sh", "-c", "echo -n durable > /mnt/dyn-empty/checkpoint")

			ginkgo.By("Stopping and restarting kubelet")
			restartKubelet := mustStopKubelet(ctx, f)
			restartKubelet(ctx)

			ginkgo.By("Verifying c1 and c2 do not restart spuriously and VolumeStatuses remain HostMounted")
			gomega.Consistently(ctx, func(g gomega.Gomega) {
				p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
				g.Expect(err).To(gomega.Succeed())
				g.Expect(p.Status.ContainerStatuses).To(gomega.HaveLen(2))
				for _, cs := range p.Status.ContainerStatuses {
					g.Expect(cs.RestartCount).To(gomega.Equal(int32(0)))
					g.Expect(cs.State.Running).NotTo(gomega.BeNil())
				}
				g.Expect(p.Status.VolumeStatuses).To(gomega.ContainElements(
					v1.PodVolumeStatus{Name: "dyn-empty", HostPhase: v1.PodVolumeHostMounted},
					v1.PodVolumeStatus{Name: "dyn-cm", HostPhase: v1.PodVolumeHostMounted},
				))
			}, 8*time.Second, 2*time.Second).Should(gomega.Succeed())

			gomega.Expect(e2epod.ExecCommandInContainer(f, pod.Name, "c2", "cat", "/mnt/dyn-empty/checkpoint")).To(gomega.Equal("durable"))
		})
	})
})
