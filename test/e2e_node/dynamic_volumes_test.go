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
	"encoding/json"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Dynamic Node-Local Ephemeral Volumes", framework.WithFeatureGate(features.DynamicNodeLocalEphemeralVolumes), func() {
	f := framework.NewDefaultFramework("dyn-eph-vols")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	updatePodDynamic := func(ctx context.Context, pod *v1.Pod) (*v1.Pod, error) {
		body, err := json.Marshal(pod)
		if err != nil {
			return nil, err
		}
		result := &v1.Pod{}
		err = f.ClientSet.CoreV1().RESTClient().Put().
			Namespace(f.Namespace.Name).
			Resource("pods").
			Name(pod.Name).
			SubResource("dynamic").
			Body(body).
			Do(ctx).
			Into(result)
		return result, err
	}

	createRunningSleepPod := func(ctx context.Context) *v1.Pod {
		return e2epod.NewPodClient(f).CreateSync(ctx, &v1.Pod{
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
						DynamicVolumeMountPolicy: &v1.ContainerDynamicVolumeMountPolicy{
							RestartPolicy: v1.DynamicVolumeMountRestartContainer,
						},
					},
				},
			},
		})
	}

	ginkgo.It("should dynamically add and remove emptyDir and configMap volumes with RestartContainer policy", func(ctx context.Context) {
		cm := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "dyn-cm-" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Data: map[string]string{
				"key1": "hello-dynamic-volume",
			},
		}
		_, err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(ctx, cm, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		pod := createRunningSleepPod(ctx)

		ginkgo.By("Adding an emptyDir and configMap volume via /dynamic subresource")
		currentPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		updated := currentPod.DeepCopy()
		updated.Spec.Volumes = append(updated.Spec.Volumes,
			v1.Volume{
				Name: "dyn-empty",
				VolumeSource: v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{},
				},
			},
			v1.Volume{
				Name: "dyn-cm",
				VolumeSource: v1.VolumeSource{
					ConfigMap: &v1.ConfigMapVolumeSource{
						LocalObjectReference: v1.LocalObjectReference{Name: cm.Name},
					},
				},
			},
		)
		updated.Spec.Containers[0].VolumeMounts = append(updated.Spec.Containers[0].VolumeMounts,
			v1.VolumeMount{Name: "dyn-empty", MountPath: "/mnt/dyn-empty"},
			v1.VolumeMount{Name: "dyn-cm", MountPath: "/mnt/dyn-cm"},
		)
		_, err = updatePodDynamic(ctx, updated)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for container c1 to restart in-place and mount both volumes with HostMounted phase")
		gomega.Eventually(ctx, func(g gomega.Gomega) {
			p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			g.Expect(err).To(gomega.Succeed())
			g.Expect(p.Status.ContainerStatuses).To(gomega.HaveLen(1))
			g.Expect(p.Status.ContainerStatuses[0].RestartCount).To(gomega.BeNumerically(">=", 1))
			g.Expect(p.Status.ContainerStatuses[0].State.Running).NotTo(gomega.BeNil())
			g.Expect(p.Status.VolumeStatuses).To(gomega.ContainElements(
				v1.PodVolumeStatus{Name: "dyn-empty", HostPhase: v1.PodVolumeHostMounted},
				v1.PodVolumeStatus{Name: "dyn-cm", HostPhase: v1.PodVolumeHostMounted},
			))
		}, 60*time.Second, 2*time.Second).Should(gomega.Succeed())

		out := e2epod.ExecCommandInContainer(f, pod.Name, "c1", "cat", "/mnt/dyn-cm/key1")
		gomega.Expect(out).To(gomega.Equal("hello-dynamic-volume"))
		e2epod.ExecCommandInContainer(f, pod.Name, "c1", "sh", "-c", "echo -n persisted > /mnt/dyn-empty/keep")

		ginkgo.By("Removing dyn-cm volume via /dynamic subresource while retaining dyn-empty")
		currentPod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		updatedAfterRemove := currentPod.DeepCopy()
		var remainingVols []v1.Volume
		for _, v := range updatedAfterRemove.Spec.Volumes {
			if v.Name != "dyn-cm" {
				remainingVols = append(remainingVols, v)
			}
		}
		updatedAfterRemove.Spec.Volumes = remainingVols
		var remainingMounts []v1.VolumeMount
		for _, vm := range updatedAfterRemove.Spec.Containers[0].VolumeMounts {
			if vm.Name != "dyn-cm" {
				remainingMounts = append(remainingMounts, vm)
			}
		}
		updatedAfterRemove.Spec.Containers[0].VolumeMounts = remainingMounts
		_, err = updatePodDynamic(ctx, updatedAfterRemove)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for c1 to restart without /mnt/dyn-cm and for dyn-cm to be pruned from VolumeStatuses")
		gomega.Eventually(ctx, func(g gomega.Gomega) {
			p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			g.Expect(err).To(gomega.Succeed())
			g.Expect(p.Status.ContainerStatuses).To(gomega.HaveLen(1))
			g.Expect(p.Status.ContainerStatuses[0].RestartCount).To(gomega.BeNumerically(">=", 2))
			g.Expect(p.Status.ContainerStatuses[0].State.Running).NotTo(gomega.BeNil())
			for _, vs := range p.Status.VolumeStatuses {
				g.Expect(vs.Name).NotTo(gomega.Equal("dyn-cm"))
			}
			g.Expect(p.Status.VolumeStatuses).To(gomega.ContainElement(
				v1.PodVolumeStatus{Name: "dyn-empty", HostPhase: v1.PodVolumeHostMounted},
			))
		}, 60*time.Second, 2*time.Second).Should(gomega.Succeed())

		checkOut := e2epod.ExecCommandInContainer(f, pod.Name, "c1", "sh", "-c", "test ! -e /mnt/dyn-cm && cat /mnt/dyn-empty/keep")
		gomega.Expect(checkOut).To(gomega.Equal("persisted"))
	})

	ginkgo.It("should defer container restart while referenced ConfigMap is missing and actuate once created", func(ctx context.Context) {
		missingCMName := "deferred-cm-" + string(uuid.NewUUID())
		pod := createRunningSleepPod(ctx)

		currentPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		updated := currentPod.DeepCopy()
		updated.Spec.Volumes = append(updated.Spec.Volumes, v1.Volume{
			Name: "deferred-cm",
			VolumeSource: v1.VolumeSource{
				ConfigMap: &v1.ConfigMapVolumeSource{
					LocalObjectReference: v1.LocalObjectReference{Name: missingCMName},
				},
			},
		})
		updated.Spec.Containers[0].VolumeMounts = append(updated.Spec.Containers[0].VolumeMounts,
			v1.VolumeMount{Name: "deferred-cm", MountPath: "/mnt/deferred-cm"},
		)
		_, err = updatePodDynamic(ctx, updated)
		framework.ExpectNoError(err)

		ginkgo.By("Verifying container c1 remains running without restarting while ConfigMap does not exist")
		gomega.Consistently(ctx, func(g gomega.Gomega) {
			p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			g.Expect(err).To(gomega.Succeed())
			g.Expect(p.Status.ContainerStatuses).To(gomega.HaveLen(1))
			g.Expect(p.Status.ContainerStatuses[0].RestartCount).To(gomega.Equal(int32(0)))
			g.Expect(p.Status.ContainerStatuses[0].State.Running).NotTo(gomega.BeNil())
			g.Expect(p.Status.VolumeStatuses).To(gomega.ContainElement(
				v1.PodVolumeStatus{Name: "deferred-cm", HostPhase: v1.PodVolumeHostPending},
			))
		}, 6*time.Second, time.Second).Should(gomega.Succeed())

		ginkgo.By("Creating the missing ConfigMap and verifying c1 restarts and mounts it")
		_, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(ctx, &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      missingCMName,
				Namespace: f.Namespace.Name,
			},
			Data: map[string]string{"ready": "yes"},
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		gomega.Eventually(ctx, func(g gomega.Gomega) {
			p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			g.Expect(err).To(gomega.Succeed())
			g.Expect(p.Status.ContainerStatuses).To(gomega.HaveLen(1))
			g.Expect(p.Status.ContainerStatuses[0].RestartCount).To(gomega.BeNumerically(">=", 1))
			g.Expect(p.Status.ContainerStatuses[0].State.Running).NotTo(gomega.BeNil())
			g.Expect(p.Status.VolumeStatuses).To(gomega.ContainElement(
				v1.PodVolumeStatus{Name: "deferred-cm", HostPhase: v1.PodVolumeHostMounted},
			))
		}, 60*time.Second, 2*time.Second).Should(gomega.Succeed())

		out := e2epod.ExecCommandInContainer(f, pod.Name, "c1", "cat", "/mnt/deferred-cm/ready")
		gomega.Expect(out).To(gomega.Equal("yes"))
	})

	ginkgo.It("should dynamically mount secret and projected volumes", func(ctx context.Context) {
		secName := "dyn-sec-" + string(uuid.NewUUID())
		_, err := f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(ctx, &v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      secName,
				Namespace: f.Namespace.Name,
			},
			StringData: map[string]string{"token": "s3cr3t"},
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		pod := createRunningSleepPod(ctx)
		currentPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		updated := currentPod.DeepCopy()
		updated.Spec.Volumes = append(updated.Spec.Volumes,
			v1.Volume{
				Name: "dyn-sec",
				VolumeSource: v1.VolumeSource{
					Secret: &v1.SecretVolumeSource{SecretName: secName},
				},
			},
			v1.Volume{
				Name: "dyn-proj",
				VolumeSource: v1.VolumeSource{
					Projected: &v1.ProjectedVolumeSource{
						Sources: []v1.VolumeProjection{
							{
								Secret: &v1.SecretProjection{
									LocalObjectReference: v1.LocalObjectReference{Name: secName},
								},
							},
							{
								DownwardAPI: &v1.DownwardAPIProjection{
									Items: []v1.DownwardAPIVolumeFile{
										{
											Path:     "podname",
											FieldRef: &v1.ObjectFieldSelector{FieldPath: "metadata.name"},
										},
									},
								},
							},
						},
					},
				},
			},
		)
		updated.Spec.Containers[0].VolumeMounts = append(updated.Spec.Containers[0].VolumeMounts,
			v1.VolumeMount{Name: "dyn-sec", MountPath: "/mnt/dyn-sec"},
			v1.VolumeMount{Name: "dyn-proj", MountPath: "/mnt/dyn-proj"},
		)
		_, err = updatePodDynamic(ctx, updated)
		framework.ExpectNoError(err)

		gomega.Eventually(ctx, func(g gomega.Gomega) {
			p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			g.Expect(err).To(gomega.Succeed())
			g.Expect(p.Status.ContainerStatuses).To(gomega.HaveLen(1))
			g.Expect(p.Status.ContainerStatuses[0].RestartCount).To(gomega.BeNumerically(">=", 1))
			g.Expect(p.Status.ContainerStatuses[0].State.Running).NotTo(gomega.BeNil())
			g.Expect(p.Status.VolumeStatuses).To(gomega.ContainElements(
				v1.PodVolumeStatus{Name: "dyn-sec", HostPhase: v1.PodVolumeHostMounted},
				v1.PodVolumeStatus{Name: "dyn-proj", HostPhase: v1.PodVolumeHostMounted},
			))
		}, 60*time.Second, 2*time.Second).Should(gomega.Succeed())

		gomega.Expect(e2epod.ExecCommandInContainer(f, pod.Name, "c1", "cat", "/mnt/dyn-sec/token")).To(gomega.Equal("s3cr3t"))
		gomega.Expect(e2epod.ExecCommandInContainer(f, pod.Name, "c1", "cat", "/mnt/dyn-proj/podname")).To(gomega.Equal(pod.Name))
		gomega.Expect(e2epod.ExecCommandInContainer(f, pod.Name, "c1", "cat", "/mnt/dyn-proj/token")).To(gomega.Equal("s3cr3t"))
	})

	ginkgo.Context("with OCI Image volumes", feature.ImageVolume, func() {
		ginkgo.It("should keep running container healthy when dynamic Image pull fails and restart once valid Image is pulled", func(ctx context.Context) {
			pod := createRunningSleepPod(ctx)
			currentPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Adding an invalid OCI Image volume and verifying c1 stays Running with HostPhase=Pending")
			badUpdate := currentPod.DeepCopy()
			badUpdate.Spec.Volumes = append(badUpdate.Spec.Volumes, v1.Volume{
				Name: "bad-img",
				VolumeSource: v1.VolumeSource{
					Image: &v1.ImageVolumeSource{
						Reference:  "localhost/invalid:nonexistent",
						PullPolicy: v1.PullAlways,
					},
				},
			})
			badUpdate.Spec.Containers[0].VolumeMounts = append(badUpdate.Spec.Containers[0].VolumeMounts,
				v1.VolumeMount{Name: "bad-img", MountPath: "/mnt/bad-img"},
			)
			_, err = updatePodDynamic(ctx, badUpdate)
			framework.ExpectNoError(err)

			gomega.Consistently(ctx, func(g gomega.Gomega) {
				p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
				g.Expect(err).To(gomega.Succeed())
				g.Expect(p.Status.ContainerStatuses).To(gomega.HaveLen(1))
				g.Expect(p.Status.ContainerStatuses[0].RestartCount).To(gomega.Equal(int32(0)))
				g.Expect(p.Status.ContainerStatuses[0].State.Running).NotTo(gomega.BeNil())
				g.Expect(p.Status.VolumeStatuses).To(gomega.ContainElement(
					v1.PodVolumeStatus{Name: "bad-img", HostPhase: v1.PodVolumeHostPending},
				))
			}, 8*time.Second, time.Second).Should(gomega.Succeed())

			ginkgo.By("Replacing bad-img with a valid OCI Image volume and verifying pre-pull and in-place restart")
			currentPod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			goodUpdate := currentPod.DeepCopy()
			var keptVols []v1.Volume
			for _, v := range goodUpdate.Spec.Volumes {
				if v.Name != "bad-img" {
					keptVols = append(keptVols, v)
				}
			}
			keptVols = append(keptVols, v1.Volume{
				Name: "good-img",
				VolumeSource: v1.VolumeSource{
					Image: &v1.ImageVolumeSource{
						Reference:  imageutils.GetE2EImage(imageutils.Kitten),
						PullPolicy: v1.PullIfNotPresent,
					},
				},
			})
			goodUpdate.Spec.Volumes = keptVols
			var keptMounts []v1.VolumeMount
			for _, vm := range goodUpdate.Spec.Containers[0].VolumeMounts {
				if vm.Name != "bad-img" {
					keptMounts = append(keptMounts, vm)
				}
			}
			keptMounts = append(keptMounts, v1.VolumeMount{Name: "good-img", MountPath: "/mnt/good-img"})
			goodUpdate.Spec.Containers[0].VolumeMounts = keptMounts
			_, err = updatePodDynamic(ctx, goodUpdate)
			framework.ExpectNoError(err)

			gomega.Eventually(ctx, func(g gomega.Gomega) {
				p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
				g.Expect(err).To(gomega.Succeed())
				g.Expect(p.Status.ContainerStatuses).To(gomega.HaveLen(1))
				g.Expect(p.Status.ContainerStatuses[0].RestartCount).To(gomega.BeNumerically(">=", 1))
				g.Expect(p.Status.ContainerStatuses[0].State.Running).NotTo(gomega.BeNil())
				g.Expect(p.Status.VolumeStatuses).To(gomega.ContainElement(
					v1.PodVolumeStatus{Name: "good-img", HostPhase: v1.PodVolumeHostMounted},
				))
			}, 90*time.Second, 2*time.Second).Should(gomega.Succeed())
		})
	})

	ginkgo.Context("across kubelet restart", framework.WithSerial(), framework.WithDisruptive(), func() {
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

			pod := createRunningSleepPod(ctx)
			currentPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			updated := currentPod.DeepCopy()
			updated.Spec.Volumes = append(updated.Spec.Volumes,
				v1.Volume{Name: "dyn-empty", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}},
				v1.Volume{Name: "dyn-cm", VolumeSource: v1.VolumeSource{ConfigMap: &v1.ConfigMapVolumeSource{LocalObjectReference: v1.LocalObjectReference{Name: cm.Name}}}},
			)
			updated.Spec.Containers[0].VolumeMounts = append(updated.Spec.Containers[0].VolumeMounts,
				v1.VolumeMount{Name: "dyn-empty", MountPath: "/mnt/dyn-empty"},
				v1.VolumeMount{Name: "dyn-cm", MountPath: "/mnt/dyn-cm"},
			)
			_, err = updatePodDynamic(ctx, updated)
			framework.ExpectNoError(err)

			gomega.Eventually(ctx, func(g gomega.Gomega) {
				p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
				g.Expect(err).To(gomega.Succeed())
				g.Expect(p.Status.ContainerStatuses).To(gomega.HaveLen(1))
				g.Expect(p.Status.ContainerStatuses[0].RestartCount).To(gomega.Equal(int32(1)))
				g.Expect(p.Status.VolumeStatuses).To(gomega.ContainElements(
					v1.PodVolumeStatus{Name: "dyn-empty", HostPhase: v1.PodVolumeHostMounted},
					v1.PodVolumeStatus{Name: "dyn-cm", HostPhase: v1.PodVolumeHostMounted},
				))
			}, 60*time.Second, 2*time.Second).Should(gomega.Succeed())

			e2epod.ExecCommandInContainer(f, pod.Name, "c1", "sh", "-c", "echo -n durable > /mnt/dyn-empty/checkpoint")

			ginkgo.By("Stopping and restarting kubelet")
			restartKubelet := mustStopKubelet(ctx, f)
			restartKubelet(ctx)

			ginkgo.By("Verifying c1 does not restart spuriously and VolumeStatuses remain HostMounted")
			gomega.Consistently(ctx, func(g gomega.Gomega) {
				p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
				g.Expect(err).To(gomega.Succeed())
				g.Expect(p.Status.ContainerStatuses).To(gomega.HaveLen(1))
				g.Expect(p.Status.ContainerStatuses[0].RestartCount).To(gomega.Equal(int32(1)))
				g.Expect(p.Status.VolumeStatuses).To(gomega.ContainElements(
					v1.PodVolumeStatus{Name: "dyn-empty", HostPhase: v1.PodVolumeHostMounted},
					v1.PodVolumeStatus{Name: "dyn-cm", HostPhase: v1.PodVolumeHostMounted},
				))
			}, 8*time.Second, 2*time.Second).Should(gomega.Succeed())

			gomega.Expect(e2epod.ExecCommandInContainer(f, pod.Name, "c1", "cat", "/mnt/dyn-empty/checkpoint")).To(gomega.Equal("durable"))
		})
	})
})
