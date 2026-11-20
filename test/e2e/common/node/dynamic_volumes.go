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

package node

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
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Dynamic Node-Local Ephemeral Volumes", framework.WithFeatureGate(features.DynamicNodeLocalEphemeralVolumes), framework.WithFeatureGate(features.DynamicContainers), func() {
	f := framework.NewDefaultFramework("dyn-eph-vols")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.BeforeEach(func() {
		if framework.NodeOSDistroIs("windows") {
			e2eskipper.Skipf("dynamic node-local ephemeral volumes are not supported on windows")
		}
	})

	updatePodDynamic := func(ctx context.Context, podName string, mutate func(*v1.Pod)) {
		pods := f.ClientSet.CoreV1().Pods(f.Namespace.Name)
		err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
			p, err := pods.Get(ctx, podName, metav1.GetOptions{})
			if err != nil {
				return err
			}
			mutate(p)
			_, err = pods.UpdateDynamic(ctx, podName, p, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "failed to update pod %q through pods/dynamic", podName)
	}

	createRunningSleepPod := func(ctx context.Context) *v1.Pod {
		return e2epod.NewPodClient(f).CreateSync(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "dyn-vol-pod-" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				TerminationGracePeriodSeconds: new(int64(1)),
				Containers: []v1.Container{
					{
						Name:    "c1",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "sleep 3600"},
					},
				},
			},
		})
	}

	findContainerStatus := func(statuses []v1.ContainerStatus, name string) *v1.ContainerStatus {
		for i := range statuses {
			if statuses[i].Name == name {
				return &statuses[i]
			}
		}
		return nil
	}

	ginkgo.It("should dynamically add and remove emptyDir and configMap volumes with dynamic containers", func(ctx context.Context) {
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

		ginkgo.By("Adding emptyDir and configMap volumes with dynamic containers via /dynamic")
		updatePodDynamic(ctx, pod.Name, func(updated *v1.Pod) {
			updated.Spec.Volumes = append(updated.Spec.Volumes,
				v1.Volume{
					Name:         "dyn-empty",
					VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}},
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
			updated.Spec.Containers = append(updated.Spec.Containers,
				v1.Container{
					Name:         "c-keep",
					Image:        imageutils.GetE2EImage(imageutils.BusyBox),
					Command:      []string{"sh", "-c", "sleep 3600"},
					VolumeMounts: []v1.VolumeMount{{Name: "dyn-empty", MountPath: "/mnt/dyn-empty"}},
				},
				v1.Container{
					Name:    "c2",
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"sh", "-c", "sleep 3600"},
					VolumeMounts: []v1.VolumeMount{
						{Name: "dyn-empty", MountPath: "/mnt/dyn-empty"},
						{Name: "dyn-cm", MountPath: "/mnt/dyn-cm"},
					},
				},
			)
		})

		ginkgo.By("Waiting for c2 and c-keep to start with HostMounted phase while c1 stays untouched")
		gomega.Eventually(ctx, func(g gomega.Gomega) {
			p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			g.Expect(err).To(gomega.Succeed())
			c1 := findContainerStatus(p.Status.ContainerStatuses, "c1")
			c2 := findContainerStatus(p.Status.ContainerStatuses, "c2")
			cKeep := findContainerStatus(p.Status.ContainerStatuses, "c-keep")
			g.Expect(c1).NotTo(gomega.BeNil())
			g.Expect(c1.RestartCount).To(gomega.Equal(int32(0)))
			g.Expect(c2).NotTo(gomega.BeNil())
			g.Expect(c2.State.Running).NotTo(gomega.BeNil())
			g.Expect(cKeep).NotTo(gomega.BeNil())
			g.Expect(cKeep.State.Running).NotTo(gomega.BeNil())
			g.Expect(p.Status.VolumeStatuses).To(gomega.ContainElements(
				v1.PodVolumeStatus{Name: "dyn-empty", HostPhase: v1.PodVolumeHostMounted},
				v1.PodVolumeStatus{Name: "dyn-cm", HostPhase: v1.PodVolumeHostMounted},
			))
		}, 60*time.Second, 2*time.Second).Should(gomega.Succeed())

		out := e2epod.ExecCommandInContainer(f, pod.Name, "c2", "cat", "/mnt/dyn-cm/key1")
		gomega.Expect(out).To(gomega.Equal("hello-dynamic-volume"))
		e2epod.ExecCommandInContainer(f, pod.Name, "c2", "sh", "-c", "echo -n persisted > /mnt/dyn-empty/keep")

		ginkgo.By("Jointly removing c2 and dyn-cm via /dynamic while retaining dyn-empty on c-keep")
		updatePodDynamic(ctx, pod.Name, func(updatedAfterRemove *v1.Pod) {
			var remainingVols []v1.Volume
			for _, v := range updatedAfterRemove.Spec.Volumes {
				if v.Name != "dyn-cm" {
					remainingVols = append(remainingVols, v)
				}
			}
			updatedAfterRemove.Spec.Volumes = remainingVols
			var remainingCtrs []v1.Container
			for _, c := range updatedAfterRemove.Spec.Containers {
				if c.Name != "c2" {
					remainingCtrs = append(remainingCtrs, c)
				}
			}
			updatedAfterRemove.Spec.Containers = remainingCtrs
		})

		ginkgo.By("Waiting for c2 to terminate with cleared volumeMounts and for dyn-cm to be pruned from VolumeStatuses")
		gomega.Eventually(ctx, func(g gomega.Gomega) {
			p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			g.Expect(err).To(gomega.Succeed())
			c2 := findContainerStatus(p.Status.ContainerStatuses, "c2")
			g.Expect(c2).NotTo(gomega.BeNil())
			g.Expect(c2.State.Terminated).NotTo(gomega.BeNil())
			g.Expect(c2.VolumeMounts).To(gomega.BeEmpty())
			for _, vs := range p.Status.VolumeStatuses {
				g.Expect(vs.Name).NotTo(gomega.Equal("dyn-cm"))
			}
			g.Expect(p.Status.VolumeStatuses).To(gomega.ContainElement(
				v1.PodVolumeStatus{Name: "dyn-empty", HostPhase: v1.PodVolumeHostMounted},
			))
		}, 60*time.Second, 2*time.Second).Should(gomega.Succeed())

		checkOut := e2epod.ExecCommandInContainer(f, pod.Name, "c-keep", "cat", "/mnt/dyn-empty/keep")
		gomega.Expect(checkOut).To(gomega.Equal("persisted"))
	})

	ginkgo.It("should defer dynamic and ephemeral container starts while referenced ConfigMap is missing and start them once created", func(ctx context.Context) {
		missingCMName := "deferred-cm-" + string(uuid.NewUUID())
		pod := createRunningSleepPod(ctx)

		updatePodDynamic(ctx, pod.Name, func(updated *v1.Pod) {
			updated.Spec.Volumes = append(updated.Spec.Volumes, v1.Volume{
				Name: "deferred-cm",
				VolumeSource: v1.VolumeSource{
					ConfigMap: &v1.ConfigMapVolumeSource{
						LocalObjectReference: v1.LocalObjectReference{Name: missingCMName},
					},
				},
			})
			updated.Spec.Containers = append(updated.Spec.Containers, v1.Container{
				Name:         "c-deferred",
				Image:        imageutils.GetE2EImage(imageutils.BusyBox),
				Command:      []string{"sh", "-c", "sleep 3600"},
				VolumeMounts: []v1.VolumeMount{{Name: "deferred-cm", MountPath: "/mnt/deferred-cm"}},
			})
		})

		err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
			p, getErr := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			if getErr != nil {
				return getErr
			}
			p.Spec.EphemeralContainers = append(p.Spec.EphemeralContainers, v1.EphemeralContainer{
				EphemeralContainerCommon: v1.EphemeralContainerCommon{
					Name:         "dbg-deferred",
					Image:        imageutils.GetE2EImage(imageutils.BusyBox),
					Command:      []string{"sh", "-c", "sleep 3600"},
					VolumeMounts: []v1.VolumeMount{{Name: "deferred-cm", MountPath: "/mnt/deferred-cm"}},
				},
			})
			_, updErr := f.ClientSet.CoreV1().Pods(f.Namespace.Name).UpdateEphemeralContainers(ctx, pod.Name, p, metav1.UpdateOptions{})
			return updErr
		})
		framework.ExpectNoError(err)

		ginkgo.By("Verifying c1 remains running and new containers wait without CreateContainerConfigError while ConfigMap is missing")
		gomega.Eventually(ctx, func(g gomega.Gomega) {
			p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			g.Expect(err).To(gomega.Succeed())
			g.Expect(p.Status.VolumeStatuses).To(gomega.ContainElement(
				v1.PodVolumeStatus{Name: "deferred-cm", HostPhase: v1.PodVolumeHostPending},
			))
		}, 15*time.Second, 500*time.Millisecond).Should(gomega.Succeed())
		gomega.Consistently(ctx, func(g gomega.Gomega) {
			p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			g.Expect(err).To(gomega.Succeed())
			c1 := findContainerStatus(p.Status.ContainerStatuses, "c1")
			g.Expect(c1).NotTo(gomega.BeNil())
			g.Expect(c1.RestartCount).To(gomega.Equal(int32(0)))
			g.Expect(c1.State.Running).NotTo(gomega.BeNil())
			cDef := findContainerStatus(p.Status.ContainerStatuses, "c-deferred")
			if cDef != nil && cDef.State.Waiting != nil {
				g.Expect(cDef.State.Waiting.Reason).NotTo(gomega.Equal("CreateContainerConfigError"))
			}
			g.Expect(p.Status.VolumeStatuses).To(gomega.ContainElement(
				v1.PodVolumeStatus{Name: "deferred-cm", HostPhase: v1.PodVolumeHostPending},
			))
		}, 6*time.Second, time.Second).Should(gomega.Succeed())

		ginkgo.By("Creating the missing ConfigMap and verifying both c-deferred and dbg-deferred start")
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
			cDef := findContainerStatus(p.Status.ContainerStatuses, "c-deferred")
			g.Expect(cDef).NotTo(gomega.BeNil())
			g.Expect(cDef.State.Running).NotTo(gomega.BeNil())
			dbgDef := findContainerStatus(p.Status.EphemeralContainerStatuses, "dbg-deferred")
			g.Expect(dbgDef).NotTo(gomega.BeNil())
			g.Expect(dbgDef.State.Running).NotTo(gomega.BeNil())
			g.Expect(p.Status.VolumeStatuses).To(gomega.ContainElement(
				v1.PodVolumeStatus{Name: "deferred-cm", HostPhase: v1.PodVolumeHostMounted},
			))
		}, 60*time.Second, 2*time.Second).Should(gomega.Succeed())

		gomega.Expect(e2epod.ExecCommandInContainer(f, pod.Name, "c-deferred", "cat", "/mnt/deferred-cm/ready")).To(gomega.Equal("yes"))
		gomega.Expect(e2epod.ExecCommandInContainer(f, pod.Name, "dbg-deferred", "cat", "/mnt/deferred-cm/ready")).To(gomega.Equal("yes"))
	})

	ginkgo.It("should dynamically mount and tear down secret, subPath, and projected volumes", func(ctx context.Context) {
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

		updatePodDynamic(ctx, pod.Name, func(updated *v1.Pod) {
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
			updated.Spec.Containers = append(updated.Spec.Containers, v1.Container{
				Name:    "c-sec",
				Image:   imageutils.GetE2EImage(imageutils.BusyBox),
				Command: []string{"sh", "-c", "sleep 3600"},
				VolumeMounts: []v1.VolumeMount{
					{Name: "dyn-sec", MountPath: "/mnt/token-file", SubPath: "token"},
					{Name: "dyn-proj", MountPath: "/mnt/dyn-proj"},
				},
			})
		})

		gomega.Eventually(ctx, func(g gomega.Gomega) {
			p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			g.Expect(err).To(gomega.Succeed())
			cSec := findContainerStatus(p.Status.ContainerStatuses, "c-sec")
			g.Expect(cSec).NotTo(gomega.BeNil())
			g.Expect(cSec.State.Running).NotTo(gomega.BeNil())
			g.Expect(p.Status.VolumeStatuses).To(gomega.ContainElements(
				v1.PodVolumeStatus{Name: "dyn-sec", HostPhase: v1.PodVolumeHostMounted},
				v1.PodVolumeStatus{Name: "dyn-proj", HostPhase: v1.PodVolumeHostMounted},
			))
		}, 60*time.Second, 2*time.Second).Should(gomega.Succeed())

		gomega.Expect(e2epod.ExecCommandInContainer(f, pod.Name, "c-sec", "cat", "/mnt/token-file")).To(gomega.Equal("s3cr3t"))
		gomega.Expect(e2epod.ExecCommandInContainer(f, pod.Name, "c-sec", "cat", "/mnt/dyn-proj/podname")).To(gomega.Equal(pod.Name))
		gomega.Expect(e2epod.ExecCommandInContainer(f, pod.Name, "c-sec", "cat", "/mnt/dyn-proj/token")).To(gomega.Equal("s3cr3t"))

		ginkgo.By("Removing c-sec along with dyn-sec and dyn-proj and verifying clean teardown")
		updatePodDynamic(ctx, pod.Name, func(cleaned *v1.Pod) {
			cleaned.Spec.Containers = cleaned.Spec.Containers[:1]
			var keptVols []v1.Volume
			for _, v := range cleaned.Spec.Volumes {
				if v.Name != "dyn-sec" && v.Name != "dyn-proj" {
					keptVols = append(keptVols, v)
				}
			}
			cleaned.Spec.Volumes = keptVols
		})

		gomega.Eventually(ctx, func(g gomega.Gomega) {
			p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			g.Expect(err).To(gomega.Succeed())
			for _, vs := range p.Status.VolumeStatuses {
				g.Expect(vs.Name).NotTo(gomega.Equal("dyn-sec"))
				g.Expect(vs.Name).NotTo(gomega.Equal("dyn-proj"))
			}
		}, 60*time.Second, 2*time.Second).Should(gomega.Succeed())
	})

	ginkgo.Context("with OCI Image volumes and failure cleanup", func() {
		ginkgo.It("should isolate bad OCI Image volume pulls, mount valid OCI Image volumes, and release volumes from removed RunContainerError containers", func(ctx context.Context) {
			pod := createRunningSleepPod(ctx)

			ginkgo.By("Adding bad OCI Image volume, valid OCI Image volume, and RunContainerError container via /dynamic")
			updatePodDynamic(ctx, pod.Name, func(updated *v1.Pod) {
				updated.Spec.Volumes = append(updated.Spec.Volumes,
					v1.Volume{
						Name: "bad-img",
						VolumeSource: v1.VolumeSource{
							Image: &v1.ImageVolumeSource{
								Reference:  "localhost/invalid:nonexistent",
								PullPolicy: v1.PullAlways,
							},
						},
					},
					v1.Volume{
						Name: "good-img",
						VolumeSource: v1.VolumeSource{
							Image: &v1.ImageVolumeSource{
								Reference:  imageutils.GetE2EImage(imageutils.Kitten),
								PullPolicy: v1.PullIfNotPresent,
							},
						},
					},
					v1.Volume{
						Name:         "dyn-runerr-vol",
						VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}},
					},
				)
				updated.Spec.Containers = append(updated.Spec.Containers,
					v1.Container{
						Name:         "c-bad-img",
						Image:        imageutils.GetE2EImage(imageutils.BusyBox),
						Command:      []string{"sh", "-c", "sleep 3600"},
						VolumeMounts: []v1.VolumeMount{{Name: "bad-img", MountPath: "/mnt/bad-img"}},
					},
					v1.Container{
						Name:         "c-good-img",
						Image:        imageutils.GetE2EImage(imageutils.BusyBox),
						Command:      []string{"sh", "-c", "sleep 3600"},
						VolumeMounts: []v1.VolumeMount{{Name: "good-img", MountPath: "/mnt/good-img"}},
					},
					v1.Container{
						Name:         "c-runerr",
						Image:        imageutils.GetE2EImage(imageutils.BusyBox),
						Command:      []string{"/nonexistent-binary-that-fails-start"},
						VolumeMounts: []v1.VolumeMount{{Name: "dyn-runerr-vol", MountPath: "/mnt/runerr"}},
					},
				)
			})

			gomega.Eventually(ctx, func(g gomega.Gomega) {
				p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
				g.Expect(err).To(gomega.Succeed())
				c1 := findContainerStatus(p.Status.ContainerStatuses, "c1")
				g.Expect(c1).NotTo(gomega.BeNil())
				g.Expect(c1.RestartCount).To(gomega.Equal(int32(0)))
				g.Expect(c1.State.Running).NotTo(gomega.BeNil())
				cGood := findContainerStatus(p.Status.ContainerStatuses, "c-good-img")
				g.Expect(cGood).NotTo(gomega.BeNil())
				g.Expect(cGood.State.Running).NotTo(gomega.BeNil())
				g.Expect(findContainerStatus(p.Status.ContainerStatuses, "c-bad-img")).NotTo(gomega.BeNil())
				g.Expect(findContainerStatus(p.Status.ContainerStatuses, "c-runerr")).NotTo(gomega.BeNil())
				g.Expect(p.Status.VolumeStatuses).To(gomega.ContainElements(
					v1.PodVolumeStatus{Name: "bad-img", HostPhase: v1.PodVolumeHostPending},
					v1.PodVolumeStatus{Name: "good-img", HostPhase: v1.PodVolumeHostMounted},
					v1.PodVolumeStatus{Name: "dyn-runerr-vol", HostPhase: v1.PodVolumeHostMounted},
				))
			}, 90*time.Second, 2*time.Second).Should(gomega.Succeed())

			ginkgo.By("Removing c-bad-img, c-runerr, bad-img, and dyn-runerr-vol and verifying clean teardown")
			updatePodDynamic(ctx, pod.Name, func(cleaned *v1.Pod) {
				var keptCtrs []v1.Container
				for _, c := range cleaned.Spec.Containers {
					if c.Name != "c-bad-img" && c.Name != "c-runerr" {
						keptCtrs = append(keptCtrs, c)
					}
				}
				cleaned.Spec.Containers = keptCtrs
				var keptVols []v1.Volume
				for _, v := range cleaned.Spec.Volumes {
					if v.Name != "bad-img" && v.Name != "dyn-runerr-vol" {
						keptVols = append(keptVols, v)
					}
				}
				cleaned.Spec.Volumes = keptVols
			})

			gomega.Eventually(ctx, func(g gomega.Gomega) {
				p, getErr := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
				g.Expect(getErr).To(gomega.Succeed())
				for _, vs := range p.Status.VolumeStatuses {
					g.Expect(vs.Name).NotTo(gomega.Equal("bad-img"))
					g.Expect(vs.Name).NotTo(gomega.Equal("dyn-runerr-vol"))
				}
				g.Expect(p.Status.VolumeStatuses).To(gomega.ContainElement(
					v1.PodVolumeStatus{Name: "good-img", HostPhase: v1.PodVolumeHostMounted},
				))
			}, 60*time.Second, 2*time.Second).Should(gomega.Succeed())
		})
	})
})
