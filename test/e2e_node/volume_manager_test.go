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

package e2enode

import (
	"context"
	"fmt"
	"path"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("Kubelet Volume Manager", func() {
	f := framework.NewDefaultFramework("kubelet-volume-manager")
	ginkgo.Context("with Watch ResourceChangeDetectionStrategy", func() {
		podName := "pod-with-volume"
		key := "data-1"
		createContent := "value-1"
		updateContent := "value-2"
		trueVal := true

		runPodAndVerifyVolume := func(volumeSource v1.VolumeSource, createVolumeSource, updateVolumeSource func()) {
			var pod *v1.Pod

			ginkgo.By("Creating a pod with the volume mounted", func() {
				volumes := []v1.Volume{
					{
						Name:         volumeName,
						VolumeSource: volumeSource,
					},
				}
				mounts := []v1.VolumeMount{
					{
						Name:      volumeName,
						MountPath: volumeMountPath,
					},
				}
				mounttestArgs := []string{"mounttest", "--break_on_expected_content=false", fmt.Sprintf("--file_content_in_loop=%s", path.Join(volumeMountPath, key))}
				pod = e2epod.NewAgnhostPod(f.Namespace.Name, podName, volumes, mounts, nil, mounttestArgs...)
				pod = f.PodClient().CreateSync(pod)
			})

			ginkgo.By("Creating the volume source", createVolumeSource)

			ginkgo.By("Verifying volume is updated on source creation", func() {
				gomega.Eventually(func() error {
					return f.PodClient().MatchContainerOutput(pod.Name, pod.Spec.Containers[0].Name, createContent)
				}, 5*time.Second, 1*time.Second).Should(gomega.BeNil())
			})

			ginkgo.By("Updating the volume source", updateVolumeSource)

			ginkgo.By("Verifying volume is updated on source update", func() {
				gomega.Eventually(func() error {
					return f.PodClient().MatchContainerOutput(pod.Name, pod.Spec.Containers[0].Name, updateContent)
				}, 5*time.Second, 1*time.Second).Should(gomega.BeNil())
			})
		}

		ginkgo.BeforeEach(func() {
			if framework.TestContext.KubeletConfig.ConfigMapAndSecretChangeDetectionStrategy != config.WatchChangeDetectionStrategy {
				e2eskipper.Skipf("This test is meant to run with ConfigMapAndSecretChangeDetectionStrategy set to Watch")
			}
		})

		ginkgo.It("ConfigMap update should be reflected in volume immediately", func() {
			configMapName := "configmap-" + string(uuid.NewUUID())
			volumeSource := v1.VolumeSource{
				ConfigMap: &v1.ConfigMapVolumeSource{
					LocalObjectReference: v1.LocalObjectReference{
						Name: configMapName,
					},
					Optional: &trueVal,
				},
			}
			configMap := &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: f.Namespace.Name,
					Name:      configMapName,
				},
				Data: map[string]string{
					"data-1": createContent,
				},
			}
			createConfigMap := func() {
				ginkgo.By(fmt.Sprintf("Creating configMap %v", configMap.Name))
				if _, err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configMap, metav1.CreateOptions{}); err != nil {
					framework.Failf("Failed to create configMap %s: %v", configMap.Name, err)
				}
			}
			updateConfigMap := func() {
				toUpdate := configMap.DeepCopy()
				toUpdate.Data["data-1"] = updateContent
				ginkgo.By(fmt.Sprintf("Updating configMap %v", configMap.Name))
				if _, err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Update(context.TODO(), toUpdate, metav1.UpdateOptions{}); err != nil {
					framework.Failf("Failed to update configMap %s: %v", toUpdate.Name, err)
				}
			}
			runPodAndVerifyVolume(volumeSource, createConfigMap, updateConfigMap)
			f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Delete(context.TODO(), configMapName, metav1.DeleteOptions{})
		})

		ginkgo.It("Secret update should be reflected in volume immediately", func() {
			secretName := "secret-" + string(uuid.NewUUID())
			volumeSource := v1.VolumeSource{
				Secret: &v1.SecretVolumeSource{
					SecretName: secretName,
					Optional:   &trueVal,
				},
			}
			secret := &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: f.Namespace.Name,
					Name:      secretName,
				},
				StringData: map[string]string{
					"data-1": createContent,
				},
			}
			createSecret := func() {
				ginkgo.By(fmt.Sprintf("Creating secret %v", secret.Name))
				if _, err := f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(context.TODO(), secret, metav1.CreateOptions{}); err != nil {
					framework.Failf("Failed to create serect %s: %v", secret.Name, err)
				}
			}
			updateSecret := func() {
				toUpdate := secret.DeepCopy()
				toUpdate.StringData["data-1"] = updateContent
				ginkgo.By(fmt.Sprintf("Updating secret %v", secret.Name))
				if _, err := f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Update(context.TODO(), toUpdate, metav1.UpdateOptions{}); err != nil {
					framework.Failf("Failed to update secret %s: %v", toUpdate.Name, err)
				}
			}
			runPodAndVerifyVolume(volumeSource, createSecret, updateSecret)
			f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Delete(context.TODO(), secretName, metav1.DeleteOptions{})
		})

		ginkgo.AfterEach(func() {
			f.PodClient().DeleteSync(podName, metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)
		})
	})

	ginkgo.Describe("Volume Manager", func() {
		ginkgo.Context("On termination of pod with memory backed volume", func() {
			ginkgo.It("should remove the volume from the node [NodeConformance]", func() {
				var (
					memoryBackedPod *v1.Pod
					volumeName      string
				)
				ginkgo.By("Creating a pod with a memory backed volume that exits success without restart", func() {
					volumeName = "memory-volume"
					memoryBackedPod = f.PodClient().Create(&v1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "pod" + string(uuid.NewUUID()),
							Namespace: f.Namespace.Name,
						},
						Spec: v1.PodSpec{
							RestartPolicy: v1.RestartPolicyNever,
							Containers: []v1.Container{
								{
									Image:   busyboxImage,
									Name:    "container" + string(uuid.NewUUID()),
									Command: []string{"sh", "-c", "echo"},
									VolumeMounts: []v1.VolumeMount{
										{
											Name:      volumeName,
											MountPath: "/tmp",
										},
									},
								},
							},
							Volumes: []v1.Volume{
								{
									Name: volumeName,
									VolumeSource: v1.VolumeSource{
										EmptyDir: &v1.EmptyDirVolumeSource{Medium: v1.StorageMediumMemory},
									},
								},
							},
						},
					})
					err := e2epod.WaitForPodSuccessInNamespace(f.ClientSet, memoryBackedPod.Name, f.Namespace.Name)
					framework.ExpectNoError(err)
				})
				ginkgo.By("Verifying the memory backed volume was removed from node", func() {
					volumePath := fmt.Sprintf("/tmp/%s/volumes/kubernetes.io~empty-dir/%s", string(memoryBackedPod.UID), volumeName)
					var err error
					for i := 0; i < 10; i++ {
						// need to create a new verification pod on each pass since updates
						//to the HostPath volume aren't propogated to the pod
						pod := f.PodClient().Create(&v1.Pod{
							ObjectMeta: metav1.ObjectMeta{
								Name:      "pod" + string(uuid.NewUUID()),
								Namespace: f.Namespace.Name,
							},
							Spec: v1.PodSpec{
								RestartPolicy: v1.RestartPolicyNever,
								Containers: []v1.Container{
									{
										Image:   busyboxImage,
										Name:    "container" + string(uuid.NewUUID()),
										Command: []string{"sh", "-c", "if [ -d " + volumePath + " ]; then exit 1; fi;"},
										VolumeMounts: []v1.VolumeMount{
											{
												Name:      "kubelet-pods",
												MountPath: "/tmp",
											},
										},
									},
								},
								Volumes: []v1.Volume{
									{
										Name: "kubelet-pods",
										VolumeSource: v1.VolumeSource{
											// TODO: remove hardcoded kubelet volume directory path
											// framework.TestContext.KubeVolumeDir is currently not populated for node e2e
											HostPath: &v1.HostPathVolumeSource{Path: "/var/lib/kubelet/pods"},
										},
									},
								},
							},
						})
						err = e2epod.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
						gp := int64(1)
						f.PodClient().Delete(context.TODO(), pod.Name, metav1.DeleteOptions{GracePeriodSeconds: &gp})
						if err == nil {
							break
						}
						<-time.After(10 * time.Second)
					}
					framework.ExpectNoError(err)
				})
			})
		})
	})
})
