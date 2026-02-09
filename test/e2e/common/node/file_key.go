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

package node

import (
	"bytes"
	"context"
	"fmt"
	"time"

	"k8s.io/utils/ptr"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epodoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("FileKeyRef", framework.WithFeatureGate(features.EnvFiles), func() {
	f := framework.NewDefaultFramework("filekeyref")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	/*
		Release: v1.34
		Testname: FileKeyRef, basic functionality
		Description: Create a Pod with an init container that writes environment variables to a file,
		and a main container that reads those environment variables using FileKeyRef.
		The main container should be able to access the environment variables from the file.
	*/
	framework.It("should be consumable via FileKeyRef", func(ctx context.Context) {
		podName := "filekeyref-test-" + string(uuid.NewUUID())

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				InitContainers: []v1.Container{
					{
						Name:    "setup-envfile",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", `echo CONFIG_1=\'value1\' > /data/config.env && echo CONFIG_2=\'value2\' >> /data/config.env`},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "config",
								MountPath: "/data",
							},
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:    "use-envfile",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "env | grep -E '(CONFIG_1|CONFIG_2)' | sort"},
						Env: []v1.EnvVar{
							{
								Name: "CONFIG_1",
								ValueFrom: &v1.EnvVarSource{
									FileKeyRef: &v1.FileKeySelector{
										VolumeName: "config",
										Path:       "config.env",
										Key:        "CONFIG_1",
									},
								},
							},
							{
								Name: "CONFIG_2",
								ValueFrom: &v1.EnvVarSource{
									FileKeyRef: &v1.FileKeySelector{
										VolumeName: "config",
										Path:       "config.env",
										Key:        "CONFIG_2",
									},
								},
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				Volumes: []v1.Volume{
					{
						Name: "config",
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{},
						},
					},
				},
			},
		}

		e2epodoutput.TestContainerOutput(ctx, f, "consume FileKeyRef", pod, 0, []string{
			"CONFIG_1=value1",
			"CONFIG_2=value2",
		})
	})

	/*
		Release: v1.34
		Testname: FileKeyRef, multiple containers
		Description: Create a Pod with an init container that writes environment variables to a file,
		and multiple containers that read different environment variables using FileKeyRef.
		Each container should be able to access its own environment variables from the file.
	*/
	framework.It("should be consumable by multiple containers", func(ctx context.Context) {
		podName := "filekeyref-multi-" + string(uuid.NewUUID())

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				InitContainers: []v1.Container{
					{
						Name:    "setup-envfile",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", `echo CONFIG_1=\'value1\' > /data/config.env && echo CONFIG_2=\'value2\' >> /data/config.env && echo CONFIG_3=\'value3\' >> /data/config.env`},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "config",
								MountPath: "/data",
							},
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:    "container-1",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "echo $CONFIG_1"},
						Env: []v1.EnvVar{
							{
								Name: "CONFIG_1",
								ValueFrom: &v1.EnvVarSource{
									FileKeyRef: &v1.FileKeySelector{
										VolumeName: "config",
										Path:       "config.env",
										Key:        "CONFIG_1",
									},
								},
							},
						},
					},
					{
						Name:    "container-2",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "echo $CONFIG_2"},
						Env: []v1.EnvVar{
							{
								Name: "CONFIG_2",
								ValueFrom: &v1.EnvVarSource{
									FileKeyRef: &v1.FileKeySelector{
										VolumeName: "config",
										Path:       "config.env",
										Key:        "CONFIG_2",
									},
								},
							},
						},
					},
					{
						Name:    "container-3",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "echo $CONFIG_3"},
						Env: []v1.EnvVar{
							{
								Name: "CONFIG_3",
								ValueFrom: &v1.EnvVarSource{
									FileKeyRef: &v1.FileKeySelector{
										VolumeName: "config",
										Path:       "config.env",
										Key:        "CONFIG_3",
									},
								},
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				Volumes: []v1.Volume{
					{
						Name: "config",
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{},
						},
					},
				},
			},
		}

		e2epodoutput.TestContainerOutput(ctx, f, "consume FileKeyRef in container-1", pod, 0, []string{
			"value1",
		})
		e2epodoutput.TestContainerOutput(ctx, f, "consume FileKeyRef in container-2", pod, 1, []string{
			"value2",
		})
		e2epodoutput.TestContainerOutput(ctx, f, "consume FileKeyRef in container-3", pod, 2, []string{
			"value3",
		})
	})

	/*
		Release: v1.34
		Testname: FileKeyRef, optional key
		Description: Test FileKeyRef with optional keys that may not exist in the file.
		When a key is marked as optional and doesn't exist, the environment variable should be skipped.
	*/
	framework.It("should handle optional keys", func(ctx context.Context) {
		podName := "filekeyref-optional-" + string(uuid.NewUUID())

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				InitContainers: []v1.Container{
					{
						Name:    "setup-envfile",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", `echo EXISTING_KEY=\'existing_value\' > /data/config.env`},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "config",
								MountPath: "/data",
							},
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:    "optional-container",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "env | grep -E '(EXISTING_KEY|MISSING_KEY)' | sort"},
						Env: []v1.EnvVar{
							{
								Name: "EXISTING_KEY",
								ValueFrom: &v1.EnvVarSource{
									FileKeyRef: &v1.FileKeySelector{
										VolumeName: "config",
										Path:       "config.env",
										Key:        "EXISTING_KEY",
									},
								},
							},
							{
								Name: "MISSING_KEY",
								ValueFrom: &v1.EnvVarSource{
									FileKeyRef: &v1.FileKeySelector{
										VolumeName: "config",
										Path:       "config.env",
										Key:        "MISSING_KEY",
										Optional:   ptr.To(true),
									},
								},
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				Volumes: []v1.Volume{
					{
						Name: "config",
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{},
						},
					},
				},
			},
		}

		e2epodoutput.TestContainerOutput(ctx, f, "consume FileKeyRef with optional keys", pod, 0, []string{
			"EXISTING_KEY=existing_value",
		})
	})

	/*
		Release: v1.34
		Testname: FileKeyRef, lifecycle hooks
		Description: Create a Pod with an init container that writes environment variables to a file,
		and a main container with postStart hooks that use FileKeyRef to access environment variables.
		The postStart lifecycle hooks should be able to access environment variables from the file.
	*/
	framework.It("should be consumable in postStart lifecycle hooks", func(ctx context.Context) {
		podName := "filekeyref-lifecycle-" + string(uuid.NewUUID())
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				InitContainers: []v1.Container{
					{
						Name:    "setup-envfile",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", `echo HOOK_CONFIG=\'hook_value\' > /data/config.env`},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "config",
								MountPath: "/data",
							},
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:    "main-container",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "sleep 3 && cat /tmp/config.env"},
						Lifecycle: &v1.Lifecycle{
							PostStart: &v1.LifecycleHandler{
								Exec: &v1.ExecAction{
									Command: []string{"sh", "-c", "echo PostStart: $HOOK_CONFIG > /tmp/config.env"},
								},
							},
						},
						Env: []v1.EnvVar{
							{
								Name: "HOOK_CONFIG",
								ValueFrom: &v1.EnvVarSource{
									FileKeyRef: &v1.FileKeySelector{
										VolumeName: "config",
										Path:       "config.env",
										Key:        "HOOK_CONFIG",
									},
								},
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				Volumes: []v1.Volume{
					{
						Name: "config",
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{},
						},
					},
				},
			},
		}

		podClient := e2epod.NewPodClient(f)
		pod = podClient.Create(ctx, pod)
		ginkgo.By("Waiting for pod to complete")
		err := e2epod.WaitForPodNoLongerRunningInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
		framework.ExpectNoError(err)

		// Check logs for lifecycle hook output
		rc, err := podClient.GetLogs(podName, &v1.PodLogOptions{}).Stream(ctx)
		framework.ExpectNoError(err)
		defer func() { _ = rc.Close() }()
		buf := new(bytes.Buffer)
		_, _ = buf.ReadFrom(rc)
		output := buf.String()
		gomega.Expect(output).To(gomega.ContainSubstring("PostStart: hook_value"))
	})
	/*
		Release: v1.34
		Testname: FileKeyRef, lifecycle hooks
		Description: Create a Pod with an init container that writes environment variables to a file,
		and a main container with preStop hooks that use FileKeyRef to access environment variables.
		The preStop lifecycle hooks should be able to access environment variables from the file.
	*/
	framework.It("should be consumable in preStop lifecycle hooks", func(ctx context.Context) {
		podName := "filekeyref-lifecycle-" + string(uuid.NewUUID())
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				TerminationGracePeriodSeconds: ptr.To(int64(60)),
				InitContainers: []v1.Container{
					{
						Name:    "setup-envfile",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", `echo HOOK_CONFIG=\'hook_value\' > /data/config.env`},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "config",
								MountPath: "/data",
							},
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:    "main-container",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "trap 'sleep 30' TERM; while true; do sleep 1; done"},
						Lifecycle: &v1.Lifecycle{
							PreStop: &v1.LifecycleHandler{
								Exec: &v1.ExecAction{
									Command: []string{"sh", "-c", "echo PreStop: $HOOK_CONFIG >> /proc/1/fd/1 && sleep 10"},
								},
							},
						},
						Env: []v1.EnvVar{
							{
								Name: "HOOK_CONFIG",
								ValueFrom: &v1.EnvVarSource{
									FileKeyRef: &v1.FileKeySelector{
										VolumeName: "config",
										Path:       "config.env",
										Key:        "HOOK_CONFIG",
									},
								},
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				Volumes: []v1.Volume{
					{
						Name: "config",
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{},
						},
					},
				},
			},
		}

		podClient := e2epod.NewPodClient(f)
		pod = podClient.CreateSync(ctx, pod)
		err := podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		gomega.Eventually(ctx, func(g gomega.Gomega) {
			rc, err := podClient.GetLogs(pod.Name, &v1.PodLogOptions{}).Stream(ctx)
			framework.ExpectNoError(err)
			defer func() { _ = rc.Close() }()
			buf := new(bytes.Buffer)
			_, _ = buf.ReadFrom(rc)
			output := buf.String()
			g.Expect(output).To(gomega.ContainSubstring("PreStop: hook_value"))
		}).WithTimeout(2 * time.Minute).WithPolling(2 * time.Second).Should(gomega.Succeed())
	})

	/*
		Release: v1.34
		Testname: FileKeyRef, invalid volume name
		Description: Test that Pod creation fails when the volumeName specified in FileKeySelector field does not exist.
		The Pod should fail to be created with an appropriate error message.
	*/
	framework.It("should fail when volumeName does not exist", func(ctx context.Context) {
		podName := "filekeyref-invalid-volume-" + string(uuid.NewUUID())

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				InitContainers: []v1.Container{
					{
						Name:    "setup-envfile",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", `echo CONFIG_1=\'value1\' > /data/config.env`},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "config",
								MountPath: "/data",
							},
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:    "use-envfile",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "echo $CONFIG_1"},
						Env: []v1.EnvVar{
							{
								Name: "CONFIG_1",
								ValueFrom: &v1.EnvVarSource{
									FileKeyRef: &v1.FileKeySelector{
										VolumeName: "nonexistent-volume",
										Path:       "config.env",
										Key:        "CONFIG_1",
									},
								},
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				Volumes: []v1.Volume{
					{
						Name: "config",
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{},
						},
					},
				},
			},
		}

		_, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
		gomega.Expect(err).To(gomega.HaveOccurred())
	})

	/*
		Release: v1.34
		Testname: FileKeyRef, non-emptyDir volume
		Description: Test that Pod creation fails when the volumeName specified in FileKeySelector is not an emptyDir.
		The Pod should fail to be created with an appropriate error message.
	*/
	framework.It("should fail when volume is not emptyDir", func(ctx context.Context) {
		secretName := "filekeyref-non-emptydir-" + string(uuid.NewUUID())

		secret := secretForTest(f.Namespace.Name, secretName)
		ginkgo.By(fmt.Sprintf("Creating secret with name %s", secret.Name))
		if secret, err := f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(ctx, secret, metav1.CreateOptions{}); err != nil {
			framework.Failf("unable to create test secret %s: %v", secret.Name, err)
		}

		podName := "filekeyref-non-emptydir-" + string(uuid.NewUUID())

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				InitContainers: []v1.Container{
					{
						Name:    "setup-envfile",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", `echo CONFIG_1=\'value1\' > /data/config.env`},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "config",
								MountPath: "/data",
							},
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:    "use-envfile",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "echo $CONFIG_1"},
						Env: []v1.EnvVar{
							{
								Name: "CONFIG_1",
								ValueFrom: &v1.EnvVarSource{
									FileKeyRef: &v1.FileKeySelector{
										VolumeName: "config",
										Path:       "config.env",
										Key:        "CONFIG_1",
									},
								},
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				Volumes: []v1.Volume{
					{
						Name: "config",
						VolumeSource: v1.VolumeSource{Secret: &v1.SecretVolumeSource{
							SecretName: secret.Name,
						}},
					},
				},
			},
		}

		_, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
		gomega.Expect(err).To(gomega.HaveOccurred())
	})

	/*
		Release: v1.34
		Testname: FileKeyRef, missing file or key
		Description: Test that when either the filepath or key specified in FileKeySelector field does not exist,
		the Pod is created but the Container fails to start and an error message appears in events.
	*/
	framework.It("should fail when file or key does not exist", func(ctx context.Context) {
		podName := "filekeyref-missing-file-key-" + string(uuid.NewUUID())

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				InitContainers: []v1.Container{
					{
						Name:    "setup-envfile",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", `echo EXISTING_KEY=\'value\' > /data/config.env`},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "config",
								MountPath: "/data",
							},
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:    "use-envfile",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "echo $MISSING_KEY"},
						Env: []v1.EnvVar{
							{
								Name: "MISSING_KEY",
								ValueFrom: &v1.EnvVarSource{
									FileKeyRef: &v1.FileKeySelector{
										VolumeName: "config",
										Path:       "config.env",
										Key:        "MISSING_KEY",
									},
								},
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				Volumes: []v1.Volume{
					{
						Name: "config",
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{},
						},
					},
				},
			},
		}

		podClient := e2epod.NewPodClient(f)
		pod = podClient.Create(ctx, pod)
		framework.ExpectNoError(e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "container not ready", time.Minute, func(pod *v1.Pod) (bool, error) {
			for _, c := range append(pod.Status.InitContainerStatuses, pod.Status.ContainerStatuses...) {
				if c.Name == "use-envfile" {
					if c.State.Waiting != nil && c.State.Waiting.Reason == "CreateContainerConfigError" {
						return true, nil
					}
				}
			}
			return false, nil
		}))
	})

	/*
		Release: v1.34
		Testname: FileKeyRef, optional missing file or key
		Description: Test that when either the filepath or key specified in FileKeySelector field does not exist
		but the optional field is set to true, the Pod is created and the Container starts but env vars are not populated.
	*/
	framework.It("should handle optional missing file or key", func(ctx context.Context) {
		podName := "filekeyref-optional-missing-" + string(uuid.NewUUID())

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				InitContainers: []v1.Container{
					{
						Name:    "setup-envfile",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", `echo EXISTING_KEY=\'value\' > /data/config.env`},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "config",
								MountPath: "/data",
							},
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:    "use-envfile",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "env | grep -E '(EXISTING_KEY|OPTIONAL_MISSING_KEY)' | sort"},
						Env: []v1.EnvVar{
							{
								Name: "EXISTING_KEY",
								ValueFrom: &v1.EnvVarSource{
									FileKeyRef: &v1.FileKeySelector{
										VolumeName: "config",
										Path:       "config.env",
										Key:        "EXISTING_KEY",
									},
								},
							},
							{
								Name: "OPTIONAL_MISSING_KEY",
								ValueFrom: &v1.EnvVarSource{
									FileKeyRef: &v1.FileKeySelector{
										VolumeName: "config",
										Path:       "config.env",
										Key:        "OPTIONAL_MISSING_KEY",
										Optional:   ptr.To(true),
									},
								},
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				Volumes: []v1.Volume{
					{
						Name: "config",
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{},
						},
					},
				},
			},
		}

		e2epodoutput.TestContainerOutput(ctx, f, "consume FileKeyRef with optional missing key", pod, 0, []string{
			"EXISTING_KEY=value",
		})
	})

	/*
		Release: v1.34
		Testname: FileKeyRef, initContainer consumes fileKeyRef
		Description: Create a Pod with an init container that writes environment variables to a file, and another init container that consumes the variable using FileKeyRef. The consuming init container should be able to access the environment variable from the file.
	*/
	framework.It("should allow initContainer to consume fileKeyRef", func(ctx context.Context) {
		podName := "filekeyref-initconsumer-" + string(uuid.NewUUID())
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: podName},
			Spec: v1.PodSpec{
				InitContainers: []v1.Container{
					{
						Name:         "setup-envfile",
						Image:        imageutils.GetE2EImage(imageutils.BusyBox),
						Command:      []string{"sh", "-c", `echo CONFIG_1=\'value1\' > /data/config.env`},
						VolumeMounts: []v1.VolumeMount{{Name: "config", MountPath: "/data"}},
					},
					{
						Name:    "use-envfile1",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "env | grep CONFIG_1"},
						Env: []v1.EnvVar{{
							Name: "CONFIG_1",
							ValueFrom: &v1.EnvVarSource{
								FileKeyRef: &v1.FileKeySelector{
									VolumeName: "config",
									Path:       "config.env",
									Key:        "CONFIG_1",
									Optional:   ptr.To(false),
								},
							},
						}},
						VolumeMounts: []v1.VolumeMount{{Name: "config", MountPath: "/data"}},
					},
				},
				Containers: []v1.Container{{
					Name:    "main",
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"sh", "-c", "sleep 1"},
				}},
				RestartPolicy: v1.RestartPolicyNever,
				Volumes: []v1.Volume{{
					Name:         "config",
					VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}},
				}},
			},
		}
		podClient := e2epod.NewPodClient(f)
		pod = podClient.Create(ctx, pod)
		err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
		framework.ExpectNoError(err)
		logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "use-envfile1")
		framework.ExpectNoError(err)
		gomega.Expect(logs).To(gomega.ContainSubstring("CONFIG_1=value1"))
	})

	/*
		Release: v1.34
		Testname: FileKeyRef, initContainer fails to consume fileKeyRef from non-emptyDir
		Description: Create a Pod with an init container that writes environment variables to a file, and another init container that tries to consume the variable using FileKeyRef from a non-emptyDir volume. Pod creation should fail.
	*/
	framework.It("should fail when initContainer consumes fileKeyRef from non-emptyDir volume", func(ctx context.Context) {
		podName := "filekeyref-initfail-" + string(uuid.NewUUID())
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				InitContainers: []v1.Container{
					{
						Name:         "setup-envfile",
						Image:        imageutils.GetE2EImage(imageutils.BusyBox),
						Command:      []string{"sh", "-c", `echo CONFIG_INIT=\'fail\' > /data/config.env`},
						VolumeMounts: []v1.VolumeMount{{Name: "config", MountPath: "/data"}},
					},
					{
						Name:    "use-envfile1",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "env | grep CONFIG_MAIN"},
						Env: []v1.EnvVar{{
							Name: "CONFIG_MAIN",
							ValueFrom: &v1.EnvVarSource{
								FileKeyRef: &v1.FileKeySelector{
									VolumeName: "config",
									Path:       "config.env",
									Key:        "CONFIG_INIT",
									Optional:   ptr.To(false),
								},
							},
						}},
						VolumeMounts: []v1.VolumeMount{{Name: "config", MountPath: "/data"}},
					},
				},
				Containers: []v1.Container{{
					Name:    "main",
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"sh", "-c", "sleep 1"},
				}},
				RestartPolicy: v1.RestartPolicyNever,
				Volumes: []v1.Volume{{
					Name:         "config",
					VolumeSource: v1.VolumeSource{HostPath: &v1.HostPathVolumeSource{Path: "/tmp"}},
				}},
			},
		}
		_, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
		gomega.Expect(err).To(gomega.HaveOccurred())
	})
	/*
		Release: v1.34
		Testname: FileKeyRef, ephemeralContainer consumes fileKeyRef
		Description: Create a Pod with an init container that writes environment variables to a file, and an ephemeral container that consumes the variable using FileKeyRef. The ephemeral container should be able to access the environment variable from the file.
	*/
	framework.It("should allow ephemeralContainer to consume fileKeyRef", func(ctx context.Context) {
		podName := "filekeyref-ephemeralconsumer-" + string(uuid.NewUUID())
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: podName},
			Spec: v1.PodSpec{
				InitContainers: []v1.Container{{
					Name:         "setup-envfile",
					Image:        imageutils.GetE2EImage(imageutils.BusyBox),
					Command:      []string{"sh", "-c", `echo CONFIG_EPH=\'ephemeral\' > /data/config.env`},
					VolumeMounts: []v1.VolumeMount{{Name: "config", MountPath: "/data"}},
				}},
				Containers: []v1.Container{{
					Name:    "main",
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"sh", "-c", "sleep 1000"},
				}},
				Volumes: []v1.Volume{{
					Name:         "config",
					VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}},
				}},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}
		podClient := e2epod.NewPodClient(f)
		pod = podClient.CreateSync(ctx, pod)
		ec := v1.EphemeralContainer{
			EphemeralContainerCommon: v1.EphemeralContainerCommon{
				Name:    "debugger",
				Image:   imageutils.GetE2EImage(imageutils.BusyBox),
				Command: []string{"sh", "-c", "env | grep CONFIG_EPH_MAIN && sleep 10"},
				Env: []v1.EnvVar{{
					Name: "CONFIG_EPH_MAIN",
					ValueFrom: &v1.EnvVarSource{
						FileKeyRef: &v1.FileKeySelector{
							VolumeName: "config",
							Path:       "config.env",
							Key:        "CONFIG_EPH",
							Optional:   ptr.To(false),
						},
					},
				}},
				VolumeMounts: []v1.VolumeMount{{Name: "config", MountPath: "/data"}},
			},
		}
		err := podClient.AddEphemeralContainerSync(ctx, pod, &ec, time.Minute)
		framework.ExpectNoError(err)
		logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "debugger")
		framework.ExpectNoError(err)
		gomega.Expect(logs).To(gomega.ContainSubstring("CONFIG_EPH_MAIN=ephemeral"))
	})
})
