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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epodoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("FileKeyRef", feature.EnvFiles, func() {
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
						Command: []string{"sh", "-c", "echo 'DATABASE=mydb' > /data/config.env && echo 'API_KEY=secret123' >> /data/config.env"},
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
						Command: []string{"sh", "-c", "env | grep -E '(DATABASE|API_KEY)' | sort"},
						Env: []v1.EnvVar{
							{
								Name: "DATABASE",
								ValueFrom: &v1.EnvVarSource{
									FileKeyRef: &v1.FileKeySelector{
										VolumeName: "config",
										Path:       "config.env",
										Key:        "DATABASE",
									},
								},
							},
							{
								Name: "API_KEY",
								ValueFrom: &v1.EnvVarSource{
									FileKeyRef: &v1.FileKeySelector{
										VolumeName: "config",
										Path:       "config.env",
										Key:        "API_KEY",
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
			"API_KEY=secret123",
			"DATABASE=mydb",
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
						Command: []string{"sh", "-c", "echo 'CONFIG_1=value1' > /data/config.env && echo 'CONFIG_2=value2' >> /data/config.env && echo 'CONFIG_3=value3' >> /data/config.env"},
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
						Command: []string{"sh", "-c", "echo 'EXISTING_KEY=existing_value' > /data/config.env"},
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
										Optional:   &[]bool{true}[0],
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
						Command: []string{"sh", "-c", "echo 'HOOK_CONFIG=hook_value' > /data/config.env"},
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
									Command: []string{"sh", "-c", "echo 'PostStart: ' $HOOK_CONFIG > /tmp/config.env"},
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
		defer func() {
			_ = rc.Close()
		}()
		buf := new(bytes.Buffer)
		_, _ = buf.ReadFrom(rc)
		output := buf.String()
		gomega.Expect(output).To(gomega.ContainSubstring("PostStart: hook_value"))
	})
})
