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
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("Pod Extended (container restart policy)", framework.WithFeatureGate(features.ContainerRestartRules), func() {
	f := framework.NewDefaultFramework("pods")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.Describe("Container Restart Rules", func() {
		var (
			containerRestartPolicyAlways = v1.ContainerRestartPolicyAlways
			containerRestartPolicyNever  = v1.ContainerRestartPolicyNever
		)

		ginkgo.It("should restart container on rule match", func(ctx context.Context) {
			podName := "restart-rules-exit-code-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Name:          "main-container",
							Image:         imageutils.GetE2EImage(imageutils.BusyBox),
							Command:       []string{"/bin/sh", "-c", "exit 42"},
							RestartPolicy: &containerRestartPolicyNever,
							RestartPolicyRules: []v1.ContainerRestartRule{
								{
									Action: v1.ContainerRestartRuleActionRestart,
									ExitCodes: &v1.ContainerRestartRuleOnExitCodes{
										Operator: v1.ContainerRestartRuleOnExitCodesOpIn,
										Values:   []int32{42},
									},
								},
							},
						},
					},
				},
			}

			createAndValidateRestartableContainer(ctx, f, pod, podName, "main-container")
		})

		ginkgo.It("should not restart container on rule mismatch, container restart policy Never", func(ctx context.Context) {
			podName := "restart-rules-no-restart-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Name:          "main-container",
							Image:         imageutils.GetE2EImage(imageutils.BusyBox),
							Command:       []string{"/bin/sh", "-c", "exit 1"},
							RestartPolicy: &containerRestartPolicyNever,
							RestartPolicyRules: []v1.ContainerRestartRule{
								{
									Action: v1.ContainerRestartRuleActionRestart,
									ExitCodes: &v1.ContainerRestartRuleOnExitCodes{
										Operator: v1.ContainerRestartRuleOnExitCodesOpIn,
										Values:   []int32{42},
									},
								},
							},
						},
					},
				},
			}

			createAndValidateNonRestartableContainer(ctx, f, pod, podName, "main-container")
		})

		ginkgo.It("should restart container on container-level restart policy Never", func(ctx context.Context) {
			podName := "restart-rules-no-restart-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyAlways,
					Containers: []v1.Container{
						{
							Name:          "main-container",
							Image:         imageutils.GetE2EImage(imageutils.BusyBox),
							Command:       []string{"/bin/sh", "-c", "exit 1"},
							RestartPolicy: &containerRestartPolicyNever,
						},
					},
				},
			}

			createAndValidateNonRestartableContainer(ctx, f, pod, podName, "main-container")
		})

		ginkgo.It("should restart container on container-level restart policy Always", func(ctx context.Context) {
			podName := "restart-rules-no-restart-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Name:          "main-container",
							Image:         imageutils.GetE2EImage(imageutils.BusyBox),
							Command:       []string{"/bin/sh", "-c", "exit 1"},
							RestartPolicy: &containerRestartPolicyAlways,
						},
					},
				},
			}

			createAndValidateRestartableContainer(ctx, f, pod, podName, "main-container")
		})

		ginkgo.It("should restart container on pod-level restart policy Always when no container-level restart policy", func(ctx context.Context) {
			podName := "restart-rules-no-match-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyAlways,
					Containers: []v1.Container{
						{
							Name:    "main-container",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"/bin/sh", "-c", "exit 1"},
						},
					},
				},
			}

			createAndValidateRestartableContainer(ctx, f, pod, podName, "main-container")
		})
	})
})

var _ = SIGDescribe("Pod Extended (RestartAllContainers)", framework.WithFeatureGate(features.ContainerRestartRules), framework.WithFeatureGate(features.RestartAllContainersOnContainerExits), func() {
	f := framework.NewDefaultFramework("pods")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.Describe("RestartAllContainers", func() {
		var (
			containerRestartPolicyAlways = v1.ContainerRestartPolicyAlways
			containerRestartPolicyNever  = v1.ContainerRestartPolicyNever
		)

		restartAllContainersRules := []v1.ContainerRestartRule{
			{
				Action: v1.ContainerRestartRuleActionRestartAllContainers,
				ExitCodes: &v1.ContainerRestartRuleOnExitCodes{
					Operator: v1.ContainerRestartRuleOnExitCodesOpIn,
					Values:   []int32{42},
				},
			},
		}

		ginkgo.It("should restart all containers on regular container exit", func(ctx context.Context) {
			podName := "restart-rules-exit-code-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					InitContainers: []v1.Container{
						{
							Name:    "init",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"/bin/sh", "-c", "exit 0"},
						},
						{
							Name:          "sidecar",
							Image:         imageutils.GetE2EImage(imageutils.BusyBox),
							Command:       []string{"/bin/sh", "-c", "sleep 10000"},
							RestartPolicy: &containerRestartPolicyAlways,
						},
					},
					Containers: []v1.Container{
						{
							Name:               "source-container",
							Image:              imageutils.GetE2EImage(imageutils.BusyBox),
							Command:            []string{"/bin/sh", "-c", "if [ -f /mnt/restart-complete ]; then sleep 10000; else touch /mnt/restart-complete; exit 42; fi"},
							RestartPolicy:      &containerRestartPolicyNever,
							RestartPolicyRules: restartAllContainersRules,
							VolumeMounts: []v1.VolumeMount{
								{
									Name:      "workdir",
									MountPath: "/mnt",
								},
							},
						},
						{
							Name:    "regular",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"/bin/sh", "-c", "sleep 10000"},
						},
					},
					Volumes: []v1.Volume{
						{
							Name: "workdir",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{},
							},
						},
					},
				},
			}

			// All containers should be restarted once
			podClient := e2epod.NewPodClient(f)
			podClient.Create(ctx, pod)
			ginkgo.DeferCleanup(func(ctx context.Context) error {
				ginkgo.By("deleting the pod")
				return podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{})
			})
			validateAllContainersRestarted(ctx, f, pod, []string{"init", "sidecar", "source-container", "regular"})
			framework.ExpectNoError(e2epod.WaitForContainerRunning(ctx, f.ClientSet, f.Namespace.Name, podName, "source-container", 3*time.Minute))
			framework.ExpectNoError(e2epod.WaitForContainerRunning(ctx, f.ClientSet, f.Namespace.Name, podName, "regular", 3*time.Minute))
		})

		ginkgo.It("should restart all containers on sidecar container exit", func(ctx context.Context) {
			podName := "restart-rules-exit-code-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					InitContainers: []v1.Container{
						{
							Name:    "init",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"/bin/sh", "-c", "exit 0"},
						},
						{
							Name:          "sidecar",
							Image:         imageutils.GetE2EImage(imageutils.BusyBox),
							Command:       []string{"/bin/sh", "-c", "sleep 10000"},
							RestartPolicy: &containerRestartPolicyAlways,
						},
						{
							Name:               "source-sidecar",
							Image:              imageutils.GetE2EImage(imageutils.BusyBox),
							Command:            []string{"/bin/sh", "-c", "if [ -f /mnt/init-complete ]; then sleep 10000; else touch /mnt/init-complete; sleep 30; exit 42; fi"},
							RestartPolicy:      &containerRestartPolicyAlways,
							RestartPolicyRules: restartAllContainersRules,
							VolumeMounts: []v1.VolumeMount{
								{
									Name:      "workdir",
									MountPath: "/mnt",
								},
							},
						},
					},
					Containers: []v1.Container{
						{
							Name:    "regular",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"/bin/sh", "-c", "sleep 10000"},
						},
					},
					Volumes: []v1.Volume{
						{
							Name: "workdir",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{},
							},
						},
					},
				},
			}

			// All containers should be restarted once
			podClient := e2epod.NewPodClient(f)
			podClient.Create(ctx, pod)
			ginkgo.DeferCleanup(func(ctx context.Context) error {
				ginkgo.By("deleting the pod")
				return podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{})
			})
			validateAllContainersRestarted(ctx, f, pod, []string{"init", "sidecar", "source-sidecar", "regular"})
			framework.ExpectNoError(e2epod.WaitForContainerRunning(ctx, f.ClientSet, f.Namespace.Name, podName, "regular", 3*time.Minute))
			framework.ExpectNoError(e2epod.WaitForContainerRunning(ctx, f.ClientSet, f.Namespace.Name, podName, "source-sidecar", 3*time.Minute))
		})

		ginkgo.It("should restart init and sidecar containers on init container exit", func(ctx context.Context) {
			podName := "restart-rules-exit-code-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					InitContainers: []v1.Container{
						{
							Name:          "sidecar",
							Image:         imageutils.GetE2EImage(imageutils.BusyBox),
							Command:       []string{"/bin/sh", "-c", "sleep 10000"},
							RestartPolicy: &containerRestartPolicyAlways,
						},
						{
							Name:    "init",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"/bin/sh", "-c", "exit 0"},
						},
						{
							Name:               "source-init",
							Image:              imageutils.GetE2EImage(imageutils.BusyBox),
							Command:            []string{"/bin/sh", "-c", "if [ -f /mnt/init-complete ]; then exit 0; else touch /mnt/init-complete; exit 42; fi"},
							RestartPolicy:      &containerRestartPolicyNever,
							RestartPolicyRules: restartAllContainersRules,
							VolumeMounts: []v1.VolumeMount{
								{
									Name:      "workdir",
									MountPath: "/mnt",
								},
							},
						},
					},
					Containers: []v1.Container{
						{
							Name:    "regular",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"/bin/sh", "-c", "sleep 10000"},
						},
					},
					Volumes: []v1.Volume{
						{
							Name: "workdir",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{},
							},
						},
					},
				},
			}

			// All containers should be restarted once
			podClient := e2epod.NewPodClient(f)
			podClient.Create(ctx, pod)
			ginkgo.DeferCleanup(func(ctx context.Context) error {
				ginkgo.By("deleting the pod")
				return podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{})
			})
			validateAllContainersRestarted(ctx, f, pod, []string{"init", "sidecar", "source-init"})
			framework.ExpectNoError(e2epod.WaitForContainerRunning(ctx, f.ClientSet, f.Namespace.Name, podName, "regular", 3*time.Minute))
		})

		ginkgo.It("should allow multiple RestartAllContainers actions and not introduce a loop", func(ctx context.Context) {
			podName := "restart-rules-exit-code-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Name:               "source-container",
							Image:              imageutils.GetE2EImage(imageutils.BusyBox),
							Command:            []string{"/bin/sh", "-c", "if [ -f /mnt/restart-complete ]; then sleep 10000; else touch /mnt/restart-complete; sleep 10; exit 42; fi"},
							RestartPolicy:      &containerRestartPolicyNever,
							RestartPolicyRules: restartAllContainersRules,
							VolumeMounts: []v1.VolumeMount{
								{
									Name:      "workdir",
									MountPath: "/mnt",
								},
							},
						},
						{
							Name:               "regular",
							Image:              imageutils.GetE2EImage(imageutils.BusyBox),
							Command:            []string{"/bin/sh", "-c", "sleep 10000"},
							RestartPolicy:      &containerRestartPolicyNever,
							RestartPolicyRules: restartAllContainersRules,
						},
					},
					Volumes: []v1.Volume{
						{
							Name: "workdir",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{},
							},
						},
					},
				},
			}

			// All containers should be restarted once
			podClient := e2epod.NewPodClient(f)
			podClient.Create(ctx, pod)
			ginkgo.DeferCleanup(func(ctx context.Context) error {
				ginkgo.By("deleting the pod")
				return podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{})
			})
			validateAllContainersRestarted(ctx, f, pod, []string{"source-container", "regular"})
			framework.ExpectNoError(e2epod.WaitForContainerRunning(ctx, f.ClientSet, f.Namespace.Name, podName, "source-container", 3*time.Minute))
			framework.ExpectNoError(e2epod.WaitForContainerRunning(ctx, f.ClientSet, f.Namespace.Name, podName, "regular", 3*time.Minute))
		})

		ginkgo.It("should restart all containers on a previously restarted regular container exit", func(ctx context.Context) {
			podName := "restart-rules-exit-code-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:               "source-container",
							Image:              imageutils.GetE2EImage(imageutils.BusyBox),
							Command:            []string{"/bin/sh", "-c", "if [ -f /mnt/restart-complete ]; then sleep 10000; elif [ -f /mnt/restart-1 ]; then touch /mnt/restart-complete; exit 42; else touch /mnt/restart-1; exit 1; fi"},
							RestartPolicy:      &containerRestartPolicyAlways,
							RestartPolicyRules: restartAllContainersRules,
							VolumeMounts: []v1.VolumeMount{
								{
									Name:      "workdir",
									MountPath: "/mnt",
								},
							},
						},
						{
							Name:    "regular",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"/bin/sh", "-c", "sleep 10000"},
						},
					},
					Volumes: []v1.Volume{
						{
							Name: "workdir",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{},
							},
						},
					},
				},
			}

			// All containers should be restarted once
			podClient := e2epod.NewPodClient(f)
			podClient.Create(ctx, pod)
			ginkgo.DeferCleanup(func(ctx context.Context) error {
				ginkgo.By("deleting the pod")
				return podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{})
			})
			validateAllContainersRestarted(ctx, f, pod, []string{"source-container", "regular"})
			framework.ExpectNoError(e2epod.WaitForContainerRunning(ctx, f.ClientSet, f.Namespace.Name, podName, "source-container", 3*time.Minute))
			framework.ExpectNoError(e2epod.WaitForContainerRunning(ctx, f.ClientSet, f.Namespace.Name, podName, "regular", 3*time.Minute))
		})
	})
})

func validateAllContainersRestarted(ctx context.Context, f *framework.Framework, pod *v1.Pod, containers []string) {
	ginkgo.By("Waiting for all containers to restart")
	err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "all containers restarted", 3*time.Minute, func(pod *v1.Pod) (bool, error) {
		restartedCount := 0
		for _, cName := range containers {
			var cStatus v1.ContainerStatus
			for _, status := range pod.Status.ContainerStatuses {
				if status.Name == cName {
					cStatus = status
				}
			}
			for _, status := range pod.Status.InitContainerStatuses {
				if status.Name == cName {
					cStatus = status
				}
			}
			if cStatus.RestartCount > 0 {
				restartedCount++
			} else {
				framework.Logf("container %s did not restart", cName)
			}
			if cStatus.LastTerminationState.Terminated == nil {
				return false, fmt.Errorf("container %s do not have lastTerminationState", cName)
			}
		}
		framework.Logf("%d out of %d containers restarted", restartedCount, len(containers))
		return restartedCount == len(containers), nil
	})
	framework.ExpectNoError(err, "failed to see all containers restart")
}

func createAndValidateRestartableContainer(ctx context.Context, f *framework.Framework, pod *v1.Pod, podName, containerName string) {
	ginkgo.By("Creating the pod")
	e2epod.NewPodClient(f).Create(ctx, pod)

	ginkgo.By("Waiting for the container to restart")
	err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, podName, "container restarted", 10*time.Minute, func(pod *v1.Pod) (bool, error) {
		for _, status := range pod.Status.ContainerStatuses {
			if status.Name == containerName && status.RestartCount > 0 {
				return true, nil
			}
		}
		return false, nil
	})
	framework.ExpectNoError(err, "failed to see container restart")
}

func createAndValidateNonRestartableContainer(ctx context.Context, f *framework.Framework, pod *v1.Pod, podName, containerName string) {
	ginkgo.By("Creating the pod")
	e2epod.NewPodClient(f).Create(ctx, pod)

	ginkgo.By("Waiting for the pod to terminate")
	err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(ctx, f.ClientSet, podName, f.Namespace.Name, 10*time.Minute)
	framework.ExpectNoError(err, "failed to wait for pod terminate")

	ginkgo.By("Checking container restart count")
	p, err := e2epod.NewPodClient(f).Get(ctx, podName, metav1.GetOptions{})
	framework.ExpectNoError(err, "failed to get pod")
	for _, status := range p.Status.ContainerStatuses {
		if status.Name == containerName {
			gomega.Expect(status.RestartCount).To(gomega.BeZero())
		}
	}
}
