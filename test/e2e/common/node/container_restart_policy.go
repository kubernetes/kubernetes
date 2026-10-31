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
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("Pod Extended (container restart policy)", framework.WithNodeConformance(), framework.WithFeatureGate(features.ContainerRestartRules), func() {
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

		ginkgo.It("should continue probing on rule match", func(ctx context.Context) {
			podName := "restart-rules-probing-" + string(uuid.NewUUID())
			containerName := "main-container"
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Name:          containerName,
							Image:         imageutils.GetE2EImage(imageutils.BusyBox),
							Command:       []string{"/bin/sh", "-c", "if [ -f /mnt/restart-complete ]; then sleep 10000; else touch /mnt/restart-complete; exit 42; fi"},
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
							// The container exits before InitialDelaySeconds elapses, so the
							// probes can only succeed against the restarted container.
							StartupProbe: &v1.Probe{
								ProbeHandler: v1.ProbeHandler{
									Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
								},
								InitialDelaySeconds: 3,
								PeriodSeconds:       2,
								FailureThreshold:    10,
							},
							// Readiness is only probed once the startup probe has succeeded, so
							// Ready also covers the readiness probe worker surviving the restart.
							ReadinessProbe: &v1.Probe{
								ProbeHandler: v1.ProbeHandler{
									Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
								},
								PeriodSeconds: 2,
							},
							VolumeMounts: []v1.VolumeMount{
								{
									Name:      "workdir",
									MountPath: "/mnt",
								},
							},
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

			podClient := e2epod.NewPodClient(f)
			podClient.Create(ctx, pod)
			ginkgo.DeferCleanup(func(ctx context.Context) error {
				ginkgo.By("deleting the pod")
				return podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{})
			})

			ginkgo.By("Waiting for the container to be restarted by the matching rule")
			framework.ExpectNoError(e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, podName, "container restarted by rule", 3*time.Minute, func(pod *v1.Pod) (bool, error) {
				for _, status := range pod.Status.ContainerStatuses {
					if status.Name != containerName || status.RestartCount == 0 {
						continue
					}
					if status.LastTerminationState.Terminated == nil || status.LastTerminationState.Terminated.ExitCode != 42 {
						return false, fmt.Errorf("container %q restarted without exiting with code 42: %+v", containerName, status.LastTerminationState)
					}
					return true, nil
				}
				return false, nil
			}), "wait for container %q in Pod %q to be restarted", containerName, podName)

			ginkgo.By("Waiting for the restarted container to pass its startup and readiness probes")
			framework.ExpectNoError(e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, podName, "restarted container started and ready", 3*time.Minute, func(pod *v1.Pod) (bool, error) {
				for _, status := range pod.Status.ContainerStatuses {
					if status.Name == containerName {
						return status.RestartCount > 0 && status.Started != nil && *status.Started && status.Ready, nil
					}
				}
				return false, nil
			}), "wait for restarted container %q in Pod %q to be started and ready", containerName, podName)
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

var _ = SIGDescribe("Pod Extended (RestartAllContainers)", framework.WithSlow(), framework.WithNodeConformance(), framework.WithFeatureGate(features.ContainerRestartRules), framework.WithFeatureGate(features.RestartAllContainersOnContainerExits), func() {
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
			framework.ExpectNoError(e2epod.WaitForContainerRunning(ctx, f.ClientSet, f.Namespace.Name, podName, "source-container", 3*time.Minute), "wait for container %q in Pod %q in namespace %q to run", "source-container", podName, f.Namespace.Name)
			framework.ExpectNoError(e2epod.WaitForContainerRunning(ctx, f.ClientSet, f.Namespace.Name, podName, "regular", 3*time.Minute), "wait for container %q in Pod %q in namespace %q to run", "regular", podName, f.Namespace.Name)
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
			framework.ExpectNoError(e2epod.WaitForContainerRunning(ctx, f.ClientSet, f.Namespace.Name, podName, "regular", 3*time.Minute), "wait for container %q in Pod %q in namespace %q to run", "regular", podName, f.Namespace.Name)
			framework.ExpectNoError(e2epod.WaitForContainerRunning(ctx, f.ClientSet, f.Namespace.Name, podName, "source-sidecar", 3*time.Minute), "wait for container %q in Pod %q in namespace %q to run", "source-sidecar", podName, f.Namespace.Name)
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
			framework.ExpectNoError(e2epod.WaitForContainerRunning(ctx, f.ClientSet, f.Namespace.Name, podName, "regular", 3*time.Minute), "wait for container %q in Pod %q in namespace %q to run", "regular", podName, f.Namespace.Name)
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
			framework.ExpectNoError(e2epod.WaitForContainerRunning(ctx, f.ClientSet, f.Namespace.Name, podName, "source-container", 3*time.Minute), "wait for container %q in Pod %q in namespace %q to run", "source-container", podName, f.Namespace.Name)
			framework.ExpectNoError(e2epod.WaitForContainerRunning(ctx, f.ClientSet, f.Namespace.Name, podName, "regular", 3*time.Minute), "wait for container %q in Pod %q in namespace %q to run", "regular", podName, f.Namespace.Name)
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
			framework.ExpectNoError(e2epod.WaitForContainerRunning(ctx, f.ClientSet, f.Namespace.Name, podName, "source-container", 3*time.Minute), "wait for container %q in Pod %q in namespace %q to run", "source-container", podName, f.Namespace.Name)
			framework.ExpectNoError(e2epod.WaitForContainerRunning(ctx, f.ClientSet, f.Namespace.Name, podName, "regular", 3*time.Minute), "wait for container %q in Pod %q in namespace %q to run", "regular", podName, f.Namespace.Name)
		})

		ginkgo.It("should not carry liveness probe failures over to a restarted container that terminated on rule mismatch", func(ctx context.Context) {
			podName := "restart-rules-probe-state-" + string(uuid.NewUUID())
			// Fails once per container, which only reaches FailureThreshold if the probe
			// worker carried the old container's failure over the restart.
			livenessCommand := "[ -f /tmp/probed ] || { touch /tmp/probed; exit 1; }"
			probePeriodSeconds := int32(3)
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					// Without a SIGTERM handler, a liveness kill would take the full 30s grace period.
					TerminationGracePeriodSeconds: ptr.To[int64](1),
					Containers: []v1.Container{
						{
							Name:          "target",
							Image:         imageutils.GetE2EImage(imageutils.BusyBox),
							Command:       []string{"/bin/sh", "-c", "if [ -f /mnt/target-started ]; then sleep 10000; else touch /mnt/target-started; until [ -f /tmp/probed ]; do sleep 0.1; done; sleep 1; exit 1; fi"},
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
							LivenessProbe: &v1.Probe{
								ProbeHandler: v1.ProbeHandler{
									Exec: &v1.ExecAction{Command: []string{"/bin/sh", "-c", livenessCommand}},
								},
								PeriodSeconds:    probePeriodSeconds,
								FailureThreshold: 2,
							},
							VolumeMounts: []v1.VolumeMount{
								{
									Name:      "workdir",
									MountPath: "/mnt",
								},
							},
						},
						{
							// Restarts every container once the test signals that the probe
							// worker has had time to observe the terminated target.
							Name:               "trigger",
							Image:              imageutils.GetE2EImage(imageutils.BusyBox),
							Command:            []string{"/bin/sh", "-c", "if [ -f /mnt/restart-complete ]; then sleep 10000; else until [ -f /mnt/restart-trigger ]; do sleep 0.1; done; touch /mnt/restart-complete; exit 42; fi"},
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

			podClient := e2epod.NewPodClient(f)
			podClient.Create(ctx, pod)
			ginkgo.DeferCleanup(func(ctx context.Context) error {
				ginkgo.By("deleting the pod")
				return podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{})
			})

			ginkgo.By("Waiting for the target container to terminate without a matching rule")
			framework.ExpectNoError(e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, podName, "target terminated", 3*time.Minute, func(pod *v1.Pod) (bool, error) {
				for _, status := range pod.Status.ContainerStatuses {
					if status.Name == "target" && status.State.Terminated != nil {
						if status.State.Terminated.ExitCode != 1 {
							return false, fmt.Errorf("target exited with code %d, expected 1", status.State.Terminated.ExitCode)
						}
						return true, nil
					}
				}
				return false, nil
			}))

			// Whether the probe worker stopped is not visible through the API. It reads the
			// same status that is reported to the API server, though, so it sees the
			// termination within one probe period from here. Two periods leave some margin.
			ginkgo.By("Verifying the target container is not restarted without a matching rule")
			gomega.Consistently(ctx, func(ctx context.Context) (bool, error) {
				pod, err := podClient.Get(ctx, podName, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				for _, status := range pod.Status.ContainerStatuses {
					if status.Name == "target" {
						return status.State.Terminated != nil && status.RestartCount == 0, nil
					}
				}
				return false, nil
			}).WithTimeout(2 * time.Duration(probePeriodSeconds) * time.Second).WithPolling(time.Second).Should(gomega.BeTrueBecause("the target container should stay terminated without a matching rule"))

			ginkgo.By("Restarting all containers")
			e2epod.ExecCommandInContainer(f, podName, "trigger", "touch", "/mnt/restart-trigger")

			validateAllContainersRestarted(ctx, f, pod, []string{"target", "trigger"})

			ginkgo.By("Waiting for the liveness probe to fail once in the restarted target container")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				pod, err := podClient.Get(ctx, podName, metav1.GetOptions{})
				if err != nil {
					return err
				}
				for _, status := range pod.Status.ContainerStatuses {
					if status.Name == "target" && status.State.Terminated != nil {
						return gomega.StopTrying(fmt.Sprintf("restarted target container terminated with exit code %d after a single liveness probe failure", status.State.Terminated.ExitCode))
					}
				}
				_, _, err = e2epod.ExecCommandInContainerWithFullOutput(f, podName, "target", "test", "-f", "/tmp/probed")
				return err
			}).WithTimeout(time.Minute).WithPolling(time.Second).Should(gomega.Succeed())

			ginkgo.By("Verifying the restarted target container is not killed by its first liveness probe failure")
			gomega.Consistently(ctx, func(ctx context.Context) (bool, error) {
				pod, err := podClient.Get(ctx, podName, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				for _, status := range pod.Status.ContainerStatuses {
					if status.Name == "target" {
						return status.State.Running != nil, nil
					}
				}
				return false, nil
			}).WithTimeout(10 * time.Second).WithPolling(time.Second).Should(gomega.BeTrueBecause("the restarted target container should survive a single liveness probe failure"))
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
	framework.ExpectNoError(err, "see all containers restart")
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
	framework.ExpectNoError(err, "see container restart")
}

func createAndValidateNonRestartableContainer(ctx context.Context, f *framework.Framework, pod *v1.Pod, podName, containerName string) {
	ginkgo.By("Creating the pod")
	e2epod.NewPodClient(f).Create(ctx, pod)

	ginkgo.By("Waiting for the pod to terminate")
	err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(ctx, f.ClientSet, podName, f.Namespace.Name, 10*time.Minute)
	framework.ExpectNoError(err, "wait for pod terminate")

	ginkgo.By("Checking container restart count")
	p, err := e2epod.NewPodClient(f).Get(ctx, podName, metav1.GetOptions{})
	framework.ExpectNoError(err, "get pod")
	for _, status := range p.Status.ContainerStatuses {
		if status.Name == containerName {
			gomega.Expect(status.RestartCount).To(gomega.BeZero())
		}
	}
}
