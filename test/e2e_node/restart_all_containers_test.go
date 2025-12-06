//go:build linux

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
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("RestartAllContainersWithKubeletRestarts",
	framework.WithSerial(),
	framework.WithDisruptive(),
	framework.WithFeatureGate(features.ContainerRestartRules),
	framework.WithFeatureGate(features.RestartAllContainersOnContainerExits),
	func() {
		const (
			containerCount = 10
		)
		var (
			containerRestartPolicyNever = v1.ContainerRestartPolicyNever
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
		f := framework.NewDefaultFramework("restart-test")
		f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

		ginkgo.It("kubelet restart during cleanup", func(ctx context.Context) {
			podName := "restart-kubelet-during-cleanup" + string(uuid.NewUUID())
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
							Command:            []string{"/bin/sh", "-c", "if [ -f /mnt/init-complete ]; then sleep 3600; else touch /mnt/init-complete; sleep 10; exit 42; fi"},
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
			// Setting up extra containers so the cleanup takes more time to coordinate with kubelet restart.
			for i := 0; i < containerCount; i++ {
				pod.Spec.Containers = append(pod.Spec.Containers, v1.Container{
					Name:    fmt.Sprintf("container-%d", i),
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"/bin/sh", "-c", "sleep 3600; exit 0"},
				})
			}

			ginkgo.By(fmt.Sprintf("Creating a pod (%v/%v)", f.Namespace.Name, podName))
			podClient := e2epod.NewPodClient(f)
			podClient.Create(ctx, pod)
			ginkgo.DeferCleanup(func(ctx context.Context) error {
				ginkgo.By("deleting the pod")
				return podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{})
			})

			ginkgo.By("waiting for pod to be marked for restart")
			gomega.Eventually(ctx, func() error {
				pod, err := getPodByName(ctx, f, podName)
				if err != nil {
					return err
				}
				for _, cond := range pod.Status.Conditions {
					if cond.Type == v1.AllContainersRestarting && cond.Status == v1.ConditionTrue {
						return nil
					}
				}
				return fmt.Errorf("pod not marked for restart")
			}, 2*time.Minute, 1*time.Second).Should(gomega.Succeed())

			ginkgo.By("restarting kubelet")
			restartKubelet := mustStopKubelet(ctx, f)
			restartKubelet(ctx)

			ginkgo.By("waiting for containers to restart")
			gomega.Eventually(ctx, func() error {
				pod, err := getPodByName(ctx, f, podName)
				if err != nil {
					return err
				}
				restartedContainers := 0
				runningContainers := 0
				for _, c := range pod.Status.ContainerStatuses {
					if c.RestartCount > 0 {
						restartedContainers++
					}
					if c.State.Running != nil {
						runningContainers++
					}
				}
				if restartedContainers < containerCount+1 {
					return fmt.Errorf("not all containers have restarted")
				}
				if runningContainers < containerCount+1 {
					return fmt.Errorf("not all containers are running")
				}
				return nil
			}, 3*time.Minute, f.Timeouts.Poll).Should(gomega.Succeed())
		})

		ginkgo.It("kubelet restart during startup", func(ctx context.Context) {
			podName := "restart-kubelet-during-cleanup" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					InitContainers: []v1.Container{
						{
							Name:  "init",
							Image: imageutils.GetE2EImage(imageutils.BusyBox),
							// Let init container sleep 20s before succeeding to coordinate with kubelet restart
							Command: []string{"/bin/sh", "-c", "sleep 20; exit 0"},
						},
					},
					Containers: []v1.Container{
						{
							Name:               "source-container",
							Image:              imageutils.GetE2EImage(imageutils.BusyBox),
							Command:            []string{"/bin/sh", "-c", "if [ -f /mnt/init-complete ]; then sleep 3600; else touch /mnt/init-complete; sleep 10; exit 42; fi"},
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
			// Setting up extra containers so the cleanup takes more time to coordinate with kubelet restart.
			for i := 0; i < containerCount; i++ {
				pod.Spec.Containers = append(pod.Spec.Containers, v1.Container{
					Name:    fmt.Sprintf("container-%d", i),
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"/bin/sh", "-c", "sleep 3600; exit 0"},
				})
			}

			ginkgo.By(fmt.Sprintf("Creating a pod (%v/%v)", f.Namespace.Name, podName))
			podClient := e2epod.NewPodClient(f)
			podClient.Create(ctx, pod)
			ginkgo.DeferCleanup(func(ctx context.Context) error {
				ginkgo.By("deleting the pod")
				return podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{})
			})

			ginkgo.By("waiting for pod to be marked for restart")
			gomega.Eventually(ctx, func() error {
				pod, err := getPodByName(ctx, f, podName)
				if err != nil {
					return err
				}
				for _, cond := range pod.Status.Conditions {
					if cond.Type == v1.AllContainersRestarting && cond.Status == v1.ConditionTrue {
						return nil
					}
				}
				return fmt.Errorf("pod not marked for restart")
			}, 2*time.Minute, 1*time.Second).Should(gomega.Succeed())

			ginkgo.By("waiting for pod cleanup finish")
			gomega.Eventually(ctx, func() error {
				pod, err := getPodByName(ctx, f, podName)
				if err != nil {
					return err
				}
				for _, cond := range pod.Status.Conditions {
					if cond.Type == v1.AllContainersRestarting && cond.Status == v1.ConditionFalse {
						return nil
					}
				}
				return fmt.Errorf("pod cleanup not finished")
			}, 2*time.Minute, 1*time.Second).Should(gomega.Succeed())

			ginkgo.By("restarting kubelet")
			restartKubelet := mustStopKubelet(ctx, f)
			restartKubelet(ctx)

			ginkgo.By("waiting for containers to restart")
			gomega.Eventually(ctx, func() error {
				pod, err := getPodByName(ctx, f, podName)
				if err != nil {
					return err
				}
				restartedContainers := 0
				runningContainers := 0
				for _, c := range pod.Status.ContainerStatuses {
					if c.RestartCount > 0 {
						restartedContainers++
					}
					if c.State.Running != nil {
						runningContainers++
					}
				}
				if restartedContainers < containerCount+1 {
					return fmt.Errorf("not all containers have restarted")
				}
				if runningContainers < containerCount+1 {
					return fmt.Errorf("not all containers are running")
				}
				return nil
			}, 3*time.Minute, f.Timeouts.Poll).Should(gomega.Succeed())
		})
	})
