/*
Copyright 2018 The Kubernetes Authors.

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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubelogs "k8s.io/kubernetes/pkg/kubelet/logs"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	testContainerLogMaxFiles        = 3
	testContainerLogMaxSize         = "40Ki"
	testContainerLogMaxWorkers      = 2
	testContainerLogMonitorInterval = 3 * time.Second
	rotationPollInterval            = 5 * time.Second
	rotationEventuallyTimeout       = 3 * time.Minute
	rotationConsistentlyTimeout     = 2 * time.Minute
)

var _ = SIGDescribe("ContainerLogRotation", framework.WithSlow(), framework.WithSerial(), framework.WithDisruptive(), func() {
	f := framework.NewDefaultFramework("container-log-rotation-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.Context("when a container generates a lot of log", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.ContainerLogMaxFiles = testContainerLogMaxFiles
			initialConfig.ContainerLogMaxSize = testContainerLogMaxSize
		})

		var logRotationPod *v1.Pod
		ginkgo.BeforeEach(func(ctx context.Context) {
			ginkgo.By("create log container")
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-container-log-rotation",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Name:  "log-container",
							Image: busyboxImage,
							Command: []string{
								"sh",
								"-c",
								// ~12Kb/s. Exceeding 40Kb in 4 seconds. Log rotation period is 10 seconds.
								"while true; do echo hello world; sleep 0.001; done;",
							},
						},
					},
				},
			}
			logRotationPod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, logRotationPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
		})

		ginkgo.It("should be rotated and limited to a fixed amount of files", func(ctx context.Context) {

			ginkgo.By("get container log path")
			gomega.Expect(logRotationPod.Status.ContainerStatuses).To(gomega.HaveLen(1), "log rotation pod should have one container")
			id := kubecontainer.ParseContainerID(logRotationPod.Status.ContainerStatuses[0].ContainerID).ID
			r, _, err := getCRIClient()
			framework.ExpectNoError(err, "should connect to CRI and obtain runtime service clients and image service client")
			resp, err := r.ContainerStatus(context.Background(), id, false)
			framework.ExpectNoError(err)
			logPath := resp.GetStatus().GetLogPath()
			ginkgo.By("wait for container log being rotated to max file limit")
			gomega.Eventually(ctx, func() (int, error) {
				logs, err := kubelogs.GetAllLogs(logPath)
				if err != nil {
					return 0, err
				}
				return len(logs), nil
			}, rotationEventuallyTimeout, rotationPollInterval).Should(gomega.Equal(testContainerLogMaxFiles), "should eventually rotate to max file limit")
			ginkgo.By("make sure container log number won't exceed max file limit")
			gomega.Consistently(ctx, func() (int, error) {
				logs, err := kubelogs.GetAllLogs(logPath)
				if err != nil {
					return 0, err
				}
				return len(logs), nil
			}, rotationConsistentlyTimeout, rotationPollInterval).Should(gomega.BeNumerically("<=", testContainerLogMaxFiles), "should never exceed max file limit")
		})
	})
})

var _ = SIGDescribe("ContainerLogRotationWithMultipleWorkers", framework.WithSlow(), framework.WithSerial(), framework.WithDisruptive(), func() {
	f := framework.NewDefaultFramework("container-log-rotation-test-multi-worker")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.Context("when a container generates a lot of logs", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.ContainerLogMaxFiles = testContainerLogMaxFiles
			initialConfig.ContainerLogMaxSize = testContainerLogMaxSize
			initialConfig.ContainerLogMaxWorkers = testContainerLogMaxWorkers
			initialConfig.ContainerLogMonitorInterval = metav1.Duration{Duration: testContainerLogMonitorInterval}
		})

		var logRotationPods []*v1.Pod
		ginkgo.BeforeEach(func(ctx context.Context) {
			ginkgo.By("create log container 1")
			for _, name := range []string{"test-container-log-rotation", "test-container-log-rotation-1"} {
				pod := &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name: name,
					},
					Spec: v1.PodSpec{
						RestartPolicy: v1.RestartPolicyNever,
						Containers: []v1.Container{
							{
								Name:  "log-container",
								Image: busyboxImage,
								Command: []string{
									"sh",
									"-c",
									// ~12Kb/s. Exceeding 40Kb in 4 seconds. Log rotation period is 10 seconds.
									"while true; do echo hello world; sleep 0.001; done;",
								},
							},
						},
					},
				}
				logRotationPod := e2epod.NewPodClient(f).CreateSync(ctx, pod)
				logRotationPods = append(logRotationPods, logRotationPod)
				ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, logRotationPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
			}
		})

		ginkgo.It("should be rotated and limited to a fixed amount of files", func(ctx context.Context) {
			ginkgo.By("get container log path")
			var logPaths []string
			for _, pod := range logRotationPods {
				gomega.Expect(pod.Status.ContainerStatuses).To(gomega.HaveLen(1), "log rotation pod should have one container")
				id := kubecontainer.ParseContainerID(pod.Status.ContainerStatuses[0].ContainerID).ID
				r, _, err := getCRIClient()
				framework.ExpectNoError(err, "should connect to CRI and obtain runtime service clients and image service client")
				resp, err := r.ContainerStatus(context.Background(), id, false)
				framework.ExpectNoError(err)
				logPaths = append(logPaths, resp.GetStatus().GetLogPath())
			}

			ginkgo.By("wait for container log being rotated to max file limit")
			gomega.Eventually(ctx, func() (int, error) {
				var logFiles []string
				for _, logPath := range logPaths {
					logs, err := kubelogs.GetAllLogs(logPath)
					if err != nil {
						return 0, err
					}
					logFiles = append(logFiles, logs...)
				}
				return len(logFiles), nil
			}, rotationEventuallyTimeout, rotationPollInterval).Should(gomega.Equal(testContainerLogMaxFiles*2), "should eventually rotate to max file limit")
			ginkgo.By("make sure container log number won't exceed max file limit")

			gomega.Consistently(ctx, func() (int, error) {
				var logFiles []string
				for _, logPath := range logPaths {
					logs, err := kubelogs.GetAllLogs(logPath)
					if err != nil {
						return 0, err
					}
					logFiles = append(logFiles, logs...)
				}
				return len(logFiles), nil
			}, rotationConsistentlyTimeout, rotationPollInterval).Should(gomega.BeNumerically("<=", testContainerLogMaxFiles*2), "should never exceed max file limit")
		})
	})
})
