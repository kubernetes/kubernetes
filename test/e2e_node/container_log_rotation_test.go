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
	"os"
	"path/filepath"
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
			r, _, err := getCRIClient(ctx)
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

var _ = SIGDescribe("ContainerLogRedundantCleanup", framework.WithSlow(), framework.WithSerial(), framework.WithDisruptive(), func() {
	f := framework.NewDefaultFramework("container-log-redundant-cleanup-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("when a container restarts many times", func() {
		const (
			redundantTestMaxFiles = 3
		)
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.ContainerLogMaxFiles = redundantTestMaxFiles
			initialConfig.ContainerLogMaxSize = testContainerLogMaxSize
		})

		ginkgo.It("should remove redundant logs from previous restarts", func(ctx context.Context) {
			ginkgo.By("create a pod that crashes and restarts multiple times")
			// The container exits after printing some output, causing restarts.
			// With RestartPolicyAlways, kubelet will restart it, creating 0.log, 1.log, 2.log, etc.
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-redundant-log-cleanup",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyAlways,
					Containers: []v1.Container{
						{
							Name:  "log-container",
							Image: busyboxImage,
							Command: []string{
								"sh",
								"-c",
								// Generate enough logs to trigger rotation check, then exit to cause restart.
								"for i in $(seq 1 100); do echo redundant log test line $i; done; exit 0",
							},
						},
					},
				},
			}
			logPod := e2epod.NewPodClient(f).Create(ctx, pod)
			ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, logPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

			ginkgo.By("wait for the container to restart enough times to exceed MaxFiles")
			// With exponential restart backoff (10s, 20s, 40s, 80s, ...),
			// MaxFiles+1 restarts (backoff sum ~70s) is sufficient to trigger
			// redundant log cleanup while staying well within the 3m timeout.
			targetRestarts := int32(redundantTestMaxFiles + 1)
			gomega.Eventually(ctx, func() (int32, error) {
				p, err := e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
				if err != nil {
					return 0, err
				}
				if len(p.Status.ContainerStatuses) == 0 {
					return 0, nil
				}
				return p.Status.ContainerStatuses[0].RestartCount, nil
			}, rotationEventuallyTimeout, rotationPollInterval).Should(gomega.BeNumerically(">=", targetRestarts),
				"container should restart enough times to generate redundant logs")

			ginkgo.By("get container log directory")
			p, err := e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(p.Status.ContainerStatuses).To(gomega.HaveLen(1))
			id := kubecontainer.ParseContainerID(p.Status.ContainerStatuses[0].ContainerID).ID
			r, _, err := getCRIClient(ctx)
			framework.ExpectNoError(err)
			resp, err := r.ContainerStatus(ctx, id, false)
			framework.ExpectNoError(err)
			logPath := resp.GetStatus().GetLogPath()
			// The container log directory is the parent of the log file (e.g., /var/log/pods/.../log-container/)
			logDir := filepath.Dir(logPath)

			ginkgo.By("verify redundant logs have been cleaned up")
			gomega.Eventually(ctx, func() (int, error) {
				entries, err := os.ReadDir(logDir)
				if err != nil {
					return 0, err
				}
				return len(entries), nil
			}, rotationEventuallyTimeout, rotationPollInterval).Should(gomega.BeNumerically("<=", redundantTestMaxFiles),
				"redundant log files from old restarts should be removed, keeping at most MaxFiles")
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
				r, _, err := getCRIClient(ctx)
				framework.ExpectNoError(err, "should connect to CRI and obtain runtime service clients and image service client")
				resp, err := r.ContainerStatus(ctx, id, false)
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
