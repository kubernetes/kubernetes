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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/kubelet"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

const (
	logString        = "This is the expected log content of this node e2e test"
	logContainerName = "logger"
)

var _ = SIGDescribe("ContainerLogPath", framework.WithNodeConformance(), func() {
	f := framework.NewDefaultFramework("kubelet-container-log-path")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var podClient *e2epod.PodClient

	ginkgo.Describe("Pod with a container", func() {
		ginkgo.Context("printed log to stdout", func() {
			makeLogPod := func(podName, log string) *v1.Pod {
				return &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name: podName,
					},
					Spec: v1.PodSpec{
						// this pod is expected to exit successfully
						RestartPolicy: v1.RestartPolicyNever,
						Containers: []v1.Container{
							{
								Image:   busyboxImage,
								Name:    logContainerName,
								Command: []string{"sh", "-c", "echo " + log},
							},
						},
					},
				}
			}

			makeLogCheckPod := func(podName, log, expectedLogPath string) *v1.Pod {
				hostPathType := new(v1.HostPathType)
				*hostPathType = v1.HostPathType(string(v1.HostPathFileOrCreate))

				return &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name: podName,
					},
					Spec: v1.PodSpec{
						// this pod is expected to exit successfully
						RestartPolicy: v1.RestartPolicyNever,
						Containers: []v1.Container{
							{
								Image: busyboxImage,
								SecurityContext: &v1.SecurityContext{
									SELinuxOptions: &v1.SELinuxOptions{
										Type: "container_logreader_t",
									},
								},
								Name: podName,
								// If we find expected log file and contains right content, exit 0
								// else, keep checking until test timeout
								Command: []string{"sh", "-c", "while true; do if [ -e " + expectedLogPath + " ] && grep -q " + log + " " + expectedLogPath + "; then exit 0; fi; sleep 1; done"},
								VolumeMounts: []v1.VolumeMount{
									{
										Name: "logdir",
										// mount ContainerLogsDir to the same path in container
										MountPath: expectedLogPath,
										ReadOnly:  true,
									},
								},
							},
						},
						Volumes: []v1.Volume{
							{
								Name: "logdir",
								VolumeSource: v1.VolumeSource{
									HostPath: &v1.HostPathVolumeSource{
										Path: expectedLogPath,
										Type: hostPathType,
									},
								},
							},
						},
					},
				}
			}

			createAndWaitPod := func(ctx context.Context, pod *v1.Pod) error {
				podClient.Create(ctx, pod)
				return e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			}

			var logPodName string
			ginkgo.BeforeEach(func(ctx context.Context) {
				podClient = e2epod.NewPodClient(f)
				logPodName = "log-pod-" + string(uuid.NewUUID())
				err := createAndWaitPod(ctx, makeLogPod(logPodName, logString))
				framework.ExpectNoError(err, "Failed waiting for pod: %s to enter success state", logPodName)
			})
			ginkgo.It("should print log to correct log path", func(ctx context.Context) {

				logDir := kubelet.ContainerLogsDir

				// get containerID from created Pod
				createdLogPod, err := podClient.Get(ctx, logPodName, metav1.GetOptions{})
				logContainerID := kubecontainer.ParseContainerID(createdLogPod.Status.ContainerStatuses[0].ContainerID)
				framework.ExpectNoError(err, "Failed to get pod: %s", logPodName)

				// build log file path
				expectedlogFile := logDir + "/" + logPodName + "_" + f.Namespace.Name + "_" + logContainerName + "-" + logContainerID.ID + ".log"

				logCheckPodName := "log-check-" + string(uuid.NewUUID())
				err = createAndWaitPod(ctx, makeLogCheckPod(logCheckPodName, logString, expectedlogFile))
				framework.ExpectNoError(err, "Failed waiting for pod: %s to enter success state", logCheckPodName)
			})

			ginkgo.It("should print log to correct cri log path", func(ctx context.Context) {

				logCRIDir := "/var/log/pods"

				// get podID from created Pod
				createdLogPod, err := podClient.Get(ctx, logPodName, metav1.GetOptions{})
				framework.ExpectNoError(err, "Failed to get pod: %s", logPodName)
				podNs := createdLogPod.Namespace
				podName := createdLogPod.Name
				podID := string(createdLogPod.UID)

				// build log cri file path
				expectedCRILogFile := logCRIDir + "/" + podNs + "_" + podName + "_" + podID + "/" + logContainerName + "/0.log"

				logCRICheckPodName := "log-cri-check-" + string(uuid.NewUUID())
				err = createAndWaitPod(ctx, makeLogCheckPod(logCRICheckPodName, logString, expectedCRILogFile))
				framework.ExpectNoError(err, "Failed waiting for pod: %s to enter success state", logCRICheckPodName)
			})
		})
	})
})
