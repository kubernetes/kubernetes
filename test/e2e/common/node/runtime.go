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

package node

import (
	"context"
	"fmt"
	"os"
	"path"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/kubelet/images"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eregistry "k8s.io/kubernetes/test/e2e/framework/registry"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	gomegatypes "github.com/onsi/gomega/types"
)

var _ = SIGDescribe("Container Runtime", func() {
	f := framework.NewDefaultFramework("container-runtime")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged // custom registry pods need HostPorts

	ginkgo.Describe("blackbox test", func() {
		ginkgo.Context("when starting a container that exits", func() {

			/*
				Release: v1.13
				Testname: Container Runtime, Restart Policy, Pod Phases
				Description: If the restart policy is set to 'Always', Pod MUST be restarted when terminated, If restart policy is 'OnFailure', Pod MUST be started only if it is terminated with non-zero exit code. If the restart policy is 'Never', Pod MUST never be restarted. All these three test cases MUST verify the restart counts accordingly.
			*/
			framework.ConformanceIt("should run with the expected status", f.WithNodeConformance(), func(ctx context.Context) {
				restartCountVolumeName := "restart-count"
				restartCountVolumePath := "/restart-count"
				testContainer := v1.Container{
					Image: imageutils.GetE2EImage(imageutils.BusyBox),
					VolumeMounts: []v1.VolumeMount{
						{
							MountPath: restartCountVolumePath,
							Name:      restartCountVolumeName,
						},
					},
				}
				testVolumes := []v1.Volume{
					{
						Name: restartCountVolumeName,
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{Medium: v1.StorageMediumMemory},
						},
					},
				}
				testCases := []struct {
					Name          string
					RestartPolicy v1.RestartPolicy
					Phase         v1.PodPhase
					State         ContainerState
					RestartCount  int32
					Ready         bool
				}{
					{"terminate-cmd-rpa", v1.RestartPolicyAlways, v1.PodRunning, ContainerStateRunning, 2, true},
					{"terminate-cmd-rpof", v1.RestartPolicyOnFailure, v1.PodSucceeded, ContainerStateTerminated, 1, false},
					{"terminate-cmd-rpn", v1.RestartPolicyNever, v1.PodFailed, ContainerStateTerminated, 0, false},
				}
				for _, testCase := range testCases {

					// It failed at the 1st run, then succeeded at 2nd run, then run forever
					cmdScripts := `
f=%s
count=$(echo 'hello' >> $f ; wc -l $f | awk {'print $1'})
if [ $count -eq 1 ]; then
	exit 1
fi
if [ $count -eq 2 ]; then
	exit 0
fi
while true; do sleep 1; done
`
					tmpCmd := fmt.Sprintf(cmdScripts, path.Join(restartCountVolumePath, "restartCount"))
					testContainer.Name = testCase.Name
					testContainer.Command = []string{"sh", "-c", tmpCmd}
					terminateContainer := ConformanceContainer{
						PodClient:     e2epod.NewPodClient(f),
						Container:     testContainer,
						RestartPolicy: testCase.RestartPolicy,
						Volumes:       testVolumes,
					}
					terminateContainer.Create(ctx)
					ginkgo.DeferCleanup(framework.IgnoreNotFound(terminateContainer.Delete))

					ginkgo.By(fmt.Sprintf("Container '%s': should get the expected 'RestartCount'", testContainer.Name))
					gomega.Eventually(ctx, func() (int32, error) {
						status, err := terminateContainer.GetStatus(ctx)
						return status.RestartCount, err
					}, ContainerStatusRetryTimeout, ContainerStatusPollInterval).Should(gomega.Equal(testCase.RestartCount))

					ginkgo.By(fmt.Sprintf("Container '%s': should get the expected 'Phase'", testContainer.Name))
					gomega.Eventually(ctx, terminateContainer.GetPhase, ContainerStatusRetryTimeout, ContainerStatusPollInterval).Should(gomega.Equal(testCase.Phase))

					ginkgo.By(fmt.Sprintf("Container '%s': should get the expected 'Ready' condition", testContainer.Name))
					gomega.Eventually(ctx, func() (bool, error) {
						return terminateContainer.IsReady(ctx)
					}, ContainerStatusRetryTimeout, ContainerStatusPollInterval).Should(gomega.Equal(testCase.Ready))

					status, err := terminateContainer.GetStatus(ctx)
					framework.ExpectNoError(err)

					ginkgo.By(fmt.Sprintf("Container '%s': should get the expected 'State'", testContainer.Name))
					gomega.Expect(GetContainerState(status.State)).To(gomega.Equal(testCase.State))

					ginkgo.By(fmt.Sprintf("Container '%s': should be possible to delete", testContainer.Name))
					gomega.Expect(terminateContainer.Delete(ctx)).To(gomega.Succeed())
					gomega.Eventually(ctx, terminateContainer.Present, ContainerStatusRetryTimeout, ContainerStatusPollInterval).Should(gomega.BeFalseBecause("container '%s': should have been terminated by now.", testContainer.Name))
				}
			})
		})

		ginkgo.Context("on terminated container", func() {
			rootUser := int64(0)
			nonRootUser := int64(10000)
			adminUserName := "ContainerAdministrator"
			nonAdminUserName := "ContainerUser"

			// Create and then terminate the container under defined PodPhase to verify if termination message matches the expected output. Lastly delete the created container.
			matchTerminationMessage := func(ctx context.Context, container v1.Container, expectedPhase v1.PodPhase, expectedMsg gomegatypes.GomegaMatcher) {
				container.Name = "termination-message-container"
				c := ConformanceContainer{
					PodClient:     e2epod.NewPodClient(f),
					Container:     container,
					RestartPolicy: v1.RestartPolicyNever,
				}

				ginkgo.By("create the container")
				c.Create(ctx)
				ginkgo.DeferCleanup(framework.IgnoreNotFound(c.Delete))

				ginkgo.By(fmt.Sprintf("wait for the container to reach %s", expectedPhase))
				gomega.Eventually(ctx, c.GetPhase, ContainerStatusRetryTimeout, ContainerStatusPollInterval).Should(gomega.Equal(expectedPhase))

				ginkgo.By("get the container status")
				status, err := c.GetStatus(ctx)
				framework.ExpectNoError(err)

				ginkgo.By("the container should be terminated")
				gomega.Expect(GetContainerState(status.State)).To(gomega.Equal(ContainerStateTerminated))

				ginkgo.By("the termination message should be set")
				framework.Logf("Expected: %v to match Container's Termination Message: %v --", expectedMsg, status.State.Terminated.Message)
				gomega.Expect(status.State.Terminated.Message).Should(expectedMsg)

				ginkgo.By("delete the container")
				gomega.Expect(c.Delete(ctx)).To(gomega.Succeed())
			}

			f.It("should report termination message if TerminationMessagePath is set", f.WithNodeConformance(), func(ctx context.Context) {
				container := v1.Container{
					Image:                  imageutils.GetE2EImage(imageutils.BusyBox),
					Command:                []string{"/bin/sh", "-c"},
					Args:                   []string{"/bin/echo -n DONE > /dev/termination-log"},
					TerminationMessagePath: "/dev/termination-log",
					SecurityContext:        &v1.SecurityContext{},
				}
				if framework.NodeOSDistroIs("windows") {
					container.SecurityContext.WindowsOptions = &v1.WindowsSecurityContextOptions{RunAsUserName: &adminUserName}
				} else {
					container.SecurityContext.RunAsUser = &rootUser
				}
				matchTerminationMessage(ctx, container, v1.PodSucceeded, gomega.Equal("DONE"))
			})

			/*
				Release: v1.15
				Testname: Container Runtime, TerminationMessagePath, non-root user and non-default path
				Description: Create a pod with a container to run it as a non-root user with a custom TerminationMessagePath set. Pod redirects the output to the provided path successfully. When the container is terminated, the termination message MUST match the expected output logged in the provided custom path.
			*/
			framework.ConformanceIt("should report termination message if TerminationMessagePath is set as non-root user and at a non-default path", f.WithNodeConformance(), func(ctx context.Context) {
				container := v1.Container{
					Image:                  imageutils.GetE2EImage(imageutils.BusyBox),
					Command:                []string{"/bin/sh", "-c"},
					Args:                   []string{"/bin/echo -n DONE > /dev/termination-custom-log"},
					TerminationMessagePath: "/dev/termination-custom-log",
					SecurityContext:        &v1.SecurityContext{},
				}
				if framework.NodeOSDistroIs("windows") {
					container.SecurityContext.WindowsOptions = &v1.WindowsSecurityContextOptions{RunAsUserName: &nonAdminUserName}
				} else {
					container.SecurityContext.RunAsUser = &nonRootUser
				}
				matchTerminationMessage(ctx, container, v1.PodSucceeded, gomega.Equal("DONE"))
			})

			/*
				Release: v1.15
				Testname: Container Runtime, TerminationMessage, from container's log output of failing container
				Description: Create a pod with an container. Container's output is recorded in log and container exits with an error. When container is terminated, termination message MUST match the expected output recorded from container's log.
			*/
			framework.ConformanceIt("should report termination message from log output if TerminationMessagePolicy FallbackToLogsOnError is set", f.WithNodeConformance(), func(ctx context.Context) {
				container := v1.Container{
					Image:                    imageutils.GetE2EImage(imageutils.BusyBox),
					Command:                  []string{"/bin/sh", "-c"},
					Args:                     []string{"/bin/echo -n DONE; /bin/false"},
					TerminationMessagePath:   "/dev/termination-log",
					TerminationMessagePolicy: v1.TerminationMessageFallbackToLogsOnError,
				}
				matchTerminationMessage(ctx, container, v1.PodFailed, gomega.Equal("DONE"))
			})

			/*
				Release: v1.15
				Testname: Container Runtime, TerminationMessage, from log output of succeeding container
				Description: Create a pod with an container. Container's output is recorded in log and container exits successfully without an error. When container is terminated, terminationMessage MUST have no content as container succeed.
			*/
			framework.ConformanceIt("should report termination message as empty when pod succeeds and TerminationMessagePolicy FallbackToLogsOnError is set", f.WithNodeConformance(), func(ctx context.Context) {
				container := v1.Container{
					Image:                    imageutils.GetE2EImage(imageutils.BusyBox),
					Command:                  []string{"/bin/sh", "-c"},
					Args:                     []string{"/bin/echo -n DONE; /bin/true"},
					TerminationMessagePath:   "/dev/termination-log",
					TerminationMessagePolicy: v1.TerminationMessageFallbackToLogsOnError,
				}
				matchTerminationMessage(ctx, container, v1.PodSucceeded, gomega.Equal(""))
			})

			/*
				Release: v1.15
				Testname: Container Runtime, TerminationMessage, from file of succeeding container
				Description: Create a pod with an container. Container's output is recorded in a file and the container exits successfully without an error. When container is terminated, terminationMessage MUST match with the content from file.
			*/
			framework.ConformanceIt("should report termination message from file when pod succeeds and TerminationMessagePolicy FallbackToLogsOnError is set", f.WithNodeConformance(), func(ctx context.Context) {
				container := v1.Container{
					Image:                    imageutils.GetE2EImage(imageutils.BusyBox),
					Command:                  []string{"/bin/sh", "-c"},
					Args:                     []string{"/bin/echo -n OK > /dev/termination-log; /bin/echo DONE; /bin/true"},
					TerminationMessagePath:   "/dev/termination-log",
					TerminationMessagePolicy: v1.TerminationMessageFallbackToLogsOnError,
				}
				matchTerminationMessage(ctx, container, v1.PodSucceeded, gomega.Equal("OK"))
			})
		})

		framework.Context("when running a container with a new image", framework.WithSerial(), func() {
			var registryAddress string
			var registryNodeNames []string
			var pullSecret *v1.Secret
			ginkgo.BeforeEach(func(ctx context.Context) {
				var err error

				registryAddress, registryNodeNames, err = e2eregistry.SetupRegistry(ctx, f, true)
				framework.ExpectNoError(err)
				gomega.Expect(registryNodeNames).ToNot(gomega.BeEmpty())
				// we need to wait for the registry to be removed and so we need to delete the whole NS ourselves
				ginkgo.DeferCleanup(func(ctx context.Context) {
					f.DeleteNamespace(ctx, f.Namespace.Name)
				})

				secret := e2eregistry.User1DockerSecret(registryAddress)
				secret.Name = "image-pull-secret-" + string(uuid.NewUUID())
				// we might be told to use a different docker config JSON.
				if framework.TestContext.DockerConfigFile != "" {
					contents, err := os.ReadFile(framework.TestContext.DockerConfigFile)
					framework.ExpectNoError(err)
					secret.Data[v1.DockerConfigJsonKey] = contents
				}
				ginkgo.By("create image pull secret")
				pullSecret, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(ctx, secret, metav1.CreateOptions{})
				framework.ExpectNoError(err)
			})

			f.It("should not be able to pull image from invalid registry", f.WithNodeConformance(), func(ctx context.Context) {
				image := imageutils.GetE2EImage(imageutils.InvalidRegistryImage)
				_ = ImagePullTest(ctx, f, image, v1.PullAlways, nil, registryNodeNames[0], v1.PodPending, true)
			})

			f.It("should be able to pull image", f.WithNodeConformance(), func(ctx context.Context) {
				image := imageutils.GetE2EImage(imageutils.Agnhost)
				_ = ImagePullTest(ctx, f, image, v1.PullAlways, nil, registryNodeNames[0], v1.PodRunning, false)
			})

			f.It("should not be able to pull from private registry without secret", f.WithNodeConformance(), func(ctx context.Context) {
				image := registryAddress + "/pause:testing"
				_ = ImagePullTest(ctx, f, image, v1.PullAlways, nil, registryNodeNames[0], v1.PodPending, true)
			})

			f.It("should be able to pull from private registry with secret", f.WithNodeConformance(), func(ctx context.Context) {
				image := registryAddress + "/pause:testing"
				_ = ImagePullTest(ctx, f, image, v1.PullAlways, pullSecret, registryNodeNames[0], v1.PodRunning, false)
			})
		})
	})
})

// Images used by imagePullTest are not added to NodePrePullImageList because this test is
// specifically testing image pulling behavior. The containers use ImagePullPolicy values other
// than PullNever, so the framework allows them to be pulled during the test.
func ImagePullTest(ctx context.Context, f *framework.Framework, image string, imagePullPolicy v1.PullPolicy, pullSecret *v1.Secret, nodeName string, expectedPhase v1.PodPhase, expectedPullStatus bool) *v1.Pod {
	container := ConformanceContainer{
		PodClient: e2epod.NewPodClient(f),
		NodeName:  nodeName,
		Container: v1.Container{
			Name:            "image-pull-test",
			Image:           image,
			ImagePullPolicy: imagePullPolicy,
		},
		RestartPolicy: v1.RestartPolicyNever,
	}
	if pullSecret != nil {
		container.ImagePullSecrets = []string{pullSecret.Name}
	}
	// checkContainerStatus checks whether the container status matches expectation.
	checkContainerStatus := func(ctx context.Context) error {
		status, err := container.GetStatus(ctx)
		if err != nil {
			return fmt.Errorf("failed to get container status: %w", err)
		}
		// We need to check container state first. The default pod status is pending, If we check pod phase first,
		// and the expected pod phase is Pending, the container status may not even show up when we check it.
		// Check container state
		if !expectedPullStatus {
			if status.State.Running == nil {
				return fmt.Errorf("expected container state: Running, got: %q",
					GetContainerState(status.State))
			}
		}
		if expectedPullStatus {
			if status.State.Waiting == nil {
				gomega.Expect(status.State.Running).To(gomega.BeNil())
				return fmt.Errorf("expected container state: Waiting, got: %q",
					GetContainerState(status.State))
			}
			reason := status.State.Waiting.Reason
			if reason != images.ErrImagePull.Error() &&
				reason != images.ErrImagePullBackOff.Error() &&
				reason != images.ErrImageNeverPull.Error() {
				return fmt.Errorf("unexpected waiting reason: %q", reason)
			}
		}
		// Check pod phase
		phase, err := container.GetPhase(ctx)
		if err != nil {
			return fmt.Errorf("failed to get pod phase: %w", err)
		}
		if phase != expectedPhase {
			return fmt.Errorf("expected pod phase: %q, got: %q", expectedPhase, phase)
		}
		return nil
	}

	ginkgo.By("create the container")
	testPod := container.Create(ctx)
	ginkgo.DeferCleanup(func(ctx context.Context) {
		ginkgo.By("delete the conformance container")
		if err := container.Delete(ctx); err != nil {
			framework.Logf("error deleting a conformance container: %v", err)
		}
	})

	ginkgo.By("check the container status")
	var latestErr error
	err := wait.PollUntilContextCancel(ctx, ContainerStatusPollInterval, true, func(ctx context.Context) (bool, error) {
		if latestErr = checkContainerStatus(ctx); latestErr != nil {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		framework.Failf("Failed to read container status: %v; last observed error from wait loop: %v", err, latestErr)
	}

	return testPod
}
