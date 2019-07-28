/*
Copyright 2017 The Kubernetes Authors.

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

package common

import (
	"fmt"
	"strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	"k8s.io/utils/pointer"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Security Context", func() {
	f := framework.NewDefaultFramework("security-context-test")
	var podClient *framework.PodClient
	ginkgo.BeforeEach(func() {
		podClient = f.PodClient()
	})

	ginkgo.Context("When creating a container with runAsUser", func() {
		makeUserPod := func(podName, image string, command []string, userid int64) *v1.Pod {
			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Image:   image,
							Name:    podName,
							Command: command,
							SecurityContext: &v1.SecurityContext{
								RunAsUser: &userid,
							},
						},
					},
				},
			}
		}
		createAndWaitUserPod := func(userid int64) {
			podName := fmt.Sprintf("busybox-user-%d-%s", userid, uuid.NewUUID())
			podClient.Create(makeUserPod(podName,
				framework.BusyBoxImage,
				[]string{"sh", "-c", fmt.Sprintf("test $(id -u) -eq %d", userid)},
				userid,
			))

			podClient.WaitForSuccess(podName, framework.PodStartTimeout)
		}

		/*
			Release : v1.15
			Testname: Security Context, runAsUser=65534
			Description: Container is created with runAsUser option by passing uid 65534 to run as unpriviledged user. Pod MUST be in Succeeded phase.
			[LinuxOnly]: This test is marked as LinuxOnly since Windows does not support running as UID / GID.
		*/
		framework.ConformanceIt("should run the container with uid 65534 [LinuxOnly] [NodeConformance]", func() {
			createAndWaitUserPod(65534)
		})

		/*
			Release : v1.15
			Testname: Security Context, runAsUser=0
			Description: Container is created with runAsUser option by passing uid 0 to run as root priviledged user. Pod MUST be in Succeeded phase.
			This e2e can not be promoted to Conformance because a Conformant platform may not allow to run containers with 'uid 0' or running privileged operations.
			[LinuxOnly]: This test is marked as LinuxOnly since Windows does not support running as UID / GID.
		*/
		ginkgo.It("should run the container with uid 0 [LinuxOnly] [NodeConformance]", func() {
			createAndWaitUserPod(0)
		})
	})

	ginkgo.Context("When creating a container with runAsNonRoot", func() {
		rootImage := imageutils.GetE2EImage(imageutils.BusyBox)
		nonRootImage := imageutils.GetE2EImage(imageutils.NonRoot)
		makeNonRootPod := func(podName, image string, userid *int64) *v1.Pod {
			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Image:   image,
							Name:    podName,
							Command: []string{"id", "-u"}, // Print UID and exit
							SecurityContext: &v1.SecurityContext{
								RunAsNonRoot: pointer.BoolPtr(true),
								RunAsUser:    userid,
							},
						},
					},
				},
			}
		}

		ginkgo.It("should run with an explicit non-root user ID", func() {
			name := "explicit-nonroot-uid"
			pod := makeNonRootPod(name, rootImage, pointer.Int64Ptr(1234))
			pod = podClient.Create(pod)

			podClient.WaitForSuccess(name, framework.PodStartTimeout)
			framework.ExpectNoError(podClient.MatchContainerOutput(name, name, "1234"))
		})
		ginkgo.It("should not run with an explicit root user ID", func() {
			name := "explicit-root-uid"
			pod := makeNonRootPod(name, nonRootImage, pointer.Int64Ptr(0))
			pod = podClient.Create(pod)

			ev, err := podClient.WaitForErrorEventOrSuccess(pod)
			framework.ExpectNoError(err)
			gomega.Expect(ev).NotTo(gomega.BeNil())
			framework.ExpectEqual(ev.Reason, events.FailedToCreateContainer)
		})
		ginkgo.It("should run with an image specified user ID", func() {
			name := "implicit-nonroot-uid"
			pod := makeNonRootPod(name, nonRootImage, nil)
			pod = podClient.Create(pod)

			podClient.WaitForSuccess(name, framework.PodStartTimeout)
			framework.ExpectNoError(podClient.MatchContainerOutput(name, name, "1234"))
		})
		ginkgo.It("should not run without a specified user ID", func() {
			name := "implicit-root-uid"
			pod := makeNonRootPod(name, rootImage, nil)
			pod = podClient.Create(pod)

			ev, err := podClient.WaitForErrorEventOrSuccess(pod)
			framework.ExpectNoError(err)
			gomega.Expect(ev).NotTo(gomega.BeNil())
			framework.ExpectEqual(ev.Reason, events.FailedToCreateContainer)
		})
	})

	ginkgo.Context("When creating a pod with readOnlyRootFilesystem", func() {
		makeUserPod := func(podName, image string, command []string, readOnlyRootFilesystem bool) *v1.Pod {
			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Image:   image,
							Name:    podName,
							Command: command,
							SecurityContext: &v1.SecurityContext{
								ReadOnlyRootFilesystem: &readOnlyRootFilesystem,
							},
						},
					},
				},
			}
		}
		createAndWaitUserPod := func(readOnlyRootFilesystem bool) string {
			podName := fmt.Sprintf("busybox-readonly-%v-%s", readOnlyRootFilesystem, uuid.NewUUID())
			podClient.Create(makeUserPod(podName,
				framework.BusyBoxImage,
				[]string{"sh", "-c", "touch checkfile"},
				readOnlyRootFilesystem,
			))

			if readOnlyRootFilesystem {
				podClient.WaitForFailure(podName, framework.PodStartTimeout)
			} else {
				podClient.WaitForSuccess(podName, framework.PodStartTimeout)
			}

			return podName
		}

		/*
			Release : v1.15
			Testname: Security Context, readOnlyRootFilesystem=true.
			Description: Container is configured to run with readOnlyRootFilesystem to true which will force containers to run with a read only root file system.
			Write operation MUST NOT be allowed and Pod MUST be in Failed state.
			At this moment we are not considering this test for Conformance due to use of SecurityContext.
			[LinuxOnly]: This test is marked as LinuxOnly since Windows does not support creating containers with read-only access.
		*/
		ginkgo.It("should run the container with readonly rootfs when readOnlyRootFilesystem=true [LinuxOnly] [NodeConformance]", func() {
			createAndWaitUserPod(true)
		})

		/*
			Release : v1.15
			Testname: Security Context, readOnlyRootFilesystem=false.
			Description: Container is configured to run with readOnlyRootFilesystem to false.
			Write operation MUST be allowed and Pod MUST be in Succeeded state.
		*/
		framework.ConformanceIt("should run the container with writable rootfs when readOnlyRootFilesystem=false [NodeConformance]", func() {
			createAndWaitUserPod(false)
		})
	})

	ginkgo.Context("When creating a pod with privileged", func() {
		makeUserPod := func(podName, image string, command []string, privileged bool) *v1.Pod {
			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Image:   image,
							Name:    podName,
							Command: command,
							SecurityContext: &v1.SecurityContext{
								Privileged: &privileged,
							},
						},
					},
				},
			}
		}
		createAndWaitUserPod := func(privileged bool) string {
			podName := fmt.Sprintf("busybox-privileged-%v-%s", privileged, uuid.NewUUID())
			podClient.Create(makeUserPod(podName,
				framework.BusyBoxImage,
				[]string{"sh", "-c", "ip link add dummy0 type dummy || true"},
				privileged,
			))
			podClient.WaitForSuccess(podName, framework.PodStartTimeout)
			return podName
		}
		/*
			Release : v1.15
			Testname: Security Context, privileged=false.
			Description: Create a container to run in unprivileged mode by setting pod's SecurityContext Privileged option as false. Pod MUST be in Succeeded phase.
			[LinuxOnly]: This test is marked as LinuxOnly since it runs a Linux-specific command.
		*/
		framework.ConformanceIt("should run the container as unprivileged when false [LinuxOnly] [NodeConformance]", func() {
			podName := createAndWaitUserPod(false)
			logs, err := e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, podName, podName)
			if err != nil {
				e2elog.Failf("GetPodLogs for pod %q failed: %v", podName, err)
			}

			e2elog.Logf("Got logs for pod %q: %q", podName, logs)
			if !strings.Contains(logs, "Operation not permitted") {
				e2elog.Failf("unprivileged container shouldn't be able to create dummy device")
			}
		})
	})

	ginkgo.Context("when creating containers with AllowPrivilegeEscalation", func() {
		makeAllowPrivilegeEscalationPod := func(podName string, allowPrivilegeEscalation *bool, uid int64) *v1.Pod {
			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Image: imageutils.GetE2EImage(imageutils.Nonewprivs),
							Name:  podName,
							SecurityContext: &v1.SecurityContext{
								AllowPrivilegeEscalation: allowPrivilegeEscalation,
								RunAsUser:                &uid,
							},
						},
					},
				},
			}
		}
		createAndMatchOutput := func(podName, output string, allowPrivilegeEscalation *bool, uid int64) error {
			podClient.Create(makeAllowPrivilegeEscalationPod(podName,
				allowPrivilegeEscalation,
				uid,
			))
			podClient.WaitForSuccess(podName, framework.PodStartTimeout)
			return podClient.MatchContainerOutput(podName, podName, output)
		}

		/*
			Release : v1.15
			Testname: Security Context, allowPrivilegeEscalation unset, uid != 0.
			Description: Configuring the allowPrivilegeEscalation unset, allows the privilege escalation operation.
			A container is configured with allowPrivilegeEscalation not specified (nil) and a given uid which is not 0.
			When the container is run, container's output MUST match with expected output verifying container ran with uid=0.
			This e2e Can not be promoted to Conformance as it is Container Runtime dependent and not all conformant platforms will require this behavior.
			[LinuxOnly]: This test is marked LinuxOnly since Windows does not support running as UID / GID, or privilege escalation.
		*/
		ginkgo.It("should allow privilege escalation when not explicitly set and uid != 0 [LinuxOnly] [NodeConformance]", func() {
			podName := "alpine-nnp-nil-" + string(uuid.NewUUID())
			if err := createAndMatchOutput(podName, "Effective uid: 0", nil, 1000); err != nil {
				e2elog.Failf("Match output for pod %q failed: %v", podName, err)
			}
		})

		/*
			Release : v1.15
			Testname: Security Context, allowPrivilegeEscalation=false.
			Description: Configuring the allowPrivilegeEscalation to false, does not allow the privilege escalation operation.
			A container is configured with allowPrivilegeEscalation=false and a given uid (1000) which is not 0.
			When the container is run, container's output MUST match with expected output verifying container ran with given uid i.e. uid=1000.
			[LinuxOnly]: This test is marked LinuxOnly since Windows does not support running as UID / GID, or privilege escalation.
		*/
		framework.ConformanceIt("should not allow privilege escalation when false [LinuxOnly] [NodeConformance]", func() {
			podName := "alpine-nnp-false-" + string(uuid.NewUUID())
			apeFalse := false
			if err := createAndMatchOutput(podName, "Effective uid: 1000", &apeFalse, 1000); err != nil {
				e2elog.Failf("Match output for pod %q failed: %v", podName, err)
			}
		})

		/*
			Release : v1.15
			Testname: Security Context, allowPrivilegeEscalation=true.
			Description: Configuring the allowPrivilegeEscalation to true, allows the privilege escalation operation.
			A container is configured with allowPrivilegeEscalation=true and a given uid (1000) which is not 0.
			When the container is run, container's output MUST match with expected output verifying container ran with uid=0 (making use of the privilege escalation).
			This e2e Can not be promoted to Conformance as it is Container Runtime dependent and runtime may not allow to run.
			[LinuxOnly]: This test is marked LinuxOnly since Windows does not support running as UID / GID.
		*/
		ginkgo.It("should allow privilege escalation when true [LinuxOnly] [NodeConformance]", func() {
			podName := "alpine-nnp-true-" + string(uuid.NewUUID())
			apeTrue := true
			if err := createAndMatchOutput(podName, "Effective uid: 0", &apeTrue, 1000); err != nil {
				e2elog.Failf("Match output for pod %q failed: %v", podName, err)
			}
		})
	})
})
