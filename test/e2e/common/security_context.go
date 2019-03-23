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

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	. "github.com/onsi/ginkgo"
)

var _ = framework.KubeDescribe("Security Context", func() {
	f := framework.NewDefaultFramework("security-context-test")
	var podClient *framework.PodClient
	BeforeEach(func() {
		podClient = f.PodClient()
	})

	Context("When creating a container with runAsUser", func() {
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
		  Release : v1.12
		  Testname: Security Context: runAsUser (id:65534)
		  Description: Container created with runAsUser option, passing an id (id:65534) uses that
		  given id when running the container.
		  This test is marked LinuxOnly since Windows does not support running as UID / GID.
		*/
		It("should run the container with uid 65534 [LinuxOnly] [NodeConformance]", func() {
			createAndWaitUserPod(65534)
		})

		/*
		  Release : v1.12
		  Testname: Security Context: runAsUser (id:0)
		  Description: Container created with runAsUser option, passing an id (id:0) uses that
		  given id when running the container.
		  This test is marked LinuxOnly since Windows does not support running as UID / GID.
		*/
		It("should run the container with uid 0 [LinuxOnly] [NodeConformance]", func() {
			createAndWaitUserPod(0)
		})
	})

	Context("When creating a pod with readOnlyRootFilesystem", func() {
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
		  Release : v1.12
		  Testname: Security Context: readOnlyRootFilesystem=true.
		  Description: when a container has configured readOnlyRootFilesystem to true, write operations are not allowed.
		  This test is marked LinuxOnly since Windows does not support creating containers with read-only access.
		*/
		It("should run the container with readonly rootfs when readOnlyRootFilesystem=true [LinuxOnly] [NodeConformance]", func() {
			createAndWaitUserPod(true)
		})

		/*
		  Release : v1.12
		  Testname: Security Context: readOnlyRootFilesystem=false.
		  Description: when a container has configured readOnlyRootFilesystem to false, write operations are allowed.
		*/
		It("should run the container with writable rootfs when readOnlyRootFilesystem=false [NodeConformance]", func() {
			createAndWaitUserPod(false)
		})
	})

	Context("When creating a pod with privileged", func() {
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

		It("should run the container as unprivileged when false [LinuxOnly] [NodeConformance]", func() {
			// This test is marked LinuxOnly since it runs a Linux-specific command, and Windows does not support Windows escalation.
			podName := createAndWaitUserPod(false)
			logs, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, podName, podName)
			if err != nil {
				framework.Failf("GetPodLogs for pod %q failed: %v", podName, err)
			}

			framework.Logf("Got logs for pod %q: %q", podName, logs)
			if !strings.Contains(logs, "Operation not permitted") {
				framework.Failf("unprivileged container shouldn't be able to create dummy device")
			}
		})
	})

	Context("when creating containers with AllowPrivilegeEscalation", func() {
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
		  Testname: allowPrivilegeEscalation unset and uid != 0.
		  Description: Configuring the allowPrivilegeEscalation unset, allows the privilege escalation operation.
		  A container is configured with allowPrivilegeEscalation not specified (nil) and a given uid which is not 0.
		  When the container is run, the container is run using uid=0.
		  This test is marked LinuxOnly since Windows does not support running as UID / GID, or privilege escalation.
		*/
		It("should allow privilege escalation when not explicitly set and uid != 0 [LinuxOnly] [NodeConformance]", func() {
			podName := "alpine-nnp-nil-" + string(uuid.NewUUID())
			if err := createAndMatchOutput(podName, "Effective uid: 0", nil, 1000); err != nil {
				framework.Failf("Match output for pod %q failed: %v", podName, err)
			}
		})

		/*
		  Testname: allowPrivilegeEscalation=false.
		  Description: Configuring the allowPrivilegeEscalation to false, does not allow the privilege escalation operation.
		  A container is configured with allowPrivilegeEscalation=false and a given uid (1000) which is not 0.
		  When the container is run, the container is run using uid=1000.
		  This test is marked LinuxOnly since Windows does not support running as UID / GID, or privilege escalation.
		*/
		It("should not allow privilege escalation when false [LinuxOnly] [NodeConformance]", func() {
			podName := "alpine-nnp-false-" + string(uuid.NewUUID())
			apeFalse := false
			if err := createAndMatchOutput(podName, "Effective uid: 1000", &apeFalse, 1000); err != nil {
				framework.Failf("Match output for pod %q failed: %v", podName, err)
			}
		})

		/*
		  Testname: allowPrivilegeEscalation=true.
		  Description: Configuring the allowPrivilegeEscalation to true, allows the privilege escalation operation.
		  A container is configured with allowPrivilegeEscalation=true and a given uid (1000) which is not 0.
		  When the container is run, the container is run using uid=0 (making use of the privilege escalation).
		  This test is marked LinuxOnly since Windows does not support running as UID / GID.
		*/
		It("should allow privilege escalation when true [LinuxOnly] [NodeConformance]", func() {
			podName := "alpine-nnp-true-" + string(uuid.NewUUID())
			apeTrue := true
			if err := createAndMatchOutput(podName, "Effective uid: 0", &apeTrue, 1000); err != nil {
				framework.Failf("Match output for pod %q failed: %v", podName, err)
			}
		})
	})
})
