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

package node

import (
	"context"
	"fmt"
	"os/exec"
	"reflect"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epodoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/pointer"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gcustom"
)

var (
	// non-root UID used in tests.
	nonRootTestUserID = int64(1000)

	// kubelet user used for userns mapping.
	kubeletUserForUsernsMapping = "kubelet"
	getsubuidsBinary            = "getsubids"
)

var _ = SIGDescribe("Security Context", func() {
	f := framework.NewDefaultFramework("security-context-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var podClient *e2epod.PodClient
	ginkgo.BeforeEach(func() {
		podClient = e2epod.NewPodClient(f)
	})

	ginkgo.Context("When creating a pod with HostUsers", func() {
		ginkgo.BeforeEach(func() {
			e2eskipper.SkipIfNodeOSDistroIs("windows")
		})

		containerName := "userns-test"
		makePod := func(hostUsers bool) *v1.Pod {
			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "userns-" + string(uuid.NewUUID()),
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    containerName,
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"cat", "/proc/self/uid_map"},
						},
					},
					RestartPolicy: v1.RestartPolicyNever,
					HostUsers:     &hostUsers,
				},
			}
		}

		f.It("must create the user namespace if set to false [LinuxOnly]", feature.UserNamespacesSupport, func(ctx context.Context) {
			// with hostUsers=false the pod must use a new user namespace
			podClient := e2epod.PodClientNS(f, f.Namespace.Name)

			createdPod1 := podClient.Create(ctx, makePod(false))
			createdPod2 := podClient.Create(ctx, makePod(false))
			ginkgo.DeferCleanup(func(ctx context.Context) {
				ginkgo.By("delete the pods")
				podClient.DeleteSync(ctx, createdPod1.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
				podClient.DeleteSync(ctx, createdPod2.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
			})
			getLogs := func(pod *v1.Pod) (string, error) {
				err := e2epod.WaitForPodSuccessInNamespaceTimeout(ctx, f.ClientSet, pod.Name, f.Namespace.Name, f.Timeouts.PodStart)
				if err != nil {
					return "", err
				}
				podStatus, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
				if err != nil {
					return "", err
				}
				return e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, podStatus.Name, containerName)
			}

			logs1, err := getLogs(createdPod1)
			framework.ExpectNoError(err)
			logs2, err := getLogs(createdPod2)
			framework.ExpectNoError(err)

			// 65536 is the size used for a user namespace.  Verify that the value is present
			// in the /proc/self/uid_map file.
			if !strings.Contains(logs1, "65536") || !strings.Contains(logs2, "65536") {
				framework.Failf("user namespace not created")
			}
			if logs1 == logs2 {
				framework.Failf("two different pods are running with the same user namespace configuration")
			}
		})

		f.It("must create the user namespace in the configured hostUID/hostGID range [LinuxOnly]", feature.UserNamespacesSupport, func(ctx context.Context) {
			// We need to check with the binary "getsubuids" the mappings for the kubelet.
			// If something is not present, we skip the test as the node wasn't configured to run this test.
			id, length, err := kubeletUsernsMappings(getsubuidsBinary)
			if err != nil {
				e2eskipper.Skipf("node is not setup for userns with kubelet mappings: %v", err)
			}

			for i := 0; i < 4; i++ {
				// makePod(false) creates the pod with user namespace
				podClient := e2epod.PodClientNS(f, f.Namespace.Name)
				createdPod := podClient.Create(ctx, makePod(false))
				ginkgo.DeferCleanup(func(ctx context.Context) {
					ginkgo.By("delete the pods")
					podClient.DeleteSync(ctx, createdPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
				})
				getLogs := func(pod *v1.Pod) (string, error) {
					err := e2epod.WaitForPodSuccessInNamespaceTimeout(ctx, f.ClientSet, createdPod.Name, f.Namespace.Name, f.Timeouts.PodStart)
					if err != nil {
						return "", err
					}
					podStatus, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
					if err != nil {
						return "", err
					}
					return e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, podStatus.Name, containerName)
				}

				logs, err := getLogs(createdPod)
				framework.ExpectNoError(err)

				// The hostUID is the second field in the /proc/self/uid_map file.
				hostMap := strings.Fields(logs)
				if len(hostMap) != 3 {
					framework.Failf("can't detect hostUID for container, is the format of /proc/self/uid_map correct?")
				}

				tmp, err := strconv.ParseUint(hostMap[1], 10, 32)
				if err != nil {
					framework.Failf("can't convert hostUID to int: %v", err)
				}
				hostUID := uint32(tmp)

				// Here we check the pod got a userns mapping within the range
				// configured for the kubelet.
				// To make sure the pod mapping doesn't fall within range by chance,
				// we do the following:
				// * The configured kubelet range as small as possible (enough to
				// fit 110 pods, the default of the kubelet) to minimize the chance
				// of this range being used "by chance" in the node configuration.
				// * We also run this in a loop, so it is less likely to get lucky
				// several times in a row.
				//
				// There are 65536 ranges possible and we configured the kubelet to
				// use 110 of them. The chances of this test passing by chance 4
				// times in a row and the kubelet not using only the configured
				// range are:
				//
				//	(110/65536) ^ 4 = 4.73e-12. IOW, less than 1 in a trillion.
				//
				// Furthermore, the unit tests would also need to be buggy and not
				// detect the bug. We expect to catch off-by-one errors there.
				if hostUID < id || hostUID > id+length {
					framework.Failf("user namespace created outside of the configured range. Expected range: %v-%v, got: %v", id, id+length, hostUID)
				}
			}
		})

		f.It("must not create the user namespace if set to true [LinuxOnly]", feature.UserNamespacesSupport, func(ctx context.Context) {
			// with hostUsers=true the pod must use the host user namespace
			pod := makePod(true)
			// When running in the host's user namespace, the /proc/self/uid_map file content looks like:
			// 0          0 4294967295
			// Verify the value 4294967295 is present in the output.
			e2epodoutput.TestContainerOutput(ctx, f, "read namespace", pod, 0, []string{
				"4294967295",
			})
		})

		f.It("should mount all volumes with proper permissions with hostUsers=false [LinuxOnly]", feature.UserNamespacesSupport, func(ctx context.Context) {
			// Create configmap.
			name := "userns-volumes-test-" + string(uuid.NewUUID())
			configMap := newConfigMap(f, name)
			ginkgo.By(fmt.Sprintf("Creating configMap %v/%v", f.Namespace.Name, configMap.Name))
			var err error
			if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(ctx, configMap, metav1.CreateOptions{}); err != nil {
				framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
			}

			// Create secret.
			secret := secretForTest(f.Namespace.Name, name)
			ginkgo.By(fmt.Sprintf("Creating secret with name %s", secret.Name))
			if secret, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(ctx, secret, metav1.CreateOptions{}); err != nil {
				framework.Failf("unable to create test secret %s: %v", secret.Name, err)
			}

			// downwardAPI definition.
			downwardVolSource := &v1.DownwardAPIVolumeSource{
				Items: []v1.DownwardAPIVolumeFile{
					{
						Path: "name",
						FieldRef: &v1.ObjectFieldSelector{
							APIVersion: "v1",
							FieldPath:  "metadata.name",
						},
					},
				},
			}

			// Create a pod with all the volumes
			falseVar := false
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod-userns-volumes-" + string(uuid.NewUUID()),
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "userns-file-permissions",
							Image: imageutils.GetE2EImage(imageutils.BusyBox),
							// Print (numeric) GID of the files in /vol/.
							// We limit to "type f" as kubelet uses symlinks to those files, but we
							// don't care about the owner of the symlink itself, just the files.
							Command: []string{"sh", "-c", "stat -c='%g' $(find /vol/ -type f)"},
							VolumeMounts: []v1.VolumeMount{
								{
									Name:      "cfg",
									MountPath: "/vol/cfg/",
								},
								{
									Name:      "secret",
									MountPath: "/vol/secret/",
								},
								{
									Name:      "downward",
									MountPath: "/vol/downward/",
								},
								{
									Name:      "projected",
									MountPath: "/vol/projected/",
								},
							},
						},
					},
					Volumes: []v1.Volume{
						{
							Name: "cfg",
							VolumeSource: v1.VolumeSource{
								ConfigMap: &v1.ConfigMapVolumeSource{
									LocalObjectReference: v1.LocalObjectReference{Name: configMap.Name},
								},
							},
						},
						{
							Name: "secret",
							VolumeSource: v1.VolumeSource{
								Secret: &v1.SecretVolumeSource{
									SecretName: secret.Name,
								},
							},
						},
						{
							Name: "downward",
							VolumeSource: v1.VolumeSource{
								DownwardAPI: downwardVolSource,
							},
						},
						{
							Name: "projected",
							VolumeSource: v1.VolumeSource{
								Projected: &v1.ProjectedVolumeSource{
									Sources: []v1.VolumeProjection{
										{
											DownwardAPI: &v1.DownwardAPIProjection{
												Items: downwardVolSource.Items,
											},
										},
										{
											Secret: &v1.SecretProjection{
												LocalObjectReference: v1.LocalObjectReference{Name: secret.Name},
											},
										},
									},
								},
							},
						},
					},
					HostUsers:     &falseVar,
					RestartPolicy: v1.RestartPolicyNever,
				},
			}

			// Expect one line for each file on all the volumes.
			// Each line should be "=0" that means root inside the container is the owner of the file.
			downwardAPIVolFiles := 1
			projectedFiles := len(secret.Data) + downwardAPIVolFiles
			e2epodoutput.TestContainerOutput(ctx, f, "check file permissions", pod, 0, []string{
				strings.Repeat("=0\n", len(secret.Data)+len(configMap.Data)+downwardAPIVolFiles+projectedFiles),
			})
		})

		f.It("should set FSGroup to user inside the container with hostUsers=false [LinuxOnly]", feature.UserNamespacesSupport, func(ctx context.Context) {
			// Create configmap.
			name := "userns-volumes-test-" + string(uuid.NewUUID())
			configMap := newConfigMap(f, name)
			ginkgo.By(fmt.Sprintf("Creating configMap %v/%v", f.Namespace.Name, configMap.Name))
			var err error
			if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(ctx, configMap, metav1.CreateOptions{}); err != nil {
				framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
			}

			// Create a pod with hostUsers=false
			falseVar := false
			fsGroup := int64(200)
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod-userns-fsgroup-" + string(uuid.NewUUID()),
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "userns-fsgroup",
							Image: imageutils.GetE2EImage(imageutils.BusyBox),
							// Print (numeric) GID of the files in /vol/.
							// We limit to "type f" as kubelet uses symlinks to those files, but we
							// don't care about the owner of the symlink itself, just the files.
							Command: []string{"sh", "-c", "stat -c='%g' $(find /vol/ -type f)"},
							VolumeMounts: []v1.VolumeMount{
								{
									Name:      "cfg",
									MountPath: "/vol/cfg/",
								},
							},
						},
					},
					Volumes: []v1.Volume{
						{
							Name: "cfg",
							VolumeSource: v1.VolumeSource{
								ConfigMap: &v1.ConfigMapVolumeSource{
									LocalObjectReference: v1.LocalObjectReference{Name: configMap.Name},
								},
							},
						},
					},
					HostUsers:     &falseVar,
					RestartPolicy: v1.RestartPolicyNever,
					SecurityContext: &v1.PodSecurityContext{
						FSGroup: &fsGroup,
					},
				},
			}

			// Expect one line for each file on all the volumes.
			// Each line should be "=200" (fsGroup) that means it was mapped to the
			// right user inside the container.
			e2epodoutput.TestContainerOutput(ctx, f, "check FSGroup is mapped correctly", pod, 0, []string{
				strings.Repeat(fmt.Sprintf("=%v\n", fsGroup), len(configMap.Data)),
			})
		})
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
		createAndWaitUserPod := func(ctx context.Context, userid int64) {
			podName := fmt.Sprintf("busybox-user-%d-%s", userid, uuid.NewUUID())
			podClient.Create(ctx, makeUserPod(podName,
				imageutils.GetE2EImage(imageutils.BusyBox),
				[]string{"sh", "-c", fmt.Sprintf("test $(id -u) -eq %d", userid)},
				userid,
			))

			podClient.WaitForSuccess(ctx, podName, framework.PodStartTimeout)
		}

		/*
			Release: v1.15
			Testname: Security Context, runAsUser=65534
			Description: Container is created with runAsUser option by passing uid 65534 to run as unpriviledged user. Pod MUST be in Succeeded phase.
			[LinuxOnly]: This test is marked as LinuxOnly since Windows does not support running as UID / GID.
		*/
		framework.ConformanceIt("should run the container with uid 65534 [LinuxOnly]", f.WithNodeConformance(), func(ctx context.Context) {
			createAndWaitUserPod(ctx, 65534)
		})

		/*
			Release: v1.15
			Testname: Security Context, runAsUser=0
			Description: Container is created with runAsUser option by passing uid 0 to run as root privileged user. Pod MUST be in Succeeded phase.
			This e2e can not be promoted to Conformance because a Conformant platform may not allow to run containers with 'uid 0' or running privileged operations.
			[LinuxOnly]: This test is marked as LinuxOnly since Windows does not support running as UID / GID.
		*/
		f.It("should run the container with uid 0 [LinuxOnly]", f.WithNodeConformance(), func(ctx context.Context) {
			createAndWaitUserPod(ctx, 0)
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

		ginkgo.It("should run with an explicit non-root user ID [LinuxOnly]", func(ctx context.Context) {
			// creates a pod with RunAsUser, which is not supported on Windows.
			e2eskipper.SkipIfNodeOSDistroIs("windows")
			name := "explicit-nonroot-uid"
			pod := makeNonRootPod(name, rootImage, pointer.Int64Ptr(nonRootTestUserID))
			podClient.Create(ctx, pod)

			podClient.WaitForSuccess(ctx, name, framework.PodStartTimeout)
			framework.ExpectNoError(podClient.MatchContainerOutput(ctx, name, name, "1000"))
		})
		ginkgo.It("should not run with an explicit root user ID [LinuxOnly]", func(ctx context.Context) {
			// creates a pod with RunAsUser, which is not supported on Windows.
			e2eskipper.SkipIfNodeOSDistroIs("windows")
			name := "explicit-root-uid"
			pod := makeNonRootPod(name, nonRootImage, pointer.Int64Ptr(0))
			pod = podClient.Create(ctx, pod)

			ev, err := podClient.WaitForErrorEventOrSuccess(ctx, pod)
			framework.ExpectNoError(err)
			gomega.Expect(ev).NotTo(gomega.BeNil())
			gomega.Expect(ev.Reason).To(gomega.Equal(events.FailedToCreateContainer))
		})
		ginkgo.It("should run with an image specified user ID", func(ctx context.Context) {
			name := "implicit-nonroot-uid"
			pod := makeNonRootPod(name, nonRootImage, nil)
			podClient.Create(ctx, pod)

			podClient.WaitForSuccess(ctx, name, framework.PodStartTimeout)
			framework.ExpectNoError(podClient.MatchContainerOutput(ctx, name, name, "1234"))
		})
		ginkgo.It("should not run without a specified user ID", func(ctx context.Context) {
			name := "implicit-root-uid"
			pod := makeNonRootPod(name, rootImage, nil)
			pod = podClient.Create(ctx, pod)

			ev, err := podClient.WaitForErrorEventOrSuccess(ctx, pod)
			framework.ExpectNoError(err)
			gomega.Expect(ev).NotTo(gomega.BeNil())
			gomega.Expect(ev.Reason).To(gomega.Equal(events.FailedToCreateContainer))
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
		createAndWaitUserPod := func(ctx context.Context, readOnlyRootFilesystem bool) string {
			podName := fmt.Sprintf("busybox-readonly-%v-%s", readOnlyRootFilesystem, uuid.NewUUID())
			podClient.Create(ctx, makeUserPod(podName,
				imageutils.GetE2EImage(imageutils.BusyBox),
				[]string{"sh", "-c", "touch checkfile"},
				readOnlyRootFilesystem,
			))

			if readOnlyRootFilesystem {
				waitForFailure(ctx, f, podName, framework.PodStartTimeout)
			} else {
				podClient.WaitForSuccess(ctx, podName, framework.PodStartTimeout)
			}

			return podName
		}

		/*
			Release: v1.15
			Testname: Security Context, readOnlyRootFilesystem=true.
			Description: Container is configured to run with readOnlyRootFilesystem to true which will force containers to run with a read only root file system.
			Write operation MUST NOT be allowed and Pod MUST be in Failed state.
			At this moment we are not considering this test for Conformance due to use of SecurityContext.
			[LinuxOnly]: This test is marked as LinuxOnly since Windows does not support creating containers with read-only access.
		*/
		f.It("should run the container with readonly rootfs when readOnlyRootFilesystem=true [LinuxOnly]", f.WithNodeConformance(), func(ctx context.Context) {
			createAndWaitUserPod(ctx, true)
		})

		/*
			Release: v1.15
			Testname: Security Context, readOnlyRootFilesystem=false.
			Description: Container is configured to run with readOnlyRootFilesystem to false.
			Write operation MUST be allowed and Pod MUST be in Succeeded state.
		*/
		framework.ConformanceIt("should run the container with writable rootfs when readOnlyRootFilesystem=false", f.WithNodeConformance(), func(ctx context.Context) {
			createAndWaitUserPod(ctx, false)
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
		createAndWaitUserPod := func(ctx context.Context, privileged bool) string {
			podName := fmt.Sprintf("busybox-privileged-%v-%s", privileged, uuid.NewUUID())
			podClient.Create(ctx, makeUserPod(podName,
				imageutils.GetE2EImage(imageutils.BusyBox),
				[]string{"sh", "-c", "ip link add dummy0 type dummy || true"},
				privileged,
			))
			podClient.WaitForSuccess(ctx, podName, framework.PodStartTimeout)
			return podName
		}
		/*
			Release: v1.15
			Testname: Security Context, privileged=false.
			Description: Create a container to run in unprivileged mode by setting pod's SecurityContext Privileged option as false. Pod MUST be in Succeeded phase.
			[LinuxOnly]: This test is marked as LinuxOnly since it runs a Linux-specific command.
		*/
		framework.ConformanceIt("should run the container as unprivileged when false [LinuxOnly]", f.WithNodeConformance(), func(ctx context.Context) {
			podName := createAndWaitUserPod(ctx, false)
			logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, podName, podName)
			if err != nil {
				framework.Failf("GetPodLogs for pod %q failed: %v", podName, err)
			}

			framework.Logf("Got logs for pod %q: %q", podName, logs)
			if !strings.Contains(logs, "Operation not permitted") {
				framework.Failf("unprivileged container shouldn't be able to create dummy device")
			}
		})

		f.It("should run the container as privileged when true [LinuxOnly]", feature.HostAccess, func(ctx context.Context) {
			podName := createAndWaitUserPod(ctx, true)
			logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, podName, podName)
			if err != nil {
				framework.Failf("GetPodLogs for pod %q failed: %v", podName, err)
			}

			framework.Logf("Got logs for pod %q: %q", podName, logs)
			if strings.Contains(logs, "Operation not permitted") {
				framework.Failf("privileged container should be able to create dummy device")
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
		createAndMatchOutput := func(ctx context.Context, podName, output string, allowPrivilegeEscalation *bool, uid int64) error {
			podClient.Create(ctx, makeAllowPrivilegeEscalationPod(podName,
				allowPrivilegeEscalation,
				uid,
			))
			podClient.WaitForSuccess(ctx, podName, framework.PodStartTimeout)
			return podClient.MatchContainerOutput(ctx, podName, podName, output)
		}

		/*
			Release: v1.15
			Testname: Security Context, allowPrivilegeEscalation unset, uid != 0.
			Description: Configuring the allowPrivilegeEscalation unset, allows the privilege escalation operation.
			A container is configured with allowPrivilegeEscalation not specified (nil) and a given uid which is not 0.
			When the container is run, container's output MUST match with expected output verifying container ran with uid=0.
			This e2e Can not be promoted to Conformance as it is Container Runtime dependent and not all conformant platforms will require this behavior.
			[LinuxOnly]: This test is marked LinuxOnly since Windows does not support running as UID / GID, or privilege escalation.
		*/
		f.It("should allow privilege escalation when not explicitly set and uid != 0 [LinuxOnly]", f.WithNodeConformance(), func(ctx context.Context) {
			podName := "alpine-nnp-nil-" + string(uuid.NewUUID())
			if err := createAndMatchOutput(ctx, podName, "Effective uid: 0", nil, nonRootTestUserID); err != nil {
				framework.Failf("Match output for pod %q failed: %v", podName, err)
			}
		})

		/*
			Release: v1.15
			Testname: Security Context, allowPrivilegeEscalation=false.
			Description: Configuring the allowPrivilegeEscalation to false, does not allow the privilege escalation operation.
			A container is configured with allowPrivilegeEscalation=false and a given uid (1000) which is not 0.
			When the container is run, container's output MUST match with expected output verifying container ran with given uid i.e. uid=1000.
			[LinuxOnly]: This test is marked LinuxOnly since Windows does not support running as UID / GID, or privilege escalation.
		*/
		framework.ConformanceIt("should not allow privilege escalation when false [LinuxOnly]", f.WithNodeConformance(), func(ctx context.Context) {
			podName := "alpine-nnp-false-" + string(uuid.NewUUID())
			apeFalse := false
			if err := createAndMatchOutput(ctx, podName, fmt.Sprintf("Effective uid: %d", nonRootTestUserID), &apeFalse, nonRootTestUserID); err != nil {
				framework.Failf("Match output for pod %q failed: %v", podName, err)
			}
		})

		/*
			Release: v1.15
			Testname: Security Context, allowPrivilegeEscalation=true.
			Description: Configuring the allowPrivilegeEscalation to true, allows the privilege escalation operation.
			A container is configured with allowPrivilegeEscalation=true and a given uid (1000) which is not 0.
			When the container is run, container's output MUST match with expected output verifying container ran with uid=0 (making use of the privilege escalation).
			This e2e Can not be promoted to Conformance as it is Container Runtime dependent and runtime may not allow to run.
			[LinuxOnly]: This test is marked LinuxOnly since Windows does not support running as UID / GID.
		*/
		f.It("should allow privilege escalation when true [LinuxOnly]", f.WithNodeConformance(), func(ctx context.Context) {
			podName := "alpine-nnp-true-" + string(uuid.NewUUID())
			apeTrue := true
			if err := createAndMatchOutput(ctx, podName, "Effective uid: 0", &apeTrue, nonRootTestUserID); err != nil {
				framework.Failf("Match output for pod %q failed: %v", podName, err)
			}
		})
	})

	f.Context("SupplementalGroupsPolicy [LinuxOnly]", feature.SupplementalGroupsPolicy, framework.WithFeatureGate(features.SupplementalGroupsPolicy), func() {
		timeout := 1 * time.Minute

		agnhostImage := imageutils.GetE2EImage(imageutils.Agnhost)
		uidInImage := int64(1000)
		gidDefinedInImage := int64(50000)
		supplementalGroup := int64(60000)

		mkPod := func(policy *v1.SupplementalGroupsPolicy) *v1.Pod {
			// In specified image(agnhost E2E image),
			// - user-defined-in-image(uid=1000) is defined
			// - user-defined-in-image belongs to group-defined-in-image(gid=50000)
			// thus, resultant supplementary group of the container processes should be
			// - 1000 : self
			// - 50000: pre-defined groups defined in the container image(/etc/group) of self(uid=1000)
			// - 60000: specified in SupplementalGroups
			// $ id -G
			// 1000 50000 60000 (if SupplementalGroupsPolicy=Merge or not set)
			// 1000 60000       (if SupplementalGroupsPolicy=Strict)
			podName := "sppl-grp-plcy-" + string(uuid.NewUUID())
			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:        podName,
					Labels:      map[string]string{"name": podName},
					Annotations: map[string]string{},
				},
				Spec: v1.PodSpec{
					SecurityContext: &v1.PodSecurityContext{
						RunAsUser:                &uidInImage,
						SupplementalGroups:       []int64{supplementalGroup},
						SupplementalGroupsPolicy: policy,
					},
					Containers: []v1.Container{
						{
							Name:    "test-container",
							Image:   agnhostImage,
							Command: []string{"sh", "-c", "id -G; while :; do sleep 1; done"},
						},
					},
					RestartPolicy: v1.RestartPolicyNever,
				},
			}
		}

		nodeSupportsSupplementalGroupsPolicy := func(ctx context.Context, f *framework.Framework, nodeName string) bool {
			node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(node).NotTo(gomega.BeNil())
			if node.Status.Features != nil {
				supportsSupplementalGroupsPolicy := node.Status.Features.SupplementalGroupsPolicy
				if supportsSupplementalGroupsPolicy != nil && *supportsSupplementalGroupsPolicy {
					return true
				}
			}
			return false
		}
		waitForContainerUser := func(ctx context.Context, f *framework.Framework, podName string, containerName string, expectedContainerUser *v1.ContainerUser) error {
			return framework.Gomega().Eventually(ctx,
				framework.RetryNotFound(framework.GetObject(f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get, podName, metav1.GetOptions{}))).
				WithTimeout(timeout).
				Should(gcustom.MakeMatcher(func(p *v1.Pod) (bool, error) {
					for _, s := range p.Status.ContainerStatuses {
						if s.Name == containerName {
							return reflect.DeepEqual(s.User, expectedContainerUser), nil
						}
					}
					return false, nil
				}))
		}
		waitForPodLogs := func(ctx context.Context, f *framework.Framework, podName string, containerName string, expectedLog string) error {
			return framework.Gomega().Eventually(ctx,
				framework.RetryNotFound(framework.GetObject(f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get, podName, metav1.GetOptions{}))).
				WithTimeout(timeout).
				Should(gcustom.MakeMatcher(func(p *v1.Pod) (bool, error) {
					podLogs, err := e2epod.GetPodLogs(ctx, f.ClientSet, p.Namespace, p.Name, containerName)
					if err != nil {
						return false, err
					}
					return podLogs == expectedLog, nil
				}))
		}
		expectMergePolicyInEffect := func(ctx context.Context, f *framework.Framework, podName string, containerName string, featureSupportedOnNode bool) {
			expectedOutput := fmt.Sprintf("%d %d %d", uidInImage, gidDefinedInImage, supplementalGroup)
			expectedContainerUser := &v1.ContainerUser{
				Linux: &v1.LinuxContainerUser{
					UID:                uidInImage,
					GID:                uidInImage,
					SupplementalGroups: []int64{uidInImage, gidDefinedInImage, supplementalGroup},
				},
			}

			if featureSupportedOnNode {
				framework.ExpectNoError(waitForContainerUser(ctx, f, podName, containerName, expectedContainerUser))
			}
			framework.ExpectNoError(waitForPodLogs(ctx, f, podName, containerName, expectedOutput+"\n"))

			stdout := e2epod.ExecCommandInContainer(f, podName, containerName, "id", "-G")
			gomega.Expect(stdout).To(gomega.Equal(expectedOutput))
		}
		expectStrictPolicyInEffect := func(ctx context.Context, f *framework.Framework, podName string, containerName string, featureSupportedOnNode bool) {
			expectedOutput := fmt.Sprintf("%d %d", uidInImage, supplementalGroup)
			expectedContainerUser := &v1.ContainerUser{
				Linux: &v1.LinuxContainerUser{
					UID:                uidInImage,
					GID:                uidInImage,
					SupplementalGroups: []int64{uidInImage, supplementalGroup},
				},
			}

			if featureSupportedOnNode {
				framework.ExpectNoError(waitForContainerUser(ctx, f, podName, containerName, expectedContainerUser))
			}
			framework.ExpectNoError(waitForPodLogs(ctx, f, podName, containerName, expectedOutput+"\n"))

			stdout := e2epod.ExecCommandInContainer(f, podName, containerName, "id", "-G")
			gomega.Expect(stdout).To(gomega.Equal(expectedOutput))
		}
		expectRejectionEventIssued := func(ctx context.Context, f *framework.Framework, pod *v1.Pod) {
			framework.ExpectNoError(
				framework.Gomega().Eventually(ctx,
					framework.HandleRetry(framework.ListObjects(
						f.ClientSet.CoreV1().Events(pod.Namespace).List,
						metav1.ListOptions{
							FieldSelector: fields.Set{
								"type":                      core.EventTypeWarning,
								"reason":                    lifecycle.SupplementalGroupsPolicyNotSupported,
								"involvedObject.kind":       "Pod",
								"involvedObject.apiVersion": v1.SchemeGroupVersion.String(),
								"involvedObject.name":       pod.Name,
								"involvedObject.uid":        string(pod.UID),
							}.AsSelector().String(),
						},
					))).
					WithTimeout(timeout).
					Should(gcustom.MakeMatcher(func(eventList *v1.EventList) (bool, error) {
						return len(eventList.Items) == 1, nil
					})),
			)
		}

		ginkgo.When("SupplementalGroupsPolicy nil in SecurityContext", func() {
			ginkgo.When("if the container's primary UID belongs to some groups in the image", func() {
				var pod *v1.Pod
				ginkgo.BeforeEach(func(ctx context.Context) {
					ginkgo.By("creating a pod", func() {
						pod = e2epod.NewPodClient(f).CreateSync(ctx, mkPod(ptr.To(v1.SupplementalGroupsPolicyMerge)))
					})
				})
				ginkgo.When("scheduled node does not support SupplementalGroupsPolicy", func() {
					ginkgo.BeforeEach(func(ctx context.Context) {
						// ensure the scheduled node does not support SupplementalGroupsPolicy
						if nodeSupportsSupplementalGroupsPolicy(ctx, f, pod.Spec.NodeName) {
							e2eskipper.Skipf("scheduled node does support SupplementalGroupsPolicy")
						}
					})
					ginkgo.It("it should add SupplementalGroups to them [LinuxOnly]", func(ctx context.Context) {
						expectMergePolicyInEffect(ctx, f, pod.Name, pod.Spec.Containers[0].Name, false)
					})
				})
				ginkgo.When("scheduled node supports SupplementalGroupsPolicy", func() {
					ginkgo.BeforeEach(func(ctx context.Context) {
						// ensure the scheduled node does support SupplementalGroupsPolicy
						if !nodeSupportsSupplementalGroupsPolicy(ctx, f, pod.Spec.NodeName) {
							e2eskipper.Skipf("scheduled node does not support SupplementalGroupsPolicy")
						}
					})
					ginkgo.It("it should add SupplementalGroups to them [LinuxOnly]", func(ctx context.Context) {
						expectMergePolicyInEffect(ctx, f, pod.Name, pod.Spec.Containers[0].Name, true)
					})
				})
			})
		})
		ginkgo.When("SupplementalGroupsPolicy was set to Merge in PodSpec", func() {
			ginkgo.When("the container's primary UID belongs to some groups in the image", func() {
				var pod *v1.Pod
				ginkgo.BeforeEach(func(ctx context.Context) {
					ginkgo.By("creating a pod", func() {
						pod = e2epod.NewPodClient(f).CreateSync(ctx, mkPod(nil))
					})
				})
				ginkgo.When("scheduled node does not support SupplementalGroupsPolicy", func() {
					ginkgo.BeforeEach(func(ctx context.Context) {
						// ensure the scheduled node does not support SupplementalGroupsPolicy
						if nodeSupportsSupplementalGroupsPolicy(ctx, f, pod.Spec.NodeName) {
							e2eskipper.Skipf("scheduled node does support SupplementalGroupsPolicy")
						}
					})
					ginkgo.It("it should add SupplementalGroups to them [LinuxOnly]", func(ctx context.Context) {
						expectMergePolicyInEffect(ctx, f, pod.Name, pod.Spec.Containers[0].Name, false)
					})
				})
				ginkgo.When("scheduled node supports SupplementalGroupsPolicy", func() {
					ginkgo.BeforeEach(func(ctx context.Context) {
						// ensure the scheduled node does support SupplementalGroupsPolicy
						if !nodeSupportsSupplementalGroupsPolicy(ctx, f, pod.Spec.NodeName) {
							e2eskipper.Skipf("scheduled node does not support SupplementalGroupsPolicy")
						}
					})
					ginkgo.It("it should add SupplementalGroups to them [LinuxOnly]", func(ctx context.Context) {
						expectMergePolicyInEffect(ctx, f, pod.Name, pod.Spec.Containers[0].Name, true)
					})
				})
			})
		})
		ginkgo.When("SupplementalGroupsPolicy was set to Strict in PodSpec", func() {
			ginkgo.When("the container's primary UID belongs to some groups in the image", func() {
				var pod *v1.Pod
				ginkgo.BeforeEach(func(ctx context.Context) {
					ginkgo.By("creating a pod", func() {
						pod = e2epod.NewPodClient(f).Create(ctx, mkPod(ptr.To(v1.SupplementalGroupsPolicyStrict)))
						framework.ExpectNoError(e2epod.WaitForPodScheduled(ctx, f.ClientSet, pod.Namespace, pod.Name))
						var err error
						pod, err = e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
						framework.ExpectNoError(err)
					})
				})
				ginkgo.When("scheduled node does not support SupplementalGroupsPolicy", func() {
					ginkgo.BeforeEach(func(ctx context.Context) {
						// ensure the scheduled node does not support SupplementalGroupsPolicy
						if nodeSupportsSupplementalGroupsPolicy(ctx, f, pod.Spec.NodeName) {
							e2eskipper.Skipf("scheduled node does support SupplementalGroupsPolicy")
						}
					})
					ginkgo.It("it should reject the pod [LinuxOnly]", func(ctx context.Context) {
						expectRejectionEventIssued(ctx, f, pod)
					})
				})
				ginkgo.When("scheduled node supports SupplementalGroupsPolicy", func() {
					ginkgo.BeforeEach(func(ctx context.Context) {
						// ensure the scheduled node does support SupplementalGroupsPolicy
						if !nodeSupportsSupplementalGroupsPolicy(ctx, f, pod.Spec.NodeName) {
							e2eskipper.Skipf("scheduled node does not support SupplementalGroupsPolicy")
						}
						framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod))
					})
					ginkgo.It("it should NOT add SupplementalGroups to them [LinuxOnly]", func(ctx context.Context) {
						expectStrictPolicyInEffect(ctx, f, pod.Name, pod.Spec.Containers[0].Name, true)
					})
				})
			})
		})
	})
})

var _ = SIGDescribe("User Namespaces for Pod Security Standards [LinuxOnly]", func() {
	ginkgo.BeforeEach(func() {
		e2eskipper.SkipIfNodeOSDistroIs("windows")
	})

	f := framework.NewDefaultFramework("user-namespaces-pss-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelRestricted

	ginkgo.Context("with UserNamespacesSupport and UserNamespacesPodSecurityStandards enabled", func() {
		f.It("should allow pod", feature.UserNamespacesPodSecurityStandards, func(ctx context.Context) {
			name := "pod-user-namespaces-pss-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: name},
				Spec: v1.PodSpec{
					RestartPolicy:   v1.RestartPolicyNever,
					HostUsers:       ptr.To(false),
					SecurityContext: &v1.PodSecurityContext{},
					Containers: []v1.Container{
						{
							Name:    name,
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"whoami"},
							SecurityContext: &v1.SecurityContext{
								AllowPrivilegeEscalation: ptr.To(false),
								Capabilities:             &v1.Capabilities{Drop: []v1.Capability{"ALL"}},
								SeccompProfile:           &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault},
							},
						},
					},
				},
			}

			e2epodoutput.TestContainerOutput(ctx, f, "RunAsUser-RunAsNonRoot", pod, 0, []string{"root"})
		})
	})
})

// waitForFailure waits for pod to fail.
func waitForFailure(ctx context.Context, f *framework.Framework, name string, timeout time.Duration) {
	gomega.Expect(e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, name, fmt.Sprintf("%s or %s", v1.PodSucceeded, v1.PodFailed), timeout,
		func(pod *v1.Pod) (bool, error) {
			switch pod.Status.Phase {
			case v1.PodFailed:
				return true, nil
			case v1.PodSucceeded:
				return true, fmt.Errorf("pod %q succeeded with reason: %q, message: %q", name, pod.Status.Reason, pod.Status.Message)
			default:
				return false, nil
			}
		},
	)).To(gomega.Succeed(), "wait for pod %q to fail", name)
}

// parseGetSubIdsOutput parses the output from the `getsubids` tool, which is used to query subordinate user or group ID ranges for
// a given user or group. getsubids produces a line for each mapping configured.
// Here we expect that there is a single mapping, and the same values are used for the subordinate user and group ID ranges.
// The output is something like:
// $ getsubids kubelet
// 0: kubelet 65536 2147483648
// $ getsubids -g kubelet
// 0: kubelet 65536 2147483648
// XXX: this is a c&p from pkg/kubelet/kubelet_pods.go. It is simpler to c&p than to try to reuse it.
func parseGetSubIdsOutput(input string) (uint32, uint32, error) {
	lines := strings.Split(strings.Trim(input, "\n"), "\n")
	if len(lines) != 1 {
		return 0, 0, fmt.Errorf("error parsing line %q: it must contain only one line", input)
	}

	parts := strings.Fields(lines[0])
	if len(parts) != 4 {
		return 0, 0, fmt.Errorf("invalid line %q", input)
	}

	// Parsing the numbers
	num1, err := strconv.ParseUint(parts[2], 10, 32)
	if err != nil {
		return 0, 0, fmt.Errorf("error parsing line %q: %w", input, err)
	}

	num2, err := strconv.ParseUint(parts[3], 10, 32)
	if err != nil {
		return 0, 0, fmt.Errorf("error parsing line %q: %w", input, err)
	}

	return uint32(num1), uint32(num2), nil
}

func kubeletUsernsMappings(subuidBinary string) (uint32, uint32, error) {
	cmd, err := exec.LookPath(getsubuidsBinary)
	if err != nil {
		return 0, 0, fmt.Errorf("getsubids binary not found in PATH")
	}
	outUids, err := exec.Command(cmd, kubeletUserForUsernsMapping).Output()
	if err != nil {
		return 0, 0, fmt.Errorf("no additional uids for user %q: %w", kubeletUserForUsernsMapping, err)
	}
	outGids, err := exec.Command(cmd, "-g", kubeletUserForUsernsMapping).Output()
	if err != nil {
		return 0, 0, fmt.Errorf("no additional gids for user %q", kubeletUserForUsernsMapping)
	}
	if string(outUids) != string(outGids) {
		return 0, 0, fmt.Errorf("mismatched subuids and subgids for user %q", kubeletUserForUsernsMapping)
	}

	return parseGetSubIdsOutput(string(outUids))
}
