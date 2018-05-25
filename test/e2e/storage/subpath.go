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

package storage

import (
	"fmt"
	"path/filepath"
	"strings"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var (
	volumePath      = "/test-volume"
	volumeName      = "test-volume"
	probeVolumePath = "/probe-volume"
	probeFilePath   = probeVolumePath + "/probe-file"
	fileName        = "test-file"
	retryDuration   = 10
	mountImage      = imageutils.GetE2EImage(imageutils.Mounttest)
)

type volInfo struct {
	source *v1.VolumeSource
	node   string
}

type volSource interface {
	createVolume(f *framework.Framework) volInfo
	cleanupVolume(f *framework.Framework)
	getReadOnlyVolumeSpec() *v1.VolumeSource
}

var initVolSources = map[string]func() volSource{
	"hostPath":         initHostpath,
	"hostPathSymlink":  initHostpathSymlink,
	"emptyDir":         initEmptydir,
	"gcePDPVC":         initGCEPDPVC,
	"gcePDPartitioned": initGCEPDPartition,
	"nfs":              initNFS,
	"nfsPVC":           initNFSPVC,
	"gluster":          initGluster,
}

var _ = utils.SIGDescribe("Subpath", func() {
	var (
		subPath           string
		subPathDir        string
		filePathInSubpath string
		filePathInVolume  string
		pod               *v1.Pod
		vol               volSource
	)

	f := framework.NewDefaultFramework("subpath")

	Context("Atomic writer volumes", func() {
		var err error

		BeforeEach(func() {
			By("Setting up data")
			secret := &v1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "my-secret"}, Data: map[string][]byte{"secret-key": []byte("secret-value")}}
			secret, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(secret)
			if err != nil && !apierrors.IsAlreadyExists(err) {
				Expect(err).ToNot(HaveOccurred(), "while creating secret")
			}

			configmap := &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "my-configmap"}, Data: map[string]string{"configmap-key": "configmap-value"}}
			configmap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(configmap)
			if err != nil && !apierrors.IsAlreadyExists(err) {
				Expect(err).ToNot(HaveOccurred(), "while creating configmap")
			}
		})

		It("should support subpaths with secret pod", func() {
			pod := testPodSubpath(f, "secret-key", "secret", &v1.VolumeSource{Secret: &v1.SecretVolumeSource{SecretName: "my-secret"}})
			testBasicSubpath(f, "secret-value", pod)
		})

		It("should support subpaths with configmap pod", func() {
			pod := testPodSubpath(f, "configmap-key", "configmap", &v1.VolumeSource{ConfigMap: &v1.ConfigMapVolumeSource{LocalObjectReference: v1.LocalObjectReference{Name: "my-configmap"}}})
			testBasicSubpath(f, "configmap-value", pod)
		})

		It("should support subpaths with configmap pod with mountPath of existing file", func() {
			pod := testPodSubpath(f, "configmap-key", "configmap", &v1.VolumeSource{ConfigMap: &v1.ConfigMapVolumeSource{LocalObjectReference: v1.LocalObjectReference{Name: "my-configmap"}}})
			file := "/etc/resolv.conf"
			pod.Spec.Containers[0].VolumeMounts[0].MountPath = file
			testBasicSubpathFile(f, "configmap-value", pod, file)
		})

		It("should support subpaths with downward pod", func() {
			pod := testPodSubpath(f, "downward/podname", "downwardAPI", &v1.VolumeSource{
				DownwardAPI: &v1.DownwardAPIVolumeSource{
					Items: []v1.DownwardAPIVolumeFile{{Path: "downward/podname", FieldRef: &v1.ObjectFieldSelector{APIVersion: "v1", FieldPath: "metadata.name"}}},
				},
			})
			testBasicSubpath(f, pod.Name, pod)
		})

		It("should support subpaths with projected pod", func() {
			pod := testPodSubpath(f, "projected/configmap-key", "projected", &v1.VolumeSource{
				Projected: &v1.ProjectedVolumeSource{
					Sources: []v1.VolumeProjection{
						{ConfigMap: &v1.ConfigMapProjection{
							LocalObjectReference: v1.LocalObjectReference{Name: "my-configmap"},
							Items:                []v1.KeyToPath{{Path: "projected/configmap-key", Key: "configmap-key"}},
						}},
					},
				},
			})
			testBasicSubpath(f, "configmap-value", pod)
		})
	})

	for volType, volInit := range initVolSources {
		curVolType := volType
		curVolInit := volInit

		Context(fmt.Sprintf("[Volume type: %v]", curVolType), func() {
			BeforeEach(func() {
				By(fmt.Sprintf("Initializing %s volume", curVolType))
				vol = curVolInit()
				subPath = f.Namespace.Name
				subPathDir = filepath.Join(volumePath, subPath)
				filePathInSubpath = filepath.Join(volumePath, fileName)
				filePathInVolume = filepath.Join(subPathDir, fileName)
				volInfo := vol.createVolume(f)
				pod = testPodSubpath(f, subPath, curVolType, volInfo.source)
				pod.Spec.NodeName = volInfo.node
			})

			AfterEach(func() {
				By("Deleting pod")
				err := framework.DeletePodWithWait(f, f.ClientSet, pod)
				Expect(err).ToNot(HaveOccurred(), "while deleting pod")

				By("Cleaning up volume")
				vol.cleanupVolume(f)
			})

			It("should support non-existent path", func() {
				// Write the file in the subPath from container 0
				setWriteCommand(filePathInSubpath, &pod.Spec.Containers[0])

				// Read it from outside the subPath from container 1
				testReadFile(f, filePathInVolume, pod, 1)
			})

			It("should support existing directory", func() {
				// Create the directory
				setInitCommand(pod, fmt.Sprintf("mkdir -p %s", subPathDir))

				// Write the file in the subPath from container 0
				setWriteCommand(filePathInSubpath, &pod.Spec.Containers[0])

				// Read it from outside the subPath from container 1
				testReadFile(f, filePathInVolume, pod, 1)
			})

			It("should support existing single file", func() {
				// Create the file in the init container
				setInitCommand(pod, fmt.Sprintf("mkdir -p %s; echo \"mount-tester new file\" > %s", subPathDir, filePathInVolume))

				// Read it from inside the subPath from container 0
				testReadFile(f, filePathInSubpath, pod, 0)
			})

			It("should support file as subpath", func() {
				// Create the file in the init container
				setInitCommand(pod, fmt.Sprintf("echo %s > %s", f.Namespace.Name, subPathDir))

				testBasicSubpath(f, f.Namespace.Name, pod)
			})

			It("should fail if subpath directory is outside the volume [Slow]", func() {
				// Create the subpath outside the volume
				setInitCommand(pod, fmt.Sprintf("ln -s /bin %s", subPathDir))

				// Pod should fail
				testPodFailSubpath(f, pod)
			})

			It("should fail if subpath file is outside the volume [Slow]", func() {
				// Create the subpath outside the volume
				setInitCommand(pod, fmt.Sprintf("ln -s /bin/sh %s", subPathDir))

				// Pod should fail
				testPodFailSubpath(f, pod)
			})

			It("should fail if non-existent subpath is outside the volume [Slow]", func() {
				// Create the subpath outside the volume
				setInitCommand(pod, fmt.Sprintf("ln -s /bin/notanexistingpath %s", subPathDir))

				// Pod should fail
				testPodFailSubpath(f, pod)
			})

			It("should fail if subpath with backstepping is outside the volume [Slow]", func() {
				// Create the subpath outside the volume
				setInitCommand(pod, fmt.Sprintf("ln -s ../ %s", subPathDir))

				// Pod should fail
				testPodFailSubpath(f, pod)
			})

			It("should support creating multiple subpath from same volumes [Slow]", func() {
				subpathDir1 := filepath.Join(volumePath, "subpath1")
				subpathDir2 := filepath.Join(volumePath, "subpath2")
				filepath1 := filepath.Join("/test-subpath1", fileName)
				filepath2 := filepath.Join("/test-subpath2", fileName)
				setInitCommand(pod, fmt.Sprintf("mkdir -p %s; mkdir -p %s", subpathDir1, subpathDir2))

				addSubpathVolumeContainer(&pod.Spec.Containers[0], v1.VolumeMount{
					Name:      volumeName,
					MountPath: "/test-subpath1",
					SubPath:   "subpath1",
				})
				addSubpathVolumeContainer(&pod.Spec.Containers[0], v1.VolumeMount{
					Name:      volumeName,
					MountPath: "/test-subpath2",
					SubPath:   "subpath2",
				})

				addMultipleWrites(&pod.Spec.Containers[0], filepath1, filepath2)
				testMultipleReads(f, pod, 0, filepath1, filepath2)
			})

			It("should support restarting containers using directory as subpath [Slow]", func() {
				// Create the directory
				setInitCommand(pod, fmt.Sprintf("mkdir -p %v; touch %v", subPathDir, probeFilePath))

				testPodContainerRestart(f, pod)
			})

			It("should support restarting containers using file as subpath [Slow]", func() {
				// Create the file
				setInitCommand(pod, fmt.Sprintf("touch %v; touch %v", subPathDir, probeFilePath))

				testPodContainerRestart(f, pod)
			})

			It("should unmount if pod is gracefully deleted while kubelet is down [Disruptive][Slow]", func() {
				testSubpathReconstruction(f, pod, false)
			})

			It("should unmount if pod is force deleted while kubelet is down [Disruptive][Slow]", func() {
				if curVolType == "hostPath" || curVolType == "hostPathSymlink" {
					framework.Skipf("%s volume type does not support reconstruction, skipping", curVolType)
				}
				testSubpathReconstruction(f, pod, true)
			})

			It("should support readOnly directory specified in the volumeMount", func() {
				// Create the directory
				setInitCommand(pod, fmt.Sprintf("mkdir -p %s", subPathDir))

				// Write the file in the volume from container 1
				setWriteCommand(filePathInVolume, &pod.Spec.Containers[1])

				// Read it from inside the subPath from container 0
				pod.Spec.Containers[0].VolumeMounts[0].ReadOnly = true
				testReadFile(f, filePathInSubpath, pod, 0)
			})

			It("should support readOnly file specified in the volumeMount", func() {
				// Create the file
				setInitCommand(pod, fmt.Sprintf("touch %s", subPathDir))

				// Write the file in the volume from container 1
				setWriteCommand(subPathDir, &pod.Spec.Containers[1])

				// Read it from inside the subPath from container 0
				pod.Spec.Containers[0].VolumeMounts[0].ReadOnly = true
				testReadFile(f, volumePath, pod, 0)
			})

			It("should support existing directories when readOnly specified in the volumeSource", func() {
				roVol := vol.getReadOnlyVolumeSpec()
				if roVol == nil {
					framework.Skipf("Volume type %v doesn't support readOnly source", curVolType)
				}

				// Initialize content in the volume while it's writable
				initVolumeContent(f, pod, filePathInVolume, filePathInSubpath)

				// Set volume source to read only
				pod.Spec.Volumes[0].VolumeSource = *roVol

				// Read it from inside the subPath from container 0
				testReadFile(f, filePathInSubpath, pod, 0)
			})

			It("should fail for new directories when readOnly specified in the volumeSource", func() {
				roVol := vol.getReadOnlyVolumeSpec()
				if roVol == nil {
					framework.Skipf("Volume type %v doesn't support readOnly source", curVolType)
				}

				// Set volume source to read only
				pod.Spec.Volumes[0].VolumeSource = *roVol

				// Pod should fail
				testPodFailSubpathError(f, pod, "")
			})
		})
	}

	// TODO: add a test case for the same disk with two partitions
})

func testBasicSubpath(f *framework.Framework, contents string, pod *v1.Pod) {
	testBasicSubpathFile(f, contents, pod, volumePath)
}

func testBasicSubpathFile(f *framework.Framework, contents string, pod *v1.Pod, filepath string) {
	setReadCommand(filepath, &pod.Spec.Containers[0])

	By(fmt.Sprintf("Creating pod %s", pod.Name))
	f.TestContainerOutput("atomic-volume-subpath", pod, 0, []string{contents})

	By(fmt.Sprintf("Deleting pod %s", pod.Name))
	err := framework.DeletePodWithWait(f, f.ClientSet, pod)
	Expect(err).NotTo(HaveOccurred(), "while deleting pod")
}

func testPodSubpath(f *framework.Framework, subpath, volumeType string, source *v1.VolumeSource) *v1.Pod {
	var (
		suffix          = strings.ToLower(fmt.Sprintf("%s-%s", volumeType, rand.String(4)))
		privileged      = true
		gracePeriod     = int64(1)
		probeVolumeName = "liveness-probe-volume"
	)
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("pod-subpath-test-%s", suffix),
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{
				{
					Name:  fmt.Sprintf("init-volume-%s", suffix),
					Image: "busybox",
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: volumePath,
						},
						{
							Name:      probeVolumeName,
							MountPath: probeVolumePath,
						},
					},
					SecurityContext: &v1.SecurityContext{
						Privileged: &privileged,
					},
				},
			},
			Containers: []v1.Container{
				{
					Name:  fmt.Sprintf("test-container-subpath-%s", suffix),
					Image: mountImage,
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: volumePath,
							SubPath:   subpath,
						},
						{
							Name:      probeVolumeName,
							MountPath: probeVolumePath,
						},
					},
					SecurityContext: &v1.SecurityContext{
						Privileged: &privileged,
					},
				},
				{
					Name:  fmt.Sprintf("test-container-volume-%s", suffix),
					Image: mountImage,
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: volumePath,
						},
						{
							Name:      probeVolumeName,
							MountPath: probeVolumePath,
						},
					},
					SecurityContext: &v1.SecurityContext{
						Privileged: &privileged,
					},
				},
			},
			RestartPolicy:                 v1.RestartPolicyNever,
			TerminationGracePeriodSeconds: &gracePeriod,
			Volumes: []v1.Volume{
				{
					Name:         volumeName,
					VolumeSource: *source,
				},
				{
					Name: probeVolumeName,
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{},
					},
				},
			},
			SecurityContext: &v1.PodSecurityContext{
				SELinuxOptions: &v1.SELinuxOptions{
					Level: "s0:c0,c1",
				},
			},
		},
	}
}

func clearSubpathPodCommands(pod *v1.Pod) {
	pod.Spec.InitContainers[0].Command = nil
	pod.Spec.Containers[0].Args = nil
	pod.Spec.Containers[1].Args = nil
}

func setInitCommand(pod *v1.Pod, command string) {
	pod.Spec.InitContainers[0].Command = []string{"/bin/sh", "-ec", command}
}

func setWriteCommand(file string, container *v1.Container) {
	container.Args = []string{
		fmt.Sprintf("--new_file_0644=%v", file),
		fmt.Sprintf("--file_mode=%v", file),
	}
}

func addSubpathVolumeContainer(container *v1.Container, volumeMount v1.VolumeMount) {
	existingMounts := container.VolumeMounts
	container.VolumeMounts = append(existingMounts, volumeMount)
}

func addMultipleWrites(container *v1.Container, file1 string, file2 string) {
	container.Args = []string{
		fmt.Sprintf("--new_file_0644=%v", file1),
		fmt.Sprintf("--new_file_0666=%v", file2),
	}
}

func testMultipleReads(f *framework.Framework, pod *v1.Pod, containerIndex int, file1 string, file2 string) {
	By(fmt.Sprintf("Creating pod %s", pod.Name))
	f.TestContainerOutput("multi_subpath", pod, containerIndex, []string{
		"content of file \"" + file1 + "\": mount-tester new file",
		"content of file \"" + file2 + "\": mount-tester new file",
	})
}

func setReadCommand(file string, container *v1.Container) {
	container.Args = []string{
		fmt.Sprintf("--file_content_in_loop=%v", file),
		fmt.Sprintf("--retry_time=%d", retryDuration),
	}
}

func testReadFile(f *framework.Framework, file string, pod *v1.Pod, containerIndex int) {
	setReadCommand(file, &pod.Spec.Containers[containerIndex])

	By(fmt.Sprintf("Creating pod %s", pod.Name))
	f.TestContainerOutput("subpath", pod, containerIndex, []string{
		"content of file \"" + file + "\": mount-tester new file",
	})

	By(fmt.Sprintf("Deleting pod %s", pod.Name))
	err := framework.DeletePodWithWait(f, f.ClientSet, pod)
	Expect(err).NotTo(HaveOccurred(), "while deleting pod")
}

func testPodFailSubpath(f *framework.Framework, pod *v1.Pod) {
	testPodFailSubpathError(f, pod, "subPath")
}

func testPodFailSubpathError(f *framework.Framework, pod *v1.Pod, errorMsg string) {
	By(fmt.Sprintf("Creating pod %s", pod.Name))
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
	Expect(err).ToNot(HaveOccurred(), "while creating pod")
	defer func() {
		framework.DeletePodWithWait(f, f.ClientSet, pod)
	}()
	err = framework.WaitTimeoutForPodRunningInNamespace(f.ClientSet, pod.Name, pod.Namespace, time.Minute)
	Expect(err).To(HaveOccurred(), "while waiting for pod to be running")

	By("Checking for subpath error event")
	selector := fields.Set{
		"involvedObject.kind":      "Pod",
		"involvedObject.name":      pod.Name,
		"involvedObject.namespace": f.Namespace.Name,
		"reason":                   "Failed",
	}.AsSelector().String()
	options := metav1.ListOptions{FieldSelector: selector}
	events, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(options)
	Expect(err).NotTo(HaveOccurred(), "while getting pod events")
	Expect(len(events.Items)).NotTo(Equal(0), "no events found")
	Expect(events.Items[0].Message).To(ContainSubstring(errorMsg), fmt.Sprintf("%q error not found", errorMsg))
}

// Tests that the existing subpath mount is detected when a container restarts
func testPodContainerRestart(f *framework.Framework, pod *v1.Pod) {
	pod.Spec.RestartPolicy = v1.RestartPolicyOnFailure

	pod.Spec.Containers[0].Image = "busybox"
	pod.Spec.Containers[0].Command = []string{"/bin/sh", "-ec", "sleep 100000"}
	pod.Spec.Containers[1].Image = "busybox"
	pod.Spec.Containers[1].Command = []string{"/bin/sh", "-ec", "sleep 100000"}

	// Add liveness probe to subpath container
	pod.Spec.Containers[0].LivenessProbe = &v1.Probe{
		Handler: v1.Handler{
			Exec: &v1.ExecAction{
				Command: []string{"cat", probeFilePath},
			},
		},
		InitialDelaySeconds: 1,
		FailureThreshold:    1,
		PeriodSeconds:       2,
	}

	// Start pod
	By(fmt.Sprintf("Creating pod %s", pod.Name))
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
	Expect(err).ToNot(HaveOccurred(), "while creating pod")

	err = framework.WaitTimeoutForPodRunningInNamespace(f.ClientSet, pod.Name, pod.Namespace, time.Minute)
	Expect(err).ToNot(HaveOccurred(), "while waiting for pod to be running")

	By("Failing liveness probe")
	out, err := podContainerExec(pod, 1, fmt.Sprintf("rm %v", probeFilePath))
	framework.Logf("Pod exec output: %v", out)
	Expect(err).ToNot(HaveOccurred(), "while failing liveness probe")

	// Check that container has restarted
	By("Waiting for container to restart")
	restarts := int32(0)
	err = wait.PollImmediate(10*time.Second, 2*time.Minute, func() (bool, error) {
		pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(pod.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		for _, status := range pod.Status.ContainerStatuses {
			if status.Name == pod.Spec.Containers[0].Name {
				framework.Logf("Container %v, restarts: %v", status.Name, status.RestartCount)
				restarts = status.RestartCount
				if restarts > 0 {
					framework.Logf("Container has restart count: %v", restarts)
					return true, nil
				}
			}
		}
		return false, nil
	})
	Expect(err).ToNot(HaveOccurred(), "while waiting for container to restart")

	// Fix liveness probe
	By("Rewriting the file")
	writeCmd := fmt.Sprintf("echo test-after > %v", probeFilePath)
	out, err = podContainerExec(pod, 1, writeCmd)
	framework.Logf("Pod exec output: %v", out)
	Expect(err).ToNot(HaveOccurred(), "while rewriting the probe file")

	// Wait for container restarts to stabilize
	By("Waiting for container to stop restarting")
	stableCount := int(0)
	stableThreshold := int(time.Minute / framework.Poll)
	err = wait.PollImmediate(framework.Poll, 2*time.Minute, func() (bool, error) {
		pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(pod.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		for _, status := range pod.Status.ContainerStatuses {
			if status.Name == pod.Spec.Containers[0].Name {
				if status.RestartCount == restarts {
					stableCount++
					if stableCount > stableThreshold {
						framework.Logf("Container restart has stabilized")
						return true, nil
					}
				} else {
					restarts = status.RestartCount
					stableCount = 0
					framework.Logf("Container has restart count: %v", restarts)
				}
				break
			}
		}
		return false, nil
	})
	Expect(err).ToNot(HaveOccurred(), "while waiting for container to stabilize")
}

func testSubpathReconstruction(f *framework.Framework, pod *v1.Pod, forceDelete bool) {
	// This is mostly copied from TestVolumeUnmountsFromDeletedPodWithForceOption()

	// Change to busybox
	pod.Spec.Containers[0].Image = "busybox"
	pod.Spec.Containers[0].Command = []string{"/bin/sh", "-ec", "sleep 100000"}
	pod.Spec.Containers[1].Image = "busybox"
	pod.Spec.Containers[1].Command = []string{"/bin/sh", "-ec", "sleep 100000"}

	// If grace period is too short, then there is not enough time for the volume
	// manager to cleanup the volumes
	gracePeriod := int64(30)
	pod.Spec.TerminationGracePeriodSeconds = &gracePeriod

	By(fmt.Sprintf("Creating pod %s", pod.Name))
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
	Expect(err).ToNot(HaveOccurred(), "while creating pod")

	err = framework.WaitTimeoutForPodRunningInNamespace(f.ClientSet, pod.Name, pod.Namespace, time.Minute)
	Expect(err).ToNot(HaveOccurred(), "while waiting for pod to be running")

	pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(pod.Name, metav1.GetOptions{})
	Expect(err).ToNot(HaveOccurred(), "while getting pod")

	utils.TestVolumeUnmountsFromDeletedPodWithForceOption(f.ClientSet, f, pod, forceDelete, true)
}

func initVolumeContent(f *framework.Framework, pod *v1.Pod, volumeFilepath, subpathFilepath string) {
	setWriteCommand(volumeFilepath, &pod.Spec.Containers[1])
	setReadCommand(subpathFilepath, &pod.Spec.Containers[0])

	By(fmt.Sprintf("Creating pod to write volume content %s", pod.Name))
	f.TestContainerOutput("subpath", pod, 0, []string{
		"content of file \"" + subpathFilepath + "\": mount-tester new file",
	})

	By(fmt.Sprintf("Deleting pod %s", pod.Name))
	err := framework.DeletePodWithWait(f, f.ClientSet, pod)
	Expect(err).NotTo(HaveOccurred(), "while deleting pod")

	// This pod spec is going to be reused; reset all the commands
	clearSubpathPodCommands(pod)
}

func podContainerExec(pod *v1.Pod, containerIndex int, bashExec string) (string, error) {
	return framework.RunKubectl("exec", fmt.Sprintf("--namespace=%s", pod.Namespace), pod.Name, "--container", pod.Spec.Containers[containerIndex].Name, "--", "/bin/sh", "-c", bashExec)
}

type hostpathSource struct {
}

func initHostpath() volSource {
	return &hostpathSource{}
}

func (s *hostpathSource) createVolume(f *framework.Framework) volInfo {
	return volInfo{
		source: &v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: "/tmp",
			},
		},
	}
}

func (s *hostpathSource) getReadOnlyVolumeSpec() *v1.VolumeSource {
	return nil
}

func (s *hostpathSource) cleanupVolume(f *framework.Framework) {
}

type hostpathSymlinkSource struct {
}

func initHostpathSymlink() volSource {
	return &hostpathSymlinkSource{}
}

func (s *hostpathSymlinkSource) createVolume(f *framework.Framework) volInfo {
	nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
	Expect(len(nodes.Items)).NotTo(BeZero(), "No available nodes for scheduling")

	node0 := &nodes.Items[0]
	sourcePath := fmt.Sprintf("/tmp/%v", f.Namespace.Name)
	targetPath := fmt.Sprintf("/tmp/%v-link", f.Namespace.Name)
	cmd := fmt.Sprintf("mkdir %v -m 777 && ln -s %v %v", sourcePath, sourcePath, targetPath)
	privileged := true

	// Launch pod to initialize hostpath directory and symlink
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("hostpath-symlink-prep-%s", f.Namespace.Name),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    fmt.Sprintf("init-volume-%s", f.Namespace.Name),
					Image:   "busybox",
					Command: []string{"/bin/sh", "-ec", cmd},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: "/tmp",
						},
					},
					SecurityContext: &v1.SecurityContext{
						Privileged: &privileged,
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name: volumeName,
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/tmp",
						},
					},
				},
			},
			NodeName: node0.Name,
		},
	}
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
	Expect(err).ToNot(HaveOccurred(), "while creating hostpath init pod")

	err = framework.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, pod.Namespace)
	Expect(err).ToNot(HaveOccurred(), "while waiting for hostpath init pod to succeed")

	err = framework.DeletePodWithWait(f, f.ClientSet, pod)
	Expect(err).ToNot(HaveOccurred(), "while deleting hostpath init pod")

	return volInfo{
		source: &v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: targetPath,
			},
		},
		node: node0.Name,
	}
}

func (s *hostpathSymlinkSource) getReadOnlyVolumeSpec() *v1.VolumeSource {
	return nil
}

func (s *hostpathSymlinkSource) cleanupVolume(f *framework.Framework) {
}

type emptydirSource struct {
}

func initEmptydir() volSource {
	return &emptydirSource{}
}

func (s *emptydirSource) createVolume(f *framework.Framework) volInfo {
	return volInfo{
		source: &v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
		},
	}
}

func (s *emptydirSource) getReadOnlyVolumeSpec() *v1.VolumeSource {
	return nil
}

func (s *emptydirSource) cleanupVolume(f *framework.Framework) {
}

type gcepdPVCSource struct {
	pvc *v1.PersistentVolumeClaim
}

func initGCEPDPVC() volSource {
	framework.SkipUnlessProviderIs("gce", "gke")
	return &gcepdPVCSource{}
}

func (s *gcepdPVCSource) createVolume(f *framework.Framework) volInfo {
	var err error

	framework.Logf("Creating GCE PD volume via dynamic provisioning")
	testCase := storageClassTest{
		name:      "subpath",
		claimSize: "2G",
	}

	pvc := newClaim(testCase, f.Namespace.Name, "subpath")
	s.pvc, err = framework.CreatePVC(f.ClientSet, f.Namespace.Name, pvc)
	framework.ExpectNoError(err, "Error creating PVC")

	// Launch pod to format the PD first
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("gcepd-prep-%s", f.Namespace.Name),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    fmt.Sprintf("init-volume-%s", f.Namespace.Name),
					Image:   "busybox",
					Command: []string{"/bin/sh", "-ec", "echo nothing"},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: "/vol",
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name: volumeName,
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: s.pvc.Name,
						},
					},
				},
			},
		},
	}
	pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
	Expect(err).ToNot(HaveOccurred(), "while creating gce pd init pod")

	err = framework.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, pod.Namespace)
	Expect(err).ToNot(HaveOccurred(), "while waiting for gce pd init pod to succeed")

	err = framework.DeletePodWithWait(f, f.ClientSet, pod)
	Expect(err).ToNot(HaveOccurred(), "while deleting gce pd init pod")

	return volInfo{
		source: &v1.VolumeSource{
			PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
				ClaimName: s.pvc.Name,
			},
		},
	}
}

func (s *gcepdPVCSource) getReadOnlyVolumeSpec() *v1.VolumeSource {
	return &v1.VolumeSource{
		PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
			ClaimName: s.pvc.Name,
			ReadOnly:  true,
		},
	}
}

func (s *gcepdPVCSource) cleanupVolume(f *framework.Framework) {
	if s.pvc != nil {
		err := f.ClientSet.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Delete(s.pvc.Name, nil)
		framework.ExpectNoError(err, "Error deleting PVC")
	}
}

type gcepdPartitionSource struct {
	diskName string
}

func initGCEPDPartition() volSource {
	// Need to manually create, attach, partition, detach the GCE PD
	// with disk name "subpath-partitioned-disk" before running this test
	manual := true
	if manual {
		framework.Skipf("Skipping manual GCE PD partition test")
	}
	framework.SkipUnlessProviderIs("gce", "gke")
	return &gcepdPartitionSource{diskName: "subpath-partitioned-disk"}
}

func (s *gcepdPartitionSource) createVolume(f *framework.Framework) volInfo {
	// TODO: automate partitioned of GCE PD once it supports raw block volumes
	// framework.Logf("Creating GCE PD volume")
	// s.diskName, err = framework.CreatePDWithRetry()
	// framework.ExpectNoError(err, "Error creating PD")

	return volInfo{
		source: &v1.VolumeSource{
			GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
				PDName:    s.diskName,
				Partition: 1,
			},
		},
	}
}

func (s *gcepdPartitionSource) getReadOnlyVolumeSpec() *v1.VolumeSource {
	return nil
}

func (s *gcepdPartitionSource) cleanupVolume(f *framework.Framework) {
	if s.diskName != "" {
		// err := framework.DeletePDWithRetry(s.diskName)
		// framework.ExpectNoError(err, "Error deleting PD")
	}
}

type nfsSource struct {
	serverPod *v1.Pod
	serverIP  string
}

func initNFS() volSource {
	return &nfsSource{}
}

func (s *nfsSource) createVolume(f *framework.Framework) volInfo {
	framework.Logf("Creating NFS server")
	_, s.serverPod, s.serverIP = framework.NewNFSServer(f.ClientSet, f.Namespace.Name, []string{"-G", "777", "/exports"})

	return volInfo{
		source: &v1.VolumeSource{
			NFS: &v1.NFSVolumeSource{
				Server: s.serverIP,
				Path:   "/exports",
			},
		},
	}
}

func (s *nfsSource) getReadOnlyVolumeSpec() *v1.VolumeSource {
	return &v1.VolumeSource{
		NFS: &v1.NFSVolumeSource{
			Server:   s.serverIP,
			Path:     "/exports",
			ReadOnly: true,
		},
	}
}

func (s *nfsSource) cleanupVolume(f *framework.Framework) {
	if s.serverPod != nil {
		framework.DeletePodWithWait(f, f.ClientSet, s.serverPod)
	}
}

type glusterSource struct {
	serverPod *v1.Pod
}

func initGluster() volSource {
	framework.SkipUnlessNodeOSDistroIs("gci", "ubuntu")
	return &glusterSource{}
}

func (s *glusterSource) createVolume(f *framework.Framework) volInfo {
	framework.Logf("Creating GlusterFS server")
	_, s.serverPod, _ = framework.NewGlusterfsServer(f.ClientSet, f.Namespace.Name)

	return volInfo{
		source: &v1.VolumeSource{
			Glusterfs: &v1.GlusterfsVolumeSource{
				EndpointsName: "gluster-server",
				Path:          "test_vol",
			},
		},
	}
}

func (s *glusterSource) getReadOnlyVolumeSpec() *v1.VolumeSource {
	return &v1.VolumeSource{
		Glusterfs: &v1.GlusterfsVolumeSource{
			EndpointsName: "gluster-server",
			Path:          "test_vol",
			ReadOnly:      true,
		},
	}
}

func (s *glusterSource) cleanupVolume(f *framework.Framework) {
	if s.serverPod != nil {
		framework.DeletePodWithWait(f, f.ClientSet, s.serverPod)
		err := f.ClientSet.CoreV1().Endpoints(f.Namespace.Name).Delete("gluster-server", nil)
		Expect(err).NotTo(HaveOccurred(), "Gluster delete endpoints failed")
	}
}

// TODO: need a better way to wrap PVC.  A generic framework should support both static and dynamic PV.
// For static PV, can reuse createVolume methods for inline volumes
type nfsPVCSource struct {
	serverPod *v1.Pod
	pvc       *v1.PersistentVolumeClaim
	pv        *v1.PersistentVolume
}

func initNFSPVC() volSource {
	return &nfsPVCSource{}
}

func (s *nfsPVCSource) createVolume(f *framework.Framework) volInfo {
	var serverIP string

	framework.Logf("Creating NFS server")
	_, s.serverPod, serverIP = framework.NewNFSServer(f.ClientSet, f.Namespace.Name, []string{"-G", "777", "/exports"})

	pvConfig := framework.PersistentVolumeConfig{
		NamePrefix:       "nfs-",
		StorageClassName: f.Namespace.Name,
		PVSource: v1.PersistentVolumeSource{
			NFS: &v1.NFSVolumeSource{
				Server: serverIP,
				Path:   "/exports",
			},
		},
	}
	pvcConfig := framework.PersistentVolumeClaimConfig{
		StorageClassName: &f.Namespace.Name,
	}

	framework.Logf("Creating PVC and PV")
	pv, pvc, err := framework.CreatePVCPV(f.ClientSet, pvConfig, pvcConfig, f.Namespace.Name, false)
	Expect(err).NotTo(HaveOccurred(), "PVC, PV creation failed")

	err = framework.WaitOnPVandPVC(f.ClientSet, f.Namespace.Name, pv, pvc)
	Expect(err).NotTo(HaveOccurred(), "PVC, PV failed to bind")

	s.pvc = pvc
	s.pv = pv

	return volInfo{
		source: &v1.VolumeSource{
			PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
				ClaimName: pvc.Name,
			},
		},
	}
}

func (s *nfsPVCSource) getReadOnlyVolumeSpec() *v1.VolumeSource {
	return &v1.VolumeSource{
		PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
			ClaimName: s.pvc.Name,
			ReadOnly:  true,
		},
	}
}

func (s *nfsPVCSource) cleanupVolume(f *framework.Framework) {
	if s.pvc != nil || s.pv != nil {
		if errs := framework.PVPVCCleanup(f.ClientSet, f.Namespace.Name, s.pv, s.pvc); len(errs) != 0 {
			framework.Failf("Failed to delete PVC or PV: %v", utilerrors.NewAggregate(errs))
		}
	}
	if s.serverPod != nil {
		framework.DeletePodWithWait(f, f.ClientSet, s.serverPod)
	}
}
