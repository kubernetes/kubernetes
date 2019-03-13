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

package testsuites

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
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
	retryDuration   = 20
	mountImage      = imageutils.GetE2EImage(imageutils.Mounttest)
)

type subPathTestSuite struct {
	tsInfo TestSuiteInfo
}

var _ TestSuite = &subPathTestSuite{}

// InitSubPathTestSuite returns subPathTestSuite that implements TestSuite interface
func InitSubPathTestSuite() TestSuite {
	return &subPathTestSuite{
		tsInfo: TestSuiteInfo{
			name: "subPath",
			testPatterns: []testpatterns.TestPattern{
				testpatterns.DefaultFsInlineVolume,
				testpatterns.DefaultFsPreprovisionedPV,
				testpatterns.DefaultFsDynamicPV,
			},
		},
	}
}

func (s *subPathTestSuite) getTestSuiteInfo() TestSuiteInfo {
	return s.tsInfo
}

func (s *subPathTestSuite) defineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	type local struct {
		config      *PerTestConfig
		testCleanup func()

		resource          *genericVolumeTestResource
		roVolSource       *v1.VolumeSource
		pod               *v1.Pod
		formatPod         *v1.Pod
		subPathDir        string
		filePathInSubpath string
		filePathInVolume  string
	}
	var l local

	// No preconditions to test. Normally they would be in a BeforeEach here.

	// This intentionally comes after checking the preconditions because it
	// registers its own BeforeEach which creates the namespace. Beware that it
	// also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewDefaultFramework("provisioning")

	init := func() {
		l = local{}

		// Now do the more expensive test initialization.
		l.config, l.testCleanup = driver.PrepareTest(f)
		l.resource = createGenericVolumeTestResource(driver, l.config, pattern)

		// Setup subPath test dependent resource
		volType := pattern.VolType
		switch volType {
		case testpatterns.InlineVolume:
			if iDriver, ok := driver.(InlineVolumeTestDriver); ok {
				l.roVolSource = iDriver.GetVolumeSource(true, pattern.FsType, l.resource.volume)
			}
		case testpatterns.PreprovisionedPV:
			l.roVolSource = &v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: l.resource.pvc.Name,
					ReadOnly:  true,
				},
			}
		case testpatterns.DynamicPV:
			l.roVolSource = &v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: l.resource.pvc.Name,
					ReadOnly:  true,
				},
			}
		default:
			framework.Failf("SubPath test doesn't support: %s", volType)
		}

		subPath := f.Namespace.Name
		l.pod = SubpathTestPod(f, subPath, l.resource.volType, l.resource.volSource, true)
		l.pod.Spec.NodeName = l.config.ClientNodeName
		l.pod.Spec.NodeSelector = l.config.ClientNodeSelector

		l.formatPod = volumeFormatPod(f, l.resource.volSource)
		l.formatPod.Spec.NodeName = l.config.ClientNodeName
		l.formatPod.Spec.NodeSelector = l.config.ClientNodeSelector

		l.subPathDir = filepath.Join(volumePath, subPath)
		l.filePathInSubpath = filepath.Join(volumePath, fileName)
		l.filePathInVolume = filepath.Join(l.subPathDir, fileName)
	}

	cleanup := func() {
		if l.pod != nil {
			By("Deleting pod")
			err := framework.DeletePodWithWait(f, f.ClientSet, l.pod)
			Expect(err).ToNot(HaveOccurred(), "while deleting pod")
			l.pod = nil
		}

		if l.resource != nil {
			l.resource.cleanupResource()
			l.resource = nil
		}

		if l.testCleanup != nil {
			l.testCleanup()
			l.testCleanup = nil
		}
	}

	It("should support non-existent path", func() {
		init()
		defer cleanup()

		// Write the file in the subPath from init container 1
		setWriteCommand(l.filePathInSubpath, &l.pod.Spec.InitContainers[1])

		// Read it from outside the subPath from container 1
		testReadFile(f, l.filePathInVolume, l.pod, 1)
	})

	It("should support existing directory", func() {
		init()
		defer cleanup()

		// Create the directory
		setInitCommand(l.pod, fmt.Sprintf("mkdir -p %s", l.subPathDir))

		// Write the file in the subPath from init container 1
		setWriteCommand(l.filePathInSubpath, &l.pod.Spec.InitContainers[1])

		// Read it from outside the subPath from container 1
		testReadFile(f, l.filePathInVolume, l.pod, 1)
	})

	It("should support existing single file", func() {
		init()
		defer cleanup()

		// Create the file in the init container
		setInitCommand(l.pod, fmt.Sprintf("mkdir -p %s; echo \"mount-tester new file\" > %s", l.subPathDir, l.filePathInVolume))

		// Read it from inside the subPath from container 0
		testReadFile(f, l.filePathInSubpath, l.pod, 0)
	})

	It("should support file as subpath", func() {
		init()
		defer cleanup()

		// Create the file in the init container
		setInitCommand(l.pod, fmt.Sprintf("echo %s > %s", f.Namespace.Name, l.subPathDir))

		TestBasicSubpath(f, f.Namespace.Name, l.pod)
	})

	It("should fail if subpath directory is outside the volume [Slow]", func() {
		init()
		defer cleanup()

		// Create the subpath outside the volume
		setInitCommand(l.pod, fmt.Sprintf("ln -s /bin %s", l.subPathDir))

		// Pod should fail
		testPodFailSubpath(f, l.pod, false)
	})

	It("should fail if subpath file is outside the volume [Slow]", func() {
		init()
		defer cleanup()

		// Create the subpath outside the volume
		setInitCommand(l.pod, fmt.Sprintf("ln -s /bin/sh %s", l.subPathDir))

		// Pod should fail
		testPodFailSubpath(f, l.pod, false)
	})

	It("should fail if non-existent subpath is outside the volume [Slow]", func() {
		init()
		defer cleanup()

		// Create the subpath outside the volume
		setInitCommand(l.pod, fmt.Sprintf("ln -s /bin/notanexistingpath %s", l.subPathDir))

		// Pod should fail
		testPodFailSubpath(f, l.pod, false)
	})

	It("should fail if subpath with backstepping is outside the volume [Slow]", func() {
		init()
		defer cleanup()

		// Create the subpath outside the volume
		setInitCommand(l.pod, fmt.Sprintf("ln -s ../ %s", l.subPathDir))

		// Pod should fail
		testPodFailSubpath(f, l.pod, false)
	})

	It("should support creating multiple subpath from same volumes [Slow]", func() {
		init()
		defer cleanup()

		subpathDir1 := filepath.Join(volumePath, "subpath1")
		subpathDir2 := filepath.Join(volumePath, "subpath2")
		filepath1 := filepath.Join("/test-subpath1", fileName)
		filepath2 := filepath.Join("/test-subpath2", fileName)
		setInitCommand(l.pod, fmt.Sprintf("mkdir -p %s; mkdir -p %s", subpathDir1, subpathDir2))

		addSubpathVolumeContainer(&l.pod.Spec.Containers[0], v1.VolumeMount{
			Name:      volumeName,
			MountPath: "/test-subpath1",
			SubPath:   "subpath1",
		})
		addSubpathVolumeContainer(&l.pod.Spec.Containers[0], v1.VolumeMount{
			Name:      volumeName,
			MountPath: "/test-subpath2",
			SubPath:   "subpath2",
		})

		// Write the files from container 0 and instantly read them back
		addMultipleWrites(&l.pod.Spec.Containers[0], filepath1, filepath2)
		testMultipleReads(f, l.pod, 0, filepath1, filepath2)
	})

	It("should support restarting containers using directory as subpath [Slow]", func() {
		init()
		defer cleanup()

		// Create the directory
		setInitCommand(l.pod, fmt.Sprintf("mkdir -p %v; touch %v", l.subPathDir, probeFilePath))

		testPodContainerRestart(f, l.pod)
	})

	It("should support restarting containers using file as subpath [Slow]", func() {
		init()
		defer cleanup()

		// Create the file
		setInitCommand(l.pod, fmt.Sprintf("touch %v; touch %v", l.subPathDir, probeFilePath))

		testPodContainerRestart(f, l.pod)
	})

	It("should unmount if pod is gracefully deleted while kubelet is down [Disruptive][Slow]", func() {
		init()
		defer cleanup()

		testSubpathReconstruction(f, l.pod, false)
	})

	It("should unmount if pod is force deleted while kubelet is down [Disruptive][Slow]", func() {
		init()
		defer cleanup()

		if strings.HasPrefix(l.resource.volType, "hostPath") || strings.HasPrefix(l.resource.volType, "csi-hostpath") {
			// TODO: This skip should be removed once #61446 is fixed
			framework.Skipf("%s volume type does not support reconstruction, skipping", l.resource.volType)
		}

		testSubpathReconstruction(f, l.pod, true)
	})

	It("should support readOnly directory specified in the volumeMount", func() {
		init()
		defer cleanup()

		// Create the directory
		setInitCommand(l.pod, fmt.Sprintf("mkdir -p %s", l.subPathDir))

		// Write the file in the volume from init container 2
		setWriteCommand(l.filePathInVolume, &l.pod.Spec.InitContainers[2])

		// Read it from inside the subPath from container 0
		l.pod.Spec.Containers[0].VolumeMounts[0].ReadOnly = true
		testReadFile(f, l.filePathInSubpath, l.pod, 0)
	})

	It("should support readOnly file specified in the volumeMount", func() {
		init()
		defer cleanup()

		// Create the file
		setInitCommand(l.pod, fmt.Sprintf("touch %s", l.subPathDir))

		// Write the file in the volume from init container 2
		setWriteCommand(l.subPathDir, &l.pod.Spec.InitContainers[2])

		// Read it from inside the subPath from container 0
		l.pod.Spec.Containers[0].VolumeMounts[0].ReadOnly = true
		testReadFile(f, volumePath, l.pod, 0)
	})

	It("should support existing directories when readOnly specified in the volumeSource", func() {
		init()
		defer cleanup()
		if l.roVolSource == nil {
			framework.Skipf("Volume type %v doesn't support readOnly source", l.resource.volType)
		}

		origpod := l.pod.DeepCopy()

		// Create the directory
		setInitCommand(l.pod, fmt.Sprintf("mkdir -p %s", l.subPathDir))

		// Write the file in the subPath from init container 1
		setWriteCommand(l.filePathInSubpath, &l.pod.Spec.InitContainers[1])

		// Read it from inside the subPath from container 0
		testReadFile(f, l.filePathInSubpath, l.pod, 0)

		// Reset the pod
		l.pod = origpod

		// Set volume source to read only
		l.pod.Spec.Volumes[0].VolumeSource = *l.roVolSource

		// Read it from inside the subPath from container 0
		testReadFile(f, l.filePathInSubpath, l.pod, 0)
	})

	It("should verify container cannot write to subpath readonly volumes", func() {
		init()
		defer cleanup()
		if l.roVolSource == nil {
			framework.Skipf("Volume type %v doesn't support readOnly source", l.resource.volType)
		}

		// Format the volume while it's writable
		formatVolume(f, l.formatPod)

		// Set volume source to read only
		l.pod.Spec.Volumes[0].VolumeSource = *l.roVolSource

		// Write the file in the volume from container 0
		setWriteCommand(l.subPathDir, &l.pod.Spec.Containers[0])

		// Pod should fail
		testPodFailSubpath(f, l.pod, true)
	})

	It("should be able to unmount after the subpath directory is deleted", func() {
		init()
		defer cleanup()

		// Change volume container to busybox so we can exec later
		l.pod.Spec.Containers[1].Image = imageutils.GetE2EImage(imageutils.BusyBox)
		l.pod.Spec.Containers[1].Command = []string{"/bin/sh", "-ec", "sleep 100000"}

		By(fmt.Sprintf("Creating pod %s", l.pod.Name))
		removeUnusedContainers(l.pod)
		pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(l.pod)
		Expect(err).ToNot(HaveOccurred(), "while creating pod")
		defer func() {
			By(fmt.Sprintf("Deleting pod %s", pod.Name))
			framework.DeletePodWithWait(f, f.ClientSet, pod)
		}()

		// Wait for pod to be running
		err = framework.WaitForPodRunningInNamespace(f.ClientSet, l.pod)
		Expect(err).ToNot(HaveOccurred(), "while waiting for pod to be running")

		// Exec into container that mounted the volume, delete subpath directory
		rmCmd := fmt.Sprintf("rm -rf %s", l.subPathDir)
		_, err = podContainerExec(l.pod, 1, rmCmd)
		Expect(err).ToNot(HaveOccurred(), "while removing subpath directory")

		// Delete pod (from defer) and wait for it to be successfully deleted
	})

	// TODO: add a test case for the same disk with two partitions
}

// TestBasicSubpath runs basic subpath test
func TestBasicSubpath(f *framework.Framework, contents string, pod *v1.Pod) {
	TestBasicSubpathFile(f, contents, pod, volumePath)
}

// TestBasicSubpathFile runs basic subpath file test
func TestBasicSubpathFile(f *framework.Framework, contents string, pod *v1.Pod, filepath string) {
	setReadCommand(filepath, &pod.Spec.Containers[0])

	By(fmt.Sprintf("Creating pod %s", pod.Name))
	removeUnusedContainers(pod)
	f.TestContainerOutput("atomic-volume-subpath", pod, 0, []string{contents})

	By(fmt.Sprintf("Deleting pod %s", pod.Name))
	err := framework.DeletePodWithWait(f, f.ClientSet, pod)
	Expect(err).NotTo(HaveOccurred(), "while deleting pod")
}

func generateSuffixForPodName(s string) string {
	// Pod name must:
	//   1. consist of lower case alphanumeric characters or '-',
	//   2. start and end with an alphanumeric character.
	// (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')
	// Therefore, suffix is generated by following steps:
	//   1. all strings other than [A-Za-z0-9] is replaced with "-",
	//   2. add lower case alphanumeric characters at the end ('-[a-z0-9]{4}' is added),
	//   3. convert the entire strings to lower case.
	re := regexp.MustCompile("[^A-Za-z0-9]")
	return strings.ToLower(fmt.Sprintf("%s-%s", re.ReplaceAllString(s, "-"), rand.String(4)))
}

// SubpathTestPod returns a pod spec for subpath tests
func SubpathTestPod(f *framework.Framework, subpath, volumeType string, source *v1.VolumeSource, privilegedSecurityContext bool) *v1.Pod {
	var (
		suffix          = generateSuffixForPodName(volumeType)
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
					Image: imageutils.GetE2EImage(imageutils.BusyBox),
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
						Privileged: &privilegedSecurityContext,
					},
				},
				{
					Name:  fmt.Sprintf("test-init-subpath-%s", suffix),
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
						Privileged: &privilegedSecurityContext,
					},
				},
				{
					Name:  fmt.Sprintf("test-init-volume-%s", suffix),
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
						Privileged: &privilegedSecurityContext,
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
						Privileged: &privilegedSecurityContext,
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
						Privileged: &privilegedSecurityContext,
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

func containerIsUnused(container *v1.Container) bool {
	// mountImage with nil Args does nothing. Leave everything else
	return container.Image == mountImage && container.Args == nil
}

// removeUnusedContainers removes containers from a SubpathTestPod that aren't
// needed for a test. e.g. to test for subpath mount failure, only one
// container needs to run and get its status checked.
func removeUnusedContainers(pod *v1.Pod) {
	initContainers := []v1.Container{}
	containers := []v1.Container{}
	if pod.Spec.InitContainers[0].Command != nil {
		initContainers = append(initContainers, pod.Spec.InitContainers[0])
	}
	for _, ic := range pod.Spec.InitContainers[1:] {
		if !containerIsUnused(&ic) {
			initContainers = append(initContainers, ic)
		}
	}
	containers = append(containers, pod.Spec.Containers[0])
	if !containerIsUnused(&pod.Spec.Containers[1]) {
		containers = append(containers, pod.Spec.Containers[1])
	}
	pod.Spec.InitContainers = initContainers
	pod.Spec.Containers = containers
}

// volumeFormatPod returns a Pod that does nothing but will cause the plugin to format a filesystem
// on first use
func volumeFormatPod(f *framework.Framework, volumeSource *v1.VolumeSource) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("volume-prep-%s", f.Namespace.Name),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    fmt.Sprintf("init-volume-%s", f.Namespace.Name),
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
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
					Name:         volumeName,
					VolumeSource: *volumeSource,
				},
			},
		},
	}
}

func clearSubpathPodCommands(pod *v1.Pod) {
	pod.Spec.InitContainers[0].Command = nil
	pod.Spec.InitContainers[1].Args = nil
	pod.Spec.InitContainers[2].Args = nil
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
	removeUnusedContainers(pod)
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
	removeUnusedContainers(pod)
	f.TestContainerOutput("subpath", pod, containerIndex, []string{
		"content of file \"" + file + "\": mount-tester new file",
	})

	By(fmt.Sprintf("Deleting pod %s", pod.Name))
	err := framework.DeletePodWithWait(f, f.ClientSet, pod)
	Expect(err).NotTo(HaveOccurred(), "while deleting pod")
}

func testPodFailSubpath(f *framework.Framework, pod *v1.Pod, allowContainerTerminationError bool) {
	testPodFailSubpathError(f, pod, "subPath", allowContainerTerminationError)
}

func testPodFailSubpathError(f *framework.Framework, pod *v1.Pod, errorMsg string, allowContainerTerminationError bool) {
	By(fmt.Sprintf("Creating pod %s", pod.Name))
	removeUnusedContainers(pod)
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
	Expect(err).ToNot(HaveOccurred(), "while creating pod")
	defer func() {
		framework.DeletePodWithWait(f, f.ClientSet, pod)
	}()
	By("Checking for subpath error in container status")
	err = waitForPodSubpathError(f, pod, allowContainerTerminationError)
	Expect(err).NotTo(HaveOccurred(), "while waiting for subpath failure")
}

func findSubpathContainerName(pod *v1.Pod) string {
	for _, container := range pod.Spec.Containers {
		for _, mount := range container.VolumeMounts {
			if mount.SubPath != "" {
				return container.Name
			}
		}
	}
	return ""
}

func waitForPodSubpathError(f *framework.Framework, pod *v1.Pod, allowContainerTerminationError bool) error {
	subpathContainerName := findSubpathContainerName(pod)
	if subpathContainerName == "" {
		return fmt.Errorf("failed to find container that uses subpath")
	}

	return wait.PollImmediate(framework.Poll, framework.PodStartTimeout, func() (bool, error) {
		pod, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		for _, status := range pod.Status.ContainerStatuses {
			// 0 is the container that uses subpath
			if status.Name == subpathContainerName {
				switch {
				case status.State.Terminated != nil:
					if status.State.Terminated.ExitCode != 0 && allowContainerTerminationError {
						return true, nil
					}
					return false, fmt.Errorf("subpath container unexpectedly terminated")
				case status.State.Waiting != nil:
					if status.State.Waiting.Reason == "CreateContainerConfigError" &&
						strings.Contains(status.State.Waiting.Message, "subPath") {
						return true, nil
					}
					return false, nil
				default:
					return false, nil
				}
			}
		}
		return false, nil
	})
}

// Tests that the existing subpath mount is detected when a container restarts
func testPodContainerRestart(f *framework.Framework, pod *v1.Pod) {
	pod.Spec.RestartPolicy = v1.RestartPolicyOnFailure

	pod.Spec.Containers[0].Image = imageutils.GetE2EImage(imageutils.BusyBox)
	pod.Spec.Containers[0].Command = []string{"/bin/sh", "-ec", "sleep 100000"}
	pod.Spec.Containers[1].Image = imageutils.GetE2EImage(imageutils.BusyBox)
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
	removeUnusedContainers(pod)
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
	Expect(err).ToNot(HaveOccurred(), "while creating pod")
	defer func() {
		framework.DeletePodWithWait(f, f.ClientSet, pod)
	}()
	err = framework.WaitForPodRunningInNamespace(f.ClientSet, pod)
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
	pod.Spec.Containers[0].Image = imageutils.GetE2EImage(imageutils.BusyBox)
	pod.Spec.Containers[0].Command = []string{"/bin/sh", "-ec", "sleep 100000"}
	pod.Spec.Containers[1].Image = imageutils.GetE2EImage(imageutils.BusyBox)
	pod.Spec.Containers[1].Command = []string{"/bin/sh", "-ec", "sleep 100000"}

	// If grace period is too short, then there is not enough time for the volume
	// manager to cleanup the volumes
	gracePeriod := int64(30)
	pod.Spec.TerminationGracePeriodSeconds = &gracePeriod

	By(fmt.Sprintf("Creating pod %s", pod.Name))
	removeUnusedContainers(pod)
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
	Expect(err).ToNot(HaveOccurred(), "while creating pod")

	err = framework.WaitForPodRunningInNamespace(f.ClientSet, pod)
	Expect(err).ToNot(HaveOccurred(), "while waiting for pod to be running")

	pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(pod.Name, metav1.GetOptions{})
	Expect(err).ToNot(HaveOccurred(), "while getting pod")

	utils.TestVolumeUnmountsFromDeletedPodWithForceOption(f.ClientSet, f, pod, forceDelete, true)
}

func formatVolume(f *framework.Framework, pod *v1.Pod) {
	By(fmt.Sprintf("Creating pod to format volume %s", pod.Name))
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
	Expect(err).ToNot(HaveOccurred(), "while creating volume init pod")

	err = framework.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, pod.Namespace)
	Expect(err).ToNot(HaveOccurred(), "while waiting for volume init pod to succeed")

	err = framework.DeletePodWithWait(f, f.ClientSet, pod)
	Expect(err).ToNot(HaveOccurred(), "while deleting volume init pod")
}

func podContainerExec(pod *v1.Pod, containerIndex int, bashExec string) (string, error) {
	return framework.RunKubectl("exec", fmt.Sprintf("--namespace=%s", pod.Namespace), pod.Name, "--container", pod.Spec.Containers[containerIndex].Name, "--", "/bin/sh", "-c", bashExec)
}
