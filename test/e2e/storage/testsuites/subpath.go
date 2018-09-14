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
	"strings"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
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

func (s *subPathTestSuite) skipUnsupportedTest(pattern testpatterns.TestPattern, driver drivers.TestDriver) {
}

func createSubPathTestInput(pattern testpatterns.TestPattern, resource subPathTestResource) subPathTestInput {
	driver := resource.driver
	dInfo := driver.GetDriverInfo()
	f := dInfo.Framework
	subPath := f.Namespace.Name
	subPathDir := filepath.Join(volumePath, subPath)

	return subPathTestInput{
		f:                 f,
		subPathDir:        subPathDir,
		filePathInSubpath: filepath.Join(volumePath, fileName),
		filePathInVolume:  filepath.Join(subPathDir, fileName),
		volType:           resource.volType,
		pod:               resource.pod,
		formatPod:         resource.formatPod,
		volSource:         resource.genericVolumeTestResource.volSource,
		roVol:             resource.roVolSource,
	}
}

func (s *subPathTestSuite) execTest(driver drivers.TestDriver, pattern testpatterns.TestPattern) {
	Context(getTestNameStr(s, pattern), func() {
		var (
			resource     subPathTestResource
			input        subPathTestInput
			needsCleanup bool
		)

		BeforeEach(func() {
			needsCleanup = false
			// Skip unsupported tests to avoid unnecessary resource initialization
			skipUnsupportedTest(s, driver, pattern)
			needsCleanup = true

			// Setup test resource for driver and testpattern
			resource = subPathTestResource{}
			resource.setupResource(driver, pattern)

			// Create test input
			input = createSubPathTestInput(pattern, resource)
		})

		AfterEach(func() {
			if needsCleanup {
				resource.cleanupResource(driver, pattern)
			}
		})

		testSubPath(&input)
	})
}

type subPathTestResource struct {
	genericVolumeTestResource

	roVolSource *v1.VolumeSource
	pod         *v1.Pod
	formatPod   *v1.Pod
}

var _ TestResource = &subPathTestResource{}

func (s *subPathTestResource) setupResource(driver drivers.TestDriver, pattern testpatterns.TestPattern) {
	s.driver = driver
	dInfo := s.driver.GetDriverInfo()
	f := dInfo.Framework
	fsType := pattern.FsType
	volType := pattern.VolType

	// Setup generic test resource
	s.genericVolumeTestResource.setupResource(driver, pattern)

	// Setup subPath test dependent resource
	switch volType {
	case testpatterns.InlineVolume:
		if iDriver, ok := driver.(drivers.InlineVolumeTestDriver); ok {
			s.roVolSource = iDriver.GetVolumeSource(true, fsType, s.genericVolumeTestResource.driverTestResource)
		}
	case testpatterns.PreprovisionedPV:
		s.roVolSource = &v1.VolumeSource{
			PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
				ClaimName: s.genericVolumeTestResource.pvc.Name,
				ReadOnly:  true,
			},
		}
	case testpatterns.DynamicPV:
		s.roVolSource = &v1.VolumeSource{
			PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
				ClaimName: s.genericVolumeTestResource.pvc.Name,
				ReadOnly:  true,
			},
		}
	default:
		framework.Failf("SubPath test doesn't support: %s", volType)
	}

	subPath := f.Namespace.Name
	config := dInfo.Config
	s.pod = SubpathTestPod(f, subPath, s.volType, s.volSource, true)
	s.pod.Spec.NodeName = config.ClientNodeName
	s.pod.Spec.NodeSelector = config.NodeSelector

	s.formatPod = volumeFormatPod(f, s.volSource)
	s.formatPod.Spec.NodeName = config.ClientNodeName
	s.formatPod.Spec.NodeSelector = config.NodeSelector
}

func (s *subPathTestResource) cleanupResource(driver drivers.TestDriver, pattern testpatterns.TestPattern) {
	dInfo := driver.GetDriverInfo()
	f := dInfo.Framework

	// Cleanup subPath test dependent resource
	By("Deleting pod")
	err := framework.DeletePodWithWait(f, f.ClientSet, s.pod)
	Expect(err).ToNot(HaveOccurred(), "while deleting pod")

	// Cleanup generic test resource
	s.genericVolumeTestResource.cleanupResource(driver, pattern)
}

type subPathTestInput struct {
	f                 *framework.Framework
	subPathDir        string
	filePathInSubpath string
	filePathInVolume  string
	volType           string
	pod               *v1.Pod
	formatPod         *v1.Pod
	volSource         *v1.VolumeSource
	roVol             *v1.VolumeSource
}

func testSubPath(input *subPathTestInput) {
	It("should support non-existent path", func() {
		// Write the file in the subPath from container 0
		setWriteCommand(input.filePathInSubpath, &input.pod.Spec.Containers[0])

		// Read it from outside the subPath from container 1
		testReadFile(input.f, input.filePathInVolume, input.pod, 1)
	})

	It("should support existing directory", func() {
		// Create the directory
		setInitCommand(input.pod, fmt.Sprintf("mkdir -p %s", input.subPathDir))

		// Write the file in the subPath from container 0
		setWriteCommand(input.filePathInSubpath, &input.pod.Spec.Containers[0])

		// Read it from outside the subPath from container 1
		testReadFile(input.f, input.filePathInVolume, input.pod, 1)
	})

	It("should support existing single file", func() {
		// Create the file in the init container
		setInitCommand(input.pod, fmt.Sprintf("mkdir -p %s; echo \"mount-tester new file\" > %s", input.subPathDir, input.filePathInVolume))

		// Read it from inside the subPath from container 0
		testReadFile(input.f, input.filePathInSubpath, input.pod, 0)
	})

	It("should support file as subpath", func() {
		// Create the file in the init container
		setInitCommand(input.pod, fmt.Sprintf("echo %s > %s", input.f.Namespace.Name, input.subPathDir))

		TestBasicSubpath(input.f, input.f.Namespace.Name, input.pod)
	})

	It("should fail if subpath directory is outside the volume [Slow]", func() {
		// Create the subpath outside the volume
		setInitCommand(input.pod, fmt.Sprintf("ln -s /bin %s", input.subPathDir))

		// Pod should fail
		testPodFailSubpath(input.f, input.pod)
	})

	It("should fail if subpath file is outside the volume [Slow]", func() {
		// Create the subpath outside the volume
		setInitCommand(input.pod, fmt.Sprintf("ln -s /bin/sh %s", input.subPathDir))

		// Pod should fail
		testPodFailSubpath(input.f, input.pod)
	})

	It("should fail if non-existent subpath is outside the volume [Slow]", func() {
		// Create the subpath outside the volume
		setInitCommand(input.pod, fmt.Sprintf("ln -s /bin/notanexistingpath %s", input.subPathDir))

		// Pod should fail
		testPodFailSubpath(input.f, input.pod)
	})

	It("should fail if subpath with backstepping is outside the volume [Slow]", func() {
		// Create the subpath outside the volume
		setInitCommand(input.pod, fmt.Sprintf("ln -s ../ %s", input.subPathDir))

		// Pod should fail
		testPodFailSubpath(input.f, input.pod)
	})

	It("should support creating multiple subpath from same volumes [Slow]", func() {
		subpathDir1 := filepath.Join(volumePath, "subpath1")
		subpathDir2 := filepath.Join(volumePath, "subpath2")
		filepath1 := filepath.Join("/test-subpath1", fileName)
		filepath2 := filepath.Join("/test-subpath2", fileName)
		setInitCommand(input.pod, fmt.Sprintf("mkdir -p %s; mkdir -p %s", subpathDir1, subpathDir2))

		addSubpathVolumeContainer(&input.pod.Spec.Containers[0], v1.VolumeMount{
			Name:      volumeName,
			MountPath: "/test-subpath1",
			SubPath:   "subpath1",
		})
		addSubpathVolumeContainer(&input.pod.Spec.Containers[0], v1.VolumeMount{
			Name:      volumeName,
			MountPath: "/test-subpath2",
			SubPath:   "subpath2",
		})

		addMultipleWrites(&input.pod.Spec.Containers[0], filepath1, filepath2)
		testMultipleReads(input.f, input.pod, 0, filepath1, filepath2)
	})

	It("should support restarting containers using directory as subpath [Slow]", func() {
		// Create the directory
		setInitCommand(input.pod, fmt.Sprintf("mkdir -p %v; touch %v", input.subPathDir, probeFilePath))

		testPodContainerRestart(input.f, input.pod)
	})

	It("should support restarting containers using file as subpath [Slow]", func() {
		// Create the file
		setInitCommand(input.pod, fmt.Sprintf("touch %v; touch %v", input.subPathDir, probeFilePath))

		testPodContainerRestart(input.f, input.pod)
	})

	It("should unmount if pod is gracefully deleted while kubelet is down [Disruptive][Slow]", func() {
		testSubpathReconstruction(input.f, input.pod, false)
	})

	It("should unmount if pod is force deleted while kubelet is down [Disruptive][Slow]", func() {
		if input.volType == "hostPath" || input.volType == "hostPathSymlink" {
			framework.Skipf("%s volume type does not support reconstruction, skipping", input.volType)
		}
		testSubpathReconstruction(input.f, input.pod, true)
	})

	It("should support readOnly directory specified in the volumeMount", func() {
		// Create the directory
		setInitCommand(input.pod, fmt.Sprintf("mkdir -p %s", input.subPathDir))

		// Write the file in the volume from container 1
		setWriteCommand(input.filePathInVolume, &input.pod.Spec.Containers[1])

		// Read it from inside the subPath from container 0
		input.pod.Spec.Containers[0].VolumeMounts[0].ReadOnly = true
		testReadFile(input.f, input.filePathInSubpath, input.pod, 0)
	})

	It("should support readOnly file specified in the volumeMount", func() {
		// Create the file
		setInitCommand(input.pod, fmt.Sprintf("touch %s", input.subPathDir))

		// Write the file in the volume from container 1
		setWriteCommand(input.subPathDir, &input.pod.Spec.Containers[1])

		// Read it from inside the subPath from container 0
		input.pod.Spec.Containers[0].VolumeMounts[0].ReadOnly = true
		testReadFile(input.f, volumePath, input.pod, 0)
	})

	It("should support existing directories when readOnly specified in the volumeSource", func() {
		if input.roVol == nil {
			framework.Skipf("Volume type %v doesn't support readOnly source", input.volType)
		}

		// Initialize content in the volume while it's writable
		initVolumeContent(input.f, input.pod, input.filePathInVolume, input.filePathInSubpath)

		// Set volume source to read only
		input.pod.Spec.Volumes[0].VolumeSource = *input.roVol

		// Read it from inside the subPath from container 0
		testReadFile(input.f, input.filePathInSubpath, input.pod, 0)
	})

	It("should fail for new directories when readOnly specified in the volumeSource", func() {
		if input.roVol == nil {
			framework.Skipf("Volume type %v doesn't support readOnly source", input.volType)
		}

		// Format the volume while it's writable
		formatVolume(input.f, input.formatPod)

		// Set volume source to read only
		input.pod.Spec.Volumes[0].VolumeSource = *input.roVol
		// Pod should fail
		testPodFailSubpathError(input.f, input.pod, "")
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
	f.TestContainerOutput("atomic-volume-subpath", pod, 0, []string{contents})

	By(fmt.Sprintf("Deleting pod %s", pod.Name))
	err := framework.DeletePodWithWait(f, f.ClientSet, pod)
	Expect(err).NotTo(HaveOccurred(), "while deleting pod")
}

// SubpathTestPod returns a pod spec for subpath tests
func SubpathTestPod(f *framework.Framework, subpath, volumeType string, source *v1.VolumeSource, privilegedSecurityContext bool) *v1.Pod {
	var (
		suffix          = strings.ToLower(fmt.Sprintf("%s-%s", volumeType, rand.String(4)))
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
	err = framework.WaitForPodRunningInNamespace(f.ClientSet, pod)
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
