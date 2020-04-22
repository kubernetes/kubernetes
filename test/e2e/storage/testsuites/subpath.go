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
	"context"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

var (
	volumePath      = "/test-volume"
	volumeName      = "test-volume"
	probeVolumePath = "/probe-volume"
	probeFilePath   = probeVolumePath + "/probe-file"
	fileName        = "test-file"
	retryDuration   = 20
	mountImage      = imageutils.GetE2EImage(imageutils.Agnhost)
)

type subPathTestSuite struct {
	tsInfo TestSuiteInfo
}

var _ TestSuite = &subPathTestSuite{}

// InitSubPathTestSuite returns subPathTestSuite that implements TestSuite interface
func InitSubPathTestSuite() TestSuite {
	return &subPathTestSuite{
		tsInfo: TestSuiteInfo{
			Name: "subPath",
			TestPatterns: []testpatterns.TestPattern{
				testpatterns.DefaultFsInlineVolume,
				testpatterns.DefaultFsPreprovisionedPV,
				testpatterns.DefaultFsDynamicPV,
				testpatterns.NtfsDynamicPV,
			},
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
		},
	}
}

func (s *subPathTestSuite) GetTestSuiteInfo() TestSuiteInfo {
	return s.tsInfo
}

func (s *subPathTestSuite) SkipRedundantSuite(driver TestDriver, pattern testpatterns.TestPattern) {
	skipVolTypePatterns(pattern, driver, testpatterns.NewVolTypeMap(
		testpatterns.PreprovisionedPV,
		testpatterns.InlineVolume))
}

func (s *subPathTestSuite) DefineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	type local struct {
		config        *PerTestConfig
		driverCleanup func()

		hostExec          utils.HostExec
		resource          *VolumeResource
		roVolSource       *v1.VolumeSource
		pod               *v1.Pod
		formatPod         *v1.Pod
		subPathDir        string
		filePathInSubpath string
		filePathInVolume  string

		intreeOps   opCounts
		migratedOps opCounts
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
		l.config, l.driverCleanup = driver.PrepareTest(f)
		l.intreeOps, l.migratedOps = getMigrationVolumeOpCounts(f.ClientSet, driver.GetDriverInfo().InTreePluginName)
		testVolumeSizeRange := s.GetTestSuiteInfo().SupportedSizeRange
		l.resource = CreateVolumeResource(driver, l.config, pattern, testVolumeSizeRange)
		l.hostExec = utils.NewHostExec(f)

		// Setup subPath test dependent resource
		volType := pattern.VolType
		switch volType {
		case testpatterns.InlineVolume:
			if iDriver, ok := driver.(InlineVolumeTestDriver); ok {
				l.roVolSource = iDriver.GetVolumeSource(true, pattern.FsType, l.resource.Volume)
			}
		case testpatterns.PreprovisionedPV:
			l.roVolSource = &v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: l.resource.Pvc.Name,
					ReadOnly:  true,
				},
			}
		case testpatterns.DynamicPV:
			l.roVolSource = &v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: l.resource.Pvc.Name,
					ReadOnly:  true,
				},
			}
		default:
			framework.Failf("SubPath test doesn't support: %s", volType)
		}

		subPath := f.Namespace.Name
		l.pod = SubpathTestPod(f, subPath, string(volType), l.resource.VolSource, true)
		e2epod.SetNodeSelection(&l.pod.Spec, l.config.ClientNodeSelection)

		l.formatPod = volumeFormatPod(f, l.resource.VolSource)
		e2epod.SetNodeSelection(&l.formatPod.Spec, l.config.ClientNodeSelection)

		l.subPathDir = filepath.Join(volumePath, subPath)
		l.filePathInSubpath = filepath.Join(volumePath, fileName)
		l.filePathInVolume = filepath.Join(l.subPathDir, fileName)
	}

	cleanup := func() {
		var errs []error
		if l.pod != nil {
			ginkgo.By("Deleting pod")
			err := e2epod.DeletePodWithWait(f.ClientSet, l.pod)
			errs = append(errs, err)
			l.pod = nil
		}

		if l.resource != nil {
			errs = append(errs, l.resource.CleanupResource())
			l.resource = nil
		}

		errs = append(errs, tryFunc(l.driverCleanup))
		l.driverCleanup = nil
		framework.ExpectNoError(errors.NewAggregate(errs), "while cleaning up resource")

		if l.hostExec != nil {
			l.hostExec.Cleanup()
		}

		validateMigrationVolumeOpCounts(f.ClientSet, driver.GetDriverInfo().InTreePluginName, l.intreeOps, l.migratedOps)
	}

	driverName := driver.GetDriverInfo().Name

	ginkgo.It("should support non-existent path", func() {
		init()
		defer cleanup()

		// Write the file in the subPath from init container 1
		setWriteCommand(l.filePathInSubpath, &l.pod.Spec.InitContainers[1])

		// Read it from outside the subPath from container 1
		testReadFile(f, l.filePathInVolume, l.pod, 1)
	})

	ginkgo.It("should support existing directory", func() {
		init()
		defer cleanup()

		// Create the directory
		setInitCommand(l.pod, fmt.Sprintf("mkdir -p %s", l.subPathDir))

		// Write the file in the subPath from init container 1
		setWriteCommand(l.filePathInSubpath, &l.pod.Spec.InitContainers[1])

		// Read it from outside the subPath from container 1
		testReadFile(f, l.filePathInVolume, l.pod, 1)
	})

	ginkgo.It("should support existing single file [LinuxOnly]", func() {
		init()
		defer cleanup()

		// Create the file in the init container
		setInitCommand(l.pod, fmt.Sprintf("mkdir -p %s; echo \"mount-tester new file\" > %s", l.subPathDir, l.filePathInVolume))

		// Read it from inside the subPath from container 0
		testReadFile(f, l.filePathInSubpath, l.pod, 0)
	})

	ginkgo.It("should support file as subpath [LinuxOnly]", func() {
		init()
		defer cleanup()

		// Create the file in the init container
		setInitCommand(l.pod, fmt.Sprintf("echo %s > %s", f.Namespace.Name, l.subPathDir))

		TestBasicSubpath(f, f.Namespace.Name, l.pod)
	})

	ginkgo.It("should fail if subpath directory is outside the volume [Slow]", func() {
		init()
		defer cleanup()

		// Create the subpath outside the volume
		var command string
		if framework.NodeOSDistroIs("windows") {
			command = fmt.Sprintf("New-Item -ItemType SymbolicLink -Path %s -value \\Windows", l.subPathDir)
		} else {
			command = fmt.Sprintf("ln -s /bin %s", l.subPathDir)
		}
		setInitCommand(l.pod, command)
		// Pod should fail
		testPodFailSubpath(f, l.pod, false)
	})

	ginkgo.It("should fail if subpath file is outside the volume [Slow][LinuxOnly]", func() {
		init()
		defer cleanup()

		// Create the subpath outside the volume
		setInitCommand(l.pod, fmt.Sprintf("ln -s /bin/sh %s", l.subPathDir))

		// Pod should fail
		testPodFailSubpath(f, l.pod, false)
	})

	ginkgo.It("should fail if non-existent subpath is outside the volume [Slow][LinuxOnly]", func() {
		init()
		defer cleanup()

		// Create the subpath outside the volume
		setInitCommand(l.pod, fmt.Sprintf("ln -s /bin/notanexistingpath %s", l.subPathDir))

		// Pod should fail
		testPodFailSubpath(f, l.pod, false)
	})

	ginkgo.It("should fail if subpath with backstepping is outside the volume [Slow]", func() {
		init()
		defer cleanup()

		// Create the subpath outside the volume
		var command string
		if framework.NodeOSDistroIs("windows") {
			command = fmt.Sprintf("New-Item -ItemType SymbolicLink -Path %s -value ..\\", l.subPathDir)
		} else {
			command = fmt.Sprintf("ln -s ../ %s", l.subPathDir)
		}
		setInitCommand(l.pod, command)
		// Pod should fail
		testPodFailSubpath(f, l.pod, false)
	})

	ginkgo.It("should support creating multiple subpath from same volumes [Slow]", func() {
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

	ginkgo.It("should support restarting containers using directory as subpath [Slow]", func() {
		init()
		defer cleanup()

		// Create the directory
		var command string
		if framework.NodeOSDistroIs("windows") {
			command = fmt.Sprintf("mkdir -p %v; New-Item -itemtype File -path %v", l.subPathDir, probeFilePath)
		} else {
			command = fmt.Sprintf("mkdir -p %v; touch %v", l.subPathDir, probeFilePath)
		}
		setInitCommand(l.pod, command)
		testPodContainerRestart(f, l.pod)
	})

	ginkgo.It("should support restarting containers using file as subpath [Slow][LinuxOnly]", func() {
		init()
		defer cleanup()

		// Create the file
		setInitCommand(l.pod, fmt.Sprintf("touch %v; touch %v", l.subPathDir, probeFilePath))

		testPodContainerRestart(f, l.pod)
	})

	ginkgo.It("should unmount if pod is gracefully deleted while kubelet is down [Disruptive][Slow][LinuxOnly]", func() {
		init()
		defer cleanup()

		testSubpathReconstruction(f, l.hostExec, l.pod, false)
	})

	ginkgo.It("should unmount if pod is force deleted while kubelet is down [Disruptive][Slow][LinuxOnly]", func() {
		init()
		defer cleanup()

		if strings.HasPrefix(driverName, "hostPath") {
			// TODO: This skip should be removed once #61446 is fixed
			e2eskipper.Skipf("Driver %s does not support reconstruction, skipping", driverName)
		}

		testSubpathReconstruction(f, l.hostExec, l.pod, true)
	})

	ginkgo.It("should support readOnly directory specified in the volumeMount", func() {
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

	ginkgo.It("should support readOnly file specified in the volumeMount [LinuxOnly]", func() {
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

	ginkgo.It("should support existing directories when readOnly specified in the volumeSource", func() {
		init()
		defer cleanup()
		if l.roVolSource == nil {
			e2eskipper.Skipf("Driver %s on volume type %s doesn't support readOnly source", driverName, pattern.VolType)
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

	ginkgo.It("should verify container cannot write to subpath readonly volumes [Slow]", func() {
		init()
		defer cleanup()
		if l.roVolSource == nil {
			e2eskipper.Skipf("Driver %s on volume type %s doesn't support readOnly source", driverName, pattern.VolType)
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

	ginkgo.It("should be able to unmount after the subpath directory is deleted", func() {
		init()
		defer cleanup()

		// Change volume container to busybox so we can exec later
		l.pod.Spec.Containers[1].Image = e2evolume.GetTestImage(imageutils.GetE2EImage(imageutils.BusyBox))
		l.pod.Spec.Containers[1].Command = e2evolume.GenerateScriptCmd("sleep 100000")

		ginkgo.By(fmt.Sprintf("Creating pod %s", l.pod.Name))
		removeUnusedContainers(l.pod)
		pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), l.pod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "while creating pod")
		defer func() {
			ginkgo.By(fmt.Sprintf("Deleting pod %s", pod.Name))
			e2epod.DeletePodWithWait(f.ClientSet, pod)
		}()

		// Wait for pod to be running
		err = e2epod.WaitForPodRunningInNamespace(f.ClientSet, l.pod)
		framework.ExpectNoError(err, "while waiting for pod to be running")

		// Exec into container that mounted the volume, delete subpath directory
		rmCmd := fmt.Sprintf("rm -r %s", l.subPathDir)
		_, err = podContainerExec(l.pod, 1, rmCmd)
		framework.ExpectNoError(err, "while removing subpath directory")

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

	ginkgo.By(fmt.Sprintf("Creating pod %s", pod.Name))
	removeUnusedContainers(pod)
	f.TestContainerOutput("atomic-volume-subpath", pod, 0, []string{contents})

	ginkgo.By(fmt.Sprintf("Deleting pod %s", pod.Name))
	err := e2epod.DeletePodWithWait(f.ClientSet, pod)
	framework.ExpectNoError(err, "while deleting pod")
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
		seLinuxOptions  = &v1.SELinuxOptions{Level: "s0:c0,c1"}
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
					Image: e2evolume.GetTestImage(imageutils.GetE2EImage(imageutils.BusyBox)),
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
					SecurityContext: e2evolume.GenerateSecurityContext(privilegedSecurityContext),
				},
				{
					Name:  fmt.Sprintf("test-init-subpath-%s", suffix),
					Image: mountImage,
					Args:  []string{"mounttest"},
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
					SecurityContext: e2evolume.GenerateSecurityContext(privilegedSecurityContext),
				},
				{
					Name:  fmt.Sprintf("test-init-volume-%s", suffix),
					Image: mountImage,
					Args:  []string{"mounttest"},
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
					SecurityContext: e2evolume.GenerateSecurityContext(privilegedSecurityContext),
				},
			},
			Containers: []v1.Container{
				{
					Name:  fmt.Sprintf("test-container-subpath-%s", suffix),
					Image: mountImage,
					Args:  []string{"mounttest"},
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
					SecurityContext: e2evolume.GenerateSecurityContext(privilegedSecurityContext),
				},
				{
					Name:  fmt.Sprintf("test-container-volume-%s", suffix),
					Image: mountImage,
					Args:  []string{"mounttest"},
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
					SecurityContext: e2evolume.GenerateSecurityContext(privilegedSecurityContext),
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
			SecurityContext: e2evolume.GeneratePodSecurityContext(nil, seLinuxOptions),
		},
	}
}

func containerIsUnused(container *v1.Container) bool {
	// mountImage with nil Args or with just "mounttest" as Args does nothing. Leave everything else
	return container.Image == mountImage && (container.Args == nil || (len(container.Args) == 1 && container.Args[0] == "mounttest"))
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
					Image:   e2evolume.GetTestImage(imageutils.GetE2EImage(imageutils.BusyBox)),
					Command: e2evolume.GenerateScriptCmd("echo nothing"),
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

func setInitCommand(pod *v1.Pod, command string) {
	pod.Spec.InitContainers[0].Command = e2evolume.GenerateScriptCmd(command)
}

func setWriteCommand(file string, container *v1.Container) {
	container.Args = []string{
		"mounttest",
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
		"mounttest",
		fmt.Sprintf("--new_file_0644=%v", file1),
		fmt.Sprintf("--new_file_0666=%v", file2),
	}
}

func testMultipleReads(f *framework.Framework, pod *v1.Pod, containerIndex int, file1 string, file2 string) {
	ginkgo.By(fmt.Sprintf("Creating pod %s", pod.Name))
	removeUnusedContainers(pod)
	f.TestContainerOutput("multi_subpath", pod, containerIndex, []string{
		"content of file \"" + file1 + "\": mount-tester new file",
		"content of file \"" + file2 + "\": mount-tester new file",
	})
}

func setReadCommand(file string, container *v1.Container) {
	container.Args = []string{
		"mounttest",
		fmt.Sprintf("--file_content_in_loop=%v", file),
		fmt.Sprintf("--retry_time=%d", retryDuration),
	}
}

func testReadFile(f *framework.Framework, file string, pod *v1.Pod, containerIndex int) {
	setReadCommand(file, &pod.Spec.Containers[containerIndex])

	ginkgo.By(fmt.Sprintf("Creating pod %s", pod.Name))
	removeUnusedContainers(pod)
	f.TestContainerOutput("subpath", pod, containerIndex, []string{
		"content of file \"" + file + "\": mount-tester new file",
	})

	ginkgo.By(fmt.Sprintf("Deleting pod %s", pod.Name))
	err := e2epod.DeletePodWithWait(f.ClientSet, pod)
	framework.ExpectNoError(err, "while deleting pod")
}

func testPodFailSubpath(f *framework.Framework, pod *v1.Pod, allowContainerTerminationError bool) {
	testPodFailSubpathError(f, pod, "subPath", allowContainerTerminationError)
}

func testPodFailSubpathError(f *framework.Framework, pod *v1.Pod, errorMsg string, allowContainerTerminationError bool) {
	ginkgo.By(fmt.Sprintf("Creating pod %s", pod.Name))
	removeUnusedContainers(pod)
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "while creating pod")
	defer func() {
		e2epod.DeletePodWithWait(f.ClientSet, pod)
	}()
	ginkgo.By("Checking for subpath error in container status")
	err = waitForPodSubpathError(f, pod, allowContainerTerminationError)
	framework.ExpectNoError(err, "while waiting for subpath failure")
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

	waitErr := wait.PollImmediate(framework.Poll, framework.PodStartTimeout, func() (bool, error) {
		pod, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
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
	if waitErr != nil {
		return fmt.Errorf("error waiting for pod subpath error to occur: %v", waitErr)
	}
	return nil
}

// Tests that the existing subpath mount is detected when a container restarts
func testPodContainerRestart(f *framework.Framework, pod *v1.Pod) {
	pod.Spec.RestartPolicy = v1.RestartPolicyOnFailure

	pod.Spec.Containers[0].Image = e2evolume.GetTestImage(imageutils.GetE2EImage(imageutils.BusyBox))
	pod.Spec.Containers[0].Command = e2evolume.GenerateScriptCmd("sleep 100000")
	pod.Spec.Containers[1].Image = e2evolume.GetTestImage(imageutils.GetE2EImage(imageutils.BusyBox))
	pod.Spec.Containers[1].Command = e2evolume.GenerateScriptCmd("sleep 100000")
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
	ginkgo.By(fmt.Sprintf("Creating pod %s", pod.Name))
	removeUnusedContainers(pod)
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "while creating pod")
	defer func() {
		e2epod.DeletePodWithWait(f.ClientSet, pod)
	}()
	err = e2epod.WaitForPodRunningInNamespace(f.ClientSet, pod)
	framework.ExpectNoError(err, "while waiting for pod to be running")

	ginkgo.By("Failing liveness probe")
	out, err := podContainerExec(pod, 1, fmt.Sprintf("rm %v", probeFilePath))
	framework.Logf("Pod exec output: %v", out)
	framework.ExpectNoError(err, "while failing liveness probe")

	// Check that container has restarted
	ginkgo.By("Waiting for container to restart")
	restarts := int32(0)
	err = wait.PollImmediate(10*time.Second, 2*time.Minute, func() (bool, error) {
		pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), pod.Name, metav1.GetOptions{})
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
	framework.ExpectNoError(err, "while waiting for container to restart")

	// Fix liveness probe
	ginkgo.By("Rewriting the file")
	var writeCmd string
	if framework.NodeOSDistroIs("windows") {
		writeCmd = fmt.Sprintf("echo test-after | Out-File -FilePath %v", probeFilePath)
	} else {
		writeCmd = fmt.Sprintf("echo test-after > %v", probeFilePath)
	}
	out, err = podContainerExec(pod, 1, writeCmd)
	framework.Logf("Pod exec output: %v", out)
	framework.ExpectNoError(err, "while rewriting the probe file")

	// Wait for container restarts to stabilize
	ginkgo.By("Waiting for container to stop restarting")
	stableCount := int(0)
	stableThreshold := int(time.Minute / framework.Poll)
	err = wait.PollImmediate(framework.Poll, 2*time.Minute, func() (bool, error) {
		pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), pod.Name, metav1.GetOptions{})
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
	framework.ExpectNoError(err, "while waiting for container to stabilize")
}

func testSubpathReconstruction(f *framework.Framework, hostExec utils.HostExec, pod *v1.Pod, forceDelete bool) {
	// This is mostly copied from TestVolumeUnmountsFromDeletedPodWithForceOption()

	// Disruptive test run serially, we can cache all voluem global mount
	// points and verify after the test that we do not leak any global mount point.
	nodeList, err := e2enode.GetReadySchedulableNodes(f.ClientSet)
	framework.ExpectNoError(err, "while listing scheduable nodes")
	globalMountPointsByNode := make(map[string]sets.String, len(nodeList.Items))
	for _, node := range nodeList.Items {
		globalMountPointsByNode[node.Name] = utils.FindVolumeGlobalMountPoints(hostExec, &node)
	}

	// Change to busybox
	pod.Spec.Containers[0].Image = e2evolume.GetTestImage(imageutils.GetE2EImage(imageutils.BusyBox))
	pod.Spec.Containers[0].Command = e2evolume.GenerateScriptCmd("sleep 100000")
	pod.Spec.Containers[1].Image = e2evolume.GetTestImage(imageutils.GetE2EImage(imageutils.BusyBox))
	pod.Spec.Containers[1].Command = e2evolume.GenerateScriptCmd("sleep 100000")

	// If grace period is too short, then there is not enough time for the volume
	// manager to cleanup the volumes
	gracePeriod := int64(30)
	pod.Spec.TerminationGracePeriodSeconds = &gracePeriod

	ginkgo.By(fmt.Sprintf("Creating pod %s", pod.Name))
	removeUnusedContainers(pod)
	pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "while creating pod")

	err = e2epod.WaitForPodRunningInNamespace(f.ClientSet, pod)
	framework.ExpectNoError(err, "while waiting for pod to be running")

	pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "while getting pod")

	var podNode *v1.Node
	for i := range nodeList.Items {
		if nodeList.Items[i].Name == pod.Spec.NodeName {
			podNode = &nodeList.Items[i]
		}
	}
	framework.ExpectNotEqual(podNode, nil, "pod node should exist in scheduable nodes")

	utils.TestVolumeUnmountsFromDeletedPodWithForceOption(f.ClientSet, f, pod, forceDelete, true)

	if podNode != nil {
		mountPoints := globalMountPointsByNode[podNode.Name]
		mountPointsAfter := utils.FindVolumeGlobalMountPoints(hostExec, podNode)
		s1 := mountPointsAfter.Difference(mountPoints)
		s2 := mountPoints.Difference(mountPointsAfter)
		gomega.Expect(s1).To(gomega.BeEmpty(), "global mount points leaked: %v", s1)
		gomega.Expect(s2).To(gomega.BeEmpty(), "global mount points not found: %v", s2)
	}
}

func formatVolume(f *framework.Framework, pod *v1.Pod) {
	ginkgo.By(fmt.Sprintf("Creating pod to format volume %s", pod.Name))
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "while creating volume init pod")

	err = e2epod.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, pod.Namespace)
	framework.ExpectNoError(err, "while waiting for volume init pod to succeed")

	err = e2epod.DeletePodWithWait(f.ClientSet, pod)
	framework.ExpectNoError(err, "while deleting volume init pod")
}

func podContainerExec(pod *v1.Pod, containerIndex int, command string) (string, error) {
	var shell string
	var option string
	if framework.NodeOSDistroIs("windows") {
		shell = "powershell"
		option = "/c"
	} else {
		shell = "/bin/sh"
		option = "-c"
	}
	return framework.RunKubectl(pod.Namespace, "exec", fmt.Sprintf("--namespace=%s", pod.Namespace), pod.Name, "--container", pod.Spec.Containers[containerIndex].Name, "--", shell, option, command)
}
