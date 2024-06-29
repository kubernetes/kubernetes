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

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	storageutils "k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var (
	volumePath      = "/test-volume"
	volumeName      = "test-volume"
	probeVolumePath = "/probe-volume"
	probeFilePath   = probeVolumePath + "/probe-file"
	fileName        = "test-file"
	retryDuration   = 20
)

type subPathTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

// InitCustomSubPathTestSuite returns subPathTestSuite that implements TestSuite interface
// using custom test patterns
func InitCustomSubPathTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &subPathTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "subPath",
			TestPatterns: patterns,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
		},
	}
}

// InitSubPathTestSuite returns subPathTestSuite that implements TestSuite interface
// using testsuite default patterns
func InitSubPathTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.DefaultFsInlineVolume,
		storageframework.DefaultFsPreprovisionedPV,
		storageframework.DefaultFsDynamicPV,
		storageframework.NtfsDynamicPV,
	}
	return InitCustomSubPathTestSuite(patterns)
}

func (s *subPathTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return s.tsInfo
}

func (s *subPathTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	skipVolTypePatterns(pattern, driver, storageframework.NewVolTypeMap(
		storageframework.PreprovisionedPV,
		storageframework.InlineVolume))
}

func (s *subPathTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	type local struct {
		config *storageframework.PerTestConfig

		hostExec          storageutils.HostExec
		resource          *storageframework.VolumeResource
		roVolSource       *v1.VolumeSource
		pod               *v1.Pod
		formatPod         *v1.Pod
		subPathDir        string
		filePathInSubpath string
		filePathInVolume  string

		migrationCheck *migrationOpCheck
	}
	var l local

	// Beware that it also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewFrameworkWithCustomTimeouts("provisioning", storageframework.GetDriverTimeouts(driver))
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	init := func(ctx context.Context) {
		l = local{}

		// Now do the more expensive test initialization.
		l.config = driver.PrepareTest(ctx, f)
		l.migrationCheck = newMigrationOpCheck(ctx, f.ClientSet, f.ClientConfig(), driver.GetDriverInfo().InTreePluginName)
		testVolumeSizeRange := s.GetTestSuiteInfo().SupportedSizeRange
		l.resource = storageframework.CreateVolumeResource(ctx, driver, l.config, pattern, testVolumeSizeRange)
		l.hostExec = storageutils.NewHostExec(f)

		// Setup subPath test dependent resource
		volType := pattern.VolType
		switch volType {
		case storageframework.InlineVolume:
			if iDriver, ok := driver.(storageframework.InlineVolumeTestDriver); ok {
				l.roVolSource = iDriver.GetVolumeSource(true, pattern.FsType, l.resource.Volume)
			}
		case storageframework.PreprovisionedPV:
			l.roVolSource = &v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: l.resource.Pvc.Name,
					ReadOnly:  true,
				},
			}
		case storageframework.DynamicPV:
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
		l.pod = SubpathTestPod(f, subPath, string(volType), l.resource.VolSource, admissionapi.LevelPrivileged)
		e2epod.SetNodeSelection(&l.pod.Spec, l.config.ClientNodeSelection)

		l.formatPod = volumeFormatPod(f, l.resource.VolSource)
		e2epod.SetNodeSelection(&l.formatPod.Spec, l.config.ClientNodeSelection)

		l.subPathDir = filepath.Join(volumePath, subPath)
		l.filePathInSubpath = filepath.Join(volumePath, fileName)
		l.filePathInVolume = filepath.Join(l.subPathDir, fileName)
	}

	cleanup := func(ctx context.Context) {
		var errs []error
		if l.pod != nil {
			ginkgo.By("Deleting pod")
			err := e2epod.DeletePodWithWait(ctx, f.ClientSet, l.pod)
			errs = append(errs, err)
			l.pod = nil
		}

		if l.resource != nil {
			errs = append(errs, l.resource.CleanupResource(ctx))
			l.resource = nil
		}

		framework.ExpectNoError(errors.NewAggregate(errs), "while cleaning up resource")

		if l.hostExec != nil {
			l.hostExec.Cleanup(ctx)
		}

		l.migrationCheck.validateMigrationVolumeOpCounts(ctx)
	}

	driverName := driver.GetDriverInfo().Name

	ginkgo.It("should support non-existent path", func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		// Write the file in the subPath from init container 1
		setWriteCommand(l.filePathInSubpath, &l.pod.Spec.InitContainers[1])

		// Read it from outside the subPath from container 1
		testReadFile(ctx, f, l.filePathInVolume, l.pod, 1)
	})

	ginkgo.It("should support existing directory", func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		// Create the directory
		setInitCommand(l.pod, fmt.Sprintf("mkdir -p %s", l.subPathDir))

		// Write the file in the subPath from init container 1
		setWriteCommand(l.filePathInSubpath, &l.pod.Spec.InitContainers[1])

		// Read it from outside the subPath from container 1
		testReadFile(ctx, f, l.filePathInVolume, l.pod, 1)
	})

	ginkgo.It("should support existing single file [LinuxOnly]", func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		// Create the file in the init container
		setInitCommand(l.pod, fmt.Sprintf("mkdir -p %s; echo \"mount-tester new file\" > %s", l.subPathDir, l.filePathInVolume))

		// Read it from inside the subPath from container 0
		testReadFile(ctx, f, l.filePathInSubpath, l.pod, 0)
	})

	ginkgo.It("should support file as subpath [LinuxOnly]", func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		// Create the file in the init container
		setInitCommand(l.pod, fmt.Sprintf("echo %s > %s", f.Namespace.Name, l.subPathDir))

		TestBasicSubpath(ctx, f, f.Namespace.Name, l.pod)
	})

	f.It("should fail if subpath directory is outside the volume", f.WithSlow(), "[LinuxOnly]", func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		// Create the subpath outside the volume
		var command string
		if framework.NodeOSDistroIs("windows") {
			command = fmt.Sprintf("New-Item -ItemType SymbolicLink -Path %s -value \\Windows", l.subPathDir)
		} else {
			command = fmt.Sprintf("ln -s /bin %s", l.subPathDir)
		}
		setInitCommand(l.pod, command)
		// Pod should fail
		testPodFailSubpath(ctx, f, l.pod, false)
	})

	f.It("should fail if subpath file is outside the volume", f.WithSlow(), "[LinuxOnly]", func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		// Create the subpath outside the volume
		setInitCommand(l.pod, fmt.Sprintf("ln -s /bin/sh %s", l.subPathDir))

		// Pod should fail
		testPodFailSubpath(ctx, f, l.pod, false)
	})

	f.It("should fail if non-existent subpath is outside the volume", f.WithSlow(), "[LinuxOnly]", func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		// Create the subpath outside the volume
		setInitCommand(l.pod, fmt.Sprintf("ln -s /bin/notanexistingpath %s", l.subPathDir))

		// Pod should fail
		testPodFailSubpath(ctx, f, l.pod, false)
	})

	f.It("should fail if subpath with backstepping is outside the volume", f.WithSlow(), "[LinuxOnly]", func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		// Create the subpath outside the volume
		var command string
		if framework.NodeOSDistroIs("windows") {
			command = fmt.Sprintf("New-Item -ItemType SymbolicLink -Path %s -value ..\\", l.subPathDir)
		} else {
			command = fmt.Sprintf("ln -s ../ %s", l.subPathDir)
		}
		setInitCommand(l.pod, command)
		// Pod should fail
		testPodFailSubpath(ctx, f, l.pod, false)
	})

	f.It("should support creating multiple subpath from same volumes", f.WithSlow(), func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

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
		testMultipleReads(ctx, f, l.pod, 0, filepath1, filepath2)
	})

	f.It("should support restarting containers using directory as subpath", f.WithSlow(), func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		// Create the directory
		var command string
		command = fmt.Sprintf("mkdir -p %v; touch %v", l.subPathDir, probeFilePath)
		setInitCommand(l.pod, command)
		testPodContainerRestart(ctx, f, l.pod)
	})

	f.It("should support restarting containers using file as subpath", f.WithSlow(), "[LinuxOnly]", func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		// Create the file
		setInitCommand(l.pod, fmt.Sprintf("touch %v; touch %v", l.subPathDir, probeFilePath))

		testPodContainerRestart(ctx, f, l.pod)
	})

	f.It("should unmount if pod is gracefully deleted while kubelet is down", f.WithDisruptive(), f.WithSlow(), "[LinuxOnly]", func(ctx context.Context) {
		e2eskipper.SkipUnlessSSHKeyPresent()
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		if strings.HasPrefix(driverName, "hostPath") {
			// TODO: This skip should be removed once #61446 is fixed
			e2eskipper.Skipf("Driver %s does not support reconstruction, skipping", driverName)
		}

		testSubpathReconstruction(ctx, f, l.hostExec, l.pod, false)
	})

	f.It("should unmount if pod is force deleted while kubelet is down", f.WithDisruptive(), f.WithSlow(), "[LinuxOnly]", func(ctx context.Context) {
		e2eskipper.SkipUnlessSSHKeyPresent()
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		if strings.HasPrefix(driverName, "hostPath") {
			// TODO: This skip should be removed once #61446 is fixed
			e2eskipper.Skipf("Driver %s does not support reconstruction, skipping", driverName)
		}

		testSubpathReconstruction(ctx, f, l.hostExec, l.pod, true)
	})

	ginkgo.It("should support readOnly directory specified in the volumeMount", func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		// Create the directory
		setInitCommand(l.pod, fmt.Sprintf("mkdir -p %s", l.subPathDir))

		// Write the file in the volume from init container 2
		setWriteCommand(l.filePathInVolume, &l.pod.Spec.InitContainers[2])

		// Read it from inside the subPath from container 0
		l.pod.Spec.Containers[0].VolumeMounts[0].ReadOnly = true
		testReadFile(ctx, f, l.filePathInSubpath, l.pod, 0)
	})

	ginkgo.It("should support readOnly file specified in the volumeMount [LinuxOnly]", func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		// Create the file
		setInitCommand(l.pod, fmt.Sprintf("touch %s", l.subPathDir))

		// Write the file in the volume from init container 2
		setWriteCommand(l.subPathDir, &l.pod.Spec.InitContainers[2])

		// Read it from inside the subPath from container 0
		l.pod.Spec.Containers[0].VolumeMounts[0].ReadOnly = true
		testReadFile(ctx, f, volumePath, l.pod, 0)
	})

	ginkgo.It("should support existing directories when readOnly specified in the volumeSource", func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)
		if l.roVolSource == nil {
			e2eskipper.Skipf("Driver %s on volume type %s doesn't support readOnly source", driverName, pattern.VolType)
		}

		origpod := l.pod.DeepCopy()

		// Create the directory
		setInitCommand(l.pod, fmt.Sprintf("mkdir -p %s", l.subPathDir))

		// Write the file in the subPath from init container 1
		setWriteCommand(l.filePathInSubpath, &l.pod.Spec.InitContainers[1])

		// Read it from inside the subPath from container 0
		testReadFile(ctx, f, l.filePathInSubpath, l.pod, 0)

		// Reset the pod
		l.pod = origpod

		// Set volume source to read only
		l.pod.Spec.Volumes[0].VolumeSource = *l.roVolSource

		// Read it from inside the subPath from container 0
		testReadFile(ctx, f, l.filePathInSubpath, l.pod, 0)
	})

	f.It("should verify container cannot write to subpath readonly volumes", f.WithSlow(), func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)
		if l.roVolSource == nil {
			e2eskipper.Skipf("Driver %s on volume type %s doesn't support readOnly source", driverName, pattern.VolType)
		}

		// Format the volume while it's writable
		formatVolume(ctx, f, l.formatPod)

		// Set volume source to read only
		l.pod.Spec.Volumes[0].VolumeSource = *l.roVolSource

		// Write the file in the volume from container 0
		setWriteCommand(l.subPathDir, &l.pod.Spec.Containers[0])

		// Pod should fail
		testPodFailSubpath(ctx, f, l.pod, true)
	})

	// Set this test linux-only because the test will fail in Windows when
	// deleting a dir from one container while another container still use it.
	ginkgo.It("should be able to unmount after the subpath directory is deleted [LinuxOnly]", func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		// Change volume container to busybox so we can exec later
		l.pod.Spec.Containers[1].Image = e2epod.GetDefaultTestImage()
		l.pod.Spec.Containers[1].Command = e2epod.GenerateScriptCmd(e2epod.InfiniteSleepCommand)
		l.pod.Spec.Containers[1].Args = nil

		ginkgo.By(fmt.Sprintf("Creating pod %s", l.pod.Name))
		removeUnusedContainers(l.pod)
		pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, l.pod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "while creating pod")
		ginkgo.DeferCleanup(func(ctx context.Context) error {
			ginkgo.By(fmt.Sprintf("Deleting pod %s", pod.Name))
			return e2epod.DeletePodWithWait(ctx, f.ClientSet, pod)
		})

		// Wait for pod to be running
		err = e2epod.WaitTimeoutForPodRunningInNamespace(ctx, f.ClientSet, l.pod.Name, l.pod.Namespace, f.Timeouts.PodStart)
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
func TestBasicSubpath(ctx context.Context, f *framework.Framework, contents string, pod *v1.Pod) {
	TestBasicSubpathFile(ctx, f, contents, pod, volumePath)
}

// TestBasicSubpathFile runs basic subpath file test
func TestBasicSubpathFile(ctx context.Context, f *framework.Framework, contents string, pod *v1.Pod, filepath string) {
	setReadCommand(filepath, &pod.Spec.Containers[0])

	ginkgo.By(fmt.Sprintf("Creating pod %s", pod.Name))
	removeUnusedContainers(pod)
	e2eoutput.TestContainerOutput(ctx, f, "atomic-volume-subpath", pod, 0, []string{contents})

	ginkgo.By(fmt.Sprintf("Deleting pod %s", pod.Name))
	err := e2epod.DeletePodWithWait(ctx, f.ClientSet, pod)
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
func SubpathTestPod(f *framework.Framework, subpath, volumeType string, source *v1.VolumeSource, securityLevel admissionapi.Level) *v1.Pod {
	var (
		suffix          = generateSuffixForPodName(volumeType)
		gracePeriod     = int64(1)
		probeVolumeName = "liveness-probe-volume"
		seLinuxOptions  = &v1.SELinuxOptions{Level: "s0:c0,c1"}
	)

	volumeMount := v1.VolumeMount{Name: volumeName, MountPath: volumePath}
	volumeSubpathMount := v1.VolumeMount{Name: volumeName, MountPath: volumePath, SubPath: subpath}
	probeMount := v1.VolumeMount{Name: probeVolumeName, MountPath: probeVolumePath}

	initSubpathContainer := e2epod.NewAgnhostContainer(
		fmt.Sprintf("test-init-subpath-%s", suffix),
		[]v1.VolumeMount{volumeSubpathMount, probeMount}, nil, "mounttest")
	initSubpathContainer.SecurityContext = e2epod.GenerateContainerSecurityContext(securityLevel)
	initVolumeContainer := e2epod.NewAgnhostContainer(
		fmt.Sprintf("test-init-volume-%s", suffix),
		[]v1.VolumeMount{volumeMount, probeMount}, nil, "mounttest")
	initVolumeContainer.SecurityContext = e2epod.GenerateContainerSecurityContext(securityLevel)
	subpathContainer := e2epod.NewAgnhostContainer(
		fmt.Sprintf("test-container-subpath-%s", suffix),
		[]v1.VolumeMount{volumeSubpathMount, probeMount}, nil, "mounttest")
	subpathContainer.SecurityContext = e2epod.GenerateContainerSecurityContext(securityLevel)
	volumeContainer := e2epod.NewAgnhostContainer(
		fmt.Sprintf("test-container-volume-%s", suffix),
		[]v1.VolumeMount{volumeMount, probeMount}, nil, "mounttest")
	volumeContainer.SecurityContext = e2epod.GenerateContainerSecurityContext(securityLevel)

	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("pod-subpath-test-%s", suffix),
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{
				{
					Name:            fmt.Sprintf("init-volume-%s", suffix),
					Image:           e2epod.GetDefaultTestImage(),
					VolumeMounts:    []v1.VolumeMount{volumeMount, probeMount},
					SecurityContext: e2epod.GenerateContainerSecurityContext(securityLevel),
				},
				initSubpathContainer,
				initVolumeContainer,
			},
			Containers: []v1.Container{
				subpathContainer,
				volumeContainer,
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
			SecurityContext: e2epod.GeneratePodSecurityContext(nil, seLinuxOptions),
		},
	}
}

func containerIsUnused(container *v1.Container) bool {
	// agnhost image with nil command and nil Args or with just "mounttest" as Args does nothing. Leave everything else
	return container.Image == imageutils.GetE2EImage(imageutils.Agnhost) && container.Command == nil &&
		(container.Args == nil || (len(container.Args) == 1 && container.Args[0] == "mounttest"))
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
					Image:   e2epod.GetDefaultTestImage(),
					Command: e2epod.GenerateScriptCmd("echo nothing"),
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
	pod.Spec.InitContainers[0].Command = e2epod.GenerateScriptCmd(command)
}

func setWriteCommand(file string, container *v1.Container) {
	container.Args = []string{
		"mounttest",
		fmt.Sprintf("--new_file_0644=%v", file),
	}
	// See issue https://github.com/kubernetes/kubernetes/issues/94237 about file_mode
	// not working well on Windows
	// TODO: remove this check after issue is resolved
	if !framework.NodeOSDistroIs("windows") {
		container.Args = append(container.Args, fmt.Sprintf("--file_mode=%v", file))
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

func testMultipleReads(ctx context.Context, f *framework.Framework, pod *v1.Pod, containerIndex int, file1 string, file2 string) {
	ginkgo.By(fmt.Sprintf("Creating pod %s", pod.Name))
	removeUnusedContainers(pod)
	e2eoutput.TestContainerOutput(ctx, f, "multi_subpath", pod, containerIndex, []string{
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

func testReadFile(ctx context.Context, f *framework.Framework, file string, pod *v1.Pod, containerIndex int) {
	setReadCommand(file, &pod.Spec.Containers[containerIndex])

	ginkgo.By(fmt.Sprintf("Creating pod %s", pod.Name))
	removeUnusedContainers(pod)
	e2eoutput.TestContainerOutput(ctx, f, "subpath", pod, containerIndex, []string{
		"content of file \"" + file + "\": mount-tester new file",
	})

	ginkgo.By(fmt.Sprintf("Deleting pod %s", pod.Name))
	err := e2epod.DeletePodWithWait(ctx, f.ClientSet, pod)
	framework.ExpectNoError(err, "while deleting pod")
}

func testPodFailSubpath(ctx context.Context, f *framework.Framework, pod *v1.Pod, allowContainerTerminationError bool) {
	testPodFailSubpathError(ctx, f, pod, "subPath", allowContainerTerminationError)
}

func testPodFailSubpathError(ctx context.Context, f *framework.Framework, pod *v1.Pod, errorMsg string, allowContainerTerminationError bool) {
	ginkgo.By(fmt.Sprintf("Creating pod %s", pod.Name))
	removeUnusedContainers(pod)
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "while creating pod")
	ginkgo.DeferCleanup(e2epod.DeletePodWithWait, f.ClientSet, pod)
	ginkgo.By("Checking for subpath error in container status")
	err = waitForPodSubpathError(ctx, f, pod, allowContainerTerminationError)
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

func waitForPodSubpathError(ctx context.Context, f *framework.Framework, pod *v1.Pod, allowContainerTerminationError bool) error {
	subpathContainerName := findSubpathContainerName(pod)
	if subpathContainerName == "" {
		return fmt.Errorf("failed to find container that uses subpath")
	}

	waitErr := wait.PollUntilContextTimeout(ctx, framework.Poll, f.Timeouts.PodStart, true, func(ctx context.Context) (bool, error) {
		pod, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
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

type podContainerRestartHooks struct {
	AddLivenessProbeFunc  func(pod *v1.Pod, probeFilePath string)
	FailLivenessProbeFunc func(pod *v1.Pod, probeFilePath string)
	FixLivenessProbeFunc  func(pod *v1.Pod, probeFilePath string)
}

func (h *podContainerRestartHooks) AddLivenessProbe(pod *v1.Pod, probeFilePath string) {
	if h.AddLivenessProbeFunc != nil {
		h.AddLivenessProbeFunc(pod, probeFilePath)
	}
}

func (h *podContainerRestartHooks) FailLivenessProbe(pod *v1.Pod, probeFilePath string) {
	if h.FailLivenessProbeFunc != nil {
		h.FailLivenessProbeFunc(pod, probeFilePath)
	}
}

func (h *podContainerRestartHooks) FixLivenessProbe(pod *v1.Pod, probeFilePath string) {
	if h.FixLivenessProbeFunc != nil {
		h.FixLivenessProbeFunc(pod, probeFilePath)
	}
}

// testPodContainerRestartWithHooks tests that container restarts to stabilize.
// hooks wrap functions between container restarts.
func testPodContainerRestartWithHooks(ctx context.Context, f *framework.Framework, pod *v1.Pod, hooks *podContainerRestartHooks) {
	pod.Spec.RestartPolicy = v1.RestartPolicyOnFailure

	pod.Spec.Containers[0].Image = e2epod.GetDefaultTestImage()
	pod.Spec.Containers[0].Command = e2epod.GenerateScriptCmd(e2epod.InfiniteSleepCommand)
	pod.Spec.Containers[0].Args = nil
	pod.Spec.Containers[1].Image = e2epod.GetDefaultTestImage()
	pod.Spec.Containers[1].Command = e2epod.GenerateScriptCmd(e2epod.InfiniteSleepCommand)
	pod.Spec.Containers[1].Args = nil
	hooks.AddLivenessProbe(pod, probeFilePath)

	// Start pod
	ginkgo.By(fmt.Sprintf("Creating pod %s", pod.Name))
	removeUnusedContainers(pod)
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "while creating pod")
	ginkgo.DeferCleanup(e2epod.DeletePodWithWait, f.ClientSet, pod)
	err = e2epod.WaitTimeoutForPodRunningInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, f.Timeouts.PodStart)
	framework.ExpectNoError(err, "while waiting for pod to be running")

	ginkgo.By("Failing liveness probe")
	hooks.FailLivenessProbe(pod, probeFilePath)

	// Check that container has restarted. The time that this
	// might take is estimated to be lower than for "delete pod"
	// and "start pod".
	ginkgo.By("Waiting for container to restart")
	restarts := int32(0)
	err = wait.PollUntilContextTimeout(ctx, 10*time.Second, f.Timeouts.PodDelete+f.Timeouts.PodStart, true, func(ctx context.Context) (bool, error) {
		pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
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
	ginkgo.By("Fix liveness probe")
	hooks.FixLivenessProbe(pod, probeFilePath)

	// Wait for container restarts to stabilize. Estimating the
	// time for this is harder. In practice,
	// framework.PodStartTimeout = f.Timeouts.PodStart = 5min
	// turned out to be too low, therefore
	// f.Timeouts.PodStartSlow = 15min is used now.
	ginkgo.By("Waiting for container to stop restarting")
	stableCount := int(0)
	stableThreshold := int(time.Minute / framework.Poll)
	err = wait.PollUntilContextTimeout(ctx, framework.Poll, f.Timeouts.PodStartSlow, true, func(ctx context.Context) (bool, error) {
		pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
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

// testPodContainerRestart tests that the existing subpath mount is detected when a container restarts
func testPodContainerRestart(ctx context.Context, f *framework.Framework, pod *v1.Pod) {
	testPodContainerRestartWithHooks(ctx, f, pod, &podContainerRestartHooks{
		AddLivenessProbeFunc: func(p *v1.Pod, probeFilePath string) {
			p.Spec.Containers[0].LivenessProbe = &v1.Probe{
				ProbeHandler: v1.ProbeHandler{
					Exec: &v1.ExecAction{
						Command: []string{"cat", probeFilePath},
					},
				},
				InitialDelaySeconds: 1,
				FailureThreshold:    1,
				PeriodSeconds:       2,
			}
		},
		FailLivenessProbeFunc: func(p *v1.Pod, probeFilePath string) {
			out, err := podContainerExec(p, 1, fmt.Sprintf("rm %v", probeFilePath))
			framework.Logf("Pod exec output: %v", out)
			framework.ExpectNoError(err, "while failing liveness probe")
		},
		FixLivenessProbeFunc: func(p *v1.Pod, probeFilePath string) {
			ginkgo.By("Rewriting the file")
			var writeCmd string
			if framework.NodeOSDistroIs("windows") {
				writeCmd = fmt.Sprintf("echo test-after | Out-File -FilePath %v", probeFilePath)
			} else {
				writeCmd = fmt.Sprintf("echo test-after > %v", probeFilePath)
			}
			out, err := podContainerExec(pod, 1, writeCmd)
			framework.Logf("Pod exec output: %v", out)
			framework.ExpectNoError(err, "while rewriting the probe file")
		},
	})
}

// TestPodContainerRestartWithConfigmapModified tests that container can restart to stabilize when configmap has been modified.
// 1. valid container running
// 2. update configmap
// 3. container restarts
// 4. container becomes stable after configmap mounted file has been modified
func TestPodContainerRestartWithConfigmapModified(ctx context.Context, f *framework.Framework, original, modified *v1.ConfigMap) {
	ginkgo.By("Create configmap")
	_, err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(ctx, original, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		framework.ExpectNoError(err, "while creating configmap to modify")
	}

	var subpath string
	for k := range original.Data {
		subpath = k
		break
	}
	pod := SubpathTestPod(f, subpath, "configmap", &v1.VolumeSource{ConfigMap: &v1.ConfigMapVolumeSource{LocalObjectReference: v1.LocalObjectReference{Name: original.Name}}}, admissionapi.LevelBaseline)
	pod.Spec.InitContainers[0].Command = e2epod.GenerateScriptCmd(fmt.Sprintf("touch %v", probeFilePath))

	modifiedValue := modified.Data[subpath]
	testPodContainerRestartWithHooks(ctx, f, pod, &podContainerRestartHooks{
		AddLivenessProbeFunc: func(p *v1.Pod, probeFilePath string) {
			p.Spec.Containers[0].LivenessProbe = &v1.Probe{
				ProbeHandler: v1.ProbeHandler{
					Exec: &v1.ExecAction{
						// Expect probe file exist or configmap mounted file has been modified.
						Command: []string{"sh", "-c", fmt.Sprintf("cat %s || test `cat %s` = '%s'", probeFilePath, volumePath, modifiedValue)},
					},
				},
				InitialDelaySeconds: 1,
				FailureThreshold:    1,
				PeriodSeconds:       2,
			}
		},
		FailLivenessProbeFunc: func(p *v1.Pod, probeFilePath string) {
			out, err := podContainerExec(p, 1, fmt.Sprintf("rm %v", probeFilePath))
			framework.Logf("Pod exec output: %v", out)
			framework.ExpectNoError(err, "while failing liveness probe")
		},
		FixLivenessProbeFunc: func(p *v1.Pod, probeFilePath string) {
			_, err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Update(ctx, modified, metav1.UpdateOptions{})
			framework.ExpectNoError(err, "while fixing liveness probe")
		},
	})

}

func testSubpathReconstruction(ctx context.Context, f *framework.Framework, hostExec storageutils.HostExec, pod *v1.Pod, forceDelete bool) {
	// This is mostly copied from TestVolumeUnmountsFromDeletedPodWithForceOption()

	// Disruptive test run serially, we can cache all voluem global mount
	// points and verify after the test that we do not leak any global mount point.
	nodeList, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
	framework.ExpectNoError(err, "while listing schedulable nodes")
	globalMountPointsByNode := make(map[string]sets.String, len(nodeList.Items))
	for _, node := range nodeList.Items {
		globalMountPointsByNode[node.Name] = storageutils.FindVolumeGlobalMountPoints(ctx, hostExec, &node)
	}

	// Change to busybox
	pod.Spec.Containers[0].Image = e2epod.GetDefaultTestImage()
	pod.Spec.Containers[0].Command = e2epod.GenerateScriptCmd(e2epod.InfiniteSleepCommand)
	pod.Spec.Containers[0].Args = nil
	pod.Spec.Containers[1].Image = e2epod.GetDefaultTestImage()
	pod.Spec.Containers[1].Command = e2epod.GenerateScriptCmd(e2epod.InfiniteSleepCommand)
	pod.Spec.Containers[1].Args = nil
	// If grace period is too short, then there is not enough time for the volume
	// manager to cleanup the volumes
	gracePeriod := int64(30)
	pod.Spec.TerminationGracePeriodSeconds = &gracePeriod

	ginkgo.By(fmt.Sprintf("Creating pod %s", pod.Name))
	removeUnusedContainers(pod)
	pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "while creating pod")
	err = e2epod.WaitTimeoutForPodRunningInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, f.Timeouts.PodStart)
	framework.ExpectNoError(err, "while waiting for pod to be running")

	pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "while getting pod")

	var podNode *v1.Node
	for i := range nodeList.Items {
		if nodeList.Items[i].Name == pod.Spec.NodeName {
			podNode = &nodeList.Items[i]
		}
	}
	gomega.Expect(podNode).ToNot(gomega.BeNil(), "pod node should exist in schedulable nodes")

	storageutils.TestVolumeUnmountsFromDeletedPodWithForceOption(ctx, f.ClientSet, f, pod, forceDelete, true, nil, volumePath)

	if podNode != nil {
		mountPoints := globalMountPointsByNode[podNode.Name]
		mountPointsAfter := storageutils.FindVolumeGlobalMountPoints(ctx, hostExec, podNode)
		s1 := mountPointsAfter.Difference(mountPoints)
		s2 := mountPoints.Difference(mountPointsAfter)
		gomega.Expect(s1).To(gomega.BeEmpty(), "global mount points leaked: %v", s1)
		gomega.Expect(s2).To(gomega.BeEmpty(), "global mount points not found: %v", s2)
	}
}

func formatVolume(ctx context.Context, f *framework.Framework, pod *v1.Pod) {
	ginkgo.By(fmt.Sprintf("Creating pod to format volume %s", pod.Name))
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "while creating volume init pod")

	err = e2epod.WaitForPodSuccessInNamespaceTimeout(ctx, f.ClientSet, pod.Name, pod.Namespace, f.Timeouts.PodStart)
	framework.ExpectNoError(err, "while waiting for volume init pod to succeed")

	err = e2epod.DeletePodWithWait(ctx, f.ClientSet, pod)
	framework.ExpectNoError(err, "while deleting volume init pod")
}

func podContainerExec(pod *v1.Pod, containerIndex int, command string) (string, error) {
	if containerIndex > len(pod.Spec.Containers)-1 {
		return "", fmt.Errorf("container not found in pod: index %d", containerIndex)
	}
	var shell string
	var option string
	if framework.NodeOSDistroIs("windows") {
		shell = "powershell"
		option = "/c"
	} else {
		shell = "/bin/sh"
		option = "-c"
	}
	return e2ekubectl.RunKubectl(pod.Namespace, "exec", pod.Name, "--container", pod.Spec.Containers[containerIndex].Name, "--", shell, option, command)
}
