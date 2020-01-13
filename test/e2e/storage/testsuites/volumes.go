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

// This test checks that various VolumeSources are working.

// test/e2e/common/volumes.go duplicates the GlusterFS test from this file.  Any changes made to this
// test should be made there as well.

package testsuites

import (
	"fmt"
	"path/filepath"

	"github.com/onsi/ginkgo"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

type volumesTestSuite struct {
	tsInfo TestSuiteInfo
}

var _ TestSuite = &volumesTestSuite{}

// InitVolumesTestSuite returns volumesTestSuite that implements TestSuite interface
func InitVolumesTestSuite() TestSuite {
	return &volumesTestSuite{
		tsInfo: TestSuiteInfo{
			Name: "volumes",
			TestPatterns: []testpatterns.TestPattern{
				// Default fsType
				testpatterns.DefaultFsInlineVolume,
				testpatterns.DefaultFsPreprovisionedPV,
				testpatterns.DefaultFsDynamicPV,
				// ext3
				testpatterns.Ext3InlineVolume,
				testpatterns.Ext3PreprovisionedPV,
				testpatterns.Ext3DynamicPV,
				// ext4
				testpatterns.Ext4InlineVolume,
				testpatterns.Ext4PreprovisionedPV,
				testpatterns.Ext4DynamicPV,
				// xfs
				testpatterns.XfsInlineVolume,
				testpatterns.XfsPreprovisionedPV,
				testpatterns.XfsDynamicPV,
				// ntfs
				testpatterns.NtfsInlineVolume,
				testpatterns.NtfsPreprovisionedPV,
				testpatterns.NtfsDynamicPV,
				// block volumes
				testpatterns.BlockVolModePreprovisionedPV,
				testpatterns.BlockVolModeDynamicPV,
			},
			SupportedSizeRange: volume.SizeRange{
				Min: "1Mi",
			},
		},
	}
}

func (t *volumesTestSuite) GetTestSuiteInfo() TestSuiteInfo {
	return t.tsInfo
}

func (t *volumesTestSuite) SkipRedundantSuite(driver TestDriver, pattern testpatterns.TestPattern) {
}

func skipExecTest(driver TestDriver) {
	dInfo := driver.GetDriverInfo()
	if !dInfo.Capabilities[CapExec] {
		e2eskipper.Skipf("Driver %q does not support exec - skipping", dInfo.Name)
	}
}

func skipTestIfBlockNotSupported(driver TestDriver) {
	dInfo := driver.GetDriverInfo()
	if !dInfo.Capabilities[CapBlock] {
		e2eskipper.Skipf("Driver %q does not provide raw block - skipping", dInfo.Name)
	}
}

func (t *volumesTestSuite) DefineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	type local struct {
		config        *PerTestConfig
		driverCleanup func()

		resource *VolumeResource

		intreeOps   opCounts
		migratedOps opCounts
	}
	var dInfo = driver.GetDriverInfo()
	var l local

	// No preconditions to test. Normally they would be in a BeforeEach here.

	// This intentionally comes after checking the preconditions because it
	// registers its own BeforeEach which creates the namespace. Beware that it
	// also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewDefaultFramework("volume")

	init := func() {
		l = local{}

		// Now do the more expensive test initialization.
		l.config, l.driverCleanup = driver.PrepareTest(f)
		l.intreeOps, l.migratedOps = getMigrationVolumeOpCounts(f.ClientSet, dInfo.InTreePluginName)
		testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
		l.resource = CreateVolumeResource(driver, l.config, pattern, testVolumeSizeRange)
		if l.resource.VolSource == nil {
			e2eskipper.Skipf("Driver %q does not define volumeSource - skipping", dInfo.Name)
		}
	}

	cleanup := func() {
		var errs []error
		if l.resource != nil {
			errs = append(errs, l.resource.CleanupResource())
			l.resource = nil
		}

		errs = append(errs, tryFunc(l.driverCleanup))
		l.driverCleanup = nil
		framework.ExpectNoError(errors.NewAggregate(errs), "while cleaning up resource")
		validateMigrationVolumeOpCounts(f.ClientSet, dInfo.InTreePluginName, l.intreeOps, l.migratedOps)
	}

	ginkgo.It("should store data", func() {
		if pattern.VolMode == v1.PersistentVolumeBlock {
			skipTestIfBlockNotSupported(driver)
		}

		init()
		defer func() {
			volume.TestCleanup(f, convertTestConfig(l.config))
			cleanup()
		}()

		tests := []volume.Test{
			{
				Volume: *l.resource.VolSource,
				Mode:   pattern.VolMode,
				File:   "index.html",
				// Must match content
				ExpectedContent: fmt.Sprintf("Hello from %s from namespace %s",
					dInfo.Name, f.Namespace.Name),
			},
		}
		config := convertTestConfig(l.config)
		var fsGroup *int64
		if framework.NodeOSDistroIs("windows") && dInfo.Capabilities[CapFsGroup] {
			fsGroupVal := int64(1234)
			fsGroup = &fsGroupVal
		}
		// We set same fsGroup for both pods, because for same volumes (e.g.
		// local), plugin skips setting fsGroup if volume is already mounted
		// and we don't have reliable way to detect volumes are unmounted or
		// not before starting the second pod.
		volume.InjectContent(f, config, fsGroup, pattern.FsType, tests)
		if driver.GetDriverInfo().Capabilities[CapPersistence] {
			volume.TestVolumeClient(f, config, fsGroup, pattern.FsType, tests)
		} else {
			ginkgo.By("Skipping persistence check for non-persistent volume")
		}
	})

	// Exec works only on filesystem volumes
	if pattern.VolMode != v1.PersistentVolumeBlock {
		ginkgo.It("should allow exec of files on the volume", func() {
			skipExecTest(driver)
			init()
			defer cleanup()

			testScriptInPod(f, string(pattern.VolType), l.resource.VolSource, l.config)
		})
	}
}

func testScriptInPod(
	f *framework.Framework,
	volumeType string,
	source *v1.VolumeSource,
	config *PerTestConfig) {

	const (
		volPath = "/vol1"
		volName = "vol1"
	)
	suffix := generateSuffixForPodName(volumeType)
	fileName := fmt.Sprintf("test-%s", suffix)
	var content string
	if framework.NodeOSDistroIs("windows") {
		content = fmt.Sprintf("ls -n %s", volPath)
	} else {
		content = fmt.Sprintf("ls %s", volPath)
	}
	command := generateWriteandExecuteScriptFileCmd(content, fileName, volPath)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("exec-volume-test-%s", suffix),
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    fmt.Sprintf("exec-container-%s", suffix),
					Image:   volume.GetTestImage(imageutils.GetE2EImage(imageutils.Nginx)),
					Command: command,
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volName,
							MountPath: volPath,
						},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name:         volName,
					VolumeSource: *source,
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			NodeSelector:  config.ClientNodeSelector,
			NodeName:      config.ClientNodeName,
		},
	}
	ginkgo.By(fmt.Sprintf("Creating pod %s", pod.Name))
	f.TestContainerOutput("exec-volume-test", pod, 0, []string{fileName})

	ginkgo.By(fmt.Sprintf("Deleting pod %s", pod.Name))
	err := e2epod.DeletePodWithWait(f.ClientSet, pod)
	framework.ExpectNoError(err, "while deleting pod")
}

// generateWriteandExecuteScriptFileCmd generates the corresponding command lines to write a file with the given file path
// and also execute this file.
// Depending on the Node OS is Windows or linux, the command will use powershell or /bin/sh
func generateWriteandExecuteScriptFileCmd(content, fileName, filePath string) []string {
	// for windows cluster, modify the Pod spec.
	if framework.NodeOSDistroIs("windows") {
		scriptName := fmt.Sprintf("%s.ps1", fileName)
		fullPath := filepath.Join(filePath, scriptName)

		cmd := "echo \"" + content + "\" > " + fullPath + "; .\\" + fullPath
		framework.Logf("generated pod command %s", cmd)
		return []string{"powershell", "/c", cmd}
	}
	scriptName := fmt.Sprintf("%s.sh", fileName)
	fullPath := filepath.Join(filePath, scriptName)
	cmd := fmt.Sprintf("echo \"%s\" > %s; chmod u+x %s; %s;", content, fullPath, fullPath, fullPath)
	return []string{"/bin/sh", "-ec", cmd}
}
