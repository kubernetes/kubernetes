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

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
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
			name: "volumes",
			testPatterns: []testpatterns.TestPattern{
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
			},
		},
	}
}

func (t *volumesTestSuite) getTestSuiteInfo() TestSuiteInfo {
	return t.tsInfo
}

func (t *volumesTestSuite) skipUnsupportedTest(pattern testpatterns.TestPattern, driver drivers.TestDriver) {
}

func skipPersistenceTest(driver drivers.TestDriver) {
	dInfo := driver.GetDriverInfo()
	if !dInfo.Capabilities[drivers.CapPersistence] {
		framework.Skipf("Driver %q does not provide persistency - skipping", dInfo.Name)
	}
}

func skipExecTest(driver drivers.TestDriver) {
	dInfo := driver.GetDriverInfo()
	if !dInfo.Capabilities[drivers.CapExec] {
		framework.Skipf("Driver %q does not support exec - skipping", dInfo.Name)
	}
}

func createVolumesTestInput(pattern testpatterns.TestPattern, resource genericVolumeTestResource) volumesTestInput {
	var fsGroup *int64
	driver := resource.driver
	dInfo := driver.GetDriverInfo()
	f := dInfo.Framework
	volSource := resource.volSource

	if volSource == nil {
		framework.Skipf("Driver %q does not define volumeSource - skipping", dInfo.Name)
	}

	if dInfo.Capabilities[drivers.CapFsGroup] {
		fsGroupVal := int64(1234)
		fsGroup = &fsGroupVal
	}

	return volumesTestInput{
		f:        f,
		name:     dInfo.Name,
		config:   dInfo.Config,
		fsGroup:  fsGroup,
		resource: resource,
		tests: []framework.VolumeTest{
			{
				Volume: *volSource,
				File:   "index.html",
				// Must match content
				ExpectedContent: fmt.Sprintf("Hello from %s from namespace %s",
					dInfo.Name, f.Namespace.Name),
			},
		},
	}
}

func (t *volumesTestSuite) execTest(driver drivers.TestDriver, pattern testpatterns.TestPattern) {
	Context(getTestNameStr(t, pattern), func() {
		var (
			resource     genericVolumeTestResource
			input        volumesTestInput
			needsCleanup bool
		)

		BeforeEach(func() {
			needsCleanup = false
			// Skip unsupported tests to avoid unnecessary resource initialization
			skipUnsupportedTest(t, driver, pattern)
			needsCleanup = true

			// Setup test resource for driver and testpattern
			resource = genericVolumeTestResource{}
			resource.setupResource(driver, pattern)

			// Create test input
			input = createVolumesTestInput(pattern, resource)
		})

		AfterEach(func() {
			if needsCleanup {
				resource.cleanupResource(driver, pattern)
			}
		})

		testVolumes(&input)
	})
}

type volumesTestInput struct {
	f        *framework.Framework
	name     string
	config   framework.VolumeTestConfig
	fsGroup  *int64
	tests    []framework.VolumeTest
	resource genericVolumeTestResource
}

func testVolumes(input *volumesTestInput) {
	It("should be mountable", func() {
		f := input.f
		cs := f.ClientSet
		defer framework.VolumeTestCleanup(f, input.config)

		skipPersistenceTest(input.resource.driver)

		volumeTest := input.tests
		framework.InjectHtml(cs, input.config, volumeTest[0].Volume, volumeTest[0].ExpectedContent)
		framework.TestVolumeClient(cs, input.config, input.fsGroup, input.tests)
	})
	It("should allow exec of files on the volume", func() {
		f := input.f
		skipExecTest(input.resource.driver)

		testScriptInPod(f, input.resource.volType, input.resource.volSource, input.config.NodeSelector)
	})
}

func testScriptInPod(
	f *framework.Framework,
	volumeType string,
	source *v1.VolumeSource,
	nodeSelector map[string]string) {

	const (
		volPath = "/vol1"
		volName = "vol1"
	)
	suffix := generateSuffixForPodName(volumeType)
	scriptName := fmt.Sprintf("test-%s.sh", suffix)
	fullPath := filepath.Join(volPath, scriptName)
	cmd := fmt.Sprintf("echo \"ls %s\" > %s; chmod u+x %s; %s", volPath, fullPath, fullPath, fullPath)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("exec-volume-test-%s", suffix),
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    fmt.Sprintf("exec-container-%s", suffix),
					Image:   imageutils.GetE2EImage(imageutils.Nginx),
					Command: []string{"/bin/sh", "-ec", cmd},
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
			NodeSelector:  nodeSelector,
		},
	}
	By(fmt.Sprintf("Creating pod %s", pod.Name))
	f.TestContainerOutput("exec-volume-test", pod, 0, []string{scriptName})

	By(fmt.Sprintf("Deleting pod %s", pod.Name))
	err := framework.DeletePodWithWait(f, f.ClientSet, pod)
	Expect(err).NotTo(HaveOccurred(), "while deleting pod")
}
