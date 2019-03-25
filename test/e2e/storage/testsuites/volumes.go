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

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
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
				// ntfs
				testpatterns.NtfsInlineVolume,
				testpatterns.NtfsPreprovisionedPV,
				testpatterns.NtfsDynamicPV,
			},
		},
	}
}

func (t *volumesTestSuite) getTestSuiteInfo() TestSuiteInfo {
	return t.tsInfo
}

func (t *volumesTestSuite) skipUnsupportedTest(pattern testpatterns.TestPattern, driver TestDriver) {
}

func skipPersistenceTest(driver TestDriver) {
	dInfo := driver.GetDriverInfo()
	if !dInfo.Capabilities[CapPersistence] {
		framework.Skipf("Driver %q does not provide persistency - skipping", dInfo.Name)
	}
}

func skipExecTest(driver TestDriver) {
	dInfo := driver.GetDriverInfo()
	if !dInfo.Capabilities[CapExec] {
		framework.Skipf("Driver %q does not support exec - skipping", dInfo.Name)
	}
}

func (t *volumesTestSuite) defineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	type local struct {
		config      *PerTestConfig
		testCleanup func()

		resource *genericVolumeTestResource
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
		l.config, l.testCleanup = driver.PrepareTest(f)
		l.resource = createGenericVolumeTestResource(driver, l.config, pattern)
		if l.resource.volSource == nil {
			framework.Skipf("Driver %q does not define volumeSource - skipping", dInfo.Name)
		}
	}

	cleanup := func() {
		if l.resource != nil {
			l.resource.cleanupResource()
			l.resource = nil
		}

		if l.testCleanup != nil {
			l.testCleanup()
			l.testCleanup = nil
		}
	}

	It("should be mountable", func() {
		skipPersistenceTest(driver)
		init()
		defer func() {
			framework.VolumeTestCleanup(f, convertTestConfig(l.config))
			cleanup()
		}()

		tests := []framework.VolumeTest{
			{
				Volume: *l.resource.volSource,
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
		framework.InjectHtml(f.ClientSet, config, fsGroup, tests[0].Volume, tests[0].ExpectedContent)
		framework.TestVolumeClient(f.ClientSet, config, fsGroup, pattern.FsType, tests)
	})

	It("should allow exec of files on the volume", func() {
		skipExecTest(driver)
		init()
		defer cleanup()

		testScriptInPod(f, l.resource.volType, l.resource.volSource, l.config.ClientNodeSelector)
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
	fileName := fmt.Sprintf("test-%s", suffix)
	var content string
	if framework.NodeOSDistroIs("windows") {
		content = fmt.Sprintf("ls -n %s", volPath)
	} else {
		content = fmt.Sprintf("ls %s", volPath)
	}
	command := framework.GenerateWriteandExecuteScriptFileCmd(content, fileName, volPath)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("exec-volume-test-%s", suffix),
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    fmt.Sprintf("exec-container-%s", suffix),
					Image:   framework.GetTestImage(imageutils.GetE2EImage(imageutils.Nginx)),
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
			NodeSelector:  nodeSelector,
		},
	}
	By(fmt.Sprintf("Creating pod %s", pod.Name))
	f.TestContainerOutput("exec-volume-test", pod, 0, []string{fileName})

	By(fmt.Sprintf("Deleting pod %s", pod.Name))
	err := framework.DeletePodWithWait(f, f.ClientSet, pod)
	Expect(err).NotTo(HaveOccurred(), "while deleting pod")
}
