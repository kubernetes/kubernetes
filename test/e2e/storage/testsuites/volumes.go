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

package testsuites

import (
	"context"
	"fmt"
	"path/filepath"

	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

type volumesTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

var _ storageframework.TestSuite = &volumesTestSuite{}

// InitCustomVolumesTestSuite returns volumesTestSuite that implements TestSuite interface
// using custom test patterns
func InitCustomVolumesTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &volumesTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "volumes",
			TestPatterns: patterns,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
		},
	}
}

// InitVolumesTestSuite returns volumesTestSuite that implements TestSuite interface
// using testsuite default patterns
func InitVolumesTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		// Default fsType
		storageframework.DefaultFsInlineVolume,
		storageframework.DefaultFsPreprovisionedPV,
		storageframework.DefaultFsDynamicPV,
		// ext3
		storageframework.Ext3InlineVolume,
		storageframework.Ext3PreprovisionedPV,
		storageframework.Ext3DynamicPV,
		// ext4
		storageframework.Ext4InlineVolume,
		storageframework.Ext4PreprovisionedPV,
		storageframework.Ext4DynamicPV,
		// xfs
		storageframework.XfsInlineVolume,
		storageframework.XfsPreprovisionedPV,
		storageframework.XfsDynamicPV,
		// ntfs
		storageframework.NtfsInlineVolume,
		storageframework.NtfsPreprovisionedPV,
		storageframework.NtfsDynamicPV,
		// block volumes
		storageframework.BlockVolModePreprovisionedPV,
		storageframework.BlockVolModeDynamicPV,
	}
	return InitCustomVolumesTestSuite(patterns)
}

func (t *volumesTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return t.tsInfo
}

func (t *volumesTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	if pattern.VolMode == v1.PersistentVolumeBlock {
		skipTestIfBlockNotSupported(driver)
	}
}

func skipExecTest(driver storageframework.TestDriver) {
	dInfo := driver.GetDriverInfo()
	if !dInfo.Capabilities[storageframework.CapExec] {
		e2eskipper.Skipf("Driver %q does not support exec - skipping", dInfo.Name)
	}
}

func skipTestIfBlockNotSupported(driver storageframework.TestDriver) {
	dInfo := driver.GetDriverInfo()
	if !dInfo.Capabilities[storageframework.CapBlock] {
		e2eskipper.Skipf("Driver %q does not provide raw block - skipping", dInfo.Name)
	}
}

func (t *volumesTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	type local struct {
		config *storageframework.PerTestConfig

		resource *storageframework.VolumeResource

		migrationCheck *migrationOpCheck
	}
	var dInfo = driver.GetDriverInfo()
	var l local

	// Beware that it also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewFrameworkWithCustomTimeouts("volume", storageframework.GetDriverTimeouts(driver))
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	init := func(ctx context.Context) {
		l = local{}

		// Now do the more expensive test initialization.
		l.config = driver.PrepareTest(ctx, f)
		l.migrationCheck = newMigrationOpCheck(ctx, f.ClientSet, f.ClientConfig(), dInfo.InTreePluginName)
		testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
		l.resource = storageframework.CreateVolumeResource(ctx, driver, l.config, pattern, testVolumeSizeRange)
		if l.resource.VolSource == nil {
			e2eskipper.Skipf("Driver %q does not define volumeSource - skipping", dInfo.Name)
		}
	}

	cleanup := func(ctx context.Context) {
		var errs []error
		if l.resource != nil {
			errs = append(errs, l.resource.CleanupResource(ctx))
			l.resource = nil
		}

		framework.ExpectNoError(errors.NewAggregate(errs), "while cleaning up resource")
		l.migrationCheck.validateMigrationVolumeOpCounts(ctx)
	}

	ginkgo.It("should store data", func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(e2evolume.TestServerCleanup, f, storageframework.ConvertTestConfig(l.config))
		ginkgo.DeferCleanup(cleanup)

		tests := []e2evolume.Test{
			{
				Volume: *l.resource.VolSource,
				Mode:   pattern.VolMode,
				File:   "index.html",
				// Must match content
				ExpectedContent: fmt.Sprintf("Hello from %s from namespace %s",
					dInfo.Name, f.Namespace.Name),
			},
		}
		config := storageframework.ConvertTestConfig(l.config)
		var fsGroup *int64
		if framework.NodeOSDistroIs("windows") && dInfo.Capabilities[storageframework.CapFsGroup] {
			fsGroupVal := int64(1234)
			fsGroup = &fsGroupVal
		}
		// We set same fsGroup for both pods, because for same volumes (e.g.
		// local), plugin skips setting fsGroup if volume is already mounted
		// and we don't have reliable way to detect volumes are unmounted or
		// not before starting the second pod.
		e2evolume.InjectContent(ctx, f, config, fsGroup, pattern.FsType, tests)
		if driver.GetDriverInfo().Capabilities[storageframework.CapPersistence] {
			e2evolume.TestVolumeClient(ctx, f, config, fsGroup, pattern.FsType, tests)
		} else {
			ginkgo.By("Skipping persistence check for non-persistent volume")
		}
	})

	// Exec works only on filesystem volumes
	if pattern.VolMode != v1.PersistentVolumeBlock {
		ginkgo.It("should allow exec of files on the volume", func(ctx context.Context) {
			skipExecTest(driver)
			init(ctx)
			ginkgo.DeferCleanup(cleanup)

			testScriptInPod(ctx, f, string(pattern.VolType), l.resource.VolSource, l.config)
		})
	}
}

func testScriptInPod(
	ctx context.Context,
	f *framework.Framework,
	volumeType string,
	source *v1.VolumeSource,
	config *storageframework.PerTestConfig) {

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
					Image:   e2epod.GetTestImage(imageutils.Nginx),
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
		},
	}
	e2epod.SetNodeSelection(&pod.Spec, config.ClientNodeSelection)
	ginkgo.By(fmt.Sprintf("Creating pod %s", pod.Name))
	e2eoutput.TestContainerOutput(ctx, f, "exec-volume-test", pod, 0, []string{fileName})

	ginkgo.By(fmt.Sprintf("Deleting pod %s", pod.Name))
	err := e2epod.DeletePodWithWait(ctx, f.ClientSet, pod)
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
