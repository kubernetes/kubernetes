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

/*
 * This test checks that the plugin VolumeSources are working when pseudo-streaming
 * various write sizes to mounted files.
 */

package testsuites

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

// MD5 hashes of the test file corresponding to each file size.
// Test files are generated in testVolumeIO()
// If test file generation algorithm changes, these must be recomputed.
var md5hashes = map[int64]string{
	storageframework.FileSizeSmall:  "5c34c2813223a7ca05a3c2f38c0d1710",
	storageframework.FileSizeMedium: "f2fa202b1ffeedda5f3a58bd1ae81104",
	storageframework.FileSizeLarge:  "8d763edc71bd16217664793b5a15e403",
}

const mountPath = "/opt"

type volumeIOTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

// InitCustomVolumeIOTestSuite returns volumeIOTestSuite that implements TestSuite interface
// using custom test patterns
func InitCustomVolumeIOTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &volumeIOTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "volumeIO",
			TestPatterns: patterns,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
		},
	}
}

// InitVolumeIOTestSuite returns volumeIOTestSuite that implements TestSuite interface
// using testsuite default patterns
func InitVolumeIOTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.DefaultFsInlineVolume,
		storageframework.DefaultFsPreprovisionedPV,
		storageframework.DefaultFsDynamicPV,
		storageframework.NtfsDynamicPV,
	}
	return InitCustomVolumeIOTestSuite(patterns)
}

func (t *volumeIOTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return t.tsInfo
}

func (t *volumeIOTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	skipVolTypePatterns(pattern, driver, storageframework.NewVolTypeMap(
		storageframework.PreprovisionedPV,
		storageframework.InlineVolume))
}

func (t *volumeIOTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	type local struct {
		config *storageframework.PerTestConfig

		resource *storageframework.VolumeResource

		migrationCheck *migrationOpCheck
	}
	var (
		dInfo = driver.GetDriverInfo()
		l     local
	)

	// Beware that it also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewFrameworkWithCustomTimeouts("volumeio", storageframework.GetDriverTimeouts(driver))
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

	f.It("should write files of various sizes, verify size, validate content", f.WithSlow(), func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		cs := f.ClientSet
		fileSizes := createFileSizes(dInfo.MaxFileSize)
		testFile := fmt.Sprintf("%s_io_test_%s", dInfo.Name, f.Namespace.Name)
		var fsGroup *int64
		if !framework.NodeOSDistroIs("windows") && dInfo.Capabilities[storageframework.CapFsGroup] {
			fsGroupVal := int64(1234)
			fsGroup = &fsGroupVal
		}
		podSec := v1.PodSecurityContext{
			FSGroup: fsGroup,
		}
		err := testVolumeIO(ctx, f, cs, storageframework.ConvertTestConfig(l.config), *l.resource.VolSource, &podSec, testFile, fileSizes)
		framework.ExpectNoError(err)
	})
}

func createFileSizes(maxFileSize int64) []int64 {
	allFileSizes := []int64{
		storageframework.FileSizeSmall,
		storageframework.FileSizeMedium,
		storageframework.FileSizeLarge,
	}
	fileSizes := []int64{}

	for _, size := range allFileSizes {
		if size <= maxFileSize {
			fileSizes = append(fileSizes, size)
		}
	}

	return fileSizes
}

// Return the plugin's client pod spec. Use an InitContainer to setup the file i/o test env.
func makePodSpec(config e2evolume.TestConfig, initCmd string, volsrc v1.VolumeSource, podSecContext *v1.PodSecurityContext) *v1.Pod {
	var gracePeriod int64 = 1
	volName := fmt.Sprintf("io-volume-%s", config.Namespace)
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: config.Prefix + "-io-client",
			Labels: map[string]string{
				"role": config.Prefix + "-io-client",
			},
		},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{
				{
					Name:  config.Prefix + "-io-init",
					Image: e2epod.GetDefaultTestImage(),
					Command: []string{
						"/bin/sh",
						"-c",
						initCmd,
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volName,
							MountPath: mountPath,
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Name:  config.Prefix + "-io-client",
					Image: e2epod.GetDefaultTestImage(),
					Command: []string{
						"/bin/sh",
						"-c",
						"sleep 3600", // keep pod alive until explicitly deleted
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volName,
							MountPath: mountPath,
						},
					},
				},
			},
			TerminationGracePeriodSeconds: &gracePeriod,
			SecurityContext:               podSecContext,
			Volumes: []v1.Volume{
				{
					Name:         volName,
					VolumeSource: volsrc,
				},
			},
			RestartPolicy: v1.RestartPolicyNever, // want pod to fail if init container fails
		},
	}

	e2epod.SetNodeSelection(&pod.Spec, config.ClientNodeSelection)
	return pod
}

// Write `fsize` bytes to `fpath` in the pod, using dd and the `ddInput` file.
func writeToFile(f *framework.Framework, pod *v1.Pod, fpath, ddInput string, fsize int64) error {
	ginkgo.By(fmt.Sprintf("writing %d bytes to test file %s", fsize, fpath))
	loopCnt := fsize / storageframework.MinFileSize
	writeCmd := fmt.Sprintf("i=0; while [ $i -lt %d ]; do dd if=%s bs=%d >>%s 2>/dev/null; let i+=1; done", loopCnt, ddInput, storageframework.MinFileSize, fpath)
	stdout, stderr, err := e2evolume.PodExec(f, pod, writeCmd)
	if err != nil {
		return fmt.Errorf("error writing to volume using %q: %s\nstdout: %s\nstderr: %s", writeCmd, err, stdout, stderr)
	}
	return err
}

// Verify that the test file is the expected size and contains the expected content.
func verifyFile(f *framework.Framework, pod *v1.Pod, fpath string, expectSize int64, ddInput string) error {
	ginkgo.By("verifying file size")
	rtnstr, stderr, err := e2evolume.PodExec(f, pod, fmt.Sprintf("stat -c %%s %s", fpath))
	if err != nil || rtnstr == "" {
		return fmt.Errorf("unable to get file size via `stat %s`: %v\nstdout: %s\nstderr: %s", fpath, err, rtnstr, stderr)
	}
	size, err := strconv.Atoi(strings.TrimSuffix(rtnstr, "\n"))
	if err != nil {
		return fmt.Errorf("unable to convert string %q to int: %w", rtnstr, err)
	}
	if int64(size) != expectSize {
		return fmt.Errorf("size of file %s is %d, expected %d", fpath, size, expectSize)
	}

	ginkgo.By("verifying file hash")
	rtnstr, stderr, err = e2evolume.PodExec(f, pod, fmt.Sprintf("md5sum %s | cut -d' ' -f1", fpath))
	if err != nil {
		return fmt.Errorf("unable to test file hash via `md5sum %s`: %v\nstdout: %s\nstderr: %s", fpath, err, rtnstr, stderr)
	}
	actualHash := strings.TrimSuffix(rtnstr, "\n")
	expectedHash, ok := md5hashes[expectSize]
	if !ok {
		return fmt.Errorf("File hash is unknown for file size %d. Was a new file size added to the test suite?",
			expectSize)
	}
	if actualHash != expectedHash {
		return fmt.Errorf("MD5 hash is incorrect for file %s with size %d. Expected: `%s`; Actual: `%s`",
			fpath, expectSize, expectedHash, actualHash)
	}

	return nil
}

// Delete `fpath` to save some disk space on host. Delete errors are logged but ignored.
func deleteFile(f *framework.Framework, pod *v1.Pod, fpath string) {
	ginkgo.By(fmt.Sprintf("deleting test file %s...", fpath))
	stdout, stderr, err := e2evolume.PodExec(f, pod, fmt.Sprintf("rm -f %s", fpath))
	if err != nil {
		// keep going, the test dir will be deleted when the volume is unmounted
		framework.Logf("unable to delete test file %s: %v\nerror ignored, continuing test\nstdout: %s\nstderr: %s", fpath, err, stdout, stderr)
	}
}

// Create the client pod and create files of the sizes passed in by the `fsizes` parameter. Delete the
// client pod and the new files when done.
// Note: the file name is appended to "/opt/<Prefix>/<namespace>", eg. "/opt/nfs/e2e-.../<file>".
// Note: nil can be passed for the podSecContext parm, in which case it is ignored.
// Note: `fsizes` values are enforced to each be at least `MinFileSize` and a multiple of `MinFileSize`
//
//	bytes.
func testVolumeIO(ctx context.Context, f *framework.Framework, cs clientset.Interface, config e2evolume.TestConfig, volsrc v1.VolumeSource, podSecContext *v1.PodSecurityContext, file string, fsizes []int64) (err error) {
	ddInput := filepath.Join(mountPath, fmt.Sprintf("%s-%s-dd_if", config.Prefix, config.Namespace))
	writeBlk := strings.Repeat("abcdefghijklmnopqrstuvwxyz123456", 32) // 1KiB value
	loopCnt := storageframework.MinFileSize / int64(len(writeBlk))
	// initContainer cmd to create and fill dd's input file. The initContainer is used to create
	// the `dd` input file which is currently 1MiB. Rather than store a 1MiB go value, a loop is
	// used to create a 1MiB file in the target directory.
	initCmd := fmt.Sprintf("i=0; while [ $i -lt %d ]; do echo -n %s >>%s; let i+=1; done", loopCnt, writeBlk, ddInput)

	clientPod := makePodSpec(config, initCmd, volsrc, podSecContext)

	ginkgo.By(fmt.Sprintf("starting %s", clientPod.Name))
	podsNamespacer := cs.CoreV1().Pods(config.Namespace)
	clientPod, err = podsNamespacer.Create(ctx, clientPod, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("failed to create client pod %q: %w", clientPod.Name, err)
	}
	ginkgo.DeferCleanup(func(ctx context.Context) {
		deleteFile(f, clientPod, ddInput)
		ginkgo.By(fmt.Sprintf("deleting client pod %q...", clientPod.Name))
		e := e2epod.DeletePodWithWait(ctx, cs, clientPod)
		if e != nil {
			framework.Logf("client pod failed to delete: %v", e)
			if err == nil { // delete err is returned if err is not set
				err = e
			}
		} else {
			framework.Logf("sleeping a bit so kubelet can unmount and detach the volume")
			time.Sleep(e2evolume.PodCleanupTimeout)
		}
	})

	err = e2epod.WaitTimeoutForPodRunningInNamespace(ctx, cs, clientPod.Name, clientPod.Namespace, f.Timeouts.PodStart)
	if err != nil {
		return fmt.Errorf("client pod %q not running: %w", clientPod.Name, err)
	}

	// create files of the passed-in file sizes and verify test file size and content
	for _, fsize := range fsizes {
		// file sizes must be a multiple of `MinFileSize`
		if math.Mod(float64(fsize), float64(storageframework.MinFileSize)) != 0 {
			fsize = fsize/storageframework.MinFileSize + storageframework.MinFileSize
		}
		fpath := filepath.Join(mountPath, fmt.Sprintf("%s-%d", file, fsize))
		defer func() {
			deleteFile(f, clientPod, fpath)
		}()
		if err = writeToFile(f, clientPod, fpath, ddInput, fsize); err != nil {
			return err
		}
		if err = verifyFile(f, clientPod, fpath, fsize, ddInput); err != nil {
			return err
		}
	}

	return
}
