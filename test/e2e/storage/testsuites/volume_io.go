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
	"fmt"
	"math"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

// MD5 hashes of the test file corresponding to each file size.
// Test files are generated in testVolumeIO()
// If test file generation algorithm changes, these must be recomputed.
var md5hashes = map[int64]string{
	testpatterns.FileSizeSmall:  "5c34c2813223a7ca05a3c2f38c0d1710",
	testpatterns.FileSizeMedium: "f2fa202b1ffeedda5f3a58bd1ae81104",
	testpatterns.FileSizeLarge:  "8d763edc71bd16217664793b5a15e403",
}

const mountPath = "/opt"

type volumeIOTestSuite struct {
	tsInfo TestSuiteInfo
}

var _ TestSuite = &volumeIOTestSuite{}

// InitVolumeIOTestSuite returns volumeIOTestSuite that implements TestSuite interface
func InitVolumeIOTestSuite() TestSuite {
	return &volumeIOTestSuite{
		tsInfo: TestSuiteInfo{
			name: "volumeIO",
			testPatterns: []testpatterns.TestPattern{
				testpatterns.DefaultFsInlineVolume,
				testpatterns.DefaultFsPreprovisionedPV,
				testpatterns.DefaultFsDynamicPV,
			},
		},
	}
}

func (t *volumeIOTestSuite) getTestSuiteInfo() TestSuiteInfo {
	return t.tsInfo
}

func (t *volumeIOTestSuite) skipUnsupportedTest(pattern testpatterns.TestPattern, driver drivers.TestDriver) {
}

func createVolumeIOTestInput(pattern testpatterns.TestPattern, resource genericVolumeTestResource) volumeIOTestInput {
	var fsGroup *int64
	driver := resource.driver
	dInfo := driver.GetDriverInfo()
	f := dInfo.Framework
	fileSizes := createFileSizes(dInfo.MaxFileSize)
	volSource := resource.volSource

	if volSource == nil {
		framework.Skipf("Driver %q does not define volumeSource - skipping", dInfo.Name)
	}

	if dInfo.Capabilities[drivers.CapFsGroup] {
		fsGroupVal := int64(1234)
		fsGroup = &fsGroupVal
	}

	return volumeIOTestInput{
		f:         f,
		name:      dInfo.Name,
		config:    dInfo.Config,
		volSource: *volSource,
		testFile:  fmt.Sprintf("%s_io_test_%s", dInfo.Name, f.Namespace.Name),
		podSec: v1.PodSecurityContext{
			FSGroup: fsGroup,
		},
		fileSizes: fileSizes,
	}
}

func (t *volumeIOTestSuite) execTest(driver drivers.TestDriver, pattern testpatterns.TestPattern) {
	Context(getTestNameStr(t, pattern), func() {
		var (
			resource     genericVolumeTestResource
			input        volumeIOTestInput
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
			input = createVolumeIOTestInput(pattern, resource)
		})

		AfterEach(func() {
			if needsCleanup {
				resource.cleanupResource(driver, pattern)
			}
		})

		execTestVolumeIO(&input)
	})
}

type volumeIOTestInput struct {
	f         *framework.Framework
	name      string
	config    framework.VolumeTestConfig
	volSource v1.VolumeSource
	testFile  string
	podSec    v1.PodSecurityContext
	fileSizes []int64
}

func execTestVolumeIO(input *volumeIOTestInput) {
	It("should write files of various sizes, verify size, validate content [Slow]", func() {
		f := input.f
		cs := f.ClientSet

		err := testVolumeIO(f, cs, input.config, input.volSource, &input.podSec, input.testFile, input.fileSizes)
		Expect(err).NotTo(HaveOccurred())
	})
}

func createFileSizes(maxFileSize int64) []int64 {
	allFileSizes := []int64{
		testpatterns.FileSizeSmall,
		testpatterns.FileSizeMedium,
		testpatterns.FileSizeLarge,
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
func makePodSpec(config framework.VolumeTestConfig, initCmd string, volsrc v1.VolumeSource, podSecContext *v1.PodSecurityContext) *v1.Pod {
	var gracePeriod int64 = 1
	volName := fmt.Sprintf("io-volume-%s", config.Namespace)
	return &v1.Pod{
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
					Image: framework.BusyBoxImage,
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
					Image: framework.BusyBoxImage,
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
			NodeName:      config.ClientNodeName,
			NodeSelector:  config.NodeSelector,
		},
	}
}

// Write `fsize` bytes to `fpath` in the pod, using dd and the `ddInput` file.
func writeToFile(pod *v1.Pod, fpath, ddInput string, fsize int64) error {
	By(fmt.Sprintf("writing %d bytes to test file %s", fsize, fpath))
	loopCnt := fsize / testpatterns.MinFileSize
	writeCmd := fmt.Sprintf("i=0; while [ $i -lt %d ]; do dd if=%s bs=%d >>%s 2>/dev/null; let i+=1; done", loopCnt, ddInput, testpatterns.MinFileSize, fpath)
	_, err := utils.PodExec(pod, writeCmd)

	return err
}

// Verify that the test file is the expected size and contains the expected content.
func verifyFile(pod *v1.Pod, fpath string, expectSize int64, ddInput string) error {
	By("verifying file size")
	rtnstr, err := utils.PodExec(pod, fmt.Sprintf("stat -c %%s %s", fpath))
	if err != nil || rtnstr == "" {
		return fmt.Errorf("unable to get file size via `stat %s`: %v", fpath, err)
	}
	size, err := strconv.Atoi(strings.TrimSuffix(rtnstr, "\n"))
	if err != nil {
		return fmt.Errorf("unable to convert string %q to int: %v", rtnstr, err)
	}
	if int64(size) != expectSize {
		return fmt.Errorf("size of file %s is %d, expected %d", fpath, size, expectSize)
	}

	By("verifying file hash")
	rtnstr, err = utils.PodExec(pod, fmt.Sprintf("md5sum %s | cut -d' ' -f1", fpath))
	if err != nil {
		return fmt.Errorf("unable to test file hash via `md5sum %s`: %v", fpath, err)
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
func deleteFile(pod *v1.Pod, fpath string) {
	By(fmt.Sprintf("deleting test file %s...", fpath))
	_, err := utils.PodExec(pod, fmt.Sprintf("rm -f %s", fpath))
	if err != nil {
		// keep going, the test dir will be deleted when the volume is unmounted
		framework.Logf("unable to delete test file %s: %v\nerror ignored, continuing test", fpath, err)
	}
}

// Create the client pod and create files of the sizes passed in by the `fsizes` parameter. Delete the
// client pod and the new files when done.
// Note: the file name is appended to "/opt/<Prefix>/<namespace>", eg. "/opt/nfs/e2e-.../<file>".
// Note: nil can be passed for the podSecContext parm, in which case it is ignored.
// Note: `fsizes` values are enforced to each be at least `MinFileSize` and a multiple of `MinFileSize`
//   bytes.
func testVolumeIO(f *framework.Framework, cs clientset.Interface, config framework.VolumeTestConfig, volsrc v1.VolumeSource, podSecContext *v1.PodSecurityContext, file string, fsizes []int64) (err error) {
	ddInput := filepath.Join(mountPath, fmt.Sprintf("%s-%s-dd_if", config.Prefix, config.Namespace))
	writeBlk := strings.Repeat("abcdefghijklmnopqrstuvwxyz123456", 32) // 1KiB value
	loopCnt := testpatterns.MinFileSize / int64(len(writeBlk))
	// initContainer cmd to create and fill dd's input file. The initContainer is used to create
	// the `dd` input file which is currently 1MiB. Rather than store a 1MiB go value, a loop is
	// used to create a 1MiB file in the target directory.
	initCmd := fmt.Sprintf("i=0; while [ $i -lt %d ]; do echo -n %s >>%s; let i+=1; done", loopCnt, writeBlk, ddInput)

	clientPod := makePodSpec(config, initCmd, volsrc, podSecContext)

	By(fmt.Sprintf("starting %s", clientPod.Name))
	podsNamespacer := cs.CoreV1().Pods(config.Namespace)
	clientPod, err = podsNamespacer.Create(clientPod)
	if err != nil {
		return fmt.Errorf("failed to create client pod %q: %v", clientPod.Name, err)
	}
	defer func() {
		deleteFile(clientPod, ddInput)
		By(fmt.Sprintf("deleting client pod %q...", clientPod.Name))
		e := framework.DeletePodWithWait(f, cs, clientPod)
		if e != nil {
			framework.Logf("client pod failed to delete: %v", e)
			if err == nil { // delete err is returned if err is not set
				err = e
			}
		} else {
			framework.Logf("sleeping a bit so kubelet can unmount and detach the volume")
			time.Sleep(framework.PodCleanupTimeout)
		}
	}()

	err = framework.WaitForPodRunningInNamespace(cs, clientPod)
	if err != nil {
		return fmt.Errorf("client pod %q not running: %v", clientPod.Name, err)
	}

	// create files of the passed-in file sizes and verify test file size and content
	for _, fsize := range fsizes {
		// file sizes must be a multiple of `MinFileSize`
		if math.Mod(float64(fsize), float64(testpatterns.MinFileSize)) != 0 {
			fsize = fsize/testpatterns.MinFileSize + testpatterns.MinFileSize
		}
		fpath := filepath.Join(mountPath, fmt.Sprintf("%s-%d", file, fsize))
		defer func() {
			deleteFile(clientPod, fpath)
		}()
		if err = writeToFile(clientPod, fpath, ddInput, fsize); err != nil {
			return err
		}
		if err = verifyFile(clientPod, fpath, fsize, ddInput); err != nil {
			return err
		}
	}

	return
}
