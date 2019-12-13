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
	"math"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo"

	"k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	"k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

// snapshot CRD api group
const snapshotGroup = "snapshot.storage.k8s.io"

// snapshot CRD api version
const snapshotAPIVersion = "snapshot.storage.k8s.io/v1beta1"

var (
	snapshotGVR        = schema.GroupVersionResource{Group: snapshotGroup, Version: "v1beta1", Resource: "volumesnapshots"}
	snapshotClassGVR   = schema.GroupVersionResource{Group: snapshotGroup, Version: "v1beta1", Resource: "volumesnapshotclasses"}
	snapshotContentGVR = schema.GroupVersionResource{Group: snapshotGroup, Version: "v1beta1", Resource: "volumesnapshotcontents"}
)

type snapshottableTestSuite struct {
	tsInfo TestSuiteInfo
}

var _ TestSuite = &snapshottableTestSuite{}

// InitSnapshottableTestSuite returns snapshottableTestSuite that implements TestSuite interface
func InitSnapshottableTestSuite() TestSuite {
	return &snapshottableTestSuite{
		tsInfo: TestSuiteInfo{
			Name: "snapshottable",
			TestPatterns: []testpatterns.TestPattern{
				testpatterns.DynamicSnapshot,
			},
			SupportedSizeRange: volume.SizeRange{
				Min: "1Mi",
			},
		},
	}
}

func (s *snapshottableTestSuite) GetTestSuiteInfo() TestSuiteInfo {
	return s.tsInfo
}

func (s *snapshottableTestSuite) SkipRedundantSuite(driver TestDriver, pattern testpatterns.TestPattern) {
}

func (s *snapshottableTestSuite) DefineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	var (
		sDriver SnapshottableTestDriver
		dDriver DynamicPVTestDriver
	)

	ginkgo.BeforeEach(func() {
		// Check preconditions.
		framework.ExpectEqual(pattern.SnapshotType, testpatterns.DynamicCreatedSnapshot)
		dInfo := driver.GetDriverInfo()
		ok := false
		sDriver, ok = driver.(SnapshottableTestDriver)
		if !dInfo.Capabilities[CapSnapshotDataSource] || !ok {
			framework.Skipf("Driver %q does not support snapshots - skipping", dInfo.Name)
		}
		dDriver, ok = driver.(DynamicPVTestDriver)
		if !ok {
			framework.Skipf("Driver %q does not support dynamic provisioning - skipping", driver.GetDriverInfo().Name)
		}
	})

	// This intentionally comes after checking the preconditions because it
	// registers its own BeforeEach which creates the namespace. Beware that it
	// also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewDefaultFramework("snapshotting")

	ginkgo.It("should create snapshot with defaults [Feature:VolumeSnapshotDataSource]", func() {
		testFile := fmt.Sprintf("%s_io_test_%s", driver.GetDriverInfo().Name, f.Namespace.Name)
		var fsGroup *int64
		if !framework.NodeOSDistroIs("windows") && driver.GetDriverInfo().Capabilities[CapFsGroup] {
			fsGroupVal := int64(1234)
			fsGroup = &fsGroupVal
		}
		podSec := v1.PodSecurityContext{
			FSGroup: fsGroup,
		}

		cs := f.ClientSet
		dc := f.DynamicClient

		// Now do the more expensive test initialization.
		config, driverCleanup := driver.PrepareTest(f)
		defer func() {
			err := tryFunc(driverCleanup)
			framework.ExpectNoError(err, "while cleaning up driver")
		}()

		vsc := sDriver.GetSnapshotClass(config)
		class := dDriver.GetDynamicProvisionStorageClass(config, "")
		if class == nil {
			framework.Skipf("Driver %q does not define Dynamic Provision StorageClass - skipping", driver.GetDriverInfo().Name)
		}
		testVolumeSizeRange := s.GetTestSuiteInfo().SupportedSizeRange
		driverVolumeSizeRange := dDriver.GetDriverInfo().SupportedSizeRange
		claimSize, err := getSizeRangesIntersection(testVolumeSizeRange, driverVolumeSizeRange)
		framework.ExpectNoError(err, "determine intersection of test size range %+v and driver size range %+v", testVolumeSizeRange, driverVolumeSizeRange)
		pvc := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			ClaimSize:        claimSize,
			StorageClassName: &(class.Name),
		}, config.Framework.Namespace.Name)

		framework.Logf("In creating storage class object and pvc object for driver - sc: %v, pvc: %v", class, pvc)

		ginkgo.By("creating a StorageClass " + class.Name)
		class, err = cs.StorageV1().StorageClasses().Create(class)
		framework.ExpectNoError(err)
		defer func() {
			framework.Logf("deleting storage class %s", class.Name)
			framework.ExpectNoError(cs.StorageV1().StorageClasses().Delete(class.Name, nil))
		}()

		ginkgo.By("creating a claim")
		pvc, err = cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		framework.ExpectNoError(err)
		defer func() {
			framework.Logf("deleting claim %q/%q", pvc.Namespace, pvc.Name)
			// typically this claim has already been deleted
			err = cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Delete(pvc.Name, nil)
			if err != nil && !apierrs.IsNotFound(err) {
				framework.Failf("Error deleting claim %q. Error: %v", pvc.Name, err)
			}
		}()

		ginkgo.By("starting a pod to use the claim")
		command := "echo 'hello world' > /mnt/test/data"
		pod := StartInPodWithVolume(cs, pvc.Namespace, pvc.Name, "pvc-snapshottable-tester", command, e2epod.NodeSelection{Name: config.ClientNodeName})
		defer StopPod(cs, pod)

		err = e2epv.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, cs, pvc.Namespace, pvc.Name, framework.Poll, framework.ClaimProvisionTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("checking the claim")
		// Get new copy of the claim
		pvc, err = cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pvc.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		// Get the bound PV
		_, err = cs.CoreV1().PersistentVolumes().Get(pvc.Spec.VolumeName, metav1.GetOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("create pod and write data to claim")
		volSource := v1.VolumeSource{
			PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
				ClaimName: pvc.Name,
				ReadOnly:  false,
			},
		}
		oldHash, err := CreatePodAndWriteDataToVolume(f, cs, convertTestConfig(config), volSource, &podSec, testFile, testpatterns.FileSizeSmall, true)
		framework.ExpectNoError(err)

		ginkgo.By("creating a SnapshotClass")
		vsc, err = dc.Resource(snapshotClassGVR).Create(vsc, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		defer func() {
			framework.Logf("deleting SnapshotClass %s", vsc.GetName())
			framework.ExpectNoError(dc.Resource(snapshotClassGVR).Delete(vsc.GetName(), nil))
		}()

		ginkgo.By("creating a snapshot")
		snapshot := getSnapshot(pvc.Name, pvc.Namespace, vsc.GetName())

		snapshot, err = dc.Resource(snapshotGVR).Namespace(snapshot.GetNamespace()).Create(snapshot, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		defer func() {
			framework.Logf("deleting snapshot %q/%q", snapshot.GetNamespace(), snapshot.GetName())
			// typically this snapshot has already been deleted
			err = dc.Resource(snapshotGVR).Namespace(snapshot.GetNamespace()).Delete(snapshot.GetName(), nil)
			if err != nil && !apierrs.IsNotFound(err) {
				framework.Failf("Error deleting snapshot %q. Error: %v", pvc.Name, err)
			}
		}()
		err = WaitForSnapshotReady(dc, snapshot.GetNamespace(), snapshot.GetName(), framework.Poll, framework.SnapshotCreateTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("checking the snapshot")
		// Get new copy of the snapshot
		snapshot, err = dc.Resource(snapshotGVR).Namespace(snapshot.GetNamespace()).Get(snapshot.GetName(), metav1.GetOptions{})
		framework.ExpectNoError(err)

		// Get the bound snapshotContent
		snapshotStatus := snapshot.Object["status"].(map[string]interface{})
		snapshotContentName := snapshotStatus["boundVolumeSnapshotContentName"].(string)
		snapshotContent, err := dc.Resource(snapshotContentGVR).Get(snapshotContentName, metav1.GetOptions{})
		framework.ExpectNoError(err)

		snapshotContentSpec := snapshotContent.Object["spec"].(map[string]interface{})
		volumeSnapshotRef := snapshotContentSpec["volumeSnapshotRef"].(map[string]interface{})

		// Create an new pod and write more data to old claim
		ginkgo.By("create an new pod and write more data to old claim")
		_, err = CreatePodAndWriteDataToVolume(f, cs, convertTestConfig(config), volSource, &podSec, testFile, testpatterns.FileSizeLarge, true)
		framework.ExpectNoError(err)

		// Check SnapshotContent properties
		ginkgo.By("checking the SnapshotContent")
		framework.ExpectEqual(snapshotContentSpec["volumeSnapshotClassName"], vsc.GetName())
		framework.ExpectEqual(volumeSnapshotRef["name"], snapshot.GetName())
		framework.ExpectEqual(volumeSnapshotRef["namespace"], snapshot.GetNamespace())

		ginkgo.By("creating a new claim from snapshot")
		nPvc := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			ClaimSize:        claimSize,
			StorageClassName: &(class.Name),
		}, config.Framework.Namespace.Name)

		dataSource := &v1.TypedLocalObjectReference{}
		dataSource.Name = snapshot.GetName()
		apiGroup := snapshotGroup
		dataSource.APIGroup = &apiGroup
		dataSource.Kind = "VolumeSnapshot"
		nPvc.Spec.DataSource = dataSource
		ginkgo.By(fmt.Sprintf("nPvc:%v", nPvc))
		nPvc, err = cs.CoreV1().PersistentVolumeClaims(nPvc.Namespace).Create(nPvc)
		framework.ExpectNoError(err)
		defer func() {
			e2elog.Logf("deleting claim %q/%q", nPvc.Namespace, nPvc.Name)
			// typically this claim has already been deleted
			err = cs.CoreV1().PersistentVolumeClaims(nPvc.Namespace).Delete(nPvc.Name, nil)
			if err != nil && !apierrs.IsNotFound(err) {
				framework.Failf("Error deleting claim %q. Error: %v", nPvc.Name, err)
			}
		}()
		err = e2epv.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, cs, nPvc.Namespace, nPvc.Name, framework.Poll, framework.ClaimProvisionTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("checking the claim")
		// Get new copy of the claim
		nPvc, err = cs.CoreV1().PersistentVolumeClaims(nPvc.Namespace).Get(nPvc.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		// Get the bound PV
		_, err = cs.CoreV1().PersistentVolumes().Get(nPvc.Spec.VolumeName, metav1.GetOptions{})
		framework.ExpectNoError(err)

		// Create a new pod with the new pvc that restore from snapshot
		ginkgo.By("Create a new pod with the new pvc that restore from snapshot but not write data")
		newVolSource := v1.VolumeSource{
			PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
				ClaimName: nPvc.Name,
				ReadOnly:  false,
			},
		}
		newHash, err := CreatePodAndWriteDataToVolume(f, cs, convertTestConfig(config), newVolSource, &podSec, testFile, testpatterns.FileSizeSmall, false)
		framework.ExpectNoError(err)

		// Validate data
		if oldHash != newHash {
			err = fmt.Errorf("data inconsis")
			framework.ExpectNoError(err)
		}
	})
}

// WaitForSnapshotReady waits for a VolumeSnapshot to be ready to use or until timeout occurs, whichever comes first.
func WaitForSnapshotReady(c dynamic.Interface, ns string, snapshotName string, Poll, timeout time.Duration) error {
	framework.Logf("Waiting up to %v for VolumeSnapshot %s to become ready", timeout, snapshotName)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		snapshot, err := c.Resource(snapshotGVR).Namespace(ns).Get(snapshotName, metav1.GetOptions{})
		if err != nil {
			framework.Logf("Failed to get claim %q, retrying in %v. Error: %v", snapshotName, Poll, err)
			continue
		} else {
			status := snapshot.Object["status"]
			if status == nil {
				framework.Logf("VolumeSnapshot %s found but is not ready.", snapshotName)
				continue
			}
			value := status.(map[string]interface{})
			if value["readyToUse"] == true {
				framework.Logf("VolumeSnapshot %s found and is ready after %v", snapshotName, time.Since(start))
				return nil
			} else if value["ready"] == true {
				framework.Logf("VolumeSnapshot %s found and is ready after %v", snapshotName, time.Since(start))
				return nil
			} else {
				framework.Logf("VolumeSnapshot %s found but is not ready.", snapshotName)
			}
		}
	}
	return fmt.Errorf("VolumeSnapshot %s is not ready within %v", snapshotName, timeout)
}

// Write `fsize` bytes to `fpath` in the pod, using dd and the `ddInput` file.
func writeFileToPod(f *framework.Framework, pod *v1.Pod, fpath, ddInput string, fsize int64) error {
	ginkgo.By(fmt.Sprintf("writing %d bytes to test file %s", fsize, fpath))
	loopCnt := fsize / testpatterns.MinFileSize
	rtnstr, err := utils.PodExec(f, pod, fmt.Sprintf("stat -c %%s %s", fpath))
	if err == nil && rtnstr != "" {
		_, err := utils.PodExec(f, pod, fmt.Sprintf("> %s", fpath))
		if err != nil {
			return err
		}
	}
	writeCmd := fmt.Sprintf("i=0; while [ $i -lt %d ]; do dd if=%s bs=%d count=1 >>%s 2>/dev/null; let i+=1; done", loopCnt, ddInput, testpatterns.MinFileSize, fpath)
	_, err = utils.PodExec(f, pod, writeCmd)
	return err
}

// Get the test file actual Hash
func getFileHash(f *framework.Framework, pod *v1.Pod, fpath string) (string, error) {

	ginkgo.By("Get file Hash")
	rtnstr, err := utils.PodExec(f, pod, fmt.Sprintf("md5sum %s | cut -d' ' -f1", fpath))
	if err != nil {
		return "", fmt.Errorf("unable to test file hash via `md5sum %s`: %v", fpath, err)
	}
	actualHash := strings.TrimSuffix(rtnstr, "\n")
	return actualHash, nil
}

// Verify that the test file is the expected size and contains the expected content.
func verifyFileAndGetActualHash(f *framework.Framework, pod *v1.Pod, fpath string, expectSize int64) (string, error) {
	ginkgo.By("verifying file size")
	rtnstr, err := utils.PodExec(f, pod, fmt.Sprintf("stat -c %%s %s", fpath))
	if err != nil || rtnstr == "" {
		return "", fmt.Errorf("unable to get file size via `stat %s`: %v", fpath, err)
	}
	size, err := strconv.Atoi(strings.TrimSuffix(rtnstr, "\n"))
	if err != nil {
		return "", fmt.Errorf("unable to convert string %q to int: %v", rtnstr, err)
	}
	if int64(size) != expectSize {
		return "", fmt.Errorf("size of file %s is %d, expected %d", fpath, size, expectSize)
	}

	ginkgo.By("verifying file hash")
	rtnstr, err = utils.PodExec(f, pod, fmt.Sprintf("md5sum %s | cut -d' ' -f1", fpath))
	if err != nil {
		return "", fmt.Errorf("unable to test file hash via `md5sum %s`: %v", fpath, err)
	}
	actualHash := strings.TrimSuffix(rtnstr, "\n")
	expectedHash, ok := md5hashes[expectSize]
	if !ok {
		return "", fmt.Errorf("file hash is unknown for file size %d", expectSize)
	}
	if actualHash != expectedHash {
		return "", fmt.Errorf("MD5 hash is incorrect for file %s with size %d. Expected: `%s`; Actual: `%s`",
			fpath, expectSize, expectedHash, actualHash)
	}

	return actualHash, nil
}

// CreatePodAndWriteDataToVolume create an pod and write data to volume for create snapshot
func CreatePodAndWriteDataToVolume(f *framework.Framework, cs clientset.Interface, config volume.TestConfig, volsrc v1.VolumeSource, podSecContext *v1.PodSecurityContext, file string, fsize int64, write bool) (actualHash string, err error) {

	ddInput := filepath.Join(mountPath, fmt.Sprintf("%s-%s-dd_if", config.Prefix, config.Namespace))
	writeBlk := strings.Repeat("abcdefghijklmnopqrstuvwxyz123456", 32) // 1KiB value
	loopCnt := testpatterns.MinFileSize / int64(len(writeBlk))
	// initContainer cmd to create and fill dd's input file. The initContainer is used to create
	// the `dd` input file which is currently 1MiB. Rather than store a 1MiB go value, a loop is
	// used to create a 1MiB file in the target directory.
	initCmd := fmt.Sprintf("i=0; while [ $i -lt %d ]; do echo -n %s >>%s; let i+=1; done", loopCnt, writeBlk, ddInput)

	clientPod := makePodSpec(config, initCmd, volsrc, podSecContext)

	ginkgo.By(fmt.Sprintf("starting %s", clientPod.Name))
	podsNamespacer := cs.CoreV1().Pods(config.Namespace)
	clientPod, err = podsNamespacer.Create(clientPod)
	if err != nil {
		return "", fmt.Errorf("failed to create client pod %q: %v", clientPod.Name, err)
	}

	defer func() {

		ginkgo.By(fmt.Sprintf("deleting client pod %q...", clientPod.Name))
		e := e2epod.DeletePodWithWait(cs, clientPod)
		if e != nil {
			e2elog.Logf("client pod failed to delete: %v", e)
			if err == nil { // delete err is returned if err is not set
				err = e
			}
		} else {
			e2elog.Logf("sleeping a bit so kubelet can unmount and detach the volume")
			time.Sleep(volume.PodCleanupTimeout)
		}
	}()

	err = e2epod.WaitForPodRunningInNamespace(cs, clientPod)
	if err != nil {
		return "", fmt.Errorf("client pod %q not running: %v", clientPod.Name, err)
	}

	fpath := filepath.Join(mountPath, fmt.Sprintf("%s", file))
	if write {
		if math.Mod(float64(fsize), float64(testpatterns.MinFileSize)) != 0 {
			fsize = fsize/testpatterns.MinFileSize + testpatterns.MinFileSize
		}

		if err = writeFileToPod(f, clientPod, fpath, ddInput, fsize); err != nil {
			return actualHash, err
		}

		actualHash, err = verifyFileAndGetActualHash(f, clientPod, fpath, fsize)
		if err != nil {
			return actualHash, err
		}
	} else {
		actualHash, err = getFileHash(f, clientPod, fpath)
		if err != nil {
			return actualHash, err
		}
	}
	return actualHash, nil
}
