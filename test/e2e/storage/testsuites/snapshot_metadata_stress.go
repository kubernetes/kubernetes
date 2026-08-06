/*
Copyright 2026 The Kubernetes Authors.

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
	"sync"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	errors "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	storageutils "k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

type snapshotMetadataStressTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

type snapshotMetadataStressTest struct {
	config      *storageframework.PerTestConfig
	testOptions storageframework.SnapshotMetadataStressTestOptions

	pods    []*v1.Pod
	volumes []*storageframework.VolumeResource

	// Resources created by concurrent goroutines, protected by mu.
	backupPods  []*v1.Pod
	restoredPVCs []*v1.PersistentVolumeClaim
	snapshots   []*storageframework.SnapshotResource
	mu          sync.Mutex

	wg sync.WaitGroup
}

func InitCustomSnapshotMetadataStressTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &snapshotMetadataStressTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "snapshot-metadata-stress",
			TestPatterns: patterns,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
				Max: "1Gi",
			},
			TestTags: []interface{}{feature.SnapshotMetadata},
		},
	}
}

func InitSnapshotMetadataStressTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.SnapshotMetadata,
	}
	return InitCustomSnapshotMetadataStressTestSuite(patterns)
}

func (s *snapshotMetadataStressTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return s.tsInfo
}

func (s *snapshotMetadataStressTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) string {
	driverInfo := driver.GetDriverInfo()
	if driverInfo.SnapshotMetadataStressTestOptions == nil {
		return fmt.Sprintf("Driver %s doesn't specify snapshot metadata stress test options", driverInfo.Name)
	}
	if driverInfo.SnapshotMetadataStressTestOptions.NumPods <= 0 {
		return fmt.Sprintf("Driver %s has invalid NumPods in snapshot metadata stress test options: %d", driverInfo.Name, driverInfo.SnapshotMetadataStressTestOptions.NumPods)
	}
	if driverInfo.SnapshotMetadataStressTestOptions.NumSnapshotPairs <= 0 {
		return fmt.Sprintf("Driver %s has invalid NumSnapshotPairs in snapshot metadata stress test options: %d", driverInfo.Name, driverInfo.SnapshotMetadataStressTestOptions.NumSnapshotPairs)
	}
	_, ok := driver.(storageframework.SnapshotMetadataTestDriver)
	if !driverInfo.Capabilities[storageframework.CapSnapshotMetadata] || !ok {
		return fmt.Sprintf("Driver %q does not support snapshot metadata", driverInfo.Name)
	}
	if _, ok := driver.(storageframework.SnapshottableTestDriver); !ok {
		return fmt.Sprintf("Driver %q doesn't implement SnapshottableTestDriver", driverInfo.Name)
	}
	return ""
}

// extractDevicePaths finds the device paths for the given PVC names from a pod's spec.
func extractDevicePaths(pod *v1.Pod, sourcePvcName, targetPvcName string) (sourceDevicePath, targetDevicePath string) {
	volumeNameForPVC := make(map[string]string)
	for _, vol := range pod.Spec.Volumes {
		if vol.PersistentVolumeClaim != nil {
			volumeNameForPVC[vol.PersistentVolumeClaim.ClaimName] = vol.Name
		}
	}
	sourceVolName := volumeNameForPVC[sourcePvcName]
	targetVolName := volumeNameForPVC[targetPvcName]
	for _, device := range pod.Spec.Containers[0].VolumeDevices {
		if device.Name == sourceVolName {
			sourceDevicePath = device.DevicePath
		}
		if device.Name == targetVolName {
			targetDevicePath = device.DevicePath
		}
	}
	return sourceDevicePath, targetDevicePath
}

func (s *snapshotMetadataStressTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	var (
		cs         clientset.Interface
		stressTest *snapshotMetadataStressTest
	)

	f := framework.NewFrameworkWithCustomTimeouts("snapshot-metadata-stress", storageframework.GetDriverTimeouts(driver))
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	init := func(ctx context.Context) {
		cs = f.ClientSet
		config := driver.PrepareTest(ctx, f)
		driverInfo := driver.GetDriverInfo()

		ginkgo.By("Creating snapshot metadata resources")
		err := storageutils.CreateSnapshotMetadataResources(ctx, f, driverInfo.Name, config.DriverNamespace.Name)
		framework.ExpectNoError(err, "Failed to create snapshot metadata resources")

		stressTest = &snapshotMetadataStressTest{
			config:       config,
			testOptions:  *driverInfo.SnapshotMetadataStressTestOptions,
			pods:         []*v1.Pod{},
			volumes:      []*storageframework.VolumeResource{},
			backupPods:   []*v1.Pod{},
			restoredPVCs: []*v1.PersistentVolumeClaim{},
			snapshots:    []*storageframework.SnapshotResource{},
		}

		createBackupClientResources(ctx, f)
	}

	createPodsAndVolumes := func(ctx context.Context) {
		for i := 0; i < stressTest.testOptions.NumPods; i++ {
			framework.Logf("Creating resources for pod %d/%d", i, stressTest.testOptions.NumPods-1)

			volume := storageframework.CreateVolumeResource(ctx, driver, stressTest.config, pattern, s.GetTestSuiteInfo().SupportedSizeRange)
			stressTest.volumes = append(stressTest.volumes, volume)

			podConfig := e2epod.Config{
				NS:           f.Namespace.Name,
				PVCs:         []*v1.PersistentVolumeClaim{volume.Pvc},
				SeLinuxLabel: e2epv.SELinuxLabel,
			}
			pod, err := e2epod.MakeSecPod(&podConfig)
			framework.ExpectNoError(err)
			stressTest.pods = append(stressTest.pods, pod)
		}

		var wg sync.WaitGroup
		for i, pod := range stressTest.pods {
			wg.Add(1)
			go func(i int, pod *v1.Pod) {
				defer ginkgo.GinkgoRecover()
				defer wg.Done()
				if _, err := cs.CoreV1().Pods(pod.Namespace).Create(ctx, pod, metav1.CreateOptions{}); err != nil {
					framework.Failf("Failed to create pod-%d [%+v]. Error: %v", i, pod, err)
				}
				if err := e2epod.WaitForPodRunningInNamespace(ctx, cs, pod); err != nil {
					framework.Failf("Failed to wait for pod-%d [%+v] turn into running status. Error: %v", i, pod, err)
				}
				updatedPod, err := cs.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				stressTest.pods[i] = updatedPod
			}(i, pod)
		}
		wg.Wait()
	}

	cleanup := func(ctx context.Context) {
		framework.Logf("Stopping and waiting for all test routines to finish")
		stressTest.wg.Wait()

		var (
			errs []error
			mu   sync.Mutex
			wg   sync.WaitGroup
		)

		// Phase 1: Delete backup client pods
		wg.Add(len(stressTest.backupPods))
		for _, pod := range stressTest.backupPods {
			go func(pod *v1.Pod) {
				defer ginkgo.GinkgoRecover()
				defer wg.Done()
				framework.Logf("Deleting backup pod %s", pod.Name)
				err := e2epod.DeletePodWithWait(ctx, cs, pod)
				mu.Lock()
				defer mu.Unlock()
				errs = append(errs, err)
			}(pod)
		}
		wg.Wait()

		// Phase 2: Delete restored PVCs (must happen before snapshot deletion)
		wg.Add(len(stressTest.restoredPVCs))
		for _, pvc := range stressTest.restoredPVCs {
			go func(pvc *v1.PersistentVolumeClaim) {
				defer ginkgo.GinkgoRecover()
				defer wg.Done()
				framework.Logf("Deleting restored PVC %s", pvc.Name)
				err := e2epv.DeletePersistentVolumeClaim(ctx, cs, pvc.Name, pvc.Namespace)
				mu.Lock()
				defer mu.Unlock()
				errs = append(errs, err)
			}(pvc)
		}
		wg.Wait()

		// Phase 3: Delete snapshots
		wg.Add(len(stressTest.snapshots))
		for _, snapshot := range stressTest.snapshots {
			go func(snapshot *storageframework.SnapshotResource) {
				defer ginkgo.GinkgoRecover()
				defer wg.Done()
				framework.Logf("Deleting snapshot %s/%s", snapshot.Vs.GetNamespace(), snapshot.Vs.GetName())
				err := snapshot.CleanupResource(ctx, f.Timeouts)
				mu.Lock()
				defer mu.Unlock()
				errs = append(errs, err)
			}(snapshot)
		}
		wg.Wait()

		// Phase 4: Delete test pods
		wg.Add(len(stressTest.pods))
		for _, pod := range stressTest.pods {
			go func(pod *v1.Pod) {
				defer ginkgo.GinkgoRecover()
				defer wg.Done()
				framework.Logf("Deleting pod %s", pod.Name)
				err := e2epod.DeletePodWithWait(ctx, cs, pod)
				mu.Lock()
				defer mu.Unlock()
				errs = append(errs, err)
			}(pod)
		}
		wg.Wait()

		// Phase 5: Delete volumes
		wg.Add(len(stressTest.volumes))
		for _, volume := range stressTest.volumes {
			go func(volume *storageframework.VolumeResource) {
				defer ginkgo.GinkgoRecover()
				defer wg.Done()
				framework.Logf("Deleting volume %s", volume.Pvc.GetName())
				err := volume.CleanupResource(ctx)
				mu.Lock()
				defer mu.Unlock()
				errs = append(errs, err)
			}(volume)
		}
		wg.Wait()

		// Phase 6: Cleanup snapshot metadata resources (always run before asserting errors)
		if stressTest.config != nil {
			driverInfo := driver.GetDriverInfo()
			cleanupErr := storageutils.CleanupSnapshotMetadataResources(ctx, f, driverInfo.Name, stressTest.config.DriverNamespace.Name)
			errs = append(errs, cleanupErr)
		}

		framework.ExpectNoError(errors.NewAggregate(errs), "while cleaning up resources")
	}

	f.It("should stress GetMetadataDelta with concurrent pods", f.WithSlow(), f.WithSerial(), func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)
		createPodsAndVolumes(ctx)

		sDriver := driver.(storageframework.SnapshottableTestDriver)

		for i := 0; i < stressTest.testOptions.NumPods; i++ {
			stressTest.wg.Add(1)
			go func(podIndex int) {
				defer ginkgo.GinkgoRecover()
				defer stressTest.wg.Done()

				pod := stressTest.pods[podIndex]
				volume := stressTest.volumes[podIndex]
				testDevicePath := pod.Spec.Containers[0].VolumeDevices[0].DevicePath
				parameters := map[string]string{}

				for j := 0; j < stressTest.testOptions.NumSnapshotPairs; j++ {
					framework.Logf("Pod-%d, SnapshotPair %d/%d: writing initial data", podIndex, j, stressTest.testOptions.NumSnapshotPairs-1)

					writeCmd := fmt.Sprintf("exec %s -c write-pod -- dd if=/dev/urandom of=%s bs=4K count=6 oflag=direct status=none", pod.Name, testDevicePath)
					writeToDeviceInPod(pod, writeCmd)

					framework.Logf("Pod-%d, SnapshotPair %d/%d: taking snap-1", podIndex, j, stressTest.testOptions.NumSnapshotPairs-1)
					snap1 := storageframework.CreateSnapshotResource(ctx, sDriver, stressTest.config, pattern, volume.Pvc.Name, volume.Pvc.Namespace, f.Timeouts, parameters)
					stressTest.mu.Lock()
					stressTest.snapshots = append(stressTest.snapshots, snap1)
					stressTest.mu.Unlock()

					writeCmd = fmt.Sprintf("exec %s -c write-pod -- dd if=/dev/urandom of=%s bs=4K count=3 oflag=direct status=none", pod.Name, testDevicePath)
					writeToDeviceInPod(pod, writeCmd)

					framework.Logf("Pod-%d, SnapshotPair %d/%d: taking snap-2", podIndex, j, stressTest.testOptions.NumSnapshotPairs-1)
					snap2 := storageframework.CreateSnapshotResource(ctx, sDriver, stressTest.config, pattern, volume.Pvc.Name, volume.Pvc.Namespace, f.Timeouts, parameters)
					stressTest.mu.Lock()
					stressTest.snapshots = append(stressTest.snapshots, snap2)
					stressTest.mu.Unlock()

					srcPvcName := fmt.Sprintf("source-device-%d-%d", podIndex, j)
					tgtPvcName := fmt.Sprintf("target-device-%d-%d", podIndex, j)

					srcPvc, err := createPVCFromSnapshot(ctx, cs, f, srcPvcName, snap2.Vs.GetName(), volume.Pvc)
					framework.ExpectNoError(err, "Failed to create source PVC for pod-%d pair-%d", podIndex, j)
					stressTest.mu.Lock()
					stressTest.restoredPVCs = append(stressTest.restoredPVCs, srcPvc)
					stressTest.mu.Unlock()

					tgtPvc, err := createPVCFromSnapshot(ctx, cs, f, tgtPvcName, snap1.Vs.GetName(), volume.Pvc)
					framework.ExpectNoError(err, "Failed to create target PVC for pod-%d pair-%d", podIndex, j)
					stressTest.mu.Lock()
					stressTest.restoredPVCs = append(stressTest.restoredPVCs, tgtPvc)
					stressTest.mu.Unlock()

					framework.Logf("Pod-%d, SnapshotPair %d/%d: waiting for restored PVCs to bind", podIndex, j, stressTest.testOptions.NumSnapshotPairs-1)
					err = e2epv.WaitForPersistentVolumeClaimPhase(ctx, v1.ClaimBound, cs, f.Namespace.Name, srcPvcName, framework.Poll, f.Timeouts.ClaimProvision)
					framework.ExpectNoError(err, "Failed waiting for source PVC to bind for pod-%d pair-%d", podIndex, j)
					err = e2epv.WaitForPersistentVolumeClaimPhase(ctx, v1.ClaimBound, cs, f.Namespace.Name, tgtPvcName, framework.Poll, f.Timeouts.ClaimProvision)
					framework.ExpectNoError(err, "Failed waiting for target PVC to bind for pod-%d pair-%d", podIndex, j)

					backupPod, err := createBackupClientPod(ctx, f, cs, srcPvc, tgtPvc)
					framework.ExpectNoError(err, "Failed to create backup client pod for pod-%d pair-%d", podIndex, j)

					stressTest.mu.Lock()
					stressTest.backupPods = append(stressTest.backupPods, backupPod)
					stressTest.mu.Unlock()

					srcDevicePath, tgtDevicePath := extractDevicePaths(backupPod, srcPvcName, tgtPvcName)
					err = framework.Gomega().Expect(srcDevicePath).NotTo(gomega.BeEmpty())
					framework.ExpectNoError(err, "Failed to get source device path for pod-%d pair-%d", podIndex, j)
					err = framework.Gomega().Expect(tgtDevicePath).NotTo(gomega.BeEmpty())
					framework.ExpectNoError(err, "Failed to get target device path for pod-%d pair-%d", podIndex, j)

					toolCommand := fmt.Sprintf("exec %s -c write-pod -- %s",
						backupPod.Name,
						constructVerifierCommand(f.Namespace.Name, snap2.Vs.GetName(), snap1.Vs.GetName(), srcDevicePath, tgtDevicePath))
					framework.Logf("Pod-%d, SnapshotPair %d/%d: running GetMetadataDelta verifier", podIndex, j, stressTest.testOptions.NumSnapshotPairs-1)
					runSnapshotMetadataVerifier(backupPod, toolCommand)
				}
			}(i)
		}

		stressTest.wg.Wait()
	})

	f.It("should stress GetAllocatedMetadata with concurrent pods", f.WithSlow(), f.WithSerial(), func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)
		createPodsAndVolumes(ctx)

		sDriver := driver.(storageframework.SnapshottableTestDriver)

		for i := 0; i < stressTest.testOptions.NumPods; i++ {
			stressTest.wg.Add(1)
			go func(podIndex int) {
				defer ginkgo.GinkgoRecover()
				defer stressTest.wg.Done()

				pod := stressTest.pods[podIndex]
				volume := stressTest.volumes[podIndex]
				testDevicePath := pod.Spec.Containers[0].VolumeDevices[0].DevicePath
				parameters := map[string]string{}

				framework.Logf("Pod-%d: writing data", podIndex)
				writeCmd := fmt.Sprintf("exec %s -c write-pod -- dd if=/dev/urandom of=%s bs=4K count=6 oflag=direct status=none", pod.Name, testDevicePath)
				writeToDeviceInPod(pod, writeCmd)

				framework.Logf("Pod-%d: taking snapshot", podIndex)
				snap := storageframework.CreateSnapshotResource(ctx, sDriver, stressTest.config, pattern, volume.Pvc.Name, volume.Pvc.Namespace, f.Timeouts, parameters)
				stressTest.mu.Lock()
				stressTest.snapshots = append(stressTest.snapshots, snap)
				stressTest.mu.Unlock()

				srcPvcName := fmt.Sprintf("source-device-%d", podIndex)
				tgtPvcName := fmt.Sprintf("target-device-%d", podIndex)

				srcPvc, err := createPVCFromSnapshot(ctx, cs, f, srcPvcName, snap.Vs.GetName(), volume.Pvc)
				framework.ExpectNoError(err, "Failed to create source PVC for pod-%d", podIndex)
				stressTest.mu.Lock()
				stressTest.restoredPVCs = append(stressTest.restoredPVCs, srcPvc)
				stressTest.mu.Unlock()

				tgtPvcClaim := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
					Name:             tgtPvcName,
					ClaimSize:        volume.Pvc.Spec.Resources.Requests.Storage().String(),
					StorageClassName: volume.Pvc.Spec.StorageClassName,
					VolumeMode:       volume.Pvc.Spec.VolumeMode,
				}, f.Namespace.Name)
				tgtPvc, err := e2epv.CreatePVC(ctx, cs, f.Namespace.Name, tgtPvcClaim)
				framework.ExpectNoError(err, "Failed to create target PVC for pod-%d", podIndex)
				stressTest.mu.Lock()
				stressTest.restoredPVCs = append(stressTest.restoredPVCs, tgtPvc)
				stressTest.mu.Unlock()

				framework.Logf("Pod-%d: waiting for restored PVCs to bind", podIndex)
				err = e2epv.WaitForPersistentVolumeClaimPhase(ctx, v1.ClaimBound, cs, f.Namespace.Name, srcPvcName, framework.Poll, f.Timeouts.ClaimProvision)
				framework.ExpectNoError(err, "Failed waiting for source PVC to bind for pod-%d", podIndex)
				err = e2epv.WaitForPersistentVolumeClaimPhase(ctx, v1.ClaimBound, cs, f.Namespace.Name, tgtPvcName, framework.Poll, f.Timeouts.ClaimProvision)
				framework.ExpectNoError(err, "Failed waiting for target PVC to bind for pod-%d", podIndex)

				backupPod, err := createBackupClientPod(ctx, f, cs, srcPvc, tgtPvc)
				framework.ExpectNoError(err, "Failed to create backup client pod for pod-%d", podIndex)

				stressTest.mu.Lock()
				stressTest.backupPods = append(stressTest.backupPods, backupPod)
				stressTest.mu.Unlock()

				srcDevicePath, tgtDevicePath := extractDevicePaths(backupPod, srcPvcName, tgtPvcName)
				err = framework.Gomega().Expect(srcDevicePath).NotTo(gomega.BeEmpty())
				framework.ExpectNoError(err, "Failed to get source device path for pod-%d", podIndex)
				err = framework.Gomega().Expect(tgtDevicePath).NotTo(gomega.BeEmpty())
				framework.ExpectNoError(err, "Failed to get target device path for pod-%d", podIndex)

				toolCommand := fmt.Sprintf("exec %s -c write-pod -- %s",
					backupPod.Name,
					constructVerifierCommand(f.Namespace.Name, snap.Vs.GetName(), "", srcDevicePath, tgtDevicePath))
				framework.Logf("Pod-%d: running GetAllocatedMetadata verifier", podIndex)
				runSnapshotMetadataVerifier(backupPod, toolCommand)
			}(i)
		}

		stressTest.wg.Wait()
	})
}
