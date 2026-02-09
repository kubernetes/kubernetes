/*
Copyright 2024 The Kubernetes Authors.

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
	"time"

	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

type pvcDeletionPerformanceTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

var _ storageframework.TestSuite = &pvcDeletionPerformanceTestSuite{}

const pvcDeletionTestTimeout = 30 * time.Minute

// InitPvcDeletionPerformanceTestSuite returns pvcDeletionPerformanceTestSuite that implements TestSuite interface
// This test suite brings up a number of pods and PVCS (configured upstream), deletes the pods, and then deletes the PVCs.
// The main goal is to record the duration for the PVC/PV deletion process for each run, and so the test doesn't set explicit expectations to match against.
func InitPvcDeletionPerformanceTestSuite() storageframework.TestSuite {
	return &pvcDeletionPerformanceTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name: "pvc-deletion-performance",
			TestPatterns: []storageframework.TestPattern{
				storageframework.BlockVolModeDynamicPV,
			},
		},
	}
}

func (t *pvcDeletionPerformanceTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return t.tsInfo
}

func (t *pvcDeletionPerformanceTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
}

func (t *pvcDeletionPerformanceTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	type local struct {
		config  *storageframework.PerTestConfig
		cs      clientset.Interface
		ns      *v1.Namespace
		scName  string
		pvcs    []*v1.PersistentVolumeClaim
		options *storageframework.PerformanceTestOptions
		stopCh  chan struct{}
		pods    []*v1.Pod
	}
	var (
		dInfo *storageframework.DriverInfo
		l     *local
	)
	ginkgo.BeforeEach(func() {
		// Check preconditions
		dDriver := driver.(storageframework.DynamicPVTestDriver)
		if dDriver == nil {
			e2eskipper.Skipf("Test driver does not support dynamically created volumes")
		}
		dInfo = dDriver.GetDriverInfo()
		if dInfo == nil {
			e2eskipper.Skipf("Failed to get Driver info -- skipping")
		}
		if dInfo.PerformanceTestOptions == nil || dInfo.PerformanceTestOptions.ProvisioningOptions == nil {
			e2eskipper.Skipf("Driver %s doesn't specify performance test options -- skipping", dInfo.Name)
		}
	})

	// Set high QPS for the framework to avoid client-side throttling from the test itself,
	// which can interfere with measuring deletion time
	frameworkOptions := framework.Options{
		ClientQPS:   500,
		ClientBurst: 1000,
	}
	f := framework.NewFramework("pvc-deletion-performance", frameworkOptions, nil)
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.AfterEach(func(ctx context.Context) {
		if l != nil {
			if l.stopCh != nil {
				ginkgo.By("Closing informer channel")
				close(l.stopCh)
			}
			deletingStats := &performanceStats{
				mutex:             &sync.Mutex{},
				perObjectInterval: make(map[string]*interval),
				operationMetrics:  &storageframework.Metrics{},
			}
			var (
				errs []error
				mu   sync.Mutex
				wg   sync.WaitGroup
			)

			wg.Add(len(l.pods))
			for _, pod := range l.pods {
				go func(pod *v1.Pod) {
					defer ginkgo.GinkgoRecover()
					defer wg.Done()

					framework.Logf("Deleting pod %v", pod.Name)
					err := e2epod.DeletePodWithWait(ctx, l.cs, pod)
					mu.Lock()
					defer mu.Unlock()
					errs = append(errs, err)
				}(pod)
			}
			wg.Wait()

			ginkgo.By("Deleting all PVCs")

			startTime := time.Now()
			wg.Add(len(l.pvcs))
			for _, pvc := range l.pvcs {
				go func(pvc *v1.PersistentVolumeClaim) { // Start a goroutine for each PVC
					defer wg.Done() // Decrement the counter when the goroutine finishes
					startDeletingPvcTime := time.Now()
					framework.Logf("Start deleting PVC %v", pvc.GetName())
					deletingStats.mutex.Lock()
					deletingStats.perObjectInterval[pvc.Name] = &interval{
						create: startDeletingPvcTime,
					}
					deletingStats.mutex.Unlock()
					err := e2epv.DeletePersistentVolumeClaim(ctx, l.cs, pvc.Name, pvc.Namespace)
					framework.ExpectNoError(err)
					startDeletingPVTime := time.Now()
					err = e2epv.WaitForPersistentVolumeDeleted(ctx, l.cs, pvc.Spec.VolumeName, 1*time.Second, 100*time.Minute)
					framework.Logf("Deleted PV %v, PVC %v in %v", pvc.Spec.VolumeName, pvc.GetName(), time.Since(startDeletingPVTime))
					framework.ExpectNoError(err)
				}(pvc)
			}
			wg.Wait()

			endTime := time.Now() // Capture overall end time
			totalDuration := endTime.Sub(startTime)
			framework.Logf("Deleted all PVC/PVs in %v", totalDuration) // Log total deletion time

			ginkgo.By(fmt.Sprintf("Deleting Storage Class %s", l.scName))
			err := l.cs.StorageV1().StorageClasses().Delete(ctx, l.scName, metav1.DeleteOptions{})
			framework.ExpectNoError(err)

		} else {
			ginkgo.By("Local l setup is nil")
		}
	})

	f.It("should delete volumes at scale within performance constraints", f.WithSlow(), f.WithSerial(), func(ctx context.Context) {
		l = &local{
			cs:      f.ClientSet,
			ns:      f.Namespace,
			options: dInfo.PerformanceTestOptions,
		}
		l.config = driver.PrepareTest(ctx, f)

		sc := driver.(storageframework.DynamicPVTestDriver).GetDynamicProvisionStorageClass(ctx, l.config, pattern.FsType)
		ginkgo.By(fmt.Sprintf("Creating Storage Class %v", sc))
		if sc.VolumeBindingMode != nil && *sc.VolumeBindingMode == storagev1.VolumeBindingWaitForFirstConsumer {
			e2eskipper.Skipf("WaitForFirstConsumer binding mode currently not supported for this test")
		}
		ginkgo.By(fmt.Sprintf("Creating Storage Class %s", sc.Name))
		sc, err := l.cs.StorageV1().StorageClasses().Create(ctx, sc, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		l.scName = sc.Name

		// Stats for volume provisioning operation; we only need this because imported function newPVCWatch from volumeperf.go requires this as an argument
		// (this test itself doesn't use these stats)
		provisioningStats := &performanceStats{
			mutex:             &sync.Mutex{},
			perObjectInterval: make(map[string]*interval),
			operationMetrics:  &storageframework.Metrics{},
		}
		// Create a controller to watch on PVCs
		// When all PVCs provisioned by this test are in the Bound state, the controller
		// sends a signal to the channel
		controller := newPVCWatch(ctx, f, l.options.ProvisioningOptions.Count, provisioningStats)
		l.stopCh = make(chan struct{})
		go controller.Run(l.stopCh)
		waitForProvisionCh = make(chan []*v1.PersistentVolumeClaim)

		ginkgo.By(fmt.Sprintf("Creating %d PVCs of size %s", l.options.ProvisioningOptions.Count, l.options.ProvisioningOptions.VolumeSize))
		for i := 0; i < l.options.ProvisioningOptions.Count; i++ {
			pvc := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
				ClaimSize:        l.options.ProvisioningOptions.VolumeSize,
				StorageClassName: &sc.Name,
			}, l.ns.Name)
			pvc, err = l.cs.CoreV1().PersistentVolumeClaims(l.ns.Name).Create(ctx, pvc, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			// Store create time for each PVC
			provisioningStats.mutex.Lock()
			provisioningStats.perObjectInterval[pvc.Name] = &interval{
				create: pvc.CreationTimestamp.Time,
			}
			provisioningStats.mutex.Unlock()
			// Create pods
			podConfig := e2epod.Config{
				NS:           l.ns.Name,
				SeLinuxLabel: e2epv.SELinuxLabel,
			}
			pod, _ := e2epod.MakeSecPod(&podConfig)
			_, err = l.cs.CoreV1().Pods(pod.Namespace).Create(ctx, pod, metav1.CreateOptions{})
			if err != nil {
				framework.Failf("Failed to create pod [%+v]. Error: %v", pod, err)
			}
			framework.ExpectNoError(err)

			l.pods = append(l.pods, pod)
		}

		ginkgo.By("Waiting for all PVCs to be Bound...")

		select {
		case l.pvcs = <-waitForProvisionCh:
			framework.Logf("All PVCs in Bound state")
		case <-time.After(pvcDeletionTestTimeout):
			ginkgo.Fail(fmt.Sprintf("expected all PVCs to be in Bound state within %v", pvcDeletionTestTimeout.Round(time.Second)))
		}
	})

}
