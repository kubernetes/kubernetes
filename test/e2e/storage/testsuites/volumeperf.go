/*
Copyright 2021 The Kubernetes Authors.

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

	"github.com/davecgh/go-spew/spew"

	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

type volumePerformanceTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

var _ storageframework.TestSuite = &volumePerformanceTestSuite{}

const testTimeout = 15 * time.Minute

// Time intervals when a resource was created, it enters
// the desired state and the elapsed time between these
// two states
type interval struct {
	create            time.Time
	enterDesiredState time.Time
	elapsed           time.Duration
}

// Consolidates performance stats for any operation
type performanceStats struct {
	mutex             *sync.Mutex
	perObjectInterval map[string]*interval
	operationMetrics  *storageframework.Metrics
}

// waitForProvisionCh receives a signal from controller
// when all PVCs are Bound
// The signal received on this channel is the list of
// PVC objects that are created in the test
// The test blocks until waitForProvisionCh receives a signal
// or the test times out
var waitForProvisionCh chan []*v1.PersistentVolumeClaim

// InitVolumePerformanceTestSuite returns volumePerformanceTestSuite that implements TestSuite interface
func InitVolumePerformanceTestSuite() storageframework.TestSuite {
	return &volumePerformanceTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name: "volume-lifecycle-performance",
			TestPatterns: []storageframework.TestPattern{
				storageframework.FsVolModeDynamicPV,
			},
		},
	}
}

func (t *volumePerformanceTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return t.tsInfo
}

func (t *volumePerformanceTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
}

func (t *volumePerformanceTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	type local struct {
		config      *storageframework.PerTestConfig
		testCleanup func()
		cs          clientset.Interface
		ns          *v1.Namespace
		scName      string
		pvcs        []*v1.PersistentVolumeClaim
		options     *storageframework.PerformanceTestOptions
		stopCh      chan struct{}
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

	frameworkOptions := framework.Options{
		ClientQPS:   200,
		ClientBurst: 400,
	}
	f := framework.NewFramework("volume-lifecycle-performance", frameworkOptions, nil)
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	f.AddAfterEach("cleanup", func(f *framework.Framework, failed bool) {
		ginkgo.By("Closing informer channel")
		close(l.stopCh)
		ginkgo.By("Deleting all PVCs")
		for _, pvc := range l.pvcs {
			err := e2epv.DeletePersistentVolumeClaim(l.cs, pvc.Name, pvc.Namespace)
			framework.ExpectNoError(err)
			err = e2epv.WaitForPersistentVolumeDeleted(l.cs, pvc.Spec.VolumeName, 1*time.Second, 5*time.Minute)
			framework.ExpectNoError(err)
		}
		ginkgo.By(fmt.Sprintf("Deleting Storage Class %s", l.scName))
		err := l.cs.StorageV1().StorageClasses().Delete(context.TODO(), l.scName, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		l.testCleanup()
	})

	ginkgo.It("should provision volumes at scale within performance constraints [Slow] [Serial]", func() {
		l = &local{
			cs:      f.ClientSet,
			ns:      f.Namespace,
			options: dInfo.PerformanceTestOptions,
		}
		l.config, l.testCleanup = driver.PrepareTest(f)

		// Stats for volume provisioning operation
		// TODO: Add stats for attach, resize and snapshot
		provisioningStats := &performanceStats{
			mutex:             &sync.Mutex{},
			perObjectInterval: make(map[string]*interval),
			operationMetrics:  &storageframework.Metrics{},
		}
		sc := driver.(storageframework.DynamicPVTestDriver).GetDynamicProvisionStorageClass(l.config, pattern.FsType)
		ginkgo.By(fmt.Sprintf("Creating Storage Class %v", sc))
		// TODO: Add support for WaitForFirstConsumer volume binding mode
		if sc.VolumeBindingMode != nil && *sc.VolumeBindingMode == storagev1.VolumeBindingWaitForFirstConsumer {
			e2eskipper.Skipf("WaitForFirstConsumer binding mode currently not supported for performance tests")
		}
		ginkgo.By(fmt.Sprintf("Creating Storage Class %s", sc.Name))
		sc, err := l.cs.StorageV1().StorageClasses().Create(context.TODO(), sc, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		l.scName = sc.Name

		// Create a controller to watch on PVCs
		// When all PVCs provisioned by this test are in the Bound state, the controller
		// sends a signal to the channel
		controller := newPVCWatch(f, l.options.ProvisioningOptions.Count, provisioningStats)
		l.stopCh = make(chan struct{})
		go controller.Run(l.stopCh)
		waitForProvisionCh = make(chan []*v1.PersistentVolumeClaim)

		ginkgo.By(fmt.Sprintf("Creating %d PVCs of size %s", l.options.ProvisioningOptions.Count, l.options.ProvisioningOptions.VolumeSize))
		for i := 0; i < l.options.ProvisioningOptions.Count; i++ {
			pvc := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
				ClaimSize:        l.options.ProvisioningOptions.VolumeSize,
				StorageClassName: &sc.Name,
			}, l.ns.Name)
			pvc, err = l.cs.CoreV1().PersistentVolumeClaims(l.ns.Name).Create(context.TODO(), pvc, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			// Store create time for each PVC
			provisioningStats.mutex.Lock()
			provisioningStats.perObjectInterval[pvc.Name] = &interval{
				create: pvc.CreationTimestamp.Time,
			}
			provisioningStats.mutex.Unlock()
		}

		ginkgo.By("Waiting for all PVCs to be Bound...")

		select {
		case l.pvcs = <-waitForProvisionCh:
			framework.Logf("All PVCs in Bound state")
		case <-time.After(testTimeout):
			ginkgo.Fail(fmt.Sprintf("expected all PVCs to be in Bound state within %v minutes", testTimeout))
		}

		ginkgo.By("Calculating performance metrics for provisioning operations")
		createPerformanceStats(provisioningStats, l.options.ProvisioningOptions.Count, l.pvcs)

		ginkgo.By(fmt.Sprintf("Validating performance metrics for provisioning operations against baseline %v", spew.Sdump(l.options.ProvisioningOptions.ExpectedMetrics)))
		errList := validatePerformanceStats(provisioningStats.operationMetrics, l.options.ProvisioningOptions.ExpectedMetrics)
		framework.ExpectNoError(errors.NewAggregate(errList), "while validating performance metrics")
	})

}

// createPerformanceStats calculates individual metrics for an operation
// given the intervals collected during that operation
func createPerformanceStats(stats *performanceStats, provisionCount int, pvcs []*v1.PersistentVolumeClaim) {
	var min, max, sum time.Duration
	for _, pvc := range pvcs {
		pvcMetric, ok := stats.perObjectInterval[pvc.Name]
		if !ok {
			framework.Failf("PVC %s not found in perObjectInterval", pvc.Name)
		}

		elapsedTime := pvcMetric.elapsed
		sum += elapsedTime
		if elapsedTime < min || min == 0 {
			min = elapsedTime
		}
		if elapsedTime > max {
			max = elapsedTime
		}
	}
	stats.operationMetrics = &storageframework.Metrics{
		AvgLatency: time.Duration(int64(sum) / int64(provisionCount)),
		Throughput: float64(provisionCount) / max.Seconds(),
	}
}

// validatePerformanceStats validates if test performance metrics meet the baseline target
func validatePerformanceStats(operationMetrics *storageframework.Metrics, baselineMetrics *storageframework.Metrics) []error {
	var errList []error
	framework.Logf("Metrics to evaluate: %+v", spew.Sdump(operationMetrics))

	if operationMetrics.AvgLatency > baselineMetrics.AvgLatency {
		err := fmt.Errorf("expected latency to be less than %v but calculated latency %v", baselineMetrics.AvgLatency, operationMetrics.AvgLatency)
		errList = append(errList, err)
	}
	if operationMetrics.Throughput < baselineMetrics.Throughput {
		err := fmt.Errorf("expected throughput to be greater than %f but calculated throughput %f", baselineMetrics.Throughput, operationMetrics.Throughput)
		errList = append(errList, err)
	}
	return errList
}

// newPVCWatch creates an informer to check whether all PVCs are Bound
// When all PVCs are bound, the controller sends a signal to
// waitForProvisionCh to unblock the test
func newPVCWatch(f *framework.Framework, provisionCount int, pvcMetrics *performanceStats) cache.Controller {
	defer ginkgo.GinkgoRecover()
	count := 0
	countLock := &sync.Mutex{}
	ns := f.Namespace.Name
	var pvcs []*v1.PersistentVolumeClaim
	checkPVCBound := func(oldPVC *v1.PersistentVolumeClaim, newPVC *v1.PersistentVolumeClaim) {
		now := time.Now()
		pvcMetrics.mutex.Lock()
		defer pvcMetrics.mutex.Unlock()
		countLock.Lock()
		defer countLock.Unlock()

		// Check if PVC entered the bound state
		if oldPVC.Status.Phase != v1.ClaimBound && newPVC.Status.Phase == v1.ClaimBound {
			newPVCInterval, ok := pvcMetrics.perObjectInterval[newPVC.Name]
			if !ok {
				framework.Failf("PVC %s should exist in interval map already", newPVC.Name)
			}
			count++
			newPVCInterval.enterDesiredState = now
			newPVCInterval.elapsed = now.Sub(newPVCInterval.create)
			pvcs = append(pvcs, newPVC)
		}
		if count == provisionCount {
			// Number of Bound PVCs equals the number of PVCs
			// provisioned by this test
			// Send those PVCs to the channel to unblock test
			waitForProvisionCh <- pvcs
		}
	}
	_, controller := cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				obj, err := f.ClientSet.CoreV1().PersistentVolumeClaims(ns).List(context.TODO(), metav1.ListOptions{})
				return runtime.Object(obj), err
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return f.ClientSet.CoreV1().PersistentVolumeClaims(ns).Watch(context.TODO(), metav1.ListOptions{})
			},
		},
		&v1.PersistentVolumeClaim{},
		0,
		cache.ResourceEventHandlerFuncs{
			UpdateFunc: func(oldObj, newObj interface{}) {
				oldPVC, ok := oldObj.(*v1.PersistentVolumeClaim)
				if !ok {
					framework.Failf("Expected a PVC, got instead an old object of type %T", oldObj)
				}
				newPVC, ok := newObj.(*v1.PersistentVolumeClaim)
				if !ok {
					framework.Failf("Expected a PVC, got instead a new object of type %T", newObj)
				}

				checkPVCBound(oldPVC, newPVC)
			},
		},
	)
	return controller
}
