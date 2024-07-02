/*
Copyright 2020 The Kubernetes Authors.

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

// This suite tests volumes under stress conditions

package testsuites

import (
	"context"
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	errors "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

type pvcProtectionStressTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

type pvcProtectionStressTest struct {
	config *storageframework.PerTestConfig

	migrationCheck *migrationOpCheck

	volumes []*storageframework.VolumeResource
	pods    []*v1.Pod
	// stop and wait for any async routines
	wg          sync.WaitGroup
	volumesMu   sync.Mutex
	testOptions storageframework.StressTestOptions
}

var _ storageframework.TestSuite = &pvcProtectionStressTestSuite{}

// InitCustompvcProtectionStressTestSuite returns pvcProtectionStressTestSuite that implements TestSuite interface
// using custom test patterns
func InitCustomPvcProtectionStressTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &pvcProtectionStressTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "pvc-protection-stress",
			TestPatterns: patterns,
		},
	}
}

// InitPvcProtectionStressTestSuite returns pvcProtectionStressTestSuite that implements TestSuite interface
// using testsuite default patterns
func InitPvcProtectionStressTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.BlockVolModeDynamicPV,
	}
	return InitCustomPvcProtectionStressTestSuite(patterns)
}

func (t *pvcProtectionStressTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	dInfo := driver.GetDriverInfo()
	if dInfo.StressTestOptions == nil {
		e2eskipper.Skipf("Driver %s doesn't specify stress test options -- skipping", dInfo.Name)
	}
	if dInfo.StressTestOptions.NumPods <= 0 {
		framework.Failf("NumPods in stress test options must be a positive integer, received: %d", dInfo.StressTestOptions.NumPods)
	}

	if _, ok := driver.(storageframework.DynamicPVTestDriver); !ok {
		e2eskipper.Skipf("Driver %s doesn't implement DynamicPVTestDriver -- skipping", dInfo.Name)
	}
	if !driver.GetDriverInfo().Capabilities[storageframework.CapBlock] && pattern.VolMode == v1.PersistentVolumeBlock {
		e2eskipper.Skipf("Driver %q does not support block volume mode - skipping", dInfo.Name)
	}
}

func (t *pvcProtectionStressTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return t.tsInfo
}

func (t *pvcProtectionStressTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	var (
		dInfo = driver.GetDriverInfo()
		cs    clientset.Interface
		l     *pvcProtectionStressTest
	)

	// Beware that it also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewFrameworkWithCustomTimeouts("stress", storageframework.GetDriverTimeouts(driver))
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	init := func(ctx context.Context) {
		cs = f.ClientSet
		l = &pvcProtectionStressTest{}

		// Now do the more expensive test initialization.
		l.config = driver.PrepareTest(ctx, f)
		l.migrationCheck = newMigrationOpCheck(ctx, f.ClientSet, f.ClientConfig(), dInfo.InTreePluginName)
		l.volumes = []*storageframework.VolumeResource{}
		l.pods = []*v1.Pod{}
		l.testOptions = *dInfo.StressTestOptions

		//TEST CODE ONLY - REMOVE LATER
		l.testOptions.NumPods = 1000
		l.testOptions.NumRestarts = 0
	}

	createPodsAndVolumes := func(ctx context.Context) {

		// Create storage class
		dDriver, _ := driver.(storageframework.DynamicPVTestDriver)
		sc := dDriver.GetDynamicProvisionStorageClass(ctx, l.config, pattern.FsType)
		sc.AllowVolumeExpansion = &pattern.AllowExpansion

		ginkgo.By("creating a StorageClass " + sc.Name)
		var err error
		sc, err = cs.StorageV1().StorageClasses().Create(ctx, sc, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		// l.wg.Add(l.testOptions.NumPods)
		// for i := 0; i < l.testOptions.NumPods; i++ {
		// 	l.wg.Add(1)
		// 	go func(index int) {
		// 		defer l.wg.Done()

		// 		framework.Logf("Creating resources for pod %v/%v", index, l.testOptions.NumPods-1)
		// 		r := storageframework.CreateVolumeResourceForCustomStorageClass(ctx, driver, l.config, pattern, t.GetTestSuiteInfo().SupportedSizeRange, sc)

		// 		l.volumesMu.Lock() // Lock to protect concurrent writes to l.volumes
		// 		l.volumes = append(l.volumes, r)
		// 		l.volumesMu.Unlock()
		// 	}(i)
		// }

		for i := 0; i < l.testOptions.NumPods; i++ {
			framework.Logf("Creating resources for pod %v/%v", i, l.testOptions.NumPods-1)
			r := storageframework.CreateVolumeResourceForCustomStorageClass(ctx, driver, l.config, pattern, t.GetTestSuiteInfo().SupportedSizeRange, sc)
			l.volumes = append(l.volumes, r)
			podConfig := e2epod.Config{
				NS:           f.Namespace.Name,
				SeLinuxLabel: e2epv.SELinuxLabel,
			}
			pod, _ := e2epod.MakeSecPod(&podConfig)
			_, err = cs.CoreV1().Pods(pod.Namespace).Create(ctx, pod, metav1.CreateOptions{})
			if err != nil {
				framework.Failf("Failed to create pod [%+v]. Error: %v", pod, err)
			}
			framework.ExpectNoError(err)

			l.pods = append(l.pods, pod)
		}
	}

	cleanup := func(ctx context.Context) {
		framework.Logf("Stopping and waiting for all test routines to finish")
		l.wg.Wait()

		var (
			errs []error
			mu   sync.Mutex
			wg   sync.WaitGroup
		)

		// wg.Add(len(l.pods))
		// for _, pod := range l.pods {
		// 	go func(pod *v1.Pod) {
		// 		defer ginkgo.GinkgoRecover()
		// 		defer wg.Done()

		// 		framework.Logf("Deleting pod %v", pod.Name)
		// 		err := e2epod.DeletePodWithWait(ctx, cs, pod)
		// 		mu.Lock()
		// 		defer mu.Unlock()
		// 		errs = append(errs, err)
		// 	}(pod)
		// }
		// wg.Wait()

		startTime := time.Now()
		wg.Add(len(l.volumes))
		for _, volume := range l.volumes {
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

		// Delete shared storage class
		framework.Logf("Deleting storage class %s", l.volumes[0].Sc.GetName())
		l.volumes[0].CleanupStorageClass(ctx)

		endTime := time.Now() // Capture overall end time
		totalDuration := endTime.Sub(startTime)
		framework.Logf("Deleted %v volumes in %v", len(l.volumes), totalDuration) // Log total deletion time

		framework.ExpectNoError(errors.NewAggregate(errs), "while cleaning up resource")
		l.migrationCheck.validateMigrationVolumeOpCounts(ctx)
		framework.Logf("Ended test")
	}

	f.It("stress pvc protection test", func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		createPodsAndVolumes(ctx)
	})
}
