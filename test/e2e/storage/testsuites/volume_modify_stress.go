/*
Copyright 2025 The Kubernetes Authors.

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
	"sync"

	"github.com/onsi/ginkgo/v2"
	/*"github.com/onsi/gomega"*/
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

type volumeModifyStressTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

type volumeModifyStressTest struct {
	config *storageframework.PerTestConfig

	vac *storagev1.VolumeAttributesClass

	volumes []*storageframework.VolumeResource
	pods    []*v1.Pod
	// stop and wait for any async routines
	wg sync.WaitGroup

	testOptions storageframework.VolumeModifyStressTestOptions
}

var _ storageframework.TestSuite = &volumeModifyStressTestSuite{}

// InitCustomVolumeModifyStressTestSuite returns volumeModifyStressTestSuite that implements TestSuite interface
// using custom test patterns
func InitCustomVolumeModifyStressTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &volumeModifyStressTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "volume-modify-stress",
			TestPatterns: patterns,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Gi",
			},
			TestTags: []interface{}{framework.WithFeatureGate(features.VolumeAttributesClass), feature.VolumeAttributesClass},
		},
	}
}

// InitVolumeModifyStressTestSuite returns volumeModifyStressTestSuite that implements TestSuite interface
// using testsuite default patterns
func InitVolumeModifyStressTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.DefaultFsDynamicPV,
		storageframework.BlockVolModeDynamicPV,
		storageframework.NtfsDynamicPV,
	}
	return InitCustomVolumeModifyStressTestSuite(patterns)
}

func (v *volumeModifyStressTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return v.tsInfo
}

func (v *volumeModifyStressTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	driverInfo := driver.GetDriverInfo()
	if driverInfo.VolumeModifyStressTestOptions == nil {
		e2eskipper.Skipf("Driver %s doesn't specify volume modify stress test options -- skipping", driverInfo.Name)
	}
	if driverInfo.VolumeModifyStressTestOptions.NumPods <= 0 {
		framework.Failf("NumPods in modify volume stress test options must be a positive integer, received: %d", driverInfo.VolumeModifyStressTestOptions.NumPods)
	}
	_, ok := driver.(storageframework.VolumeAttributesClassTestDriver)
	if !ok {
		e2eskipper.Skipf("Driver %q does not support VolumeAttributesClass tests - skipping", driver.GetDriverInfo().Name)
	}
	// Skip block storage tests if the driver we are testing against does not support block volumes
	// TODO: This should be made generic so that it doesn't have to be re-written for every test that uses the 	BlockVolModeDynamicPV testcase
	if !driverInfo.Capabilities[storageframework.CapBlock] && pattern.VolMode == v1.PersistentVolumeBlock {
		e2eskipper.Skipf("Driver %q does not support block volume mode - skipping", driver.GetDriverInfo().Name)
	}
}

func (v *volumeModifyStressTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {

	var (
		dInfo = driver.GetDriverInfo()
		cs    clientset.Interface
		l     *volumeModifyStressTest
	)

	// Beware that it also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewFrameworkWithCustomTimeouts("volume-modify-stress", storageframework.GetDriverTimeouts(driver))
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	init := func(ctx context.Context) {
		cs = f.ClientSet
		l = &volumeModifyStressTest{}

		// Now do the more expensive test initialization.
		l.config = driver.PrepareTest(ctx, f)
		vacDriver, _ := driver.(storageframework.VolumeAttributesClassTestDriver)
		l.volumes = []*storageframework.VolumeResource{}
		l.pods = []*v1.Pod{}
		l.testOptions = *dInfo.VolumeModifyStressTestOptions
		l.vac = vacDriver.GetVolumeAttributesClass(ctx, l.config)

		if l.vac == nil {
			e2eskipper.Skipf("Driver %q returned nil VolumeAttributesClass - skipping", dInfo.Name)
		}

		ginkgo.By("Creating VolumeAttributesClass")
		_, err := f.ClientSet.StorageV1().VolumeAttributesClasses().Create(ctx, l.vac, metav1.CreateOptions{})
		framework.ExpectNoError(err, "While creating VolumeAttributesClass")
	}

	createPodsAndVolumes := func(ctx context.Context, createVolumeWithVAC bool) {
		for i := 0; i < l.testOptions.NumPods; i++ {
			framework.Logf("Creating resources for pod %v/%v", i, l.testOptions.NumPods-1)
			testVolumeSizeRange := v.GetTestSuiteInfo().SupportedSizeRange
			ginkgo.By("Creating volume")
			r := &storageframework.VolumeResource{}
			if createVolumeWithVAC {
				r = storageframework.CreateVolumeResourceWithVAC(ctx, driver, l.config, pattern, testVolumeSizeRange, &l.vac.Name)
			} else {
				r = storageframework.CreateVolumeResource(ctx, driver, l.config, pattern, testVolumeSizeRange)
			}
			l.volumes = append(l.volumes, r)
			podConfig := e2epod.Config{
				NS:           f.Namespace.Name,
				PVCs:         []*v1.PersistentVolumeClaim{r.Pvc},
				SeLinuxLabel: e2epv.SELinuxLabel,
			}
			pod, err := e2epod.MakeSecPod(&podConfig)
			framework.ExpectNoError(err)

			l.pods = append(l.pods, pod)
		}
	}

	modifyVolumes := func(ctx context.Context) {
		for i := 0; i < l.testOptions.NumPods; i++ {
			ginkgo.By("Modifying PVC via VAC")
			volume := l.volumes[i]
			volume.Pvc = SetPVCVACName(ctx, volume.Pvc, l.vac.Name, f.ClientSet, setVACWaitPeriod)

			ginkgo.By("Checking PVC status")
			err := e2epv.WaitForPersistentVolumeClaimModified(ctx, f.ClientSet, volume.Pvc, modifyVolumeWaitPeriod)
			framework.ExpectNoError(err, "While waiting for PVC to have expected VAC")
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

		wg.Add(len(l.pods))
		for _, pod := range l.pods {
			go func(pod *v1.Pod) {
				defer ginkgo.GinkgoRecover()
				defer wg.Done()

				framework.Logf("Deleting pod %v", pod.Name)
				err := e2epod.DeletePodWithWait(ctx, cs, pod)
				mu.Lock()
				defer mu.Unlock()
				errs = append(errs, err)
			}(pod)
		}
		wg.Wait()

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

		if l.vac != nil {
			ginkgo.By("Deleting VAC")
			CleanupVAC(ctx, l.vac, f.ClientSet, vacCleanupWaitPeriod)
			l.vac = nil
		}

		framework.ExpectNoError(errors.NewAggregate(errs), "while cleaning up resource")
	}

	f.It("multiple pods should provision volumes with volumeAttributesClass", f.WithSlow(), func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)
		createPodsAndVolumes(ctx, true)
		l.wg.Wait()
	})

	f.It("multiple pods should modify volumes with a different volumeAttributesClass", f.WithSlow(), func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)
		createPodsAndVolumes(ctx, false)
		modifyVolumes(ctx)
		l.wg.Wait()
	})
}
