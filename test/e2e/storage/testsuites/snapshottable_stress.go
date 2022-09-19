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

// This suite tests volume snapshots under stress conditions.

package testsuites

import (
	"context"
	"sync"

	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	errors "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	storageutils "k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

type snapshottableStressTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

type snapshottableStressTest struct {
	config        *storageframework.PerTestConfig
	testOptions   storageframework.VolumeSnapshotStressTestOptions
	driverCleanup func()

	pods      []*v1.Pod
	volumes   []*storageframework.VolumeResource
	snapshots []*storageframework.SnapshotResource
	// Because we are appending snapshot resources in parallel goroutines.
	snapshotsMutex sync.Mutex

	// Stop and wait for any async routines.
	ctx    context.Context
	wg     sync.WaitGroup
	cancel context.CancelFunc
}

// InitCustomSnapshottableStressTestSuite returns snapshottableStressTestSuite that implements TestSuite interface
// using custom test patterns
func InitCustomSnapshottableStressTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &snapshottableStressTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "snapshottable-stress",
			TestPatterns: patterns,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
			FeatureTag: "[Feature:VolumeSnapshotDataSource]",
		},
	}
}

// InitSnapshottableStressTestSuite returns snapshottableStressTestSuite that implements TestSuite interface
// using testsuite default patterns
func InitSnapshottableStressTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.DynamicSnapshotDelete,
		storageframework.DynamicSnapshotRetain,
	}
	return InitCustomSnapshottableStressTestSuite(patterns)
}

func (t *snapshottableStressTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return t.tsInfo
}

func (t *snapshottableStressTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	driverInfo := driver.GetDriverInfo()
	var ok bool
	if driverInfo.VolumeSnapshotStressTestOptions == nil {
		e2eskipper.Skipf("Driver %s doesn't specify snapshot stress test options -- skipping", driverInfo.Name)
	}
	if driverInfo.VolumeSnapshotStressTestOptions.NumPods <= 0 {
		framework.Failf("NumPods in snapshot stress test options must be a positive integer, received: %d", driverInfo.VolumeSnapshotStressTestOptions.NumPods)
	}
	if driverInfo.VolumeSnapshotStressTestOptions.NumSnapshots <= 0 {
		framework.Failf("NumSnapshots in snapshot stress test options must be a positive integer, received: %d", driverInfo.VolumeSnapshotStressTestOptions.NumSnapshots)
	}
	_, ok = driver.(storageframework.SnapshottableTestDriver)
	if !driverInfo.Capabilities[storageframework.CapSnapshotDataSource] || !ok {
		e2eskipper.Skipf("Driver %q doesn't implement SnapshottableTestDriver - skipping", driverInfo.Name)
	}

	_, ok = driver.(storageframework.DynamicPVTestDriver)
	if !ok {
		e2eskipper.Skipf("Driver %s doesn't implement DynamicPVTestDriver -- skipping", driverInfo.Name)
	}
}

func (t *snapshottableStressTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	var (
		driverInfo          *storageframework.DriverInfo
		snapshottableDriver storageframework.SnapshottableTestDriver
		cs                  clientset.Interface
		stressTest          *snapshottableStressTest
	)

	// Beware that it also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewDefaultFramework("snapshottable-stress")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	init := func() {
		driverInfo = driver.GetDriverInfo()
		snapshottableDriver, _ = driver.(storageframework.SnapshottableTestDriver)
		cs = f.ClientSet
		config, driverCleanup := driver.PrepareTest(f)
		ctx, cancel := context.WithCancel(context.Background())

		stressTest = &snapshottableStressTest{
			config:        config,
			driverCleanup: driverCleanup,
			volumes:       []*storageframework.VolumeResource{},
			snapshots:     []*storageframework.SnapshotResource{},
			pods:          []*v1.Pod{},
			testOptions:   *driverInfo.VolumeSnapshotStressTestOptions,
			ctx:           ctx,
			cancel:        cancel,
		}
	}

	createPodsAndVolumes := func() {
		for i := 0; i < stressTest.testOptions.NumPods; i++ {
			framework.Logf("Creating resources for pod %d/%d", i, stressTest.testOptions.NumPods-1)

			volume := storageframework.CreateVolumeResource(driver, stressTest.config, pattern, t.GetTestSuiteInfo().SupportedSizeRange)
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

				if _, err := cs.CoreV1().Pods(pod.Namespace).Create(context.TODO(), pod, metav1.CreateOptions{}); err != nil {
					stressTest.cancel()
					framework.Failf("Failed to create pod-%d [%+v]. Error: %v", i, pod, err)
				}
			}(i, pod)
		}
		wg.Wait()

		for i, pod := range stressTest.pods {
			if err := e2epod.WaitForPodRunningInNamespace(cs, pod); err != nil {
				stressTest.cancel()
				framework.Failf("Failed to wait for pod-%d [%+v] turn into running status. Error: %v", i, pod, err)
			}
		}
	}

	cleanup := func() {
		framework.Logf("Stopping and waiting for all test routines to finish")
		stressTest.cancel()
		stressTest.wg.Wait()

		var (
			errs []error
			mu   sync.Mutex
			wg   sync.WaitGroup
		)

		wg.Add(len(stressTest.snapshots))
		for _, snapshot := range stressTest.snapshots {
			go func(snapshot *storageframework.SnapshotResource) {
				defer ginkgo.GinkgoRecover()
				defer wg.Done()

				framework.Logf("Deleting snapshot %s/%s", snapshot.Vs.GetNamespace(), snapshot.Vs.GetName())
				err := snapshot.CleanupResource(f.Timeouts)
				mu.Lock()
				defer mu.Unlock()
				errs = append(errs, err)
			}(snapshot)
		}
		wg.Wait()

		wg.Add(len(stressTest.pods))
		for _, pod := range stressTest.pods {
			go func(pod *v1.Pod) {
				defer ginkgo.GinkgoRecover()
				defer wg.Done()

				framework.Logf("Deleting pod %s", pod.Name)
				err := e2epod.DeletePodWithWait(cs, pod)
				mu.Lock()
				defer mu.Unlock()
				errs = append(errs, err)
			}(pod)
		}
		wg.Wait()

		wg.Add(len(stressTest.volumes))
		for _, volume := range stressTest.volumes {
			go func(volume *storageframework.VolumeResource) {
				defer ginkgo.GinkgoRecover()
				defer wg.Done()

				framework.Logf("Deleting volume %s", volume.Pvc.GetName())
				err := volume.CleanupResource()
				mu.Lock()
				defer mu.Unlock()
				errs = append(errs, err)
			}(volume)
		}
		wg.Wait()

		errs = append(errs, storageutils.TryFunc(stressTest.driverCleanup))

		framework.ExpectNoError(errors.NewAggregate(errs), "while cleaning up resources")
	}

	ginkgo.BeforeEach(func() {
		init()
		ginkgo.DeferCleanup(cleanup)
		createPodsAndVolumes()
	})

	ginkgo.It("should support snapshotting of many volumes repeatedly [Slow] [Serial]", func() {
		// Repeatedly create and delete snapshots of each volume.
		for i := 0; i < stressTest.testOptions.NumPods; i++ {
			for j := 0; j < stressTest.testOptions.NumSnapshots; j++ {
				stressTest.wg.Add(1)

				go func(podIndex, snapshotIndex int) {
					defer ginkgo.GinkgoRecover()
					defer stressTest.wg.Done()

					pod := stressTest.pods[podIndex]
					volume := stressTest.volumes[podIndex]

					select {
					case <-stressTest.ctx.Done():
						return
					default:
						framework.Logf("Pod-%d [%s], Iteration %d/%d", podIndex, pod.Name, snapshotIndex, stressTest.testOptions.NumSnapshots-1)
						parameters := map[string]string{}
						snapshot := storageframework.CreateSnapshotResource(snapshottableDriver, stressTest.config, pattern, volume.Pvc.GetName(), volume.Pvc.GetNamespace(), f.Timeouts, parameters)
						stressTest.snapshotsMutex.Lock()
						defer stressTest.snapshotsMutex.Unlock()
						stressTest.snapshots = append(stressTest.snapshots, snapshot)
					}
				}(i, j)
			}
		}

		stressTest.wg.Wait()
	})
}
