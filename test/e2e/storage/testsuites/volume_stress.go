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
	storageutils "k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

type volumeStressTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

type volumeStressTest struct {
	config        *storageframework.PerTestConfig
	driverCleanup func()

	migrationCheck *migrationOpCheck

	volumes []*storageframework.VolumeResource
	pods    []*v1.Pod
	// stop and wait for any async routines
	wg     sync.WaitGroup
	ctx    context.Context
	cancel context.CancelFunc

	testOptions storageframework.StressTestOptions
}

var _ storageframework.TestSuite = &volumeStressTestSuite{}

// InitCustomVolumeStressTestSuite returns volumeStressTestSuite that implements TestSuite interface
// using custom test patterns
func InitCustomVolumeStressTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &volumeStressTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "volume-stress",
			TestPatterns: patterns,
		},
	}
}

// InitVolumeStressTestSuite returns volumeStressTestSuite that implements TestSuite interface
// using testsuite default patterns
func InitVolumeStressTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.DefaultFsDynamicPV,
		storageframework.BlockVolModeDynamicPV,
	}
	return InitCustomVolumeStressTestSuite(patterns)
}

func (t *volumeStressTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return t.tsInfo
}

func (t *volumeStressTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	dInfo := driver.GetDriverInfo()
	if dInfo.StressTestOptions == nil {
		e2eskipper.Skipf("Driver %s doesn't specify stress test options -- skipping", dInfo.Name)
	}
	if dInfo.StressTestOptions.NumPods <= 0 {
		framework.Failf("NumPods in stress test options must be a positive integer, received: %d", dInfo.StressTestOptions.NumPods)
	}
	if dInfo.StressTestOptions.NumRestarts <= 0 {
		framework.Failf("NumRestarts in stress test options must be a positive integer, received: %d", dInfo.StressTestOptions.NumRestarts)
	}

	if _, ok := driver.(storageframework.DynamicPVTestDriver); !ok {
		e2eskipper.Skipf("Driver %s doesn't implement DynamicPVTestDriver -- skipping", dInfo.Name)
	}
	if !driver.GetDriverInfo().Capabilities[storageframework.CapBlock] && pattern.VolMode == v1.PersistentVolumeBlock {
		e2eskipper.Skipf("Driver %q does not support block volume mode - skipping", dInfo.Name)
	}
}

func (t *volumeStressTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	var (
		dInfo = driver.GetDriverInfo()
		cs    clientset.Interface
		l     *volumeStressTest
	)

	// Beware that it also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewFrameworkWithCustomTimeouts("stress", storageframework.GetDriverTimeouts(driver))
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	init := func() {
		cs = f.ClientSet
		l = &volumeStressTest{}

		// Now do the more expensive test initialization.
		l.config, l.driverCleanup = driver.PrepareTest(f)
		l.migrationCheck = newMigrationOpCheck(f.ClientSet, f.ClientConfig(), dInfo.InTreePluginName)
		l.volumes = []*storageframework.VolumeResource{}
		l.pods = []*v1.Pod{}
		l.testOptions = *dInfo.StressTestOptions
		l.ctx, l.cancel = context.WithCancel(context.Background())
	}

	createPodsAndVolumes := func() {
		for i := 0; i < l.testOptions.NumPods; i++ {
			framework.Logf("Creating resources for pod %v/%v", i, l.testOptions.NumPods-1)
			r := storageframework.CreateVolumeResource(driver, l.config, pattern, t.GetTestSuiteInfo().SupportedSizeRange)
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

	cleanup := func() {
		framework.Logf("Stopping and waiting for all test routines to finish")
		l.cancel()
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
				err := e2epod.DeletePodWithWait(cs, pod)
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
				err := volume.CleanupResource()
				mu.Lock()
				defer mu.Unlock()
				errs = append(errs, err)
			}(volume)
		}
		wg.Wait()

		errs = append(errs, storageutils.TryFunc(l.driverCleanup))
		framework.ExpectNoError(errors.NewAggregate(errs), "while cleaning up resource")
		l.migrationCheck.validateMigrationVolumeOpCounts()
	}

	ginkgo.BeforeEach(func() {
		init()
		createPodsAndVolumes()
	})

	// See #96177, this is necessary for cleaning up resources when tests are interrupted.
	f.AddAfterEach("cleanup", func(f *framework.Framework, failed bool) {
		cleanup()
	})

	ginkgo.It("multiple pods should access different volumes repeatedly [Slow] [Serial]", func() {
		// Restart pod repeatedly
		for i := 0; i < l.testOptions.NumPods; i++ {
			podIndex := i
			l.wg.Add(1)
			go func() {
				defer ginkgo.GinkgoRecover()
				defer l.wg.Done()
				for j := 0; j < l.testOptions.NumRestarts; j++ {
					select {
					case <-l.ctx.Done():
						return
					default:
						pod := l.pods[podIndex]
						framework.Logf("Pod-%v [%v], Iteration %v/%v", podIndex, pod.Name, j, l.testOptions.NumRestarts-1)
						_, err := cs.CoreV1().Pods(pod.Namespace).Create(context.TODO(), pod, metav1.CreateOptions{})
						if err != nil {
							l.cancel()
							framework.Failf("Failed to create pod-%v [%+v]. Error: %v", podIndex, pod, err)
						}

						err = e2epod.WaitTimeoutForPodRunningInNamespace(cs, pod.Name, pod.Namespace, f.Timeouts.PodStart)
						if err != nil {
							l.cancel()
							framework.Failf("Failed to wait for pod-%v [%+v] turn into running status. Error: %v", podIndex, pod, err)
						}

						// TODO: write data per pod and validate it everytime

						err = e2epod.DeletePodWithWait(f.ClientSet, pod)
						if err != nil {
							l.cancel()
							framework.Failf("Failed to delete pod-%v [%+v]. Error: %v", podIndex, pod, err)
						}
					}
				}
			}()
		}

		l.wg.Wait()
	})
}
