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

	"github.com/onsi/ginkgo"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	errors "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
)

type volumeStressTestSuite struct {
	tsInfo TestSuiteInfo
}

type volumeStressTest struct {
	config        *PerTestConfig
	driverCleanup func()

	migrationCheck *migrationOpCheck

	resources []*VolumeResource
	pods      []*v1.Pod
	// stop and wait for any async routines
	wg     sync.WaitGroup
	ctx    context.Context
	cancel context.CancelFunc

	testOptions StressTestOptions
}

var _ TestSuite = &volumeStressTestSuite{}

// InitVolumeStressTestSuite returns volumeStressTestSuite that implements TestSuite interface
func InitVolumeStressTestSuite() TestSuite {
	return &volumeStressTestSuite{
		tsInfo: TestSuiteInfo{
			Name: "volume-stress",
			TestPatterns: []testpatterns.TestPattern{
				testpatterns.DefaultFsDynamicPV,
				testpatterns.BlockVolModeDynamicPV,
			},
		},
	}
}

func (t *volumeStressTestSuite) GetTestSuiteInfo() TestSuiteInfo {
	return t.tsInfo
}

func (t *volumeStressTestSuite) SkipRedundantSuite(driver TestDriver, pattern testpatterns.TestPattern) {
}

func (t *volumeStressTestSuite) DefineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	var (
		dInfo = driver.GetDriverInfo()
		cs    clientset.Interface
		l     *volumeStressTest
	)

	// Check preconditions before setting up namespace via framework below.
	ginkgo.BeforeEach(func() {
		if dInfo.StressTestOptions == nil {
			e2eskipper.Skipf("Driver %s doesn't specify stress test options -- skipping", dInfo.Name)
		}
		if dInfo.StressTestOptions.NumPods <= 0 {
			framework.Failf("NumPods in stress test options must be a positive integer, received: %d", dInfo.StressTestOptions.NumPods)
		}
		if dInfo.StressTestOptions.NumRestarts <= 0 {
			framework.Failf("NumRestarts in stress test options must be a positive integer, received: %d", dInfo.StressTestOptions.NumRestarts)
		}

		if _, ok := driver.(DynamicPVTestDriver); !ok {
			e2eskipper.Skipf("Driver %s doesn't implement DynamicPVTestDriver -- skipping", dInfo.Name)
		}

		if !driver.GetDriverInfo().Capabilities[CapBlock] && pattern.VolMode == v1.PersistentVolumeBlock {
			e2eskipper.Skipf("Driver %q does not support block volume mode - skipping", dInfo.Name)
		}
	})

	// This intentionally comes after checking the preconditions because it
	// registers its own BeforeEach which creates the namespace. Beware that it
	// also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewDefaultFramework("volume-stress")

	init := func() {
		cs = f.ClientSet
		l = &volumeStressTest{}

		// Now do the more expensive test initialization.
		l.config, l.driverCleanup = driver.PrepareTest(f)
		l.migrationCheck = newMigrationOpCheck(f.ClientSet, dInfo.InTreePluginName)
		l.resources = []*VolumeResource{}
		l.pods = []*v1.Pod{}
		l.testOptions = *dInfo.StressTestOptions
		l.ctx, l.cancel = context.WithCancel(context.Background())
	}

	createPodsAndVolumes := func() {
		for i := 0; i < l.testOptions.NumPods; i++ {
			framework.Logf("Creating resources for pod %v/%v", i, l.testOptions.NumPods-1)
			r := CreateVolumeResource(driver, l.config, pattern, t.GetTestSuiteInfo().SupportedSizeRange)
			l.resources = append(l.resources, r)
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
		var errs []error

		framework.Logf("Stopping and waiting for all test routines to finish")
		l.cancel()
		l.wg.Wait()

		for _, pod := range l.pods {
			framework.Logf("Deleting pod %v", pod.Name)
			err := e2epod.DeletePodWithWait(cs, pod)
			errs = append(errs, err)
		}

		for _, resource := range l.resources {
			errs = append(errs, resource.CleanupResource())
		}

		errs = append(errs, tryFunc(l.driverCleanup))
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

						err = e2epod.WaitForPodRunningInNamespace(cs, pod)
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
