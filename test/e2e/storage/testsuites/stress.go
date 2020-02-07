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

type stressTestSuite struct {
	tsInfo TestSuiteInfo
}

type stressTest struct {
	config        *PerTestConfig
	driverCleanup func()

	intreeOps   opCounts
	migratedOps opCounts

	resources []*VolumeResource
	pods      []*v1.Pod
	// stop and wait for any async routines
	wg      sync.WaitGroup
	stopChs []chan struct{}
}

var _ TestSuite = &stressTestSuite{}

// InitStressTestSuite returns stressTestSuite that implements TestSuite interface
func InitStressTestSuite() TestSuite {
	return &stressTestSuite{
		tsInfo: TestSuiteInfo{
			Name:       "stress",
			FeatureTag: "[Feature: VolumeStress]",
			TestPatterns: []testpatterns.TestPattern{
				testpatterns.DefaultFsDynamicPV,
				testpatterns.BlockVolModeDynamicPV,
			},
		},
	}
}

func (t *stressTestSuite) GetTestSuiteInfo() TestSuiteInfo {
	return t.tsInfo
}

func (t *stressTestSuite) SkipRedundantSuite(driver TestDriver, pattern testpatterns.TestPattern) {
}

func (t *stressTestSuite) DefineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	var (
		dInfo = driver.GetDriverInfo()
		cs    clientset.Interface
	)

	ginkgo.BeforeEach(func() {
		// Check preconditions.
		ok := false
		_, ok = driver.(DynamicPVTestDriver)
		if !ok {
			e2eskipper.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, pattern.VolType)
		}

		if !driver.GetDriverInfo().Capabilities[CapBlock] && pattern.VolMode == v1.PersistentVolumeBlock {
			e2eskipper.Skipf("Driver %q does not support block volume mode - skipping", driver.GetDriverInfo().Name)
		}
	})

	// This intentionally comes after checking the preconditions because it
	// registers its own BeforeEach which creates the namespace. Beware that it
	// also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewDefaultFramework("stress")

	init := func() *stressTest {
		cs = f.ClientSet
		l := &stressTest{}

		// Now do the more expensive test initialization.
		l.config, l.driverCleanup = driver.PrepareTest(f)
		l.intreeOps, l.migratedOps = getMigrationVolumeOpCounts(f.ClientSet, dInfo.InTreePluginName)
		l.resources = []*VolumeResource{}
		l.pods = []*v1.Pod{}
		l.stopChs = []chan struct{}{}

		return l
	}

	cleanup := func(l *stressTest) {
		var errs []error

		framework.Logf("Stopping and waiting for all test routines to finish")
		for _, stopCh := range l.stopChs {
			close(stopCh)
		}
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
		validateMigrationVolumeOpCounts(f.ClientSet, dInfo.InTreePluginName, l.intreeOps, l.migratedOps)
	}

	ginkgo.It("multiple pods should access different volumes repeatedly [Slow] [Serial]", func() {
		const (
			numPods = 10
			// number of times each Pod should start
			numPodStarts = 10
		)

		var err error

		l := init()
		defer func() {
			cleanup(l)
		}()

		for i := 0; i < numPods; i++ {
			framework.Logf("Creating resources for pod %v/%v", i, numPods)
			r := CreateVolumeResource(driver, l.config, pattern, t.GetTestSuiteInfo().SupportedSizeRange)
			l.resources = append(l.resources, r)
			l.pods = append(l.pods, e2epod.MakeSecPod(f.Namespace.Name,
				[]*v1.PersistentVolumeClaim{r.Pvc},
				nil, false, "", false, false, e2epv.SELinuxLabel, nil))
			l.stopChs = append(l.stopChs, make(chan struct{}))
		}

		// Restart pod repeatedly
		for i := 0; i < numPods; i++ {
			podIndex := i
			l.wg.Add(1)
			go func() {
				defer ginkgo.GinkgoRecover()
				defer l.wg.Done()
				for j := 0; j < numPodStarts; j++ {
					select {
					case <-l.stopChs[podIndex]:
						return
					default:
						pod := l.pods[podIndex]
						framework.Logf("Pod %v, Iteration %v/%v", podIndex, j, numPodStarts)
						_, err = cs.CoreV1().Pods(pod.Namespace).Create(context.TODO(), pod, metav1.CreateOptions{})
						framework.ExpectNoError(err)

						err = e2epod.WaitForPodRunningInNamespace(cs, pod)
						framework.ExpectNoError(err)

						// TODO: write data per pod and validate it everytime

						err = e2epod.DeletePodWithWait(f.ClientSet, pod)
						framework.ExpectNoError(err)
					}
				}
			}()
		}

		l.wg.Wait()
	})
}
