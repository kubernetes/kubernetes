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

/*

A suite of stress tests for volumes in ReadOnlyMany mode.

*/

package testsuites

import (
	"context"
	"fmt"
	"sync"

	"github.com/onsi/ginkgo"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"

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

type multiReaderStressTestSuite struct {
	tsInfo TestSuiteInfo
}

type multiReaderStressTest struct {
	config        *PerTestConfig
	driverCleanup func()

	intreeOps   opCounts
	migratedOps opCounts

	writerResource *VolumeResource
	readerPVC *v1.PersistentVolumeClaim
	readerPV  *v1.PersistentVolume
	readerPods     []*v1.Pod
	// stop and wait for any async routines
	wg      sync.WaitGroup
	stopChs []chan struct{}
}

var _ TestSuite = &multiReaderStressTestSuite{}

// InitStressTestSuite returns multiReaderStressTestSuite that implements TestSuite interface
func InitStressTestSuite() TestSuite {
	return &multiReaderStressTestSuite{
		tsInfo: TestSuiteInfo{
			Name:       "multireader-stress",
			FeatureTag: "[Feature: VolumeStress]",
			TestPatterns: []testpatterns.TestPattern{
				testpatterns.DefaultFsDynamicPV,
			},
		},
	}
}

func (t *multiReaderStressTestSuite) GetTestSuiteInfo() TestSuiteInfo {
	return t.tsInfo
}

func (t *multiReaderStressTestSuite) SkipRedundantSuite(driver TestDriver, pattern testpatterns.TestPattern) {}

func (t *multiReaderStressTestSuite) DefineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	var (
		dInfo = driver.GetDriverInfo()
		cs    clientset.Interface
	)

	ginkgo.BeforeEach(func() {
		// Check preconditions.
		_, ok := driver.(DynamicPVTestDriver)
		if !ok {
			e2eskipper.Skipf("Driver %s doesn't support dynamic PV mode -- skipping", dInfo.Name)
		}

		_, ok = driver.(ReadOnlyPVTestDriver)
		if !ok {
			e2eskipper.Skipf("Driver %s doesn't support readonly PV mode -- skipping", dInfo.Name)
		}
	})

	// This intentionally comes after checking the preconditions because it
	// registers its own BeforeEach which creates the namespace. Beware that it
	// also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewDefaultFramework("multireader-stress")

	init := func() *multiReaderStressTest {
		cs = f.ClientSet
		l := &multiReaderStressTest{}

		// Now do the more expensive test initialization.
		l.config, l.driverCleanup = driver.PrepareTest(f)
		l.intreeOps, l.migratedOps = getMigrationVolumeOpCounts(f.ClientSet, dInfo.InTreePluginName)
		l.stopChs = []chan struct{}{}

		return l
	}

	cleanup := func(l *multiReaderStressTest) {
		var errs []error

		framework.Logf("Stopping and waiting for all test routines to finish")
		for _, stopCh := range l.stopChs {
			close(stopCh)
		}
		l.wg.Wait()

		framework.Logf("Deleting reader PV and PVC")
		readerVolumeCleanupErrs := e2epv.PVPVCCleanup(cs, f.Namespace.Name, l.readerPV, l.readerPVC)
		errs = append(errs, readerVolumeCleanupErrs...)

		for _, pod := range l.readerPods {
			framework.Logf("Deleting pod %v", pod.Name)
			err := e2epod.DeletePodWithWait(cs, pod)
			errs = append(errs, err)
		}

		if l.writerResource != nil {
			framework.Logf("Deleting writer PV and PVC")
			errs = append(errs, l.writerResource.CleanupResource())
		}

		errs = append(errs, tryFunc(l.driverCleanup))
		framework.ExpectNoError(errors.NewAggregate(errs), "while cleaning up resource")
		validateMigrationVolumeOpCounts(f.ClientSet, dInfo.InTreePluginName, l.intreeOps, l.migratedOps)
	}

	ginkgo.It("multiple pods should access a single read-only volume repeatedly [Slow] [Serial]", func() {
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

		ginkgo.By("Creating writer volume")
		l.writerResource = CreateVolumeResource(driver, l.config, pattern, t.GetTestSuiteInfo().SupportedSizeRange)

		ginkgo.By("Running writer pod")
		writerConfig := e2evolume.TestConfig{
			Namespace:           f.Namespace.Name,
			Prefix:              "multireader-stress-writerpod",
		}
		tests := []e2evolume.Test{
			{
				Volume: *l.writerResource.VolSource,
				Mode:   pattern.VolMode,
				File:   "index.html",
				ExpectedContent: fmt.Sprintf("Hello from %s from namespace %s",
					dInfo.Name, f.Namespace.Name),
			},
		}
		var fsGroup *int64
		if framework.NodeOSDistroIs("windows") && dInfo.Capabilities[CapFsGroup] {
			fsGroupVal := int64(1234)
			fsGroup = &fsGroupVal
		}
		e2evolume.InjectContent(f, writerConfig, fsGroup, pattern.FsType, tests)

		ginkgo.By("Creating read-only PV and PVC")
		writerPVC, err := cs.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Get(context.TODO(), l.writerResource.Pvc.Name, metav1.GetOptions{})
		writerPV, err := cs.CoreV1().PersistentVolumes().Get(context.TODO(), writerPVC.Spec.VolumeName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		writerPVSource := &writerPV.Spec.PersistentVolumeSource
		readerPVSource := writerPVSource.DeepCopy()
		driver.(ReadOnlyPVTestDriver).UpdateReadOnlyInPVSource(
			readerPVSource,
			true /* readOnly */)

		emptyStorageClass := ""
		l.readerPV, l.readerPVC, err = e2epv.CreatePVPVC(cs,
			e2epv.PersistentVolumeConfig{
			NamePrefix: "multireader-stress-",
			PVSource: *readerPVSource,
			StorageClassName: emptyStorageClass,
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadOnlyMany},
		}, e2epv.PersistentVolumeClaimConfig{
			StorageClassName: &emptyStorageClass,
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadOnlyMany},
		}, f.Namespace.Name, true /* preBind */)

		ginkgo.By("Creating reader pods")
		for i := 0; i < numPods; i++ {
			framework.Logf("Creating reader pod %v/%v", i, numPods)
			pod, err := e2epod.MakeSecPod(&e2epod.Config{
				NS:                  f.Namespace.Name,
				PVCs: []*v1.PersistentVolumeClaim{l.readerPVC},
				SeLinuxLabel:        e2epv.SELinuxLabel,
				ReadOnly:            true,
			})
			framework.ExpectNoError(err)
			l.readerPods = append(l.readerPods, pod)
			l.stopChs = append(l.stopChs, make(chan struct{}))
		}

		ginkgo.By("Restarting pod repeatedly")
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
						pod := l.readerPods[podIndex]
						framework.Logf("Pod %v, Iteration %v/%v", podIndex, j, numPodStarts)
						_, err = cs.CoreV1().Pods(pod.Namespace).Create(context.TODO(), pod, metav1.CreateOptions{})
						framework.ExpectNoError(err, fmt.Sprintf("error creating pod %d on iteration %d", podIndex, j))

						err = e2epod.WaitForPodRunningInNamespace(cs, pod)
						framework.ExpectNoError(err, fmt.Sprintf("error waiting for pod %d running on iteration %d", podIndex, j))

						// TODO(cxing): read data per pod and validate it everytime

						err = e2epod.DeletePodWithWait(f.ClientSet, pod)
						framework.ExpectNoError(err, fmt.Sprintf("error deleting pod %d on iteration %d", podIndex, j))
					}
				}
			}()
		}

		l.wg.Wait()
	})
}