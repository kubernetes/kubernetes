/*
Copyright 2019 The Kubernetes Authors.

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
	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/api/core/v1"
	errors "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	storageutils "k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

type disruptiveTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

// InitCustomDisruptiveTestSuite returns subPathTestSuite that implements TestSuite interface
// using custom test patterns
func InitCustomDisruptiveTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &disruptiveTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "disruptive",
			FeatureTag:   "[Disruptive][LinuxOnly]",
			TestPatterns: patterns,
		},
	}
}

// InitDisruptiveTestSuite returns subPathTestSuite that implements TestSuite interface
// using test suite default patterns
func InitDisruptiveTestSuite() storageframework.TestSuite {
	testPatterns := []storageframework.TestPattern{
		// FSVolMode is already covered in subpath testsuite
		storageframework.DefaultFsInlineVolume,
		storageframework.FsVolModePreprovisionedPV,
		storageframework.FsVolModeDynamicPV,
		storageframework.BlockVolModePreprovisionedPV,
		storageframework.BlockVolModeDynamicPV,
	}
	return InitCustomDisruptiveTestSuite(testPatterns)
}

func (s *disruptiveTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return s.tsInfo
}

func (s *disruptiveTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	skipVolTypePatterns(pattern, driver, storageframework.NewVolTypeMap(storageframework.PreprovisionedPV))
	if pattern.VolMode == v1.PersistentVolumeBlock && !driver.GetDriverInfo().Capabilities[storageframework.CapBlock] {
		e2eskipper.Skipf("Driver %s doesn't support %v -- skipping", driver.GetDriverInfo().Name, pattern.VolMode)
	}
}

func (s *disruptiveTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	type local struct {
		config        *storageframework.PerTestConfig
		driverCleanup func()

		cs clientset.Interface
		ns *v1.Namespace

		// VolumeResource contains pv, pvc, sc, etc., owns cleaning that up
		resource *storageframework.VolumeResource
		pod      *v1.Pod
	}
	var l local

	// Beware that it also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewFrameworkWithCustomTimeouts("disruptive", storageframework.GetDriverTimeouts(driver))
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	init := func() {
		l = local{}
		l.ns = f.Namespace
		l.cs = f.ClientSet

		// Now do the more expensive test initialization.
		l.config, l.driverCleanup = driver.PrepareTest(f)

		testVolumeSizeRange := s.GetTestSuiteInfo().SupportedSizeRange
		l.resource = storageframework.CreateVolumeResource(driver, l.config, pattern, testVolumeSizeRange)
	}

	cleanup := func() {
		var errs []error
		if l.pod != nil {
			ginkgo.By("Deleting pod")
			err := e2epod.DeletePodWithWait(f.ClientSet, l.pod)
			errs = append(errs, err)
			l.pod = nil
		}

		if l.resource != nil {
			err := l.resource.CleanupResource()
			errs = append(errs, err)
			l.resource = nil
		}

		errs = append(errs, storageutils.TryFunc(l.driverCleanup))
		l.driverCleanup = nil
		framework.ExpectNoError(errors.NewAggregate(errs), "while cleaning up resource")
	}

	type testBody func(c clientset.Interface, f *framework.Framework, clientPod *v1.Pod)
	type disruptiveTest struct {
		testItStmt   string
		runTestFile  testBody
		runTestBlock testBody
	}
	disruptiveTestTable := []disruptiveTest{
		{
			testItStmt:   "Should test that pv written before kubelet restart is readable after restart.",
			runTestFile:  utils.TestKubeletRestartsAndRestoresMount,
			runTestBlock: utils.TestKubeletRestartsAndRestoresMap,
		},
		{
			testItStmt: "Should test that pv used in a pod that is deleted while the kubelet is down cleans up when the kubelet returns.",
			// File test is covered by subpath testsuite
			runTestBlock: utils.TestVolumeUnmapsFromDeletedPod,
		},
		{
			testItStmt: "Should test that pv used in a pod that is force deleted while the kubelet is down cleans up when the kubelet returns.",
			// File test is covered by subpath testsuite
			runTestBlock: utils.TestVolumeUnmapsFromForceDeletedPod,
		},
	}

	for _, test := range disruptiveTestTable {
		func(t disruptiveTest) {
			if (pattern.VolMode == v1.PersistentVolumeBlock && t.runTestBlock != nil) ||
				(pattern.VolMode == v1.PersistentVolumeFilesystem && t.runTestFile != nil) {
				ginkgo.It(t.testItStmt, func() {
					init()
					defer cleanup()

					var err error
					var pvcs []*v1.PersistentVolumeClaim
					var inlineSources []*v1.VolumeSource
					if pattern.VolType == storageframework.InlineVolume {
						inlineSources = append(inlineSources, l.resource.VolSource)
					} else {
						pvcs = append(pvcs, l.resource.Pvc)
					}
					ginkgo.By("Creating a pod with pvc")
					podConfig := e2epod.Config{
						NS:                  l.ns.Name,
						PVCs:                pvcs,
						InlineVolumeSources: inlineSources,
						SeLinuxLabel:        e2epv.SELinuxLabel,
						NodeSelection:       l.config.ClientNodeSelection,
						ImageID:             e2epod.GetDefaultTestImageID(),
					}
					l.pod, err = e2epod.CreateSecPodWithNodeSelection(l.cs, &podConfig, f.Timeouts.PodStart)
					framework.ExpectNoError(err, "While creating pods for kubelet restart test")

					if pattern.VolMode == v1.PersistentVolumeBlock && t.runTestBlock != nil {
						t.runTestBlock(l.cs, l.config.Framework, l.pod)
					}
					if pattern.VolMode == v1.PersistentVolumeFilesystem && t.runTestFile != nil {
						t.runTestFile(l.cs, l.config.Framework, l.pod)
					}
				})
			}
		}(test)
	}
}
