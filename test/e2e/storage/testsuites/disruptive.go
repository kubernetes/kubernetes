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
	"context"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
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
			TestTags:     []interface{}{framework.WithDisruptive(), framework.WithLabel("LinuxOnly")},
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
	e2eskipper.SkipUnlessSSHKeyPresent()
}

func (s *disruptiveTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	type local struct {
		config *storageframework.PerTestConfig

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
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	init := func(ctx context.Context, accessModes []v1.PersistentVolumeAccessMode) {
		l = local{}
		l.ns = f.Namespace
		l.cs = f.ClientSet

		// Now do the more expensive test initialization.
		l.config = driver.PrepareTest(ctx, f)

		testVolumeSizeRange := s.GetTestSuiteInfo().SupportedSizeRange
		if accessModes == nil {
			l.resource = storageframework.CreateVolumeResource(
				ctx,
				driver,
				l.config,
				pattern,
				testVolumeSizeRange)
		} else {
			l.resource = storageframework.CreateVolumeResourceWithAccessModes(
				ctx,
				driver,
				l.config,
				pattern,
				testVolumeSizeRange,
				accessModes,
				nil)
		}
	}

	cleanup := func(ctx context.Context) {
		var errs []error
		if l.pod != nil {
			ginkgo.By("Deleting pod")
			err := e2epod.DeletePodWithWait(ctx, f.ClientSet, l.pod)
			errs = append(errs, err)
			l.pod = nil
		}

		if l.resource != nil {
			err := l.resource.CleanupResource(ctx)
			errs = append(errs, err)
			l.resource = nil
		}

		framework.ExpectNoError(errors.NewAggregate(errs), "while cleaning up resource")
	}

	type singlePodTestBody func(ctx context.Context, c clientset.Interface, f *framework.Framework, clientPod *v1.Pod, mountPath string)
	type singlePodTest struct {
		testItStmt   string
		runTestFile  singlePodTestBody
		runTestBlock singlePodTestBody
	}
	singlePodTests := []singlePodTest{
		{
			testItStmt:   "Should test that pv written before kubelet restart is readable after restart.",
			runTestFile:  storageutils.TestKubeletRestartsAndRestoresMount,
			runTestBlock: storageutils.TestKubeletRestartsAndRestoresMap,
		},
		{
			testItStmt: "Should test that pv used in a pod that is deleted while the kubelet is down cleans up when the kubelet returns.",
			// File test is covered by subpath testsuite
			runTestBlock: storageutils.TestVolumeUnmapsFromDeletedPod,
		},
		{
			testItStmt: "Should test that pv used in a pod that is force deleted while the kubelet is down cleans up when the kubelet returns.",
			// File test is covered by subpath testsuite
			runTestBlock: storageutils.TestVolumeUnmapsFromForceDeletedPod,
		},
	}

	for _, test := range singlePodTests {
		func(t singlePodTest) {
			if (pattern.VolMode == v1.PersistentVolumeBlock && t.runTestBlock != nil) ||
				(pattern.VolMode == v1.PersistentVolumeFilesystem && t.runTestFile != nil) {
				ginkgo.It(t.testItStmt, func(ctx context.Context) {
					init(ctx, nil)
					ginkgo.DeferCleanup(cleanup)

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
					l.pod, err = e2epod.CreateSecPodWithNodeSelection(ctx, l.cs, &podConfig, f.Timeouts.PodStart)
					framework.ExpectNoError(err, "While creating pods for kubelet restart test")

					if pattern.VolMode == v1.PersistentVolumeBlock && t.runTestBlock != nil {
						t.runTestBlock(ctx, l.cs, l.config.Framework, l.pod, e2epod.VolumeMountPath1)
					}
					if pattern.VolMode == v1.PersistentVolumeFilesystem && t.runTestFile != nil {
						t.runTestFile(ctx, l.cs, l.config.Framework, l.pod, e2epod.VolumeMountPath1)
					}
				})
			}
		}(test)
	}
	type multiplePodTestBody func(ctx context.Context, c clientset.Interface, f *framework.Framework, pod1, pod2 *v1.Pod)
	type multiplePodTest struct {
		testItStmt            string
		changeSELinuxContexts bool
		runTestFile           multiplePodTestBody
	}
	multiplePodTests := []multiplePodTest{
		{
			testItStmt: "Should test that pv used in a pod that is deleted while the kubelet is down is usable by a new pod when kubelet returns",
			runTestFile: func(ctx context.Context, c clientset.Interface, f *framework.Framework, pod1, pod2 *v1.Pod) {
				storageutils.TestVolumeUnmountsFromDeletedPodWithForceOption(ctx, c, f, pod1, false, false, pod2, e2epod.VolumeMountPath1)
			},
		},
		{
			testItStmt: "Should test that pv used in a pod that is force deleted while the kubelet is down is usable by a new pod when kubelet returns",
			runTestFile: func(ctx context.Context, c clientset.Interface, f *framework.Framework, pod1, pod2 *v1.Pod) {
				storageutils.TestVolumeUnmountsFromDeletedPodWithForceOption(ctx, c, f, pod1, true, false, pod2, e2epod.VolumeMountPath1)
			},
		},
		{
			testItStmt:            "Should test that pv used in a pod that is deleted while the kubelet is down is usable by a new pod with a different SELinux context when kubelet returns",
			changeSELinuxContexts: true,
			runTestFile: func(ctx context.Context, c clientset.Interface, f *framework.Framework, pod1, pod2 *v1.Pod) {
				storageutils.TestVolumeUnmountsFromDeletedPodWithForceOption(ctx, c, f, pod1, false, false, pod2, e2epod.VolumeMountPath1)
			},
		},
		{
			testItStmt:            "Should test that pv used in a pod that is force deleted while the kubelet is down is usable by a new pod with a different SELinux context when kubelet returns",
			changeSELinuxContexts: true,
			runTestFile: func(ctx context.Context, c clientset.Interface, f *framework.Framework, pod1, pod2 *v1.Pod) {
				storageutils.TestVolumeUnmountsFromDeletedPodWithForceOption(ctx, c, f, pod1, true, false, pod2, e2epod.VolumeMountPath1)
			},
		},
	}

	for _, test := range multiplePodTests {
		func(t multiplePodTest) {
			if pattern.VolMode == v1.PersistentVolumeFilesystem && t.runTestFile != nil {
				f.It(t.testItStmt, feature.SELinux, func(ctx context.Context) {
					init(ctx, []v1.PersistentVolumeAccessMode{v1.ReadWriteOncePod})
					ginkgo.DeferCleanup(cleanup)

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
					l.pod, err = e2epod.CreateSecPodWithNodeSelection(ctx, l.cs, &podConfig, f.Timeouts.PodStart)
					framework.ExpectNoError(err, "While creating pods for kubelet restart test")
					if t.changeSELinuxContexts {
						// Different than e2epv.SELinuxLabel
						podConfig.SeLinuxLabel = &v1.SELinuxOptions{Level: "s0:c98,c99"}
					}
					pod2, err := e2epod.MakeSecPod(&podConfig)
					// Instantly schedule the second pod on the same node as the first one.
					pod2.Spec.NodeName = l.pod.Spec.NodeName
					framework.ExpectNoError(err, "While creating second pod for kubelet restart test")
					if pattern.VolMode == v1.PersistentVolumeFilesystem && t.runTestFile != nil {
						t.runTestFile(ctx, l.cs, l.config.Framework, l.pod, pod2)
					}
				})
			}
		}(test)
	}
}
