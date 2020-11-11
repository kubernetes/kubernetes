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

package testsuites

import (
	"fmt"
	"strconv"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	errors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	storageutils "k8s.io/kubernetes/test/e2e/storage/utils"
	utilpointer "k8s.io/utils/pointer"
)

const (
	rootDir         = "/mnt/volume1"
	rootDirFile     = "file1"
	rootDirFilePath = rootDir + "/" + rootDirFile
	subdir          = "/mnt/volume1/subdir"
	subDirFile      = "file2"
	subDirFilePath  = subdir + "/" + subDirFile
)

type fsGroupChangePolicyTestSuite struct {
	tsInfo TestSuiteInfo
}

var _ TestSuite = &fsGroupChangePolicyTestSuite{}

// InitFsGroupChangePolicyTestSuite returns fsGroupChangePolicyTestSuite that implements TestSuite interface
func InitFsGroupChangePolicyTestSuite() TestSuite {
	return &fsGroupChangePolicyTestSuite{
		tsInfo: TestSuiteInfo{
			Name: "fsgroupchangepolicy",
			TestPatterns: []testpatterns.TestPattern{
				testpatterns.DefaultFsDynamicPV,
			},
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
		},
	}
}

func (s *fsGroupChangePolicyTestSuite) GetTestSuiteInfo() TestSuiteInfo {
	return s.tsInfo
}

func (s *fsGroupChangePolicyTestSuite) SkipRedundantSuite(driver TestDriver, pattern testpatterns.TestPattern) {
	skipVolTypePatterns(pattern, driver, testpatterns.NewVolTypeMap(testpatterns.CSIInlineVolume, testpatterns.GenericEphemeralVolume))
}

func (s *fsGroupChangePolicyTestSuite) DefineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	type local struct {
		config        *PerTestConfig
		driverCleanup func()
		driver        TestDriver
		resource      *VolumeResource
	}
	var l local
	ginkgo.BeforeEach(func() {
		dInfo := driver.GetDriverInfo()
		if !dInfo.Capabilities[CapFsGroup] {
			e2eskipper.Skipf("Driver %q does not support FsGroup - skipping", dInfo.Name)
		}

		if pattern.VolMode == v1.PersistentVolumeBlock {
			e2eskipper.Skipf("Test does not support non-filesystem volume mode - skipping")
		}

		if pattern.VolType != testpatterns.DynamicPV {
			e2eskipper.Skipf("Suite %q does not support %v", s.tsInfo.Name, pattern.VolType)
		}

		_, ok := driver.(DynamicPVTestDriver)
		if !ok {
			e2eskipper.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, pattern.VolType)
		}
	})

	// This intentionally comes after checking the preconditions because it
	// registers its own BeforeEach which creates the namespace. Beware that it
	// also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewDefaultFramework("fsgroupchangepolicy")

	init := func() {
		e2eskipper.SkipIfNodeOSDistroIs("windows")
		l = local{}
		l.driver = driver
		l.config, l.driverCleanup = driver.PrepareTest(f)
		testVolumeSizeRange := s.GetTestSuiteInfo().SupportedSizeRange
		l.resource = CreateVolumeResource(l.driver, l.config, pattern, testVolumeSizeRange)
	}

	cleanup := func() {
		var errs []error
		if l.resource != nil {
			if err := l.resource.CleanupResource(); err != nil {
				errs = append(errs, err)
			}
			l.resource = nil
		}

		if l.driverCleanup != nil {
			errs = append(errs, tryFunc(l.driverCleanup))
			l.driverCleanup = nil
		}

		framework.ExpectNoError(errors.NewAggregate(errs), "while cleanup resource")
	}

	tests := []struct {
		name                              string // Test case name
		podfsGroupChangePolicy            string // 'Always' or 'OnRootMismatch'
		initialPodFsGroup                 int    // FsGroup of the initial pod
		changedRootDirFileOwnership       int    // Change the ownership of the file in the root directory (/mnt/volume1/file1), as part of the initial pod
		changedSubDirFileOwnership        int    // Change the ownership of the file in the sub directory (/mnt/volume1/subdir/file2), as part of the initial pod
		secondPodFsGroup                  int    // FsGroup of the second pod
		finalExpectedRootDirFileOwnership int    // Final expcted ownership of the file in the root directory (/mnt/volume1/file1), as part of the second pod
		finalExpectedSubDirFileOwnership  int    // Final expcted ownership of the file in the sub directory (/mnt/volume1/subdir/file2), as part of the second pod
	}{
		// Test cases for 'Always' policy
		{
			name:                              "pod created with an initial fsgroup, new pod fsgroup applied to volume contents",
			podfsGroupChangePolicy:            "Always",
			initialPodFsGroup:                 1000,
			secondPodFsGroup:                  2000,
			finalExpectedRootDirFileOwnership: 2000,
			finalExpectedSubDirFileOwnership:  2000,
		},
		{
			name:                              "pod created with an initial fsgroup, volume contents ownership changed in first pod, new pod with same fsgroup applied to the volume contents",
			podfsGroupChangePolicy:            "Always",
			initialPodFsGroup:                 1000,
			changedRootDirFileOwnership:       2000,
			changedSubDirFileOwnership:        3000,
			secondPodFsGroup:                  1000,
			finalExpectedRootDirFileOwnership: 1000,
			finalExpectedSubDirFileOwnership:  1000,
		},
		{
			name:                              "pod created with an initial fsgroup, volume contents ownership changed in first pod, new pod with different fsgroup applied to the volume contents",
			podfsGroupChangePolicy:            "Always",
			initialPodFsGroup:                 1000,
			changedRootDirFileOwnership:       2000,
			changedSubDirFileOwnership:        3000,
			secondPodFsGroup:                  4000,
			finalExpectedRootDirFileOwnership: 4000,
			finalExpectedSubDirFileOwnership:  4000,
		},
		// Test cases for 'OnRootMismatch' policy
		{
			name:                              "pod created with an initial fsgroup, new pod fsgroup applied to volume contents",
			podfsGroupChangePolicy:            "OnRootMismatch",
			initialPodFsGroup:                 1000,
			secondPodFsGroup:                  2000,
			finalExpectedRootDirFileOwnership: 2000,
			finalExpectedSubDirFileOwnership:  2000,
		},
		{
			name:                              "pod created with an initial fsgroup, volume contents ownership changed in first pod, new pod with same fsgroup skips ownership changes to the volume contents",
			podfsGroupChangePolicy:            "OnRootMismatch",
			initialPodFsGroup:                 1000,
			changedRootDirFileOwnership:       2000,
			changedSubDirFileOwnership:        3000,
			secondPodFsGroup:                  1000,
			finalExpectedRootDirFileOwnership: 2000,
			finalExpectedSubDirFileOwnership:  3000,
		},
		{
			name:                              "pod created with an initial fsgroup, volume contents ownership changed in first pod, new pod with different fsgroup applied to the volume contents",
			podfsGroupChangePolicy:            "OnRootMismatch",
			initialPodFsGroup:                 1000,
			changedRootDirFileOwnership:       2000,
			changedSubDirFileOwnership:        3000,
			secondPodFsGroup:                  4000,
			finalExpectedRootDirFileOwnership: 4000,
			finalExpectedSubDirFileOwnership:  4000,
		},
	}

	for _, t := range tests {
		test := t
		testCaseName := fmt.Sprintf("(%s)[LinuxOnly], %s", test.podfsGroupChangePolicy, test.name)
		ginkgo.It(testCaseName, func() {
			init()
			defer cleanup()

			policy := v1.PodFSGroupChangePolicy(test.podfsGroupChangePolicy)
			podConfig := e2epod.Config{
				NS:                     f.Namespace.Name,
				NodeSelection:          l.config.ClientNodeSelection,
				PVCs:                   []*v1.PersistentVolumeClaim{l.resource.Pvc},
				FsGroup:                utilpointer.Int64Ptr(int64(test.initialPodFsGroup)),
				PodFSGroupChangePolicy: &policy,
			}
			// Create initial pod and create files in root and sub-directory and verify ownership.
			pod := createPodAndVerifyContentGid(l.config.Framework, &podConfig, true /* createInitialFiles */, "" /* expectedRootDirFileOwnership */, "" /* expectedSubDirFileOwnership */)

			// Change the ownership of files in the initial pod.
			if test.changedRootDirFileOwnership != 0 {
				ginkgo.By(fmt.Sprintf("Changing the root directory file ownership to %s", strconv.Itoa(test.changedRootDirFileOwnership)))
				storageutils.ChangeFilePathGidInPod(f, rootDirFilePath, strconv.Itoa(test.changedRootDirFileOwnership), pod)
			}

			if test.changedSubDirFileOwnership != 0 {
				ginkgo.By(fmt.Sprintf("Changing the sub-directory file ownership to %s", strconv.Itoa(test.changedSubDirFileOwnership)))
				storageutils.ChangeFilePathGidInPod(f, subDirFilePath, strconv.Itoa(test.changedSubDirFileOwnership), pod)
			}

			ginkgo.By(fmt.Sprintf("Deleting Pod %s/%s", pod.Namespace, pod.Name))
			framework.ExpectNoError(e2epod.DeletePodWithWait(f.ClientSet, pod))

			// Create a second pod with existing volume and verify the contents ownership.
			podConfig.FsGroup = utilpointer.Int64Ptr(int64(test.secondPodFsGroup))
			pod = createPodAndVerifyContentGid(l.config.Framework, &podConfig, false /* createInitialFiles */, strconv.Itoa(test.finalExpectedRootDirFileOwnership), strconv.Itoa(test.finalExpectedSubDirFileOwnership))
			ginkgo.By(fmt.Sprintf("Deleting Pod %s/%s", pod.Namespace, pod.Name))
			framework.ExpectNoError(e2epod.DeletePodWithWait(f.ClientSet, pod))
		})
	}
}

func createPodAndVerifyContentGid(f *framework.Framework, podConfig *e2epod.Config, createInitialFiles bool, expectedRootDirFileOwnership, expectedSubDirFileOwnership string) *v1.Pod {
	podFsGroup := strconv.FormatInt(*podConfig.FsGroup, 10)
	ginkgo.By(fmt.Sprintf("Creating Pod in namespace %s with fsgroup %s", podConfig.NS, podFsGroup))
	pod, err := e2epod.CreateSecPodWithNodeSelection(f.ClientSet, podConfig, framework.PodStartTimeout)
	framework.ExpectNoError(err)
	framework.Logf("Pod %s/%s started successfully", pod.Namespace, pod.Name)

	if createInitialFiles {
		ginkgo.By(fmt.Sprintf("Creating a sub-directory and file, and verifying their ownership is %s", podFsGroup))
		cmd := fmt.Sprintf("touch %s", rootDirFilePath)
		var err error
		_, _, err = storageutils.PodExec(f, pod, cmd)
		framework.ExpectNoError(err)
		storageutils.VerifyFilePathGidInPod(f, rootDirFilePath, podFsGroup, pod)

		cmd = fmt.Sprintf("mkdir %s", subdir)
		_, _, err = storageutils.PodExec(f, pod, cmd)
		framework.ExpectNoError(err)
		cmd = fmt.Sprintf("touch %s", subDirFilePath)
		_, _, err = storageutils.PodExec(f, pod, cmd)
		framework.ExpectNoError(err)
		storageutils.VerifyFilePathGidInPod(f, subDirFilePath, podFsGroup, pod)
		return pod
	}

	// Verify existing contents of the volume
	ginkgo.By(fmt.Sprintf("Verifying the ownership of root directory file is %s", expectedRootDirFileOwnership))
	storageutils.VerifyFilePathGidInPod(f, rootDirFilePath, expectedRootDirFileOwnership, pod)
	ginkgo.By(fmt.Sprintf("Verifying the ownership of sub directory file is %s", expectedSubDirFileOwnership))
	storageutils.VerifyFilePathGidInPod(f, subDirFilePath, expectedSubDirFileOwnership, pod)
	return pod
}
