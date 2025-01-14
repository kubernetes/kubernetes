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
	"context"
	"fmt"
	"strconv"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	errors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	storageutils "k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
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
	tsInfo storageframework.TestSuiteInfo
}

var _ storageframework.TestSuite = &fsGroupChangePolicyTestSuite{}

// InitCustomFsGroupChangePolicyTestSuite returns fsGroupChangePolicyTestSuite that implements TestSuite interface
func InitCustomFsGroupChangePolicyTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &fsGroupChangePolicyTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "fsgroupchangepolicy",
			TestPatterns: patterns,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
		},
	}
}

// InitFsGroupChangePolicyTestSuite returns fsGroupChangePolicyTestSuite that implements TestSuite interface
func InitFsGroupChangePolicyTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.DefaultFsDynamicPV,
	}
	return InitCustomFsGroupChangePolicyTestSuite(patterns)
}

func (s *fsGroupChangePolicyTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return s.tsInfo
}

func (s *fsGroupChangePolicyTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	skipVolTypePatterns(pattern, driver, storageframework.NewVolTypeMap(storageframework.CSIInlineVolume, storageframework.GenericEphemeralVolume))
	dInfo := driver.GetDriverInfo()
	if !dInfo.Capabilities[storageframework.CapFsGroup] {
		e2eskipper.Skipf("Driver %q does not support FsGroup - skipping", dInfo.Name)
	}

	if pattern.VolMode == v1.PersistentVolumeBlock {
		e2eskipper.Skipf("Test does not support non-filesystem volume mode - skipping")
	}

	if pattern.VolType != storageframework.DynamicPV {
		e2eskipper.Skipf("Suite %q does not support %v", s.tsInfo.Name, pattern.VolType)
	}

	_, ok := driver.(storageframework.DynamicPVTestDriver)
	if !ok {
		e2eskipper.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, pattern.VolType)
	}
}

func (s *fsGroupChangePolicyTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	type local struct {
		config   *storageframework.PerTestConfig
		driver   storageframework.TestDriver
		resource *storageframework.VolumeResource
	}
	var l local

	// Beware that it also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewFrameworkWithCustomTimeouts("fsgroupchangepolicy", storageframework.GetDriverTimeouts(driver))
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	init := func(ctx context.Context) {
		e2eskipper.SkipIfNodeOSDistroIs("windows")
		l = local{}
		l.driver = driver
		l.config = driver.PrepareTest(ctx, f)
	}

	cleanup := func(ctx context.Context) {
		var errs []error
		if l.resource != nil {
			if err := l.resource.CleanupResource(ctx); err != nil {
				errs = append(errs, err)
			}
			l.resource = nil
		}

		framework.ExpectNoError(errors.NewAggregate(errs), "while cleanup resource")
	}

	rwopAccessMode := v1.ReadWriteOncePod

	tests := []struct {
		name                              string // Test case name
		podfsGroupChangePolicy            string // 'Always' or 'OnRootMismatch'
		initialPodFsGroup                 int    // FsGroup of the initial pod
		changedRootDirFileOwnership       int    // Change the ownership of the file in the root directory (/mnt/volume1/file1), as part of the initial pod
		changedSubDirFileOwnership        int    // Change the ownership of the file in the sub directory (/mnt/volume1/subdir/file2), as part of the initial pod
		secondPodFsGroup                  int    // FsGroup of the second pod
		finalExpectedRootDirFileOwnership int    // Final expected ownership of the file in the root directory (/mnt/volume1/file1), as part of the second pod
		finalExpectedSubDirFileOwnership  int    // Final expected ownership of the file in the sub directory (/mnt/volume1/subdir/file2), as part of the second pod
		// Whether the test can run for drivers that support volumeMountGroup capability.
		// For CSI drivers that support volumeMountGroup:
		// * OnRootMismatch policy is not supported.
		// * It may not be possible to chgrp after mounting a volume.
		supportsVolumeMountGroup bool
		volumeAccessMode         *v1.PersistentVolumeAccessMode
	}{
		// Test cases for 'Always' policy
		{
			name:                              "pod created with an initial fsgroup, new pod fsgroup applied to volume contents",
			podfsGroupChangePolicy:            "Always",
			initialPodFsGroup:                 1000,
			secondPodFsGroup:                  2000,
			finalExpectedRootDirFileOwnership: 2000,
			finalExpectedSubDirFileOwnership:  2000,
			supportsVolumeMountGroup:          true,
		},
		{
			name:                              "rwop pod created with an initial fsgroup, new pod fsgroup applied to volume contents",
			podfsGroupChangePolicy:            "Always",
			initialPodFsGroup:                 1000,
			secondPodFsGroup:                  2000,
			finalExpectedRootDirFileOwnership: 2000,
			finalExpectedSubDirFileOwnership:  2000,
			supportsVolumeMountGroup:          true,
			volumeAccessMode:                  &rwopAccessMode,
		},
		{
			name:                              "pod created with an initial fsgroup, volume contents ownership changed via chgrp in first pod, new pod with same fsgroup applied to the volume contents",
			podfsGroupChangePolicy:            "Always",
			initialPodFsGroup:                 1000,
			changedRootDirFileOwnership:       2000,
			changedSubDirFileOwnership:        3000,
			secondPodFsGroup:                  1000,
			finalExpectedRootDirFileOwnership: 1000,
			finalExpectedSubDirFileOwnership:  1000,
		},
		{
			name:                              "pod created with an initial fsgroup, volume contents ownership changed via chgrp in first pod, new pod with different fsgroup applied to the volume contents",
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
			name:                              "pod created with an initial fsgroup, volume contents ownership changed via chgrp in first pod, new pod with same fsgroup skips ownership changes to the volume contents",
			podfsGroupChangePolicy:            "OnRootMismatch",
			initialPodFsGroup:                 1000,
			changedRootDirFileOwnership:       2000,
			changedSubDirFileOwnership:        3000,
			secondPodFsGroup:                  1000,
			finalExpectedRootDirFileOwnership: 2000,
			finalExpectedSubDirFileOwnership:  3000,
		},
		{
			name:                              "pod created with an initial fsgroup, volume contents ownership changed via chgrp in first pod, new pod with different fsgroup applied to the volume contents",
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
		ginkgo.It(testCaseName, func(ctx context.Context) {
			dInfo := driver.GetDriverInfo()
			policy := v1.PodFSGroupChangePolicy(test.podfsGroupChangePolicy)

			if dInfo.Capabilities[storageframework.CapVolumeMountGroup] &&
				!test.supportsVolumeMountGroup {
				e2eskipper.Skipf("Driver %q supports VolumeMountGroup, which is incompatible with this test - skipping", dInfo.Name)
			}

			init(ctx)
			testVolumeSizeRange := s.GetTestSuiteInfo().SupportedSizeRange
			if test.volumeAccessMode != nil {
				accessModes := []v1.PersistentVolumeAccessMode{*test.volumeAccessMode}
				l.resource = storageframework.CreateVolumeResourceWithAccessModes(ctx, l.driver, l.config, pattern, testVolumeSizeRange, accessModes, nil)
			} else {
				l.resource = storageframework.CreateVolumeResource(ctx, l.driver, l.config, pattern, testVolumeSizeRange)
			}
			ginkgo.DeferCleanup(cleanup)
			podConfig := e2epod.Config{
				NS:                     f.Namespace.Name,
				NodeSelection:          l.config.ClientNodeSelection,
				PVCs:                   []*v1.PersistentVolumeClaim{l.resource.Pvc},
				FsGroup:                utilpointer.Int64Ptr(int64(test.initialPodFsGroup)),
				PodFSGroupChangePolicy: &policy,
			}
			// Create initial pod and create files in root and sub-directory and verify ownership.
			pod := createPodAndVerifyContentGid(ctx, l.config.Framework, &podConfig, true /* createInitialFiles */, "" /* expectedRootDirFileOwnership */, "" /* expectedSubDirFileOwnership */)

			// Change the ownership of files in the initial pod.
			if test.changedRootDirFileOwnership != 0 {
				ginkgo.By(fmt.Sprintf("Changing the root directory file ownership to %s", strconv.Itoa(test.changedRootDirFileOwnership)))
				storageutils.ChangeFilePathGidInPod(ctx, f, rootDirFilePath, strconv.Itoa(test.changedRootDirFileOwnership), pod)
			}

			if test.changedSubDirFileOwnership != 0 {
				ginkgo.By(fmt.Sprintf("Changing the sub-directory file ownership to %s", strconv.Itoa(test.changedSubDirFileOwnership)))
				storageutils.ChangeFilePathGidInPod(ctx, f, subDirFilePath, strconv.Itoa(test.changedSubDirFileOwnership), pod)
			}

			ginkgo.By(fmt.Sprintf("Deleting Pod %s/%s", pod.Namespace, pod.Name))
			framework.ExpectNoError(e2epod.DeletePodWithWait(ctx, f.ClientSet, pod))

			// Create a second pod with existing volume and verify the contents ownership.
			podConfig.FsGroup = utilpointer.Int64Ptr(int64(test.secondPodFsGroup))
			pod = createPodAndVerifyContentGid(ctx, l.config.Framework, &podConfig, false /* createInitialFiles */, strconv.Itoa(test.finalExpectedRootDirFileOwnership), strconv.Itoa(test.finalExpectedSubDirFileOwnership))
			ginkgo.By(fmt.Sprintf("Deleting Pod %s/%s", pod.Namespace, pod.Name))
			framework.ExpectNoError(e2epod.DeletePodWithWait(ctx, f.ClientSet, pod))
		})
	}
}

func createPodAndVerifyContentGid(ctx context.Context, f *framework.Framework, podConfig *e2epod.Config, createInitialFiles bool, expectedRootDirFileOwnership, expectedSubDirFileOwnership string) *v1.Pod {
	podFsGroup := strconv.FormatInt(*podConfig.FsGroup, 10)
	ginkgo.By(fmt.Sprintf("Creating Pod in namespace %s with fsgroup %s", podConfig.NS, podFsGroup))
	pod, err := e2epod.CreateSecPodWithNodeSelection(ctx, f.ClientSet, podConfig, f.Timeouts.PodStart)
	framework.ExpectNoError(err)
	framework.Logf("Pod %s/%s started successfully", pod.Namespace, pod.Name)

	if createInitialFiles {
		ginkgo.By(fmt.Sprintf("Creating a sub-directory and file, and verifying their ownership is %s", podFsGroup))
		cmd := fmt.Sprintf("touch %s", rootDirFilePath)
		var err error
		_, _, err = e2epod.ExecShellInPodWithFullOutput(ctx, f, pod.Name, cmd)
		framework.ExpectNoError(err)
		storageutils.VerifyFilePathGidInPod(ctx, f, rootDirFilePath, podFsGroup, pod)

		cmd = fmt.Sprintf("mkdir %s", subdir)
		_, _, err = e2epod.ExecShellInPodWithFullOutput(ctx, f, pod.Name, cmd)
		framework.ExpectNoError(err)
		cmd = fmt.Sprintf("touch %s", subDirFilePath)
		_, _, err = e2epod.ExecShellInPodWithFullOutput(ctx, f, pod.Name, cmd)
		framework.ExpectNoError(err)
		storageutils.VerifyFilePathGidInPod(ctx, f, subDirFilePath, podFsGroup, pod)
		return pod
	}

	// Verify existing contents of the volume
	ginkgo.By(fmt.Sprintf("Verifying the ownership of root directory file is %s", expectedRootDirFileOwnership))
	storageutils.VerifyFilePathGidInPod(ctx, f, rootDirFilePath, expectedRootDirFileOwnership, pod)
	ginkgo.By(fmt.Sprintf("Verifying the ownership of sub directory file is %s", expectedSubDirFileOwnership))
	storageutils.VerifyFilePathGidInPod(ctx, f, subDirFilePath, expectedSubDirFileOwnership, pod)
	return pod
}
