/*
Copyright 2023 The Kubernetes Authors.

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
	"time"

	"github.com/onsi/gomega"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	// pvCSIDeletionProtectionFinalizer is the finalizer added to protect PV deletion for csi volumes.
	pvCSIDeletionProtectionFinalizer = "external-provisioner.volume.kubernetes.io/finalizer"

	// pvInTreeDeletionProtectionFinalizer is the finalizer added to protect PV deletion for in-tree volumes.
	pvInTreeDeletionProtectionFinalizer = "kubernetes.io/pv-controller"
)

type VolumeDeletionTest struct {
	Client            clientset.Interface
	Timeouts          *framework.TimeoutContext
	Claim             *v1.PersistentVolumeClaim
	Class             *storagev1.StorageClass
	Name              string
	CloudProviders    []string
	Provisioner       string
	Parameters        map[string]string
	DelayBinding      bool
	ClaimSize         string
	ExpectedSize      string
	ExpectedFinalizer string
}

type pvDeletionProtectionTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

// InitTestSuite returns pvDeletionProtectionTestSuite that implements TestSuite interface
// using custom test patterns
func InitTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &pvDeletionProtectionTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "volume deletion",
			TestPatterns: patterns,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
		},
	}
}

func InitPvDeletionProtectionTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.VolumeDelete,
	}
	return InitTestSuite(patterns)
}

func (p *pvDeletionProtectionTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	// Check preconditions.
	if pattern.FeatureTag != "[Feature:HonorPVReclaimPolicy]" {
		e2eskipper.Skipf("Suite %q, limiting only to HonorPVReclaimPolicy feature test, current: %q", p.tsInfo.Name, pattern.FeatureTag)
	}
	if pattern.VolType != storageframework.DynamicPV {
		e2eskipper.Skipf("Suite %q does not support %v", p.tsInfo.Name, pattern.VolType)
	}
	dInfo := driver.GetDriverInfo()
	if dInfo.Name == "nfs" {
		e2eskipper.Skipf("Driver %s is not a in-tree volume", dInfo.Name)
	}
	if pattern.VolMode == v1.PersistentVolumeBlock && !dInfo.Capabilities[storageframework.CapBlock] {
		e2eskipper.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, pattern.VolMode)
	}
}

func (p *pvDeletionProtectionTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return p.tsInfo
}

func (p *pvDeletionProtectionTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	type local struct {
		config *storageframework.PerTestConfig

		testCase *VolumeDeletionTest
		cs       clientset.Interface
		pvc      *v1.PersistentVolumeClaim
		sc       *storagev1.StorageClass
	}
	var (
		dInfo   = driver.GetDriverInfo()
		dDriver storageframework.DynamicPVTestDriver
		l       local
	)

	// Beware that it also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewFrameworkWithCustomTimeouts("volumedeletion", storageframework.GetDriverTimeouts(driver))
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	init := func(ctx context.Context) {
		l = local{}
		dDriver, _ = driver.(storageframework.DynamicPVTestDriver)
		// Now do the more expensive test initialization.
		l.config = driver.PrepareTest(ctx, f)
		l.cs = l.config.Framework.ClientSet
		testVolumeSizeRange := p.GetTestSuiteInfo().SupportedSizeRange
		driverVolumeSizeRange := dDriver.GetDriverInfo().SupportedSizeRange
		claimSize, err := utils.GetSizeRangesIntersection(testVolumeSizeRange, driverVolumeSizeRange)
		framework.ExpectNoError(err, "determine intersection of test size range %+v and driver size range %+v", testVolumeSizeRange, driverVolumeSizeRange)

		l.sc = dDriver.GetDynamicProvisionStorageClass(ctx, l.config, pattern.FsType)
		if l.sc == nil {
			e2eskipper.Skipf("Driver %q does not define Dynamic Provision StorageClass - skipping", dInfo.Name)
		}
		// explicitly set the volume binding mode to immediate.
		immediateBinding := storagev1.VolumeBindingImmediate
		l.sc.VolumeBindingMode = &immediateBinding
		framework.Logf("GetDynamicProvisionStorageClass %q", l.sc)
		l.pvc = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			ClaimSize:        claimSize,
			StorageClassName: &(l.sc.Name),
			VolumeMode:       &pattern.VolMode,
		}, l.config.Framework.Namespace.Name)

		l.testCase = &VolumeDeletionTest{
			Client:       l.config.Framework.ClientSet,
			Claim:        l.pvc,
			Timeouts:     f.Timeouts,
			Class:        l.sc,
			Provisioner:  l.sc.Provisioner,
			ClaimSize:    claimSize,
			ExpectedSize: claimSize,
		}

		// Different finalizers are expected for in-tree and csi volumes
		if len(dDriver.GetDriverInfo().InTreePluginName) != 0 && dInfo.Name != "nfs" {
			// In Tree
			l.testCase.ExpectedFinalizer = pvInTreeDeletionProtectionFinalizer
		} else {
			// CSI
			l.testCase.ExpectedFinalizer = pvCSIDeletionProtectionFinalizer
		}
	}

	ginkgo.It("delete pv prior", func(ctx context.Context) {
		init(ctx)
		SetupStorageClass(ctx, l.testCase.Client, l.testCase.Class)
		l.testCase.TestVolumeDeletion(ctx)
	})
}

func (t VolumeDeletionTest) TestVolumeDeletion(ctx context.Context) *v1.PersistentVolume {
	var err error
	client := t.Client
	gomega.Expect(client).NotTo(gomega.BeNil(), "VolumeDeletionTest.Client is required")
	claim := t.Claim
	gomega.Expect(claim).NotTo(gomega.BeNil(), "VolumeDeletionTest.Claim is required")
	gomega.Expect(claim.GenerateName).NotTo(gomega.BeEmpty(), "VolumeDeletionTest.Claim.GenerateName must not be empty")
	class := t.Class
	gomega.Expect(class).NotTo(gomega.BeNil(), "VolumeDeletionTest.Class is required")
	class, err = client.StorageV1().StorageClasses().Get(ctx, class.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "VolumeDeletionTest.Class "+class.Name+" couldn't be fetched from the cluster")
	framework.Logf("class retrieved %q", class.Name)

	ginkgo.By(fmt.Sprintf("creating claim=%+v", claim))
	claim, err = client.CoreV1().PersistentVolumeClaims(claim.Namespace).Create(ctx, claim, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	defer func() {
		framework.Logf("deleting claim %q/%q", claim.Namespace, claim.Name)
		// typically this claim has already been deleted
		err = client.CoreV1().PersistentVolumeClaims(claim.Namespace).Delete(ctx, claim.Name, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.Failf("Error deleting claim %q. Error: %v", claim.Name, err)
		}
	}()

	// ensure that the claim refers to the provisioned StorageClass
	gomega.Expect(*claim.Spec.StorageClassName).To(gomega.Equal(class.Name))

	pv := t.checkVolumeDeletion(ctx, client, claim, class)

	return pv
}

func (t VolumeDeletionTest) checkVolumeDeletion(ctx context.Context, client clientset.Interface, claim *v1.PersistentVolumeClaim, class *storagev1.StorageClass) *v1.PersistentVolume {
	err := e2epv.WaitForPersistentVolumeClaimPhase(ctx, v1.ClaimBound, client, claim.Namespace, claim.Name, framework.Poll, t.Timeouts.ClaimProvision)
	framework.ExpectNoError(err)

	// Retrieve the PVC
	pvcBound, err := client.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(ctx, claim.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)

	// Get the bound PV
	pv, err := client.CoreV1().PersistentVolumes().Get(ctx, pvcBound.Spec.VolumeName, metav1.GetOptions{})
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Wait for finalizer %s to be added to pv %s", t.ExpectedFinalizer, pv.Name))
	pv, err = e2epv.WaitForPVFinalizer(ctx, client, pv.Name, t.ExpectedFinalizer, 1*time.Millisecond, 1*time.Minute)
	framework.ExpectNoError(err)
	framework.Logf("finalizer %q found on pv %q", t.ExpectedFinalizer, pv.Name)
	framework.Logf("Persistent Volume: %+v", pv)

	ginkgo.By("Delete pv")
	err = e2epv.DeletePersistentVolume(ctx, client, pv.Name)
	framework.ExpectNoError(err)
	framework.Logf("pv %q deleted", pv.Name)

	ginkgo.By("Delete pvc")
	err = e2epv.DeletePersistentVolumeClaim(ctx, client, claim.Name, claim.Namespace)
	framework.ExpectNoError(err)
	framework.Logf("pvc %q/%q deleted", claim.Name, claim.Namespace)

	ginkgo.By("Waiting for the pvc to be deleted")
	framework.ExpectNoError(e2epv.WaitForPersistentVolumeClaimDeleted(ctx, client, claim.Namespace, claim.Name, 2*time.Second, 60*time.Second),
		"Failed to delete PVC", claim.Name)

	ginkgo.By("Waiting for the pv to be deleted")
	framework.ExpectNoError(e2epv.WaitForPersistentVolumeDeleted(ctx, client, pv.Name, 2*time.Second, 60*time.Second),
		"Failed to delete PV ", pv.Name)
	framework.Logf("pv %q removed from the API server", pv.Name)

	return pv
}
