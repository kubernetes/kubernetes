/*
Copyright 2018 The Kubernetes Authors.

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
	"sync"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
)

// StorageClassTest represents parameters to be used by provisioning tests.
// Not all parameters are used by all tests.
type StorageClassTest struct {
	Client               clientset.Interface
	Claim                *v1.PersistentVolumeClaim
	SourceClaim          *v1.PersistentVolumeClaim
	Class                *storagev1.StorageClass
	Name                 string
	CloudProviders       []string
	Provisioner          string
	StorageClassName     string
	Parameters           map[string]string
	DelayBinding         bool
	ClaimSize            string
	ExpectedSize         string
	PvCheck              func(claim *v1.PersistentVolumeClaim)
	VolumeMode           v1.PersistentVolumeMode
	AllowVolumeExpansion bool
}

type provisioningTestSuite struct {
	tsInfo TestSuiteInfo
}

var _ TestSuite = &provisioningTestSuite{}

// InitProvisioningTestSuite returns provisioningTestSuite that implements TestSuite interface
func InitProvisioningTestSuite() TestSuite {
	return &provisioningTestSuite{
		tsInfo: TestSuiteInfo{
			Name: "provisioning",
			TestPatterns: []testpatterns.TestPattern{
				testpatterns.DefaultFsDynamicPV,
				testpatterns.BlockVolModeDynamicPV,
				testpatterns.NtfsDynamicPV,
			},
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
		},
	}
}

func (p *provisioningTestSuite) GetTestSuiteInfo() TestSuiteInfo {
	return p.tsInfo
}

func (p *provisioningTestSuite) SkipRedundantSuite(driver TestDriver, pattern testpatterns.TestPattern) {
}

func (p *provisioningTestSuite) DefineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	type local struct {
		config        *PerTestConfig
		driverCleanup func()

		testCase  *StorageClassTest
		cs        clientset.Interface
		pvc       *v1.PersistentVolumeClaim
		sourcePVC *v1.PersistentVolumeClaim
		sc        *storagev1.StorageClass

		intreeOps   opCounts
		migratedOps opCounts
	}
	var (
		dInfo   = driver.GetDriverInfo()
		dDriver DynamicPVTestDriver
		l       local
	)

	ginkgo.BeforeEach(func() {
		// Check preconditions.
		if pattern.VolType != testpatterns.DynamicPV {
			e2eskipper.Skipf("Suite %q does not support %v", p.tsInfo.Name, pattern.VolType)
		}
		if pattern.VolMode == v1.PersistentVolumeBlock && !dInfo.Capabilities[CapBlock] {
			e2eskipper.Skipf("Driver %q does not support block volumes - skipping", dInfo.Name)
		}

		ok := false
		dDriver, ok = driver.(DynamicPVTestDriver)
		if !ok {
			e2eskipper.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, pattern.VolType)
		}
	})

	// This intentionally comes after checking the preconditions because it
	// registers its own BeforeEach which creates the namespace. Beware that it
	// also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewDefaultFramework("provisioning")

	init := func() {
		l = local{}

		// Now do the more expensive test initialization.
		l.config, l.driverCleanup = driver.PrepareTest(f)
		l.intreeOps, l.migratedOps = getMigrationVolumeOpCounts(f.ClientSet, dInfo.InTreePluginName)
		l.cs = l.config.Framework.ClientSet
		testVolumeSizeRange := p.GetTestSuiteInfo().SupportedSizeRange
		driverVolumeSizeRange := dDriver.GetDriverInfo().SupportedSizeRange
		claimSize, err := getSizeRangesIntersection(testVolumeSizeRange, driverVolumeSizeRange)
		framework.ExpectNoError(err, "determine intersection of test size range %+v and driver size range %+v", testVolumeSizeRange, driverVolumeSizeRange)

		l.sc = dDriver.GetDynamicProvisionStorageClass(l.config, pattern.FsType)
		if l.sc == nil {
			e2eskipper.Skipf("Driver %q does not define Dynamic Provision StorageClass - skipping", dInfo.Name)
		}
		l.pvc = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			ClaimSize:        claimSize,
			StorageClassName: &(l.sc.Name),
			VolumeMode:       &pattern.VolMode,
		}, l.config.Framework.Namespace.Name)
		l.sourcePVC = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			ClaimSize:        claimSize,
			StorageClassName: &(l.sc.Name),
			VolumeMode:       &pattern.VolMode,
		}, l.config.Framework.Namespace.Name)
		framework.Logf("In creating storage class object and pvc objects for driver - sc: %v, pvc: %v, src-pvc: %v", l.sc, l.pvc, l.sourcePVC)
		l.testCase = &StorageClassTest{
			Client:       l.config.Framework.ClientSet,
			Claim:        l.pvc,
			SourceClaim:  l.sourcePVC,
			Class:        l.sc,
			ClaimSize:    claimSize,
			ExpectedSize: claimSize,
			VolumeMode:   pattern.VolMode,
		}
	}

	cleanup := func() {
		err := tryFunc(l.driverCleanup)
		l.driverCleanup = nil
		framework.ExpectNoError(err, "while cleaning up driver")

		validateMigrationVolumeOpCounts(f.ClientSet, dInfo.InTreePluginName, l.intreeOps, l.migratedOps)
	}

	ginkgo.It("should provision storage with mount options", func() {
		if dInfo.SupportedMountOption == nil {
			e2eskipper.Skipf("Driver %q does not define supported mount option - skipping", dInfo.Name)
		}
		if pattern.VolMode == v1.PersistentVolumeBlock {
			e2eskipper.Skipf("Block volumes do not support mount options - skipping")
		}

		init()
		defer cleanup()

		l.testCase.Class.MountOptions = dInfo.SupportedMountOption.Union(dInfo.RequiredMountOption).List()
		l.testCase.PvCheck = func(claim *v1.PersistentVolumeClaim) {
			PVWriteReadSingleNodeCheck(l.cs, claim, l.config.ClientNodeSelection)
		}
		l.testCase.TestDynamicProvisioning()
	})

	ginkgo.It("should provision storage with snapshot data source [Feature:VolumeSnapshotDataSource]", func() {
		if !dInfo.Capabilities[CapSnapshotDataSource] {
			e2eskipper.Skipf("Driver %q does not support populate data from snapshot - skipping", dInfo.Name)
		}

		sDriver, ok := driver.(SnapshottableTestDriver)
		if !ok {
			framework.Failf("Driver %q has CapSnapshotDataSource but does not implement SnapshottableTestDriver", dInfo.Name)
		}

		init()
		defer cleanup()

		dc := l.config.Framework.DynamicClient
		vsc := sDriver.GetSnapshotClass(l.config)
		testConfig := convertTestConfig(l.config)
		expectedContent := fmt.Sprintf("Hello from namespace %s", f.Namespace.Name)
		dataSource, cleanupFunc := prepareSnapshotDataSourceForProvisioning(f, testConfig, l.cs, dc, l.pvc, l.sc, vsc, pattern.VolMode, expectedContent)
		defer cleanupFunc()

		l.pvc.Spec.DataSource = dataSource
		l.testCase.PvCheck = func(claim *v1.PersistentVolumeClaim) {
			ginkgo.By("checking whether the created volume has the pre-populated data")
			tests := []e2evolume.Test{
				{
					Volume:          *createVolumeSource(claim.Name, false /* readOnly */),
					Mode:            pattern.VolMode,
					File:            "index.html",
					ExpectedContent: expectedContent,
				},
			}
			e2evolume.TestVolumeClientSlow(f, testConfig, nil, "", tests)
		}
		l.testCase.TestDynamicProvisioning()
	})

	ginkgo.It("should provision storage with pvc data source", func() {
		if !dInfo.Capabilities[CapPVCDataSource] {
			e2eskipper.Skipf("Driver %q does not support cloning - skipping", dInfo.Name)
		}
		init()
		defer cleanup()

		testConfig := convertTestConfig(l.config)
		expectedContent := fmt.Sprintf("Hello from namespace %s", f.Namespace.Name)
		dataSource, dataSourceCleanup := preparePVCDataSourceForProvisioning(f, testConfig, l.cs, l.sourcePVC, l.sc, pattern.VolMode, expectedContent)
		defer dataSourceCleanup()

		l.pvc.Spec.DataSource = dataSource
		l.testCase.PvCheck = func(claim *v1.PersistentVolumeClaim) {
			ginkgo.By("checking whether the created volume has the pre-populated data")
			tests := []e2evolume.Test{
				{
					Volume:          *createVolumeSource(claim.Name, false /* readOnly */),
					Mode:            pattern.VolMode,
					File:            "index.html",
					ExpectedContent: expectedContent,
				},
			}
			e2evolume.TestVolumeClientSlow(f, testConfig, nil, "", tests)
		}
		l.testCase.TestDynamicProvisioning()
	})

	ginkgo.It("should provision storage with pvc data source in parallel [Slow]", func() {
		// Test cloning a single volume multiple times.
		if !dInfo.Capabilities[CapPVCDataSource] {
			e2eskipper.Skipf("Driver %q does not support cloning - skipping", dInfo.Name)
		}
		if pattern.VolMode == v1.PersistentVolumeBlock && !dInfo.Capabilities[CapBlock] {
			e2eskipper.Skipf("Driver %q does not support block volumes - skipping", dInfo.Name)
		}

		init()
		defer cleanup()

		testConfig := convertTestConfig(l.config)
		expectedContent := fmt.Sprintf("Hello from namespace %s", f.Namespace.Name)
		dataSource, dataSourceCleanup := preparePVCDataSourceForProvisioning(f, testConfig, l.cs, l.sourcePVC, l.sc, pattern.VolMode, expectedContent)
		defer dataSourceCleanup()
		l.pvc.Spec.DataSource = dataSource

		var wg sync.WaitGroup
		for i := 0; i < 5; i++ {
			wg.Add(1)
			go func(i int) {
				defer ginkgo.GinkgoRecover()
				defer wg.Done()
				ginkgo.By(fmt.Sprintf("Cloning volume nr. %d", i))
				// Each go routine must have its own pod prefix
				myTestConfig := testConfig
				myTestConfig.Prefix = fmt.Sprintf("%s-%d", myTestConfig.Prefix, i)

				// Each go routine must have its own testCase copy to store their claim
				myTestCase := *l.testCase
				myTestCase.Claim = myTestCase.Claim.DeepCopy()
				myTestCase.Class = nil // Do not create/delete the storage class in TestDynamicProvisioning, it already exists.
				myTestCase.PvCheck = func(claim *v1.PersistentVolumeClaim) {
					ginkgo.By(fmt.Sprintf("checking whether the created volume %d has the pre-populated data", i))
					tests := []e2evolume.Test{
						{
							Volume:          *createVolumeSource(claim.Name, false /* readOnly */),
							Mode:            pattern.VolMode,
							File:            "index.html",
							ExpectedContent: expectedContent,
						},
					}
					e2evolume.TestVolumeClientSlow(f, myTestConfig, nil, "", tests)
				}
				myTestCase.TestDynamicProvisioning()
			}(i)
		}
		wg.Wait()
	})
}

// TestDynamicProvisioning tests dynamic provisioning with specified StorageClassTest
func (t StorageClassTest) TestDynamicProvisioning() *v1.PersistentVolume {
	client := t.Client
	gomega.Expect(client).NotTo(gomega.BeNil(), "StorageClassTest.Client is required")
	claim := t.Claim
	gomega.Expect(claim).NotTo(gomega.BeNil(), "StorageClassTest.Claim is required")
	class := t.Class

	var err error
	if class != nil {
		framework.ExpectEqual(*claim.Spec.StorageClassName, class.Name)
		ginkgo.By("creating a StorageClass " + class.Name)
		_, err = client.StorageV1().StorageClasses().Create(context.TODO(), class, metav1.CreateOptions{})
		// The "should provision storage with snapshot data source" test already has created the class.
		// TODO: make class creation optional and remove the IsAlreadyExists exception
		framework.ExpectEqual(err == nil || apierrors.IsAlreadyExists(err), true)
		class, err = client.StorageV1().StorageClasses().Get(context.TODO(), class.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		defer func() {
			framework.Logf("deleting storage class %s", class.Name)
			framework.ExpectNoError(client.StorageV1().StorageClasses().Delete(context.TODO(), class.Name, metav1.DeleteOptions{}))
		}()
	}

	ginkgo.By("creating a claim")
	claim, err = client.CoreV1().PersistentVolumeClaims(claim.Namespace).Create(context.TODO(), claim, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	defer func() {
		framework.Logf("deleting claim %q/%q", claim.Namespace, claim.Name)
		// typically this claim has already been deleted
		err = client.CoreV1().PersistentVolumeClaims(claim.Namespace).Delete(context.TODO(), claim.Name, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.Failf("Error deleting claim %q. Error: %v", claim.Name, err)
		}
	}()

	// Run the checker
	if t.PvCheck != nil {
		t.PvCheck(claim)
	}

	pv := t.checkProvisioning(client, claim, class)

	ginkgo.By(fmt.Sprintf("deleting claim %q/%q", claim.Namespace, claim.Name))
	framework.ExpectNoError(client.CoreV1().PersistentVolumeClaims(claim.Namespace).Delete(context.TODO(), claim.Name, metav1.DeleteOptions{}))

	// Wait for the PV to get deleted if reclaim policy is Delete. (If it's
	// Retain, there's no use waiting because the PV won't be auto-deleted and
	// it's expected for the caller to do it.) Technically, the first few delete
	// attempts may fail, as the volume is still attached to a node because
	// kubelet is slowly cleaning up the previous pod, however it should succeed
	// in a couple of minutes. Wait 20 minutes to recover from random cloud
	// hiccups.
	if pv != nil && pv.Spec.PersistentVolumeReclaimPolicy == v1.PersistentVolumeReclaimDelete {
		ginkgo.By(fmt.Sprintf("deleting the claim's PV %q", pv.Name))
		framework.ExpectNoError(e2epv.WaitForPersistentVolumeDeleted(client, pv.Name, 5*time.Second, 20*time.Minute))
	}

	return pv
}

// getBoundPV returns a PV details.
func getBoundPV(client clientset.Interface, pvc *v1.PersistentVolumeClaim) (*v1.PersistentVolume, error) {
	// Get new copy of the claim
	claim, err := client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(context.TODO(), pvc.Name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}

	// Get the bound PV
	pv, err := client.CoreV1().PersistentVolumes().Get(context.TODO(), claim.Spec.VolumeName, metav1.GetOptions{})
	return pv, err
}

// checkProvisioning verifies that the claim is bound and has the correct properities
func (t StorageClassTest) checkProvisioning(client clientset.Interface, claim *v1.PersistentVolumeClaim, class *storagev1.StorageClass) *v1.PersistentVolume {
	err := e2epv.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, client, claim.Namespace, claim.Name, framework.Poll, framework.ClaimProvisionTimeout)
	framework.ExpectNoError(err)

	ginkgo.By("checking the claim")
	pv, err := getBoundPV(client, claim)
	framework.ExpectNoError(err)

	// Check sizes
	expectedCapacity := resource.MustParse(t.ExpectedSize)
	pvCapacity := pv.Spec.Capacity[v1.ResourceName(v1.ResourceStorage)]
	framework.ExpectEqual(pvCapacity.Value(), expectedCapacity.Value(), "pvCapacity is not equal to expectedCapacity")

	requestedCapacity := resource.MustParse(t.ClaimSize)
	claimCapacity := claim.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	framework.ExpectEqual(claimCapacity.Value(), requestedCapacity.Value(), "claimCapacity is not equal to requestedCapacity")

	// Check PV properties
	ginkgo.By("checking the PV")

	// Every access mode in PV should be in PVC
	gomega.Expect(pv.Spec.AccessModes).NotTo(gomega.BeZero())
	for _, pvMode := range pv.Spec.AccessModes {
		found := false
		for _, pvcMode := range claim.Spec.AccessModes {
			if pvMode == pvcMode {
				found = true
				break
			}
		}
		framework.ExpectEqual(found, true)
	}

	framework.ExpectEqual(pv.Spec.ClaimRef.Name, claim.ObjectMeta.Name)
	framework.ExpectEqual(pv.Spec.ClaimRef.Namespace, claim.ObjectMeta.Namespace)
	if class == nil {
		framework.ExpectEqual(pv.Spec.PersistentVolumeReclaimPolicy, v1.PersistentVolumeReclaimDelete)
	} else {
		framework.ExpectEqual(pv.Spec.PersistentVolumeReclaimPolicy, *class.ReclaimPolicy)
		framework.ExpectEqual(pv.Spec.MountOptions, class.MountOptions)
	}
	if claim.Spec.VolumeMode != nil {
		gomega.Expect(pv.Spec.VolumeMode).NotTo(gomega.BeNil())
		framework.ExpectEqual(*pv.Spec.VolumeMode, *claim.Spec.VolumeMode)
	}
	return pv
}

// PVWriteReadSingleNodeCheck checks that a PV retains data on a single node
// and returns the PV.
//
// It starts two pods:
// - The first pod writes 'hello word' to the /mnt/test (= the volume) on one node.
// - The second pod runs grep 'hello world' on /mnt/test on the same node.
//
// The node is selected by Kubernetes when scheduling the first
// pod. It's then selected via its name for the second pod.
//
// If both succeed, Kubernetes actually allocated something that is
// persistent across pods.
//
// This is a common test that can be called from a StorageClassTest.PvCheck.
func PVWriteReadSingleNodeCheck(client clientset.Interface, claim *v1.PersistentVolumeClaim, node e2epod.NodeSelection) *v1.PersistentVolume {
	ginkgo.By(fmt.Sprintf("checking the created volume is writable on node %+v", node))
	command := "echo 'hello world' > /mnt/test/data"
	pod := StartInPodWithVolume(client, claim.Namespace, claim.Name, "pvc-volume-tester-writer", command, node)
	defer func() {
		// pod might be nil now.
		StopPod(client, pod)
	}()
	framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespaceSlow(client, pod.Name, pod.Namespace))
	runningPod, err := client.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "get pod")
	actualNodeName := runningPod.Spec.NodeName
	StopPod(client, pod)
	pod = nil // Don't stop twice.

	// Get a new copy of the PV
	e2evolume, err := getBoundPV(client, claim)
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("checking the created volume has the correct mount options, is readable and retains data on the same node %q", actualNodeName))
	command = "grep 'hello world' /mnt/test/data"

	// We give the second pod the additional responsibility of checking the volume has
	// been mounted with the PV's mount options, if the PV was provisioned with any
	for _, option := range e2evolume.Spec.MountOptions {
		// Get entry, get mount options at 6th word, replace brackets with commas
		command += fmt.Sprintf(" && ( mount | grep 'on /mnt/test' | awk '{print $6}' | sed 's/^(/,/; s/)$/,/' | grep -q ,%s, )", option)
	}
	command += " || (mount | grep 'on /mnt/test'; false)"

	if framework.NodeOSDistroIs("windows") {
		command = "select-string 'hello world' /mnt/test/data"
	}
	RunInPodWithVolume(client, claim.Namespace, claim.Name, "pvc-volume-tester-reader", command, e2epod.NodeSelection{Name: actualNodeName})

	return e2evolume
}

// PVMultiNodeCheck checks that a PV retains data when moved between nodes.
//
// It starts these pods:
// - The first pod writes 'hello word' to the /mnt/test (= the volume) on one node.
// - The second pod runs grep 'hello world' on /mnt/test on another node.
//
// The first node is selected by Kubernetes when scheduling the first pod. The second pod uses the same criteria, except that a special anti-affinity
// for the first node gets added. This test can only pass if the cluster has more than one
// suitable node. The caller has to ensure that.
//
// If all succeeds, Kubernetes actually allocated something that is
// persistent across pods and across nodes.
//
// This is a common test that can be called from a StorageClassTest.PvCheck.
func PVMultiNodeCheck(client clientset.Interface, claim *v1.PersistentVolumeClaim, node e2epod.NodeSelection) {
	framework.ExpectEqual(node.Name, "", "this test only works when not locked onto a single node")

	var pod *v1.Pod
	defer func() {
		// passing pod = nil is okay.
		StopPod(client, pod)
	}()

	ginkgo.By(fmt.Sprintf("checking the created volume is writable on node %+v", node))
	command := "echo 'hello world' > /mnt/test/data"
	pod = StartInPodWithVolume(client, claim.Namespace, claim.Name, "pvc-writer-node1", command, node)
	framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespaceSlow(client, pod.Name, pod.Namespace))
	runningPod, err := client.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "get pod")
	actualNodeName := runningPod.Spec.NodeName
	StopPod(client, pod)
	pod = nil // Don't stop twice.

	// Add node-anti-affinity.
	secondNode := node
	e2epod.SetAntiAffinity(&secondNode, actualNodeName)
	ginkgo.By(fmt.Sprintf("checking the created volume is readable and retains data on another node %+v", secondNode))
	command = "grep 'hello world' /mnt/test/data"
	if framework.NodeOSDistroIs("windows") {
		command = "select-string 'hello world' /mnt/test/data"
	}
	pod = StartInPodWithVolume(client, claim.Namespace, claim.Name, "pvc-reader-node2", command, secondNode)
	framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespaceSlow(client, pod.Name, pod.Namespace))
	runningPod, err = client.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "get pod")
	framework.ExpectNotEqual(runningPod.Spec.NodeName, actualNodeName, "second pod should have run on a different node")
	StopPod(client, pod)
	pod = nil
}

// TestBindingWaitForFirstConsumerMultiPVC tests the binding with WaitForFirstConsumer mode
func (t StorageClassTest) TestBindingWaitForFirstConsumerMultiPVC(claims []*v1.PersistentVolumeClaim, nodeSelector map[string]string, expectUnschedulable bool) ([]*v1.PersistentVolume, *v1.Node) {
	var err error
	framework.ExpectNotEqual(len(claims), 0)
	namespace := claims[0].Namespace

	ginkgo.By("creating a storage class " + t.Class.Name)
	class, err := t.Client.StorageV1().StorageClasses().Create(context.TODO(), t.Class, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	defer func() {
		err = deleteStorageClass(t.Client, class.Name)
		framework.ExpectNoError(err, "While deleting storage class")
	}()

	ginkgo.By("creating claims")
	var claimNames []string
	var createdClaims []*v1.PersistentVolumeClaim
	for _, claim := range claims {
		c, err := t.Client.CoreV1().PersistentVolumeClaims(claim.Namespace).Create(context.TODO(), claim, metav1.CreateOptions{})
		claimNames = append(claimNames, c.Name)
		createdClaims = append(createdClaims, c)
		framework.ExpectNoError(err)
	}
	defer func() {
		errors := map[string]error{}
		for _, claim := range createdClaims {
			err := e2epv.DeletePersistentVolumeClaim(t.Client, claim.Name, claim.Namespace)
			if err != nil {
				errors[claim.Name] = err
			}
		}
		if len(errors) > 0 {
			for claimName, err := range errors {
				framework.Logf("Failed to delete PVC: %s due to error: %v", claimName, err)
			}
		}
	}()

	// Wait for ClaimProvisionTimeout (across all PVCs in parallel) and make sure the phase did not become Bound i.e. the Wait errors out
	ginkgo.By("checking the claims are in pending state")
	err = e2epv.WaitForPersistentVolumeClaimsPhase(v1.ClaimBound, t.Client, namespace, claimNames, 2*time.Second /* Poll */, framework.ClaimProvisionShortTimeout, true)
	framework.ExpectError(err)
	verifyPVCsPending(t.Client, createdClaims)

	ginkgo.By("creating a pod referring to the claims")
	// Create a pod referring to the claim and wait for it to get to running
	var pod *v1.Pod
	if expectUnschedulable {
		pod, err = e2epod.CreateUnschedulablePod(t.Client, namespace, nodeSelector, createdClaims, true /* isPrivileged */, "" /* command */)
	} else {
		pod, err = e2epod.CreatePod(t.Client, namespace, nil /* nodeSelector */, createdClaims, true /* isPrivileged */, "" /* command */)
	}
	framework.ExpectNoError(err)
	defer func() {
		e2epod.DeletePodOrFail(t.Client, pod.Namespace, pod.Name)
		e2epod.WaitForPodToDisappear(t.Client, pod.Namespace, pod.Name, labels.Everything(), framework.Poll, framework.PodDeleteTimeout)
	}()
	if expectUnschedulable {
		// Verify that no claims are provisioned.
		verifyPVCsPending(t.Client, createdClaims)
		return nil, nil
	}

	// collect node details
	node, err := t.Client.CoreV1().Nodes().Get(context.TODO(), pod.Spec.NodeName, metav1.GetOptions{})
	framework.ExpectNoError(err)

	ginkgo.By("re-checking the claims to see they bound")
	var pvs []*v1.PersistentVolume
	for _, claim := range createdClaims {
		// Get new copy of the claim
		claim, err = t.Client.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(context.TODO(), claim.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		// make sure claim did bind
		err = e2epv.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, t.Client, claim.Namespace, claim.Name, framework.Poll, framework.ClaimProvisionTimeout)
		framework.ExpectNoError(err)

		pv, err := t.Client.CoreV1().PersistentVolumes().Get(context.TODO(), claim.Spec.VolumeName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		pvs = append(pvs, pv)
	}
	framework.ExpectEqual(len(pvs), len(createdClaims))
	return pvs, node
}

// RunInPodWithVolume runs a command in a pod with given claim mounted to /mnt directory.
// It starts, checks, collects output and stops it.
func RunInPodWithVolume(c clientset.Interface, ns, claimName, podName, command string, node e2epod.NodeSelection) {
	pod := StartInPodWithVolume(c, ns, claimName, podName, command, node)
	defer StopPod(c, pod)
	framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespaceSlow(c, pod.Name, pod.Namespace))
}

// StartInPodWithVolume starts a command in a pod with given claim mounted to /mnt directory
// The caller is responsible for checking the pod and deleting it.
func StartInPodWithVolume(c clientset.Interface, ns, claimName, podName, command string, node e2epod.NodeSelection) *v1.Pod {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: podName + "-",
			Labels: map[string]string{
				"app": podName,
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "volume-tester",
					Image:   e2evolume.GetTestImage(framework.BusyBoxImage),
					Command: e2evolume.GenerateScriptCmd(command),
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "my-volume",
							MountPath: "/mnt/test",
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name: "my-volume",
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: claimName,
							ReadOnly:  false,
						},
					},
				},
			},
		},
	}

	e2epod.SetNodeSelection(&pod.Spec, node)
	pod, err := c.CoreV1().Pods(ns).Create(context.TODO(), pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "Failed to create pod: %v", err)
	return pod
}

// StopPod first tries to log the output of the pod's container, then deletes the pod and
// waits for that to succeed.
func StopPod(c clientset.Interface, pod *v1.Pod) {
	if pod == nil {
		return
	}
	body, err := c.CoreV1().Pods(pod.Namespace).GetLogs(pod.Name, &v1.PodLogOptions{}).Do(context.TODO()).Raw()
	if err != nil {
		framework.Logf("Error getting logs for pod %s: %v", pod.Name, err)
	} else {
		framework.Logf("Pod %s has the following logs: %s", pod.Name, body)
	}
	e2epod.DeletePodWithWait(c, pod)
}

func verifyPVCsPending(client clientset.Interface, pvcs []*v1.PersistentVolumeClaim) {
	for _, claim := range pvcs {
		// Get new copy of the claim
		claim, err := client.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(context.TODO(), claim.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(claim.Status.Phase, v1.ClaimPending)
	}
}

func prepareSnapshotDataSourceForProvisioning(
	f *framework.Framework,
	config e2evolume.TestConfig,
	client clientset.Interface,
	dynamicClient dynamic.Interface,
	initClaim *v1.PersistentVolumeClaim,
	class *storagev1.StorageClass,
	snapshotClass *unstructured.Unstructured,
	mode v1.PersistentVolumeMode,
	injectContent string,
) (*v1.TypedLocalObjectReference, func()) {
	var err error
	if class != nil {
		ginkgo.By("[Initialize dataSource]creating a StorageClass " + class.Name)
		_, err = client.StorageV1().StorageClasses().Create(context.TODO(), class, metav1.CreateOptions{})
		framework.ExpectNoError(err)
	}

	ginkgo.By("[Initialize dataSource]creating a initClaim")
	updatedClaim, err := client.CoreV1().PersistentVolumeClaims(initClaim.Namespace).Create(context.TODO(), initClaim, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	// write namespace to the /mnt/test (= the volume).
	tests := []e2evolume.Test{
		{
			Volume:          *createVolumeSource(updatedClaim.Name, false /* readOnly */),
			Mode:            mode,
			File:            "index.html",
			ExpectedContent: injectContent,
		},
	}
	e2evolume.InjectContent(f, config, nil, "", tests)

	ginkgo.By("[Initialize dataSource]creating a SnapshotClass")
	snapshotClass, err = dynamicClient.Resource(SnapshotClassGVR).Create(context.TODO(), snapshotClass, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.By("[Initialize dataSource]creating a snapshot")
	snapshot := getSnapshot(updatedClaim.Name, updatedClaim.Namespace, snapshotClass.GetName())
	snapshot, err = dynamicClient.Resource(SnapshotGVR).Namespace(updatedClaim.Namespace).Create(context.TODO(), snapshot, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	WaitForSnapshotReady(dynamicClient, snapshot.GetNamespace(), snapshot.GetName(), framework.Poll, framework.SnapshotCreateTimeout)
	framework.ExpectNoError(err)

	ginkgo.By("[Initialize dataSource]checking the snapshot")
	// Get new copy of the snapshot
	snapshot, err = dynamicClient.Resource(SnapshotGVR).Namespace(snapshot.GetNamespace()).Get(context.TODO(), snapshot.GetName(), metav1.GetOptions{})
	framework.ExpectNoError(err)
	group := "snapshot.storage.k8s.io"
	dataSourceRef := &v1.TypedLocalObjectReference{
		APIGroup: &group,
		Kind:     "VolumeSnapshot",
		Name:     snapshot.GetName(),
	}

	cleanupFunc := func() {
		framework.Logf("deleting snapshot %q/%q", snapshot.GetNamespace(), snapshot.GetName())
		err = dynamicClient.Resource(SnapshotGVR).Namespace(updatedClaim.Namespace).Delete(context.TODO(), snapshot.GetName(), metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.Failf("Error deleting snapshot %q. Error: %v", snapshot.GetName(), err)
		}

		framework.Logf("deleting initClaim %q/%q", updatedClaim.Namespace, updatedClaim.Name)
		err = client.CoreV1().PersistentVolumeClaims(updatedClaim.Namespace).Delete(context.TODO(), updatedClaim.Name, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.Failf("Error deleting initClaim %q. Error: %v", updatedClaim.Name, err)
		}

		framework.Logf("deleting SnapshotClass %s", snapshotClass.GetName())
		framework.ExpectNoError(dynamicClient.Resource(SnapshotClassGVR).Delete(context.TODO(), snapshotClass.GetName(), metav1.DeleteOptions{}))
	}

	return dataSourceRef, cleanupFunc
}

func preparePVCDataSourceForProvisioning(
	f *framework.Framework,
	config e2evolume.TestConfig,
	client clientset.Interface,
	source *v1.PersistentVolumeClaim,
	class *storagev1.StorageClass,
	mode v1.PersistentVolumeMode,
	injectContent string,
) (*v1.TypedLocalObjectReference, func()) {
	var err error
	if class != nil {
		ginkgo.By("[Initialize dataSource]creating a StorageClass " + class.Name)
		class, err = client.StorageV1().StorageClasses().Create(context.TODO(), class, metav1.CreateOptions{})
		framework.ExpectNoError(err)
	}

	ginkgo.By("[Initialize dataSource]creating a source PVC")
	sourcePVC, err := client.CoreV1().PersistentVolumeClaims(source.Namespace).Create(context.TODO(), source, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	tests := []e2evolume.Test{
		{
			Volume:          *createVolumeSource(sourcePVC.Name, false /* readOnly */),
			Mode:            mode,
			File:            "index.html",
			ExpectedContent: injectContent,
		},
	}
	e2evolume.InjectContent(f, config, nil, "", tests)

	dataSourceRef := &v1.TypedLocalObjectReference{
		Kind: "PersistentVolumeClaim",
		Name: sourcePVC.GetName(),
	}

	cleanupFunc := func() {
		framework.Logf("deleting source PVC %q/%q", sourcePVC.Namespace, sourcePVC.Name)
		err := client.CoreV1().PersistentVolumeClaims(sourcePVC.Namespace).Delete(context.TODO(), sourcePVC.Name, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.Failf("Error deleting source PVC %q. Error: %v", sourcePVC.Name, err)
		}
		if class != nil {
			framework.Logf("deleting class %q", class.Name)
			err := client.StorageV1().StorageClasses().Delete(context.TODO(), class.Name, metav1.DeleteOptions{})
			if err != nil && !apierrors.IsNotFound(err) {
				framework.Failf("Error deleting storage class %q. Error: %v", class.Name, err)
			}
		}
	}

	return dataSourceRef, cleanupFunc
}
