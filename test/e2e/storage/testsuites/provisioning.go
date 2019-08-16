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
	"fmt"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/framework/volume"
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
			name: "provisioning",
			testPatterns: []testpatterns.TestPattern{
				testpatterns.DefaultFsDynamicPV,
				testpatterns.NtfsDynamicPV,
			},
		},
	}
}

func (p *provisioningTestSuite) getTestSuiteInfo() TestSuiteInfo {
	return p.tsInfo
}

func (p *provisioningTestSuite) skipRedundantSuite(driver TestDriver, pattern testpatterns.TestPattern) {
}

func (p *provisioningTestSuite) defineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	type local struct {
		config      *PerTestConfig
		testCleanup func()

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
			framework.Skipf("Suite %q does not support %v", p.tsInfo.name, pattern.VolType)
		}
		ok := false
		dDriver, ok = driver.(DynamicPVTestDriver)
		if !ok {
			framework.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, pattern.VolType)
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
		l.config, l.testCleanup = driver.PrepareTest(f)
		l.intreeOps, l.migratedOps = getMigrationVolumeOpCounts(f.ClientSet, dInfo.InTreePluginName)
		l.cs = l.config.Framework.ClientSet
		claimSize := dDriver.GetClaimSize()
		l.sc = dDriver.GetDynamicProvisionStorageClass(l.config, pattern.FsType)
		if l.sc == nil {
			framework.Skipf("Driver %q does not define Dynamic Provision StorageClass - skipping", dInfo.Name)
		}
		l.pvc = framework.MakePersistentVolumeClaim(framework.PersistentVolumeClaimConfig{
			ClaimSize:        claimSize,
			StorageClassName: &(l.sc.Name),
		}, l.config.Framework.Namespace.Name)
		l.sourcePVC = framework.MakePersistentVolumeClaim(framework.PersistentVolumeClaimConfig{
			ClaimSize:        claimSize,
			StorageClassName: &(l.sc.Name),
		}, l.config.Framework.Namespace.Name)
		e2elog.Logf("In creating storage class object and pvc objects for driver - sc: %v, pvc: %v, src-pvc: %v", l.sc, l.pvc, l.sourcePVC)
		l.testCase = &StorageClassTest{
			Client:       l.config.Framework.ClientSet,
			Claim:        l.pvc,
			SourceClaim:  l.sourcePVC,
			Class:        l.sc,
			ClaimSize:    claimSize,
			ExpectedSize: claimSize,
		}
	}

	cleanup := func() {
		if l.testCleanup != nil {
			l.testCleanup()
			l.testCleanup = nil
		}

		validateMigrationVolumeOpCounts(f.ClientSet, dInfo.InTreePluginName, l.intreeOps, l.migratedOps)
	}

	ginkgo.It("should provision storage with mount options", func() {
		if dInfo.SupportedMountOption == nil {
			framework.Skipf("Driver %q does not define supported mount option - skipping", dInfo.Name)
		}

		init()
		defer cleanup()

		l.testCase.Class.MountOptions = dInfo.SupportedMountOption.Union(dInfo.RequiredMountOption).List()
		l.testCase.PvCheck = func(claim *v1.PersistentVolumeClaim) {
			PVWriteReadSingleNodeCheck(l.cs, claim, framework.NodeSelection{Name: l.config.ClientNodeName})
		}
		l.testCase.TestDynamicProvisioning()
	})

	ginkgo.It("should provision storage with snapshot data source [Feature:VolumeSnapshotDataSource]", func() {
		if !dInfo.Capabilities[CapSnapshotDataSource] {
			framework.Skipf("Driver %q does not support populate data from snapshot - skipping", dInfo.Name)
		}

		sDriver, ok := driver.(SnapshottableTestDriver)
		if !ok {
			e2elog.Failf("Driver %q has CapSnapshotDataSource but does not implement SnapshottableTestDriver", dInfo.Name)
		}

		init()
		defer cleanup()

		dc := l.config.Framework.DynamicClient
		vsc := sDriver.GetSnapshotClass(l.config)
		dataSource, cleanupFunc := prepareSnapshotDataSourceForProvisioning(framework.NodeSelection{Name: l.config.ClientNodeName}, l.cs, dc, l.pvc, l.sc, vsc)
		defer cleanupFunc()

		l.pvc.Spec.DataSource = dataSource
		l.testCase.PvCheck = func(claim *v1.PersistentVolumeClaim) {
			ginkgo.By("checking whether the created volume has the pre-populated data")
			command := fmt.Sprintf("grep '%s' /mnt/test/initialData", claim.Namespace)
			RunInPodWithVolume(l.cs, claim.Namespace, claim.Name, "pvc-snapshot-tester", command, framework.NodeSelection{Name: l.config.ClientNodeName})
		}
		l.testCase.TestDynamicProvisioning()
	})
	ginkgo.It("should provision storage with pvc data source [Feature:VolumePVCDataSource]", func() {
		if !dInfo.Capabilities[CapPVCDataSource] {
			framework.Skipf("Driver %q does not support cloning - skipping", dInfo.Name)
		}

		init()
		defer cleanup()

		dc := l.config.Framework.DynamicClient
		dataSource, dataSourceCleanup := preparePVCDataSourceForProvisioning(framework.NodeSelection{Name: l.config.ClientNodeName}, l.cs, dc, l.sourcePVC, l.sc)
		defer dataSourceCleanup()

		l.pvc.Spec.DataSource = dataSource
		l.testCase.PvCheck = func(claim *v1.PersistentVolumeClaim) {
			ginkgo.By("checking whether the created volume has the pre-populated data")
			command := fmt.Sprintf("grep '%s' /mnt/test/initialData", claim.Namespace)
			RunInPodWithVolume(l.cs, claim.Namespace, claim.Name, "pvc-datasource-tester", command, framework.NodeSelection{Name: l.config.ClientNodeName})
		}
		l.testCase.TestDynamicProvisioning()
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
		_, err = client.StorageV1().StorageClasses().Create(class)
		// The "should provision storage with snapshot data source" test already has created the class.
		// TODO: make class creation optional and remove the IsAlreadyExists exception
		framework.ExpectEqual(err == nil || apierrs.IsAlreadyExists(err), true)
		class, err = client.StorageV1().StorageClasses().Get(class.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		defer func() {
			e2elog.Logf("deleting storage class %s", class.Name)
			framework.ExpectNoError(client.StorageV1().StorageClasses().Delete(class.Name, nil))
		}()
	}

	ginkgo.By("creating a claim")
	claim, err = client.CoreV1().PersistentVolumeClaims(claim.Namespace).Create(claim)
	framework.ExpectNoError(err)
	defer func() {
		e2elog.Logf("deleting claim %q/%q", claim.Namespace, claim.Name)
		// typically this claim has already been deleted
		err = client.CoreV1().PersistentVolumeClaims(claim.Namespace).Delete(claim.Name, nil)
		if err != nil && !apierrs.IsNotFound(err) {
			e2elog.Failf("Error deleting claim %q. Error: %v", claim.Name, err)
		}
	}()

	// Run the checker
	if t.PvCheck != nil {
		t.PvCheck(claim)
	}

	pv := t.checkProvisioning(client, claim, class)

	ginkgo.By(fmt.Sprintf("deleting claim %q/%q", claim.Namespace, claim.Name))
	framework.ExpectNoError(client.CoreV1().PersistentVolumeClaims(claim.Namespace).Delete(claim.Name, nil))

	// Wait for the PV to get deleted if reclaim policy is Delete. (If it's
	// Retain, there's no use waiting because the PV won't be auto-deleted and
	// it's expected for the caller to do it.) Technically, the first few delete
	// attempts may fail, as the volume is still attached to a node because
	// kubelet is slowly cleaning up the previous pod, however it should succeed
	// in a couple of minutes. Wait 20 minutes to recover from random cloud
	// hiccups.
	if pv != nil && pv.Spec.PersistentVolumeReclaimPolicy == v1.PersistentVolumeReclaimDelete {
		ginkgo.By(fmt.Sprintf("deleting the claim's PV %q", pv.Name))
		framework.ExpectNoError(framework.WaitForPersistentVolumeDeleted(client, pv.Name, 5*time.Second, 20*time.Minute))
	}

	return pv
}

// checkProvisioning verifies that the claim is bound and has the correct properities
func (t StorageClassTest) checkProvisioning(client clientset.Interface, claim *v1.PersistentVolumeClaim, class *storagev1.StorageClass) *v1.PersistentVolume {
	err := framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, client, claim.Namespace, claim.Name, framework.Poll, framework.ClaimProvisionTimeout)
	framework.ExpectNoError(err)

	ginkgo.By("checking the claim")
	pv, err := framework.GetBoundPV(client, claim)
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
		gomega.Expect(found).To(gomega.BeTrue())
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
func PVWriteReadSingleNodeCheck(client clientset.Interface, claim *v1.PersistentVolumeClaim, node framework.NodeSelection) *v1.PersistentVolume {
	ginkgo.By(fmt.Sprintf("checking the created volume is writable on node %+v", node))
	command := "echo 'hello world' > /mnt/test/data"
	pod := StartInPodWithVolume(client, claim.Namespace, claim.Name, "pvc-volume-tester-writer", command, node)
	defer func() {
		// pod might be nil now.
		StopPod(client, pod)
	}()
	framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespaceSlow(client, pod.Name, pod.Namespace))
	runningPod, err := client.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "get pod")
	actualNodeName := runningPod.Spec.NodeName
	StopPod(client, pod)
	pod = nil // Don't stop twice.

	// Get a new copy of the PV
	volume, err := framework.GetBoundPV(client, claim)
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("checking the created volume has the correct mount options, is readable and retains data on the same node %q", actualNodeName))
	command = "grep 'hello world' /mnt/test/data"

	// We give the second pod the additional responsibility of checking the volume has
	// been mounted with the PV's mount options, if the PV was provisioned with any
	for _, option := range volume.Spec.MountOptions {
		// Get entry, get mount options at 6th word, replace brackets with commas
		command += fmt.Sprintf(" && ( mount | grep 'on /mnt/test' | awk '{print $6}' | sed 's/^(/,/; s/)$/,/' | grep -q ,%s, )", option)
	}
	command += " || (mount | grep 'on /mnt/test'; false)"

	if framework.NodeOSDistroIs("windows") {
		command = "select-string 'hello world' /mnt/test/data"
	}
	RunInPodWithVolume(client, claim.Namespace, claim.Name, "pvc-volume-tester-reader", command, framework.NodeSelection{Name: actualNodeName})

	return volume
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
func PVMultiNodeCheck(client clientset.Interface, claim *v1.PersistentVolumeClaim, node framework.NodeSelection) {
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
	runningPod, err := client.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "get pod")
	actualNodeName := runningPod.Spec.NodeName
	StopPod(client, pod)
	pod = nil // Don't stop twice.

	// Add node-anti-affinity.
	secondNode := node
	framework.SetAntiAffinity(&secondNode, actualNodeName)
	ginkgo.By(fmt.Sprintf("checking the created volume is readable and retains data on another node %+v", secondNode))
	command = "grep 'hello world' /mnt/test/data"
	if framework.NodeOSDistroIs("windows") {
		command = "select-string 'hello world' /mnt/test/data"
	}
	pod = StartInPodWithVolume(client, claim.Namespace, claim.Name, "pvc-reader-node2", command, secondNode)
	framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespaceSlow(client, pod.Name, pod.Namespace))
	runningPod, err = client.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "get pod")
	gomega.Expect(runningPod.Spec.NodeName).NotTo(gomega.Equal(actualNodeName), "second pod should have run on a different node")
	StopPod(client, pod)
	pod = nil
}

// TestBindingWaitForFirstConsumer tests the binding with WaitForFirstConsumer mode
func (t StorageClassTest) TestBindingWaitForFirstConsumer(nodeSelector map[string]string, expectUnschedulable bool) (*v1.PersistentVolume, *v1.Node) {
	pvs, node := t.TestBindingWaitForFirstConsumerMultiPVC([]*v1.PersistentVolumeClaim{t.Claim}, nodeSelector, expectUnschedulable)
	if pvs == nil {
		return nil, node
	}
	return pvs[0], node
}

// TestBindingWaitForFirstConsumerMultiPVC tests the binding with WaitForFirstConsumer mode
func (t StorageClassTest) TestBindingWaitForFirstConsumerMultiPVC(claims []*v1.PersistentVolumeClaim, nodeSelector map[string]string, expectUnschedulable bool) ([]*v1.PersistentVolume, *v1.Node) {
	var err error
	framework.ExpectNotEqual(len(claims), 0)
	namespace := claims[0].Namespace

	ginkgo.By("creating a storage class " + t.Class.Name)
	class, err := t.Client.StorageV1().StorageClasses().Create(t.Class)
	framework.ExpectNoError(err)
	defer deleteStorageClass(t.Client, class.Name)

	ginkgo.By("creating claims")
	var claimNames []string
	var createdClaims []*v1.PersistentVolumeClaim
	for _, claim := range claims {
		c, err := t.Client.CoreV1().PersistentVolumeClaims(claim.Namespace).Create(claim)
		claimNames = append(claimNames, c.Name)
		createdClaims = append(createdClaims, c)
		framework.ExpectNoError(err)
	}
	defer func() {
		var errors map[string]error
		for _, claim := range createdClaims {
			err := framework.DeletePersistentVolumeClaim(t.Client, claim.Name, claim.Namespace)
			if err != nil {
				errors[claim.Name] = err
			}
		}
		if len(errors) > 0 {
			for claimName, err := range errors {
				e2elog.Logf("Failed to delete PVC: %s due to error: %v", claimName, err)
			}
		}
	}()

	// Wait for ClaimProvisionTimeout (across all PVCs in parallel) and make sure the phase did not become Bound i.e. the Wait errors out
	ginkgo.By("checking the claims are in pending state")
	err = framework.WaitForPersistentVolumeClaimsPhase(v1.ClaimBound, t.Client, namespace, claimNames, 2*time.Second /* Poll */, framework.ClaimProvisionShortTimeout, true)
	framework.ExpectError(err)
	verifyPVCsPending(t.Client, createdClaims)

	ginkgo.By("creating a pod referring to the claims")
	// Create a pod referring to the claim and wait for it to get to running
	var pod *v1.Pod
	if expectUnschedulable {
		pod, err = framework.CreateUnschedulablePod(t.Client, namespace, nodeSelector, createdClaims, true /* isPrivileged */, "" /* command */)
	} else {
		pod, err = framework.CreatePod(t.Client, namespace, nil /* nodeSelector */, createdClaims, true /* isPrivileged */, "" /* command */)
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
	node, err := t.Client.CoreV1().Nodes().Get(pod.Spec.NodeName, metav1.GetOptions{})
	framework.ExpectNoError(err)

	ginkgo.By("re-checking the claims to see they binded")
	var pvs []*v1.PersistentVolume
	for _, claim := range createdClaims {
		// Get new copy of the claim
		claim, err = t.Client.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(claim.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		// make sure claim did bind
		err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, t.Client, claim.Namespace, claim.Name, framework.Poll, framework.ClaimProvisionTimeout)
		framework.ExpectNoError(err)

		pv, err := t.Client.CoreV1().PersistentVolumes().Get(claim.Spec.VolumeName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		pvs = append(pvs, pv)
	}
	framework.ExpectEqual(len(pvs), len(createdClaims))
	return pvs, node
}

// RunInPodWithVolume runs a command in a pod with given claim mounted to /mnt directory.
// It starts, checks, collects output and stops it.
func RunInPodWithVolume(c clientset.Interface, ns, claimName, podName, command string, node framework.NodeSelection) {
	pod := StartInPodWithVolume(c, ns, claimName, podName, command, node)
	defer StopPod(c, pod)
	framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespaceSlow(c, pod.Name, pod.Namespace))
}

// StartInPodWithVolume starts a command in a pod with given claim mounted to /mnt directory
// The caller is responsible for checking the pod and deleting it.
func StartInPodWithVolume(c clientset.Interface, ns, claimName, podName, command string, node framework.NodeSelection) *v1.Pod {
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
			NodeName:     node.Name,
			NodeSelector: node.Selector,
			Affinity:     node.Affinity,
			Containers: []v1.Container{
				{
					Name:    "volume-tester",
					Image:   volume.GetTestImage(framework.BusyBoxImage),
					Command: volume.GenerateScriptCmd(command),
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

	pod, err := c.CoreV1().Pods(ns).Create(pod)
	framework.ExpectNoError(err, "Failed to create pod: %v", err)
	return pod
}

// StopPod first tries to log the output of the pod's container, then deletes the pod.
func StopPod(c clientset.Interface, pod *v1.Pod) {
	if pod == nil {
		return
	}
	body, err := c.CoreV1().Pods(pod.Namespace).GetLogs(pod.Name, &v1.PodLogOptions{}).Do().Raw()
	if err != nil {
		e2elog.Logf("Error getting logs for pod %s: %v", pod.Name, err)
	} else {
		e2elog.Logf("Pod %s has the following logs: %s", pod.Name, body)
	}
	e2epod.DeletePodOrFail(c, pod.Namespace, pod.Name)
}

func verifyPVCsPending(client clientset.Interface, pvcs []*v1.PersistentVolumeClaim) {
	for _, claim := range pvcs {
		// Get new copy of the claim
		claim, err := client.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(claim.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(claim.Status.Phase, v1.ClaimPending)
	}
}

func prepareSnapshotDataSourceForProvisioning(
	node framework.NodeSelection,
	client clientset.Interface,
	dynamicClient dynamic.Interface,
	initClaim *v1.PersistentVolumeClaim,
	class *storagev1.StorageClass,
	snapshotClass *unstructured.Unstructured,
) (*v1.TypedLocalObjectReference, func()) {
	var err error
	if class != nil {
		ginkgo.By("[Initialize dataSource]creating a StorageClass " + class.Name)
		_, err = client.StorageV1().StorageClasses().Create(class)
		framework.ExpectNoError(err)
	}

	ginkgo.By("[Initialize dataSource]creating a initClaim")
	updatedClaim, err := client.CoreV1().PersistentVolumeClaims(initClaim.Namespace).Create(initClaim)
	framework.ExpectNoError(err)
	err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, client, updatedClaim.Namespace, updatedClaim.Name, framework.Poll, framework.ClaimProvisionTimeout)
	framework.ExpectNoError(err)

	ginkgo.By("[Initialize dataSource]checking the initClaim")
	// Get new copy of the initClaim
	_, err = client.CoreV1().PersistentVolumeClaims(updatedClaim.Namespace).Get(updatedClaim.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)

	// write namespace to the /mnt/test (= the volume).
	ginkgo.By("[Initialize dataSource]write data to volume")
	command := fmt.Sprintf("echo '%s' > /mnt/test/initialData", updatedClaim.GetNamespace())
	RunInPodWithVolume(client, updatedClaim.Namespace, updatedClaim.Name, "pvc-snapshot-writer", command, node)

	ginkgo.By("[Initialize dataSource]creating a SnapshotClass")
	snapshotClass, err = dynamicClient.Resource(snapshotClassGVR).Create(snapshotClass, metav1.CreateOptions{})

	ginkgo.By("[Initialize dataSource]creating a snapshot")
	snapshot := getSnapshot(updatedClaim.Name, updatedClaim.Namespace, snapshotClass.GetName())
	snapshot, err = dynamicClient.Resource(snapshotGVR).Namespace(updatedClaim.Namespace).Create(snapshot, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	WaitForSnapshotReady(dynamicClient, snapshot.GetNamespace(), snapshot.GetName(), framework.Poll, framework.SnapshotCreateTimeout)
	framework.ExpectNoError(err)

	ginkgo.By("[Initialize dataSource]checking the snapshot")
	// Get new copy of the snapshot
	snapshot, err = dynamicClient.Resource(snapshotGVR).Namespace(snapshot.GetNamespace()).Get(snapshot.GetName(), metav1.GetOptions{})
	framework.ExpectNoError(err)
	group := "snapshot.storage.k8s.io"
	dataSourceRef := &v1.TypedLocalObjectReference{
		APIGroup: &group,
		Kind:     "VolumeSnapshot",
		Name:     snapshot.GetName(),
	}

	cleanupFunc := func() {
		e2elog.Logf("deleting snapshot %q/%q", snapshot.GetNamespace(), snapshot.GetName())
		err = dynamicClient.Resource(snapshotGVR).Namespace(updatedClaim.Namespace).Delete(snapshot.GetName(), nil)
		if err != nil && !apierrs.IsNotFound(err) {
			e2elog.Failf("Error deleting snapshot %q. Error: %v", snapshot.GetName(), err)
		}

		e2elog.Logf("deleting initClaim %q/%q", updatedClaim.Namespace, updatedClaim.Name)
		err = client.CoreV1().PersistentVolumeClaims(updatedClaim.Namespace).Delete(updatedClaim.Name, nil)
		if err != nil && !apierrs.IsNotFound(err) {
			e2elog.Failf("Error deleting initClaim %q. Error: %v", updatedClaim.Name, err)
		}

		e2elog.Logf("deleting SnapshotClass %s", snapshotClass.GetName())
		framework.ExpectNoError(dynamicClient.Resource(snapshotClassGVR).Delete(snapshotClass.GetName(), nil))
	}

	return dataSourceRef, cleanupFunc
}

func preparePVCDataSourceForProvisioning(
	node framework.NodeSelection,
	client clientset.Interface,
	dynamicClient dynamic.Interface,
	source *v1.PersistentVolumeClaim,
	class *storagev1.StorageClass,
) (*v1.TypedLocalObjectReference, func()) {
	var err error
	if class != nil {
		ginkgo.By("[Initialize dataSource]creating a StorageClass " + class.Name)
		_, err = client.StorageV1().StorageClasses().Create(class)
		framework.ExpectNoError(err)
	}

	ginkgo.By("[Initialize dataSource]creating a source PVC")
	sourcePVC, err := client.CoreV1().PersistentVolumeClaims(source.Namespace).Create(source)
	framework.ExpectNoError(err)

	// write namespace to the /mnt/test (= the volume).
	ginkgo.By("[Initialize dataSource]write data to volume")
	command := fmt.Sprintf("echo '%s' > /mnt/test/initialData", sourcePVC.GetNamespace())
	RunInPodWithVolume(client, sourcePVC.Namespace, sourcePVC.Name, "pvc-datasource-writer", command, node)

	dataSourceRef := &v1.TypedLocalObjectReference{
		Kind: "PersistentVolumeClaim",
		Name: sourcePVC.GetName(),
	}

	cleanupFunc := func() {
		e2elog.Logf("deleting source PVC %q/%q", sourcePVC.Namespace, sourcePVC.Name)
		err = client.CoreV1().PersistentVolumeClaims(sourcePVC.Namespace).Delete(sourcePVC.Name, nil)
		if err != nil && !apierrs.IsNotFound(err) {
			e2elog.Failf("Error deleting source PVC %q. Error: %v", sourcePVC.Name, err)
		}
	}

	return dataSourceRef, cleanupFunc
}
