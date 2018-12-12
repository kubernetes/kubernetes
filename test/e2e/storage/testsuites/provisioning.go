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
	"sync"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// StorageClassTest represents parameters to be used by provisioning tests.
// Not all parameters are used by all tests.
type StorageClassTest struct {
	Name             string
	CloudProviders   []string
	Provisioner      string
	StorageClassName string
	Parameters       map[string]string
	DelayBinding     bool
	ClaimSize        string
	ExpectedSize     string
	PvCheck          func(claim *v1.PersistentVolumeClaim, volume *v1.PersistentVolume)
	VolumeMode       *v1.PersistentVolumeMode
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
			},
		},
	}
}

func (p *provisioningTestSuite) getTestSuiteInfo() TestSuiteInfo {
	return p.tsInfo
}

func (p *provisioningTestSuite) skipUnsupportedTest(pattern testpatterns.TestPattern, driver TestDriver) {
}

func createProvisioningTestInput(driver TestDriver, pattern testpatterns.TestPattern) (provisioningTestResource, provisioningTestInput) {
	// Setup test resource for driver and testpattern
	resource := provisioningTestResource{}
	resource.setupResource(driver, pattern)

	input := provisioningTestInput{
		testCase: StorageClassTest{
			ClaimSize:    resource.claimSize,
			ExpectedSize: resource.claimSize,
		},
		cs:       driver.GetDriverInfo().Config.Framework.ClientSet,
		dc:       driver.GetDriverInfo().Config.Framework.DynamicClient,
		pvc:      resource.pvc,
		sc:       resource.sc,
		vsc:      resource.vsc,
		dInfo:    driver.GetDriverInfo(),
		nodeName: driver.GetDriverInfo().Config.ClientNodeName,
	}

	return resource, input
}

func (p *provisioningTestSuite) execTest(driver TestDriver, pattern testpatterns.TestPattern) {
	Context(getTestNameStr(p, pattern), func() {
		var (
			resource     provisioningTestResource
			input        provisioningTestInput
			needsCleanup bool
		)

		BeforeEach(func() {
			needsCleanup = false
			// Skip unsupported tests to avoid unnecessary resource initialization
			skipUnsupportedTest(p, driver, pattern)
			needsCleanup = true

			// Create test input
			resource, input = createProvisioningTestInput(driver, pattern)
		})

		AfterEach(func() {
			if needsCleanup {
				resource.cleanupResource(driver, pattern)
			}
		})

		// Ginkgo's "Global Shared Behaviors" require arguments for a shared function
		// to be a single struct and to be passed as a pointer.
		// Please see https://onsi.github.io/ginkgo/#global-shared-behaviors for details.
		testProvisioning(&input)
	})
}

type provisioningTestResource struct {
	driver TestDriver

	claimSize string
	sc        *storage.StorageClass
	pvc       *v1.PersistentVolumeClaim
	// follow parameter is used to test provision volume from snapshot
	vsc *unstructured.Unstructured
}

var _ TestResource = &provisioningTestResource{}

func (p *provisioningTestResource) setupResource(driver TestDriver, pattern testpatterns.TestPattern) {
	// Setup provisioningTest resource
	switch pattern.VolType {
	case testpatterns.DynamicPV:
		if dDriver, ok := driver.(DynamicPVTestDriver); ok {
			p.sc = dDriver.GetDynamicProvisionStorageClass("")
			if p.sc == nil {
				framework.Skipf("Driver %q does not define Dynamic Provision StorageClass - skipping", driver.GetDriverInfo().Name)
			}
			p.driver = driver
			p.claimSize = dDriver.GetClaimSize()
			p.pvc = getClaim(p.claimSize, driver.GetDriverInfo().Config.Framework.Namespace.Name)
			p.pvc.Spec.StorageClassName = &p.sc.Name
			framework.Logf("In creating storage class object and pvc object for driver - sc: %v, pvc: %v", p.sc, p.pvc)
			if sDriver, ok := driver.(SnapshottableTestDriver); ok {
				p.vsc = sDriver.GetSnapshotClass()
			}
		}
	default:
		framework.Failf("Dynamic Provision test doesn't support: %s", pattern.VolType)
	}
}

func (p *provisioningTestResource) cleanupResource(driver TestDriver, pattern testpatterns.TestPattern) {
}

type provisioningTestInput struct {
	testCase StorageClassTest
	cs       clientset.Interface
	dc       dynamic.Interface
	pvc      *v1.PersistentVolumeClaim
	sc       *storage.StorageClass
	vsc      *unstructured.Unstructured
	dInfo    *DriverInfo
	nodeName string
}

func testProvisioning(input *provisioningTestInput) {
	// common checker for most of the test cases below
	pvcheck := func(claim *v1.PersistentVolumeClaim, volume *v1.PersistentVolume) {
		PVWriteReadSingleNodeCheck(input.cs, claim, volume, NodeSelection{Name: input.nodeName})
	}

	It("should provision storage with defaults", func() {
		input.testCase.PvCheck = pvcheck
		TestDynamicProvisioning(input.testCase, input.cs, input.pvc, input.sc)
	})

	It("should provision storage with mount options", func() {
		if input.dInfo.SupportedMountOption == nil {
			framework.Skipf("Driver %q does not define supported mount option - skipping", input.dInfo.Name)
		}

		input.sc.MountOptions = input.dInfo.SupportedMountOption.Union(input.dInfo.RequiredMountOption).List()
		input.testCase.PvCheck = pvcheck
		TestDynamicProvisioning(input.testCase, input.cs, input.pvc, input.sc)
	})

	It("should access volume from different nodes", func() {
		// The assumption is that if the test hasn't been
		// locked onto a single node, then the driver is
		// usable on all of them *and* supports accessing a volume
		// from any node.
		if input.nodeName != "" {
			framework.Skipf("Driver %q only supports testing on one node - skipping", input.dInfo.Name)
		}
		// Ensure that we actually have more than one node.
		nodes := framework.GetReadySchedulableNodesOrDie(input.cs)
		if len(nodes.Items) <= 1 {
			framework.Skipf("need more than one node - skipping")
		}
		input.testCase.PvCheck = func(claim *v1.PersistentVolumeClaim, volume *v1.PersistentVolume) {
			PVMultiNodeCheck(input.cs, claim, volume, NodeSelection{Name: input.nodeName})
		}
		TestDynamicProvisioning(input.testCase, input.cs, input.pvc, input.sc)
	})

	It("should create and delete block persistent volumes", func() {
		if !input.dInfo.Capabilities[CapBlock] {
			framework.Skipf("Driver %q does not support BlockVolume - skipping", input.dInfo.Name)
		}
		block := v1.PersistentVolumeBlock
		input.testCase.VolumeMode = &block
		input.pvc.Spec.VolumeMode = &block
		TestDynamicProvisioning(input.testCase, input.cs, input.pvc, input.sc)
	})

	It("should provision storage with snapshot data source [Feature:VolumeSnapshotDataSource]", func() {
		if !input.dInfo.Capabilities[CapDataSource] {
			framework.Skipf("Driver %q does not support populate data from snapshot - skipping", input.dInfo.Name)
		}

		dataSource, cleanupFunc := prepareDataSourceForProvisioning(NodeSelection{Name: input.nodeName}, input.cs, input.dc, input.pvc, input.sc, input.vsc)
		defer cleanupFunc()

		input.pvc.Spec.DataSource = dataSource
		input.testCase.PvCheck = func(claim *v1.PersistentVolumeClaim, volume *v1.PersistentVolume) {
			By("checking whether the created volume has the pre-populated data")
			command := fmt.Sprintf("grep '%s' /mnt/test/initialData", claim.Namespace)
			RunInPodWithVolume(input.cs, claim.Namespace, claim.Name, "pvc-snapshot-tester", command, NodeSelection{Name: input.nodeName})
		}
		TestDynamicProvisioning(input.testCase, input.cs, input.pvc, input.sc)
	})

	It("should allow concurrent writes on the same node", func() {
		if !input.dInfo.Capabilities[CapMultiPODs] {
			framework.Skipf("Driver %q does not support multiple concurrent pods - skipping", input.dInfo.Name)
		}
		input.testCase.PvCheck = func(claim *v1.PersistentVolumeClaim, volume *v1.PersistentVolume) {
			// We start two pods concurrently on the same node,
			// using the same PVC. Both wait for other to create a
			// file before returning. The pods are forced onto the
			// same node via pod affinity.
			wg := sync.WaitGroup{}
			wg.Add(2)
			firstPodName := "pvc-tester-first"
			secondPodName := "pvc-tester-second"
			run := func(podName, command string) {
				defer GinkgoRecover()
				defer wg.Done()
				node := NodeSelection{
					Name: input.nodeName,
				}
				if podName == secondPodName {
					node.Affinity = &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{LabelSelector: &metav1.LabelSelector{
									MatchLabels: map[string]string{
										// Set by RunInPodWithVolume.
										"app": firstPodName,
									},
								},
									TopologyKey: "kubernetes.io/hostname",
								},
							},
						},
					}
				}
				RunInPodWithVolume(input.cs, claim.Namespace, claim.Name, podName, command, node)
			}
			go run(firstPodName, "touch /mnt/test/first && while ! [ -f /mnt/test/second ]; do sleep 1; done")
			go run(secondPodName, "touch /mnt/test/second && while ! [ -f /mnt/test/first ]; do sleep 1; done")
			wg.Wait()
		}
		TestDynamicProvisioning(input.testCase, input.cs, input.pvc, input.sc)
	})
}

// TestDynamicProvisioning tests dynamic provisioning with specified StorageClassTest and storageClass
func TestDynamicProvisioning(t StorageClassTest, client clientset.Interface, claim *v1.PersistentVolumeClaim, class *storage.StorageClass) *v1.PersistentVolume {
	var err error
	if class != nil {
		By("creating a StorageClass " + class.Name)
		_, err = client.StorageV1().StorageClasses().Create(class)
		// The "should provision storage with snapshot data source" test already has created the class.
		// TODO: make class creation optional and remove the IsAlreadyExists exception
		Expect(err == nil || apierrs.IsAlreadyExists(err)).To(Equal(true))
		class, err = client.StorageV1().StorageClasses().Get(class.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			framework.Logf("deleting storage class %s", class.Name)
			framework.ExpectNoError(client.StorageV1().StorageClasses().Delete(class.Name, nil))
		}()
	}

	By("creating a claim")
	claim, err = client.CoreV1().PersistentVolumeClaims(claim.Namespace).Create(claim)
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		framework.Logf("deleting claim %q/%q", claim.Namespace, claim.Name)
		// typically this claim has already been deleted
		err = client.CoreV1().PersistentVolumeClaims(claim.Namespace).Delete(claim.Name, nil)
		if err != nil && !apierrs.IsNotFound(err) {
			framework.Failf("Error deleting claim %q. Error: %v", claim.Name, err)
		}
	}()
	err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, client, claim.Namespace, claim.Name, framework.Poll, framework.ClaimProvisionTimeout)
	Expect(err).NotTo(HaveOccurred())

	By("checking the claim")
	// Get new copy of the claim
	claim, err = client.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(claim.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	// Get the bound PV
	pv, err := client.CoreV1().PersistentVolumes().Get(claim.Spec.VolumeName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	// Check sizes
	expectedCapacity := resource.MustParse(t.ExpectedSize)
	pvCapacity := pv.Spec.Capacity[v1.ResourceName(v1.ResourceStorage)]
	Expect(pvCapacity.Value()).To(Equal(expectedCapacity.Value()), "pvCapacity is not equal to expectedCapacity")

	requestedCapacity := resource.MustParse(t.ClaimSize)
	claimCapacity := claim.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	Expect(claimCapacity.Value()).To(Equal(requestedCapacity.Value()), "claimCapacity is not equal to requestedCapacity")

	// Check PV properties
	By("checking the PV")
	expectedAccessModes := []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
	Expect(pv.Spec.AccessModes).To(Equal(expectedAccessModes))
	Expect(pv.Spec.ClaimRef.Name).To(Equal(claim.ObjectMeta.Name))
	Expect(pv.Spec.ClaimRef.Namespace).To(Equal(claim.ObjectMeta.Namespace))
	if class == nil {
		Expect(pv.Spec.PersistentVolumeReclaimPolicy).To(Equal(v1.PersistentVolumeReclaimDelete))
	} else {
		Expect(pv.Spec.PersistentVolumeReclaimPolicy).To(Equal(*class.ReclaimPolicy))
		Expect(pv.Spec.MountOptions).To(Equal(class.MountOptions))
	}
	if t.VolumeMode != nil {
		Expect(pv.Spec.VolumeMode).NotTo(BeNil())
		Expect(*pv.Spec.VolumeMode).To(Equal(*t.VolumeMode))
	}

	// Run the checker
	if t.PvCheck != nil {
		t.PvCheck(claim, pv)
	}

	By(fmt.Sprintf("deleting claim %q/%q", claim.Namespace, claim.Name))
	framework.ExpectNoError(client.CoreV1().PersistentVolumeClaims(claim.Namespace).Delete(claim.Name, nil))

	// Wait for the PV to get deleted if reclaim policy is Delete. (If it's
	// Retain, there's no use waiting because the PV won't be auto-deleted and
	// it's expected for the caller to do it.) Technically, the first few delete
	// attempts may fail, as the volume is still attached to a node because
	// kubelet is slowly cleaning up the previous pod, however it should succeed
	// in a couple of minutes. Wait 20 minutes to recover from random cloud
	// hiccups.
	if pv.Spec.PersistentVolumeReclaimPolicy == v1.PersistentVolumeReclaimDelete {
		By(fmt.Sprintf("deleting the claim's PV %q", pv.Name))
		framework.ExpectNoError(framework.WaitForPersistentVolumeDeleted(client, pv.Name, 5*time.Second, 20*time.Minute))
	}

	return pv
}

// PVWriteReadSingleNodeCheck checks that a PV retains data on a single node.
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
func PVWriteReadSingleNodeCheck(client clientset.Interface, claim *v1.PersistentVolumeClaim, volume *v1.PersistentVolume, node NodeSelection) {
	By(fmt.Sprintf("checking the created volume is writable and has the PV's mount options on node %+v", node))
	command := "echo 'hello world' > /mnt/test/data"
	// We give the first pod the secondary responsibility of checking the volume has
	// been mounted with the PV's mount options, if the PV was provisioned with any
	for _, option := range volume.Spec.MountOptions {
		// Get entry, get mount options at 6th word, replace brackets with commas
		command += fmt.Sprintf(" && ( mount | grep 'on /mnt/test' | awk '{print $6}' | sed 's/^(/,/; s/)$/,/' | grep -q ,%s, )", option)
	}
	command += " || (mount | grep 'on /mnt/test'; false)"
	pod := StartInPodWithVolume(client, claim.Namespace, claim.Name, "pvc-volume-tester-writer", command, node)
	defer func() {
		// pod might be nil now.
		StopPod(client, pod)
	}()
	framework.ExpectNoError(framework.WaitForPodSuccessInNamespaceSlow(client, pod.Name, pod.Namespace))
	runningPod, err := client.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred(), "get pod")
	actualNodeName := runningPod.Spec.NodeName
	StopPod(client, pod)
	pod = nil // Don't stop twice.

	By(fmt.Sprintf("checking the created volume is readable and retains data on the same node %q", actualNodeName))
	command = "grep 'hello world' /mnt/test/data"
	RunInPodWithVolume(client, claim.Namespace, claim.Name, "pvc-volume-tester-reader", command, NodeSelection{Name: actualNodeName})
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
func PVMultiNodeCheck(client clientset.Interface, claim *v1.PersistentVolumeClaim, volume *v1.PersistentVolume, node NodeSelection) {
	Expect(node.Name).To(Equal(""), "this test only works when not locked onto a single node")

	var pod *v1.Pod
	defer func() {
		// passing pod = nil is okay.
		StopPod(client, pod)
	}()

	By(fmt.Sprintf("checking the created volume is writable and has the PV's mount options on node %+v", node))
	command := "echo 'hello world' > /mnt/test/data"
	pod = StartInPodWithVolume(client, claim.Namespace, claim.Name, "pvc-writer-node1", command, node)
	framework.ExpectNoError(framework.WaitForPodSuccessInNamespaceSlow(client, pod.Name, pod.Namespace))
	runningPod, err := client.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred(), "get pod")
	actualNodeName := runningPod.Spec.NodeName
	StopPod(client, pod)
	pod = nil // Don't stop twice.

	// Add node-anti-affinity.
	secondNode := node
	if secondNode.Affinity == nil {
		secondNode.Affinity = &v1.Affinity{}
	}
	if secondNode.Affinity.NodeAffinity == nil {
		secondNode.Affinity.NodeAffinity = &v1.NodeAffinity{}
	}
	if secondNode.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
		secondNode.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution = &v1.NodeSelector{}
	}
	secondNode.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms = append(secondNode.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms,
		v1.NodeSelectorTerm{
			// https://kubernetes.io/docs/concepts/configuration/assign-pod-node/#affinity-and-anti-affinity warns
			// that "the value of kubernetes.io/hostname may be the same as the Node name in some environments and a different value in other environments".
			// So this might be cleaner:
			// MatchFields: []v1.NodeSelectorRequirement{
			// 	{Key: "name", Operator: v1.NodeSelectorOpNotIn, Values: []string{actualNodeName}},
			// },
			// However, "name", "Name", "ObjectMeta.Name" all got rejected with "not a valid field selector key".

			MatchExpressions: []v1.NodeSelectorRequirement{
				{Key: "kubernetes.io/hostname", Operator: v1.NodeSelectorOpNotIn, Values: []string{actualNodeName}},
			},
		})

	By(fmt.Sprintf("checking the created volume is readable and retains data on another node %+v", secondNode))
	command = "grep 'hello world' /mnt/test/data"
	pod = StartInPodWithVolume(client, claim.Namespace, claim.Name, "pvc-reader-node2", command, secondNode)
	framework.ExpectNoError(framework.WaitForPodSuccessInNamespaceSlow(client, pod.Name, pod.Namespace))
	runningPod, err = client.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred(), "get pod")
	Expect(runningPod.Spec.NodeName).NotTo(Equal(actualNodeName), "second pod should have run on a different node")
	StopPod(client, pod)
	pod = nil
}

func TestBindingWaitForFirstConsumer(t StorageClassTest, client clientset.Interface, claim *v1.PersistentVolumeClaim, class *storage.StorageClass, nodeSelector map[string]string, expectUnschedulable bool) (*v1.PersistentVolume, *v1.Node) {
	pvs, node := TestBindingWaitForFirstConsumerMultiPVC(t, client, []*v1.PersistentVolumeClaim{claim}, class, nodeSelector, expectUnschedulable)
	if pvs == nil {
		return nil, node
	}
	return pvs[0], node
}

func TestBindingWaitForFirstConsumerMultiPVC(t StorageClassTest, client clientset.Interface, claims []*v1.PersistentVolumeClaim, class *storage.StorageClass, nodeSelector map[string]string, expectUnschedulable bool) ([]*v1.PersistentVolume, *v1.Node) {
	var err error
	Expect(len(claims)).ToNot(Equal(0))
	namespace := claims[0].Namespace

	By("creating a storage class " + class.Name)
	class, err = client.StorageV1().StorageClasses().Create(class)
	Expect(err).NotTo(HaveOccurred())
	defer deleteStorageClass(client, class.Name)

	By("creating claims")
	var claimNames []string
	var createdClaims []*v1.PersistentVolumeClaim
	for _, claim := range claims {
		c, err := client.CoreV1().PersistentVolumeClaims(claim.Namespace).Create(claim)
		claimNames = append(claimNames, c.Name)
		createdClaims = append(createdClaims, c)
		Expect(err).NotTo(HaveOccurred())
	}
	defer func() {
		var errors map[string]error
		for _, claim := range createdClaims {
			err := framework.DeletePersistentVolumeClaim(client, claim.Name, claim.Namespace)
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
	By("checking the claims are in pending state")
	err = framework.WaitForPersistentVolumeClaimsPhase(v1.ClaimBound, client, namespace, claimNames, 2*time.Second /* Poll */, framework.ClaimProvisionShortTimeout, true)
	Expect(err).To(HaveOccurred())
	verifyPVCsPending(client, createdClaims)

	By("creating a pod referring to the claims")
	// Create a pod referring to the claim and wait for it to get to running
	var pod *v1.Pod
	if expectUnschedulable {
		pod, err = framework.CreateUnschedulablePod(client, namespace, nodeSelector, createdClaims, true /* isPrivileged */, "" /* command */)
	} else {
		pod, err = framework.CreatePod(client, namespace, nil /* nodeSelector */, createdClaims, true /* isPrivileged */, "" /* command */)
	}
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		framework.DeletePodOrFail(client, pod.Namespace, pod.Name)
		framework.WaitForPodToDisappear(client, pod.Namespace, pod.Name, labels.Everything(), framework.Poll, framework.PodDeleteTimeout)
	}()
	if expectUnschedulable {
		// Verify that no claims are provisioned.
		verifyPVCsPending(client, createdClaims)
		return nil, nil
	}

	// collect node details
	node, err := client.CoreV1().Nodes().Get(pod.Spec.NodeName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	By("re-checking the claims to see they binded")
	var pvs []*v1.PersistentVolume
	for _, claim := range createdClaims {
		// Get new copy of the claim
		claim, err = client.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(claim.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		// make sure claim did bind
		err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, client, claim.Namespace, claim.Name, framework.Poll, framework.ClaimProvisionTimeout)
		Expect(err).NotTo(HaveOccurred())

		pv, err := client.CoreV1().PersistentVolumes().Get(claim.Spec.VolumeName, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		pvs = append(pvs, pv)
	}
	Expect(len(pvs)).To(Equal(len(createdClaims)))
	return pvs, node
}

// NodeSelection specifies where to run a pod, using a combination of fixed node name,
// node selector and/or affinity.
type NodeSelection struct {
	Name     string
	Selector map[string]string
	Affinity *v1.Affinity
}

// RunInPodWithVolume runs a command in a pod with given claim mounted to /mnt directory.
// It starts, checks, collects output and stops it.
func RunInPodWithVolume(c clientset.Interface, ns, claimName, podName, command string, node NodeSelection) {
	pod := StartInPodWithVolume(c, ns, claimName, podName, command, node)
	defer StopPod(c, pod)
	framework.ExpectNoError(framework.WaitForPodSuccessInNamespaceSlow(c, pod.Name, pod.Namespace))
}

// StartInPodWithVolume starts a command in a pod with given claim mounted to /mnt directory
// The caller is responsible for checking the pod and deleting it.
func StartInPodWithVolume(c clientset.Interface, ns, claimName, podName, command string, node NodeSelection) *v1.Pod {
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
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"/bin/sh"},
					Args:    []string{"-c", command},
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
		framework.Logf("Error getting logs for pod %s: %v", pod.Name, err)
	} else {
		framework.Logf("Pod %s has the following logs: %s", pod.Name, body)
	}
	framework.DeletePodOrFail(c, pod.Namespace, pod.Name)
}

func verifyPVCsPending(client clientset.Interface, pvcs []*v1.PersistentVolumeClaim) {
	for _, claim := range pvcs {
		// Get new copy of the claim
		claim, err := client.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(claim.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		Expect(claim.Status.Phase).To(Equal(v1.ClaimPending))
	}
}

func prepareDataSourceForProvisioning(
	node NodeSelection,
	client clientset.Interface,
	dynamicClient dynamic.Interface,
	initClaim *v1.PersistentVolumeClaim,
	class *storage.StorageClass,
	snapshotClass *unstructured.Unstructured,
) (*v1.TypedLocalObjectReference, func()) {
	var err error
	if class != nil {
		By("[Initialize dataSource]creating a StorageClass " + class.Name)
		_, err = client.StorageV1().StorageClasses().Create(class)
		Expect(err).NotTo(HaveOccurred())
	}

	By("[Initialize dataSource]creating a initClaim")
	updatedClaim, err := client.CoreV1().PersistentVolumeClaims(initClaim.Namespace).Create(initClaim)
	Expect(err).NotTo(HaveOccurred())
	err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, client, updatedClaim.Namespace, updatedClaim.Name, framework.Poll, framework.ClaimProvisionTimeout)
	Expect(err).NotTo(HaveOccurred())

	By("[Initialize dataSource]checking the initClaim")
	// Get new copy of the initClaim
	_, err = client.CoreV1().PersistentVolumeClaims(updatedClaim.Namespace).Get(updatedClaim.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	// write namespace to the /mnt/test (= the volume).
	By("[Initialize dataSource]write data to volume")
	command := fmt.Sprintf("echo '%s' > /mnt/test/initialData", updatedClaim.GetNamespace())
	RunInPodWithVolume(client, updatedClaim.Namespace, updatedClaim.Name, "pvc-snapshot-writer", command, node)

	By("[Initialize dataSource]creating a SnapshotClass")
	snapshotClass, err = dynamicClient.Resource(snapshotClassGVR).Create(snapshotClass, metav1.CreateOptions{})

	By("[Initialize dataSource]creating a snapshot")
	snapshot := getSnapshot(updatedClaim.Name, updatedClaim.Namespace, snapshotClass.GetName())
	snapshot, err = dynamicClient.Resource(snapshotGVR).Namespace(updatedClaim.Namespace).Create(snapshot, metav1.CreateOptions{})
	Expect(err).NotTo(HaveOccurred())

	WaitForSnapshotReady(dynamicClient, snapshot.GetNamespace(), snapshot.GetName(), framework.Poll, framework.SnapshotCreateTimeout)
	Expect(err).NotTo(HaveOccurred())

	By("[Initialize dataSource]checking the snapshot")
	// Get new copy of the snapshot
	snapshot, err = dynamicClient.Resource(snapshotGVR).Namespace(snapshot.GetNamespace()).Get(snapshot.GetName(), metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	group := "snapshot.storage.k8s.io"
	dataSourceRef := &v1.TypedLocalObjectReference{
		APIGroup: &group,
		Kind:     "VolumeSnapshot",
		Name:     snapshot.GetName(),
	}

	cleanupFunc := func() {
		framework.Logf("deleting snapshot %q/%q", snapshot.GetNamespace(), snapshot.GetName())
		err = dynamicClient.Resource(snapshotGVR).Namespace(updatedClaim.Namespace).Delete(snapshot.GetName(), nil)
		if err != nil && !apierrs.IsNotFound(err) {
			framework.Failf("Error deleting snapshot %q. Error: %v", snapshot.GetName(), err)
		}

		framework.Logf("deleting initClaim %q/%q", updatedClaim.Namespace, updatedClaim.Name)
		err = client.CoreV1().PersistentVolumeClaims(updatedClaim.Namespace).Delete(updatedClaim.Name, nil)
		if err != nil && !apierrs.IsNotFound(err) {
			framework.Failf("Error deleting initClaim %q. Error: %v", updatedClaim.Name, err)
		}

		framework.Logf("deleting SnapshotClass %s", snapshotClass.GetName())
		framework.ExpectNoError(dynamicClient.Resource(snapshotClassGVR).Delete(snapshotClass.GetName(), nil))
	}

	return dataSourceRef, cleanupFunc
}
