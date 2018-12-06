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

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	apierrs "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"

	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// StorageClassTest represents parameters to be used by provisioning tests
type StorageClassTest struct {
	Name               string
	CloudProviders     []string
	Provisioner        string
	StorageClassName   string
	Parameters         map[string]string
	DelayBinding       bool
	ClaimSize          string
	ExpectedSize       string
	PvCheck            func(volume *v1.PersistentVolume) error
	NodeName           string
	SkipWriteReadCheck bool
	VolumeMode         *v1.PersistentVolumeMode
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

func (p *provisioningTestSuite) skipUnsupportedTest(pattern testpatterns.TestPattern, driver drivers.TestDriver) {
}

func createProvisioningTestInput(driver drivers.TestDriver, pattern testpatterns.TestPattern) (provisioningTestResource, provisioningTestInput) {
	// Setup test resource for driver and testpattern
	resource := provisioningTestResource{}
	resource.setupResource(driver, pattern)

	input := provisioningTestInput{
		testCase: StorageClassTest{
			ClaimSize:    resource.claimSize,
			ExpectedSize: resource.claimSize,
		},
		cs:    driver.GetDriverInfo().Framework.ClientSet,
		pvc:   resource.pvc,
		sc:    resource.sc,
		dInfo: driver.GetDriverInfo(),
	}

	if driver.GetDriverInfo().Config.ClientNodeName != "" {
		input.testCase.NodeName = driver.GetDriverInfo().Config.ClientNodeName
	}

	return resource, input
}

func (p *provisioningTestSuite) execTest(driver drivers.TestDriver, pattern testpatterns.TestPattern) {
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
	driver drivers.TestDriver

	claimSize string
	sc        *storage.StorageClass
	pvc       *v1.PersistentVolumeClaim
}

var _ TestResource = &provisioningTestResource{}

func (p *provisioningTestResource) setupResource(driver drivers.TestDriver, pattern testpatterns.TestPattern) {
	// Setup provisioningTest resource
	switch pattern.VolType {
	case testpatterns.DynamicPV:
		if dDriver, ok := driver.(drivers.DynamicPVTestDriver); ok {
			p.sc = dDriver.GetDynamicProvisionStorageClass("")
			if p.sc == nil {
				framework.Skipf("Driver %q does not define Dynamic Provision StorageClass - skipping", driver.GetDriverInfo().Name)
			}
			p.driver = driver
			p.claimSize = "5Gi"
			p.pvc = getClaim(p.claimSize, driver.GetDriverInfo().Framework.Namespace.Name)
			p.pvc.Spec.StorageClassName = &p.sc.Name
			framework.Logf("In creating storage class object and pvc object for driver - sc: %v, pvc: %v", p.sc, p.pvc)
		}
	default:
		framework.Failf("Dynamic Provision test doesn't support: %s", pattern.VolType)
	}
}

func (p *provisioningTestResource) cleanupResource(driver drivers.TestDriver, pattern testpatterns.TestPattern) {
}

type provisioningTestInput struct {
	testCase StorageClassTest
	cs       clientset.Interface
	pvc      *v1.PersistentVolumeClaim
	sc       *storage.StorageClass
	dInfo    *drivers.DriverInfo
}

func testProvisioning(input *provisioningTestInput) {
	It("should provision storage with defaults", func() {
		TestDynamicProvisioning(input.testCase, input.cs, input.pvc, input.sc)
	})

	It("should provision storage with mount options", func() {
		if input.dInfo.SupportedMountOption == nil {
			framework.Skipf("Driver %q does not define supported mount option - skipping", input.dInfo.Name)
		}

		input.sc.MountOptions = input.dInfo.SupportedMountOption.Union(input.dInfo.RequiredMountOption).List()
		TestDynamicProvisioning(input.testCase, input.cs, input.pvc, input.sc)
	})

	It("should create and delete block persistent volumes", func() {
		if !input.dInfo.Capabilities[drivers.CapBlock] {
			framework.Skipf("Driver %q does not support BlockVolume - skipping", input.dInfo.Name)
		}
		block := v1.PersistentVolumeBlock
		input.testCase.VolumeMode = &block
		input.testCase.SkipWriteReadCheck = true
		input.pvc.Spec.VolumeMode = &block
		TestDynamicProvisioning(input.testCase, input.cs, input.pvc, input.sc)
	})
}

// TestDynamicProvisioning tests dynamic provisioning with specified StorageClassTest and storageClass
func TestDynamicProvisioning(t StorageClassTest, client clientset.Interface, claim *v1.PersistentVolumeClaim, class *storage.StorageClass) *v1.PersistentVolume {
	var err error
	if class != nil {
		By("creating a StorageClass " + class.Name)
		class, err = client.StorageV1().StorageClasses().Create(class)
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
		err = t.PvCheck(pv)
		Expect(err).NotTo(HaveOccurred())
	}

	if !t.SkipWriteReadCheck {
		// We start two pods:
		// - The first writes 'hello word' to the /mnt/test (= the volume).
		// - The second one runs grep 'hello world' on /mnt/test.
		// If both succeed, Kubernetes actually allocated something that is
		// persistent across pods.
		By("checking the created volume is writable and has the PV's mount options")
		command := "echo 'hello world' > /mnt/test/data"
		// We give the first pod the secondary responsibility of checking the volume has
		// been mounted with the PV's mount options, if the PV was provisioned with any
		for _, option := range pv.Spec.MountOptions {
			// Get entry, get mount options at 6th word, replace brackets with commas
			command += fmt.Sprintf(" && ( mount | grep 'on /mnt/test' | awk '{print $6}' | sed 's/^(/,/; s/)$/,/' | grep -q ,%s, )", option)
		}
		command += " || (mount | grep 'on /mnt/test'; false)"
		runInPodWithVolume(client, claim.Namespace, claim.Name, t.NodeName, command)

		By("checking the created volume is readable and retains data")
		runInPodWithVolume(client, claim.Namespace, claim.Name, t.NodeName, "grep 'hello world' /mnt/test/data")
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

// runInPodWithVolume runs a command in a pod with given claim mounted to /mnt directory.
func runInPodWithVolume(c clientset.Interface, ns, claimName, nodeName, command string) {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pvc-volume-tester-",
		},
		Spec: v1.PodSpec{
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

	if len(nodeName) != 0 {
		pod.Spec.NodeName = nodeName
	}
	pod, err := c.CoreV1().Pods(ns).Create(pod)
	framework.ExpectNoError(err, "Failed to create pod: %v", err)
	defer func() {
		body, err := c.CoreV1().Pods(ns).GetLogs(pod.Name, &v1.PodLogOptions{}).Do().Raw()
		if err != nil {
			framework.Logf("Error getting logs for pod %s: %v", pod.Name, err)
		} else {
			framework.Logf("Pod %s has the following logs: %s", pod.Name, body)
		}
		framework.DeletePodOrFail(c, ns, pod.Name)
	}()
	framework.ExpectNoError(framework.WaitForPodSuccessInNamespaceSlow(c, pod.Name, pod.Namespace))
}
