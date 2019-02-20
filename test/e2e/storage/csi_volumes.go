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

package storage

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	csiclient "k8s.io/csi-api/pkg/client/clientset/versioned"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/util/rand"
)

// List of testDrivers to be executed in below loop
var csiTestDrivers = []func() testsuites.TestDriver{
	drivers.InitHostPathCSIDriver,
	func() testsuites.TestDriver { return drivers.InitGcePDCSIDriver(false /* topology enabled */) },
	drivers.InitGcePDExternalCSIDriver,
	drivers.InitHostPathV0CSIDriver,
	// Don't run tests with mock driver (drivers.InitMockCSIDriver), it does not provide persistent storage.
}

// List of testSuites to be executed in below loop
var csiTestSuites = []func() testsuites.TestSuite{
	testsuites.InitVolumesTestSuite,
	testsuites.InitVolumeIOTestSuite,
	testsuites.InitVolumeModeTestSuite,
	testsuites.InitSubPathTestSuite,
	testsuites.InitProvisioningTestSuite,
	testsuites.InitSnapshottableTestSuite,
}

// This executes testSuites for csi volumes.
var _ = utils.SIGDescribe("CSI Volumes", func() {
	for _, initDriver := range csiTestDrivers {
		curDriver := initDriver()

		Context(testsuites.GetDriverNameWithFeatureTags(curDriver), func() {
			testsuites.DefineTestSuite(curDriver, csiTestSuites)
		})
	}

	Context("CSI Topology test using GCE PD driver [Feature:CSINodeInfo]", func() {
		f := framework.NewDefaultFramework("csitopology")
		driver := drivers.InitGcePDCSIDriver(true /* topology enabled */).(testsuites.DynamicPVTestDriver) // TODO (#71289) eliminate by moving this test to common test suite.
		var (
			config      *testsuites.PerTestConfig
			testCleanup func()
		)
		BeforeEach(func() {
			config, testCleanup = driver.PrepareTest(f)
		})

		AfterEach(func() {
			if testCleanup != nil {
				testCleanup()
			}
		})

		It("should provision zonal PD with immediate volume binding and AllowedTopologies set and mount the volume to a pod", func() {
			suffix := "topology-positive"
			testTopologyPositive(config.Framework.ClientSet, suffix, config.Framework.Namespace.GetName(), false /* delayBinding */, true /* allowedTopologies */)
		})

		It("should provision zonal PD with delayed volume binding and mount the volume to a pod", func() {
			suffix := "delayed"
			testTopologyPositive(config.Framework.ClientSet, suffix, config.Framework.Namespace.GetName(), true /* delayBinding */, false /* allowedTopologies */)
		})

		It("should provision zonal PD with delayed volume binding and AllowedTopologies set and mount the volume to a pod", func() {
			suffix := "delayed-topology-positive"
			testTopologyPositive(config.Framework.ClientSet, suffix, config.Framework.Namespace.GetName(), true /* delayBinding */, true /* allowedTopologies */)
		})

		It("should fail to schedule a pod with a zone missing from AllowedTopologies; PD is provisioned with immediate volume binding", func() {
			framework.SkipUnlessMultizone(config.Framework.ClientSet)
			suffix := "topology-negative"
			testTopologyNegative(config.Framework.ClientSet, suffix, config.Framework.Namespace.GetName(), false /* delayBinding */)
		})

		It("should fail to schedule a pod with a zone missing from AllowedTopologies; PD is provisioned with delayed volume binding", func() {
			framework.SkipUnlessMultizone(config.Framework.ClientSet)
			suffix := "delayed-topology-negative"
			testTopologyNegative(config.Framework.ClientSet, suffix, config.Framework.Namespace.GetName(), true /* delayBinding */)
		})
	})
})

func testTopologyPositive(cs clientset.Interface, suffix, namespace string, delayBinding, allowedTopologies bool) {
	test := createGCEPDStorageClassTest()
	test.DelayBinding = delayBinding

	class := newStorageClass(test, namespace, suffix)
	if allowedTopologies {
		topoZone := getRandomClusterZone(cs)
		addSingleCSIZoneAllowedTopologyToStorageClass(cs, class, topoZone)
	}
	test.Client = cs
	test.Claim = newClaim(test, namespace, suffix)
	test.Claim.Spec.StorageClassName = &class.Name
	test.Class = class

	if delayBinding {
		_, node := test.TestBindingWaitForFirstConsumer(nil /* node selector */, false /* expect unschedulable */)
		Expect(node).ToNot(BeNil(), "Unexpected nil node found")
	} else {
		test.TestDynamicProvisioning()
	}
}

func testTopologyNegative(cs clientset.Interface, suffix, namespace string, delayBinding bool) {
	framework.SkipUnlessMultizone(cs)

	// Use different zones for pod and PV
	zones, err := framework.GetClusterZones(cs)
	Expect(err).ToNot(HaveOccurred())
	Expect(zones.Len()).To(BeNumerically(">=", 2))
	zonesList := zones.UnsortedList()
	podZoneIndex := rand.Intn(zones.Len())
	podZone := zonesList[podZoneIndex]
	pvZone := zonesList[(podZoneIndex+1)%zones.Len()]

	test := createGCEPDStorageClassTest()
	test.DelayBinding = delayBinding
	nodeSelector := map[string]string{v1.LabelZoneFailureDomain: podZone}

	test.Client = cs
	test.Class = newStorageClass(test, namespace, suffix)
	addSingleCSIZoneAllowedTopologyToStorageClass(cs, test.Class, pvZone)
	test.Claim = newClaim(test, namespace, suffix)
	test.Claim.Spec.StorageClassName = &test.Class.Name
	if delayBinding {
		test.TestBindingWaitForFirstConsumer(nodeSelector, true /* expect unschedulable */)
	} else {
		test.PvCheck = func(claim *v1.PersistentVolumeClaim, volume *v1.PersistentVolume) {
			// Ensure that a pod cannot be scheduled in an unsuitable zone.
			pod := testsuites.StartInPodWithVolume(cs, namespace, claim.Name, "pvc-tester-unschedulable", "sleep 100000",
				testsuites.NodeSelection{Selector: nodeSelector})
			defer testsuites.StopPod(cs, pod)
			framework.ExpectNoError(framework.WaitForPodNameUnschedulableInNamespace(cs, pod.Name, pod.Namespace), "pod should be unschedulable")
		}
		test.TestDynamicProvisioning()
	}
}

func waitForCSIDriver(csics csiclient.Interface, driverName string) error {
	timeout := 2 * time.Minute

	framework.Logf("waiting up to %v for CSIDriver %q", timeout, driverName)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(framework.Poll) {
		_, err := csics.CsiV1alpha1().CSIDrivers().Get(driverName, metav1.GetOptions{})
		if !errors.IsNotFound(err) {
			return err
		}
	}
	return fmt.Errorf("gave up after waiting %v for CSIDriver %q.", timeout, driverName)
}

func destroyCSIDriver(csics csiclient.Interface, driverName string) {
	driverGet, err := csics.CsiV1alpha1().CSIDrivers().Get(driverName, metav1.GetOptions{})
	if err == nil {
		framework.Logf("deleting %s.%s: %s", driverGet.TypeMeta.APIVersion, driverGet.TypeMeta.Kind, driverGet.ObjectMeta.Name)
		// Uncomment the following line to get full dump of CSIDriver object
		// framework.Logf("%s", framework.PrettyPrint(driverGet))
		csics.CsiV1alpha1().CSIDrivers().Delete(driverName, nil)
	}
}

func getVolumeHandle(cs clientset.Interface, claim *v1.PersistentVolumeClaim) string {
	// re-get the claim to the latest state with bound volume
	claim, err := cs.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(claim.Name, metav1.GetOptions{})
	if err != nil {
		framework.ExpectNoError(err, "Cannot get PVC")
		return ""
	}
	pvName := claim.Spec.VolumeName
	pv, err := cs.CoreV1().PersistentVolumes().Get(pvName, metav1.GetOptions{})
	if err != nil {
		framework.ExpectNoError(err, "Cannot get PV")
		return ""
	}
	if pv.Spec.CSI == nil {
		Expect(pv.Spec.CSI).NotTo(BeNil())
		return ""
	}
	return pv.Spec.CSI.VolumeHandle
}

func startPausePod(cs clientset.Interface, t testsuites.StorageClassTest, node testsuites.NodeSelection, ns string) (*storagev1.StorageClass, *v1.PersistentVolumeClaim, *v1.Pod) {
	class := newStorageClass(t, ns, "")
	class, err := cs.StorageV1().StorageClasses().Create(class)
	framework.ExpectNoError(err, "Failed to create class : %v", err)
	claim := newClaim(t, ns, "")
	claim.Spec.StorageClassName = &class.Name
	claim, err = cs.CoreV1().PersistentVolumeClaims(ns).Create(claim)
	framework.ExpectNoError(err, "Failed to create claim: %v", err)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pvc-volume-tester-",
		},
		Spec: v1.PodSpec{
			NodeName:     node.Name,
			NodeSelector: node.Selector,
			Affinity:     node.Affinity,
			Containers: []v1.Container{
				{
					Name:  "volume-tester",
					Image: imageutils.GetE2EImage(imageutils.Pause),
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
							ClaimName: claim.Name,
							ReadOnly:  false,
						},
					},
				},
			},
		},
	}

	pod, err = cs.CoreV1().Pods(ns).Create(pod)
	framework.ExpectNoError(err, "Failed to create pod: %v", err)
	return class, claim, pod
}

// checkPodInfo tests that NodePublish was called with expected volume_context
func checkPodInfo(cs clientset.Interface, namespace, driverPodName, driverContainerName string, pod *v1.Pod, expectPodInfo bool) error {
	expectedAttributes := map[string]string{
		"csi.storage.k8s.io/pod.name":            pod.Name,
		"csi.storage.k8s.io/pod.namespace":       namespace,
		"csi.storage.k8s.io/pod.uid":             string(pod.UID),
		"csi.storage.k8s.io/serviceAccount.name": "default",
	}
	// Load logs of driver pod
	log, err := framework.GetPodLogs(cs, namespace, driverPodName, driverContainerName)
	if err != nil {
		return fmt.Errorf("could not load CSI driver logs: %s", err)
	}
	framework.Logf("CSI driver logs:\n%s", log)
	// Find NodePublish in the logs
	foundAttributes := sets.NewString()
	logLines := strings.Split(log, "\n")
	for _, line := range logLines {
		if !strings.HasPrefix(line, "gRPCCall:") {
			continue
		}
		line = strings.TrimPrefix(line, "gRPCCall:")
		// Dummy structure that parses just volume_attributes out of logged CSI call
		type MockCSICall struct {
			Method  string
			Request struct {
				VolumeContext map[string]string `json:"volume_context"`
			}
		}
		var call MockCSICall
		err := json.Unmarshal([]byte(line), &call)
		if err != nil {
			framework.Logf("Could not parse CSI driver log line %q: %s", line, err)
			continue
		}
		if call.Method != "/csi.v1.Node/NodePublishVolume" {
			continue
		}
		// Check that NodePublish had expected attributes
		for k, v := range expectedAttributes {
			vv, found := call.Request.VolumeContext[k]
			if found && v == vv {
				foundAttributes.Insert(k)
				framework.Logf("Found volume attribute %s: %s", k, v)
			}
		}
		// Process just the first NodePublish, the rest of the log is useless.
		break
	}
	if expectPodInfo {
		if foundAttributes.Len() != len(expectedAttributes) {
			return fmt.Errorf("number of found volume attributes does not match, expected %d, got %d", len(expectedAttributes), foundAttributes.Len())
		}
		return nil
	} else {
		if foundAttributes.Len() != 0 {
			return fmt.Errorf("some unexpected volume attributes were found: %+v", foundAttributes.List())
		}
		return nil
	}
}

func addSingleCSIZoneAllowedTopologyToStorageClass(c clientset.Interface, sc *storagev1.StorageClass, zone string) {
	term := v1.TopologySelectorTerm{
		MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
			{
				Key:    drivers.GCEPDCSIZoneTopologyKey,
				Values: []string{zone},
			},
		},
	}
	sc.AllowedTopologies = append(sc.AllowedTopologies, term)
}

func createGCEPDStorageClassTest() testsuites.StorageClassTest {
	return testsuites.StorageClassTest{
		Name:         drivers.GCEPDCSIProvisionerName,
		Provisioner:  drivers.GCEPDCSIProvisionerName,
		Parameters:   map[string]string{"type": "pd-standard"},
		ClaimSize:    "5Gi",
		ExpectedSize: "5Gi",
	}
}
