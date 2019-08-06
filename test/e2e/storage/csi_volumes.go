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
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/util/rand"
)

// List of testDrivers to be executed in below loop
var csiTestDrivers = []func() testsuites.TestDriver{
	drivers.InitHostPathCSIDriver,
	drivers.InitGcePDCSIDriver,
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
	testsuites.InitMultiVolumeTestSuite,
	testsuites.InitDisruptiveTestSuite,
}

// This executes testSuites for csi volumes.
var _ = utils.SIGDescribe("CSI Volumes", func() {
	for _, initDriver := range csiTestDrivers {
		curDriver := initDriver()

		ginkgo.Context(testsuites.GetDriverNameWithFeatureTags(curDriver), func() {
			testsuites.DefineTestSuite(curDriver, csiTestSuites)
		})
	}

	// TODO: PD CSI driver needs to be serial because it uses a fixed name. Address as part of #71289
	ginkgo.Context("CSI Topology test using GCE PD driver [Serial]", func() {
		f := framework.NewDefaultFramework("csitopology")
		driver := drivers.InitGcePDCSIDriver().(testsuites.DynamicPVTestDriver) // TODO (#71289) eliminate by moving this test to common test suite.
		var (
			config      *testsuites.PerTestConfig
			testCleanup func()
		)
		ginkgo.BeforeEach(func() {
			driver.SkipUnsupportedTest(testpatterns.TestPattern{})
			config, testCleanup = driver.PrepareTest(f)
		})

		ginkgo.AfterEach(func() {
			if testCleanup != nil {
				testCleanup()
			}
		})

		ginkgo.It("should provision zonal PD with immediate volume binding and AllowedTopologies set and mount the volume to a pod", func() {
			suffix := "topology-positive"
			testTopologyPositive(config.Framework.ClientSet, suffix, config.Framework.Namespace.GetName(), false /* delayBinding */, true /* allowedTopologies */)
		})

		ginkgo.It("should provision zonal PD with delayed volume binding and mount the volume to a pod", func() {
			suffix := "delayed"
			testTopologyPositive(config.Framework.ClientSet, suffix, config.Framework.Namespace.GetName(), true /* delayBinding */, false /* allowedTopologies */)
		})

		ginkgo.It("should provision zonal PD with delayed volume binding and AllowedTopologies set and mount the volume to a pod", func() {
			suffix := "delayed-topology-positive"
			testTopologyPositive(config.Framework.ClientSet, suffix, config.Framework.Namespace.GetName(), true /* delayBinding */, true /* allowedTopologies */)
		})

		ginkgo.It("should fail to schedule a pod with a zone missing from AllowedTopologies; PD is provisioned with immediate volume binding", func() {
			framework.SkipUnlessMultizone(config.Framework.ClientSet)
			suffix := "topology-negative"
			testTopologyNegative(config.Framework.ClientSet, suffix, config.Framework.Namespace.GetName(), false /* delayBinding */)
		})

		ginkgo.It("should fail to schedule a pod with a zone missing from AllowedTopologies; PD is provisioned with delayed volume binding", func() {
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
	test.Claim = framework.MakePersistentVolumeClaim(framework.PersistentVolumeClaimConfig{
		ClaimSize:        test.ClaimSize,
		StorageClassName: &(class.Name),
		VolumeMode:       &test.VolumeMode,
	}, namespace)
	test.Class = class

	if delayBinding {
		_, node := test.TestBindingWaitForFirstConsumer(nil /* node selector */, false /* expect unschedulable */)
		gomega.Expect(node).ToNot(gomega.BeNil(), "Unexpected nil node found")
	} else {
		test.TestDynamicProvisioning()
	}
}

func testTopologyNegative(cs clientset.Interface, suffix, namespace string, delayBinding bool) {
	framework.SkipUnlessMultizone(cs)

	// Use different zones for pod and PV
	zones, err := framework.GetClusterZones(cs)
	framework.ExpectNoError(err)
	gomega.Expect(zones.Len()).To(gomega.BeNumerically(">=", 2))
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
	test.Claim = framework.MakePersistentVolumeClaim(framework.PersistentVolumeClaimConfig{
		ClaimSize:        test.ClaimSize,
		StorageClassName: &(test.Class.Name),
		VolumeMode:       &test.VolumeMode,
	}, namespace)
	if delayBinding {
		test.TestBindingWaitForFirstConsumer(nodeSelector, true /* expect unschedulable */)
	} else {
		test.PvCheck = func(claim *v1.PersistentVolumeClaim) {
			// Ensure that a pod cannot be scheduled in an unsuitable zone.
			pod := testsuites.StartInPodWithVolume(cs, namespace, claim.Name, "pvc-tester-unschedulable", "sleep 100000",
				framework.NodeSelection{Selector: nodeSelector})
			defer testsuites.StopPod(cs, pod)
			framework.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(cs, pod.Name, pod.Namespace), "pod should be unschedulable")
		}
		test.TestDynamicProvisioning()
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
