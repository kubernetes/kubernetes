/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"net"
	"strings"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ec2"

	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	storagev1 "k8s.io/api/storage/v1"
	storagev1beta1 "k8s.io/api/storage/v1beta1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	clientset "k8s.io/client-go/kubernetes"
	volumehelpers "k8s.io/cloud-provider/volume/helpers"
	storageutil "k8s.io/kubernetes/pkg/apis/storage/v1/util"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/auth"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/framework/providers/gce"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

const (
	// Plugin name of the external provisioner
	externalPluginName = "example.com/nfs"
	// Number of PVCs for multi PVC tests
	multiPVCcount = 3
)

func checkZoneFromLabelAndAffinity(pv *v1.PersistentVolume, zone string, matchZone bool) {
	checkZonesFromLabelAndAffinity(pv, sets.NewString(zone), matchZone)
}

// checkZoneLabelAndAffinity checks the LabelZoneFailureDomain label of PV and terms
// with key LabelZoneFailureDomain in PV's node affinity contains zone
// matchZones is used to indicate if zones should match perfectly
func checkZonesFromLabelAndAffinity(pv *v1.PersistentVolume, zones sets.String, matchZones bool) {
	ginkgo.By("checking PV's zone label and node affinity terms match expected zone")
	if pv == nil {
		e2elog.Failf("nil pv passed")
	}
	pvLabel, ok := pv.Labels[v1.LabelZoneFailureDomain]
	if !ok {
		e2elog.Failf("label %s not found on PV", v1.LabelZoneFailureDomain)
	}

	zonesFromLabel, err := volumehelpers.LabelZonesToSet(pvLabel)
	if err != nil {
		e2elog.Failf("unable to parse zone labels %s: %v", pvLabel, err)
	}
	if matchZones && !zonesFromLabel.Equal(zones) {
		e2elog.Failf("value[s] of %s label for PV: %v does not match expected zone[s]: %v", v1.LabelZoneFailureDomain, zonesFromLabel, zones)
	}
	if !matchZones && !zonesFromLabel.IsSuperset(zones) {
		e2elog.Failf("value[s] of %s label for PV: %v does not contain expected zone[s]: %v", v1.LabelZoneFailureDomain, zonesFromLabel, zones)
	}
	if pv.Spec.NodeAffinity == nil {
		e2elog.Failf("node affinity not found in PV spec %v", pv.Spec)
	}
	if len(pv.Spec.NodeAffinity.Required.NodeSelectorTerms) == 0 {
		e2elog.Failf("node selector terms not found in PV spec %v", pv.Spec)
	}

	for _, term := range pv.Spec.NodeAffinity.Required.NodeSelectorTerms {
		keyFound := false
		for _, r := range term.MatchExpressions {
			if r.Key != v1.LabelZoneFailureDomain {
				continue
			}
			keyFound = true
			zonesFromNodeAffinity := sets.NewString(r.Values...)
			if matchZones && !zonesFromNodeAffinity.Equal(zones) {
				e2elog.Failf("zones from NodeAffinity of PV: %v does not equal expected zone[s]: %v", zonesFromNodeAffinity, zones)
			}
			if !matchZones && !zonesFromNodeAffinity.IsSuperset(zones) {
				e2elog.Failf("zones from NodeAffinity of PV: %v does not contain expected zone[s]: %v", zonesFromNodeAffinity, zones)
			}
			break
		}
		if !keyFound {
			e2elog.Failf("label %s not found in term %v", v1.LabelZoneFailureDomain, term)
		}
	}
}

// checkAWSEBS checks properties of an AWS EBS. Test framework does not
// instantiate full AWS provider, therefore we need use ec2 API directly.
func checkAWSEBS(volume *v1.PersistentVolume, volumeType string, encrypted bool) error {
	diskName := volume.Spec.AWSElasticBlockStore.VolumeID

	var client *ec2.EC2

	tokens := strings.Split(diskName, "/")
	volumeID := tokens[len(tokens)-1]

	zone := framework.TestContext.CloudConfig.Zone
	if len(zone) > 0 {
		region := zone[:len(zone)-1]
		cfg := aws.Config{Region: &region}
		e2elog.Logf("using region %s", region)
		client = ec2.New(session.New(), &cfg)
	} else {
		e2elog.Logf("no region configured")
		client = ec2.New(session.New())
	}

	request := &ec2.DescribeVolumesInput{
		VolumeIds: []*string{&volumeID},
	}
	info, err := client.DescribeVolumes(request)
	if err != nil {
		return fmt.Errorf("error querying ec2 for volume %q: %v", volumeID, err)
	}
	if len(info.Volumes) == 0 {
		return fmt.Errorf("no volumes found for volume %q", volumeID)
	}
	if len(info.Volumes) > 1 {
		return fmt.Errorf("multiple volumes found for volume %q", volumeID)
	}

	awsVolume := info.Volumes[0]
	if awsVolume.VolumeType == nil {
		return fmt.Errorf("expected volume type %q, got nil", volumeType)
	}
	if *awsVolume.VolumeType != volumeType {
		return fmt.Errorf("expected volume type %q, got %q", volumeType, *awsVolume.VolumeType)
	}
	if encrypted && awsVolume.Encrypted == nil {
		return fmt.Errorf("expected encrypted volume, got no encryption")
	}
	if encrypted && !*awsVolume.Encrypted {
		return fmt.Errorf("expected encrypted volume, got %v", *awsVolume.Encrypted)
	}
	return nil
}

func checkGCEPD(volume *v1.PersistentVolume, volumeType string) error {
	cloud, err := gce.GetGCECloud()
	if err != nil {
		return err
	}
	diskName := volume.Spec.GCEPersistentDisk.PDName
	disk, err := cloud.GetDiskByNameUnknownZone(diskName)
	if err != nil {
		return err
	}

	if !strings.HasSuffix(disk.Type, volumeType) {
		return fmt.Errorf("unexpected disk type %q, expected suffix %q", disk.Type, volumeType)
	}
	return nil
}

func testZonalDelayedBinding(c clientset.Interface, ns string, specifyAllowedTopology bool, pvcCount int) {
	storageClassTestNameFmt := "Delayed binding %s storage class test %s"
	storageClassTestNameSuffix := ""
	if specifyAllowedTopology {
		storageClassTestNameSuffix += " with AllowedTopologies"
	}
	tests := []testsuites.StorageClassTest{
		{
			Name:           fmt.Sprintf(storageClassTestNameFmt, "EBS", storageClassTestNameSuffix),
			CloudProviders: []string{"aws"},
			Provisioner:    "kubernetes.io/aws-ebs",
			ClaimSize:      "2Gi",
			DelayBinding:   true,
		},
		{
			Name:           fmt.Sprintf(storageClassTestNameFmt, "GCE PD", storageClassTestNameSuffix),
			CloudProviders: []string{"gce", "gke"},
			Provisioner:    "kubernetes.io/gce-pd",
			ClaimSize:      "2Gi",
			DelayBinding:   true,
		},
	}
	for _, test := range tests {
		if !framework.ProviderIs(test.CloudProviders...) {
			e2elog.Logf("Skipping %q: cloud providers is not %v", test.Name, test.CloudProviders)
			continue
		}
		action := "creating claims with class with waitForFirstConsumer"
		suffix := "delayed"
		var topoZone string
		test.Client = c
		test.Class = newStorageClass(test, ns, suffix)
		if specifyAllowedTopology {
			action += " and allowedTopologies"
			suffix += "-topo"
			topoZone = getRandomClusterZone(c)
			addSingleZoneAllowedTopologyToStorageClass(c, test.Class, topoZone)
		}
		ginkgo.By(action)
		var claims []*v1.PersistentVolumeClaim
		for i := 0; i < pvcCount; i++ {
			claim := framework.MakePersistentVolumeClaim(framework.PersistentVolumeClaimConfig{
				ClaimSize:        test.ClaimSize,
				StorageClassName: &test.Class.Name,
				VolumeMode:       &test.VolumeMode,
			}, ns)
			claims = append(claims, claim)
		}
		pvs, node := test.TestBindingWaitForFirstConsumerMultiPVC(claims, nil /* node selector */, false /* expect unschedulable */)
		if node == nil {
			e2elog.Failf("unexpected nil node found")
		}
		zone, ok := node.Labels[v1.LabelZoneFailureDomain]
		if !ok {
			e2elog.Failf("label %s not found on Node", v1.LabelZoneFailureDomain)
		}
		if specifyAllowedTopology && topoZone != zone {
			e2elog.Failf("zone specified in allowedTopologies: %s does not match zone of node where PV got provisioned: %s", topoZone, zone)
		}
		for _, pv := range pvs {
			checkZoneFromLabelAndAffinity(pv, zone, true)
		}
	}
}

var _ = utils.SIGDescribe("Dynamic Provisioning", func() {
	f := framework.NewDefaultFramework("volume-provisioning")

	// filled in BeforeEach
	var c clientset.Interface
	var ns string

	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
	})

	ginkgo.Describe("DynamicProvisioner [Slow]", func() {
		ginkgo.It("should provision storage with different parameters", func() {

			// This test checks that dynamic provisioning can provision a volume
			// that can be used to persist data among pods.
			tests := []testsuites.StorageClassTest{
				// GCE/GKE
				{
					Name:           "SSD PD on GCE/GKE",
					CloudProviders: []string{"gce", "gke"},
					Provisioner:    "kubernetes.io/gce-pd",
					Parameters: map[string]string{
						"type": "pd-ssd",
						"zone": getRandomClusterZone(c),
					},
					ClaimSize:    "1.5Gi",
					ExpectedSize: "2Gi",
					PvCheck: func(claim *v1.PersistentVolumeClaim) {
						volume := testsuites.PVWriteReadSingleNodeCheck(c, claim, framework.NodeSelection{})
						gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")

						err := checkGCEPD(volume, "pd-ssd")
						framework.ExpectNoError(err, "checkGCEPD pd-ssd")
					},
				},
				{
					Name:           "HDD PD on GCE/GKE",
					CloudProviders: []string{"gce", "gke"},
					Provisioner:    "kubernetes.io/gce-pd",
					Parameters: map[string]string{
						"type": "pd-standard",
					},
					ClaimSize:    "1.5Gi",
					ExpectedSize: "2Gi",
					PvCheck: func(claim *v1.PersistentVolumeClaim) {
						volume := testsuites.PVWriteReadSingleNodeCheck(c, claim, framework.NodeSelection{})
						gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")

						err := checkGCEPD(volume, "pd-standard")
						framework.ExpectNoError(err, "checkGCEPD pd-standard")
					},
				},
				// AWS
				{
					Name:           "gp2 EBS on AWS",
					CloudProviders: []string{"aws"},
					Provisioner:    "kubernetes.io/aws-ebs",
					Parameters: map[string]string{
						"type": "gp2",
						"zone": getRandomClusterZone(c),
					},
					ClaimSize:    "1.5Gi",
					ExpectedSize: "2Gi",
					PvCheck: func(claim *v1.PersistentVolumeClaim) {
						volume := testsuites.PVWriteReadSingleNodeCheck(c, claim, framework.NodeSelection{})
						gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")

						err := checkAWSEBS(volume, "gp2", false)
						framework.ExpectNoError(err, "checkAWSEBS gp2")
					},
				},
				{
					Name:           "io1 EBS on AWS",
					CloudProviders: []string{"aws"},
					Provisioner:    "kubernetes.io/aws-ebs",
					Parameters: map[string]string{
						"type":      "io1",
						"iopsPerGB": "50",
					},
					ClaimSize:    "3.5Gi",
					ExpectedSize: "4Gi", // 4 GiB is minimum for io1
					PvCheck: func(claim *v1.PersistentVolumeClaim) {
						volume := testsuites.PVWriteReadSingleNodeCheck(c, claim, framework.NodeSelection{})
						gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")

						err := checkAWSEBS(volume, "io1", false)
						framework.ExpectNoError(err, "checkAWSEBS io1")
					},
				},
				{
					Name:           "sc1 EBS on AWS",
					CloudProviders: []string{"aws"},
					Provisioner:    "kubernetes.io/aws-ebs",
					Parameters: map[string]string{
						"type": "sc1",
					},
					ClaimSize:    "500Gi", // minimum for sc1
					ExpectedSize: "500Gi",
					PvCheck: func(claim *v1.PersistentVolumeClaim) {
						volume := testsuites.PVWriteReadSingleNodeCheck(c, claim, framework.NodeSelection{})
						gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")

						err := checkAWSEBS(volume, "sc1", false)
						framework.ExpectNoError(err, "checkAWSEBS sc1")
					},
				},
				{
					Name:           "st1 EBS on AWS",
					CloudProviders: []string{"aws"},
					Provisioner:    "kubernetes.io/aws-ebs",
					Parameters: map[string]string{
						"type": "st1",
					},
					ClaimSize:    "500Gi", // minimum for st1
					ExpectedSize: "500Gi",
					PvCheck: func(claim *v1.PersistentVolumeClaim) {
						volume := testsuites.PVWriteReadSingleNodeCheck(c, claim, framework.NodeSelection{})
						gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")

						err := checkAWSEBS(volume, "st1", false)
						framework.ExpectNoError(err, "checkAWSEBS st1")
					},
				},
				{
					Name:           "encrypted EBS on AWS",
					CloudProviders: []string{"aws"},
					Provisioner:    "kubernetes.io/aws-ebs",
					Parameters: map[string]string{
						"encrypted": "true",
					},
					ClaimSize:    "1Gi",
					ExpectedSize: "1Gi",
					PvCheck: func(claim *v1.PersistentVolumeClaim) {
						volume := testsuites.PVWriteReadSingleNodeCheck(c, claim, framework.NodeSelection{})
						gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")

						err := checkAWSEBS(volume, "gp2", true)
						framework.ExpectNoError(err, "checkAWSEBS gp2 encrypted")
					},
				},
				// OpenStack generic tests (works on all OpenStack deployments)
				{
					Name:           "generic Cinder volume on OpenStack",
					CloudProviders: []string{"openstack"},
					Provisioner:    "kubernetes.io/cinder",
					Parameters:     map[string]string{},
					ClaimSize:      "1.5Gi",
					ExpectedSize:   "2Gi",
					PvCheck: func(claim *v1.PersistentVolumeClaim) {
						testsuites.PVWriteReadSingleNodeCheck(c, claim, framework.NodeSelection{})
					},
				},
				{
					Name:           "Cinder volume with empty volume type and zone on OpenStack",
					CloudProviders: []string{"openstack"},
					Provisioner:    "kubernetes.io/cinder",
					Parameters: map[string]string{
						"type":         "",
						"availability": "",
					},
					ClaimSize:    "1.5Gi",
					ExpectedSize: "2Gi",
					PvCheck: func(claim *v1.PersistentVolumeClaim) {
						testsuites.PVWriteReadSingleNodeCheck(c, claim, framework.NodeSelection{})
					},
				},
				// vSphere generic test
				{
					Name:           "generic vSphere volume",
					CloudProviders: []string{"vsphere"},
					Provisioner:    "kubernetes.io/vsphere-volume",
					Parameters:     map[string]string{},
					ClaimSize:      "1.5Gi",
					ExpectedSize:   "1.5Gi",
					PvCheck: func(claim *v1.PersistentVolumeClaim) {
						testsuites.PVWriteReadSingleNodeCheck(c, claim, framework.NodeSelection{})
					},
				},
				// Azure
				{
					Name:           "Azure disk volume with empty sku and location",
					CloudProviders: []string{"azure"},
					Provisioner:    "kubernetes.io/azure-disk",
					Parameters:     map[string]string{},
					ClaimSize:      "1Gi",
					ExpectedSize:   "1Gi",
					PvCheck: func(claim *v1.PersistentVolumeClaim) {
						testsuites.PVWriteReadSingleNodeCheck(c, claim, framework.NodeSelection{})
					},
				},
			}

			var betaTest *testsuites.StorageClassTest
			for i, t := range tests {
				// Beware of clojure, use local variables instead of those from
				// outer scope
				test := t

				if !framework.ProviderIs(test.CloudProviders...) {
					e2elog.Logf("Skipping %q: cloud providers is not %v", test.Name, test.CloudProviders)
					continue
				}

				// Remember the last supported test for subsequent test of beta API
				betaTest = &test

				ginkgo.By("Testing " + test.Name)
				suffix := fmt.Sprintf("%d", i)
				test.Client = c
				test.Class = newStorageClass(test, ns, suffix)
				test.Claim = framework.MakePersistentVolumeClaim(framework.PersistentVolumeClaimConfig{
					ClaimSize:        test.ClaimSize,
					StorageClassName: &test.Class.Name,
					VolumeMode:       &test.VolumeMode,
				}, ns)
				test.TestDynamicProvisioning()
			}

			// Run the last test with storage.k8s.io/v1beta1 on pvc
			if betaTest != nil {
				ginkgo.By("Testing " + betaTest.Name + " with beta volume provisioning")
				class := newBetaStorageClass(*betaTest, "beta")
				// we need to create the class manually, testDynamicProvisioning does not accept beta class
				class, err := c.StorageV1beta1().StorageClasses().Create(class)
				framework.ExpectNoError(err)
				defer deleteStorageClass(c, class.Name)

				betaTest.Client = c
				betaTest.Class = nil
				betaTest.Claim = framework.MakePersistentVolumeClaim(framework.PersistentVolumeClaimConfig{
					ClaimSize:        betaTest.ClaimSize,
					StorageClassName: &class.Name,
					VolumeMode:       &betaTest.VolumeMode,
				}, ns)
				betaTest.Claim.Spec.StorageClassName = &(class.Name)
				(*betaTest).TestDynamicProvisioning()
			}
		})

		ginkgo.It("should provision storage with non-default reclaim policy Retain", func() {
			framework.SkipUnlessProviderIs("gce", "gke")

			test := testsuites.StorageClassTest{
				Client:         c,
				Name:           "HDD PD on GCE/GKE",
				CloudProviders: []string{"gce", "gke"},
				Provisioner:    "kubernetes.io/gce-pd",
				Parameters: map[string]string{
					"type": "pd-standard",
				},
				ClaimSize:    "1Gi",
				ExpectedSize: "1Gi",
				PvCheck: func(claim *v1.PersistentVolumeClaim) {
					volume := testsuites.PVWriteReadSingleNodeCheck(c, claim, framework.NodeSelection{})
					gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")

					err := checkGCEPD(volume, "pd-standard")
					framework.ExpectNoError(err, "checkGCEPD")
				},
			}
			test.Class = newStorageClass(test, ns, "reclaimpolicy")
			retain := v1.PersistentVolumeReclaimRetain
			test.Class.ReclaimPolicy = &retain
			test.Claim = framework.MakePersistentVolumeClaim(framework.PersistentVolumeClaimConfig{
				ClaimSize:        test.ClaimSize,
				StorageClassName: &test.Class.Name,
				VolumeMode:       &test.VolumeMode,
			}, ns)
			pv := test.TestDynamicProvisioning()

			ginkgo.By(fmt.Sprintf("waiting for the provisioned PV %q to enter phase %s", pv.Name, v1.VolumeReleased))
			framework.ExpectNoError(framework.WaitForPersistentVolumePhase(v1.VolumeReleased, c, pv.Name, 1*time.Second, 30*time.Second))

			ginkgo.By(fmt.Sprintf("deleting the storage asset backing the PV %q", pv.Name))
			framework.ExpectNoError(framework.DeletePDWithRetry(pv.Spec.GCEPersistentDisk.PDName))

			ginkgo.By(fmt.Sprintf("deleting the PV %q", pv.Name))
			framework.ExpectNoError(framework.DeletePersistentVolume(c, pv.Name), "Failed to delete PV ", pv.Name)
			framework.ExpectNoError(framework.WaitForPersistentVolumeDeleted(c, pv.Name, 1*time.Second, 30*time.Second))
		})

		ginkgo.It("should not provision a volume in an unmanaged GCE zone.", func() {
			framework.SkipUnlessProviderIs("gce", "gke")
			var suffix string = "unmananged"

			ginkgo.By("Discovering an unmanaged zone")
			allZones := sets.NewString()     // all zones in the project
			managedZones := sets.NewString() // subset of allZones

			gceCloud, err := gce.GetGCECloud()
			framework.ExpectNoError(err)

			// Get all k8s managed zones (same as zones with nodes in them for test)
			managedZones, err = gceCloud.GetAllZonesFromCloudProvider()
			framework.ExpectNoError(err)

			// Get a list of all zones in the project
			zones, err := gceCloud.ComputeServices().GA.Zones.List(framework.TestContext.CloudConfig.ProjectID).Do()
			framework.ExpectNoError(err)
			for _, z := range zones.Items {
				allZones.Insert(z.Name)
			}

			// Get the subset of zones not managed by k8s
			var unmanagedZone string
			var popped bool
			unmanagedZones := allZones.Difference(managedZones)
			// And select one of them at random.
			if unmanagedZone, popped = unmanagedZones.PopAny(); !popped {
				framework.Skipf("No unmanaged zones found.")
			}

			ginkgo.By("Creating a StorageClass for the unmanaged zone")
			test := testsuites.StorageClassTest{
				Name:        "unmanaged_zone",
				Provisioner: "kubernetes.io/gce-pd",
				Parameters:  map[string]string{"zone": unmanagedZone},
				ClaimSize:   "1Gi",
			}
			sc := newStorageClass(test, ns, suffix)
			sc, err = c.StorageV1().StorageClasses().Create(sc)
			framework.ExpectNoError(err)
			defer deleteStorageClass(c, sc.Name)

			ginkgo.By("Creating a claim and expecting it to timeout")
			pvc := framework.MakePersistentVolumeClaim(framework.PersistentVolumeClaimConfig{
				ClaimSize:        test.ClaimSize,
				StorageClassName: &sc.Name,
				VolumeMode:       &test.VolumeMode,
			}, ns)
			pvc, err = c.CoreV1().PersistentVolumeClaims(ns).Create(pvc)
			framework.ExpectNoError(err)
			defer func() {
				framework.ExpectNoError(framework.DeletePersistentVolumeClaim(c, pvc.Name, ns), "Failed to delete PVC ", pvc.Name)
			}()

			// The claim should timeout phase:Pending
			err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, c, ns, pvc.Name, 2*time.Second, framework.ClaimProvisionShortTimeout)
			framework.ExpectError(err)
			e2elog.Logf(err.Error())
		})

		ginkgo.It("should test that deleting a claim before the volume is provisioned deletes the volume.", func() {
			// This case tests for the regressions of a bug fixed by PR #21268
			// REGRESSION: Deleting the PVC before the PV is provisioned can result in the PV
			// not being deleted.
			// NOTE:  Polls until no PVs are detected, times out at 5 minutes.

			framework.SkipUnlessProviderIs("openstack", "gce", "aws", "gke", "vsphere", "azure")

			const raceAttempts int = 100
			var residualPVs []*v1.PersistentVolume
			ginkgo.By(fmt.Sprintf("Creating and deleting PersistentVolumeClaims %d times", raceAttempts))
			test := testsuites.StorageClassTest{
				Name:        "deletion race",
				Provisioner: "", // Use a native one based on current cloud provider
				ClaimSize:   "1Gi",
			}

			class := newStorageClass(test, ns, "race")
			class, err := c.StorageV1().StorageClasses().Create(class)
			framework.ExpectNoError(err)
			defer deleteStorageClass(c, class.Name)

			// To increase chance of detection, attempt multiple iterations
			for i := 0; i < raceAttempts; i++ {
				prefix := fmt.Sprintf("race-%d", i)
				claim := framework.MakePersistentVolumeClaim(framework.PersistentVolumeClaimConfig{
					NamePrefix:       prefix,
					ClaimSize:        test.ClaimSize,
					StorageClassName: &class.Name,
					VolumeMode:       &test.VolumeMode,
				}, ns)
				tmpClaim, err := framework.CreatePVC(c, ns, claim)
				framework.ExpectNoError(err)
				framework.ExpectNoError(framework.DeletePersistentVolumeClaim(c, tmpClaim.Name, ns))
			}

			ginkgo.By(fmt.Sprintf("Checking for residual PersistentVolumes associated with StorageClass %s", class.Name))
			residualPVs, err = waitForProvisionedVolumesDeleted(c, class.Name)
			framework.ExpectNoError(err)
			// Cleanup the test resources before breaking
			defer deleteProvisionedVolumesAndDisks(c, residualPVs)

			// Report indicators of regression
			if len(residualPVs) > 0 {
				e2elog.Logf("Remaining PersistentVolumes:")
				for i, pv := range residualPVs {
					e2elog.Logf("\t%d) %s", i+1, pv.Name)
				}
				e2elog.Failf("Expected 0 PersistentVolumes remaining. Found %d", len(residualPVs))
			}
			e2elog.Logf("0 PersistentVolumes remain.")
		})

		ginkgo.It("deletion should be idempotent", func() {
			// This test ensures that deletion of a volume is idempotent.
			// It creates a PV with Retain policy, deletes underlying AWS / GCE
			// volume and changes the reclaim policy to Delete.
			// PV controller should delete the PV even though the underlying volume
			// is already deleted.
			framework.SkipUnlessProviderIs("gce", "gke", "aws")
			ginkgo.By("creating PD")
			diskName, err := framework.CreatePDWithRetry()
			framework.ExpectNoError(err)

			ginkgo.By("creating PV")
			pv := framework.MakePersistentVolume(framework.PersistentVolumeConfig{
				NamePrefix: "volume-idempotent-delete-",
				// Use Retain to keep the PV, the test will change it to Delete
				// when the time comes.
				ReclaimPolicy: v1.PersistentVolumeReclaimRetain,
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
				},
				Capacity: "1Gi",
				// PV is bound to non-existing PVC, so it's reclaim policy is
				// executed immediately
				Prebind: &v1.PersistentVolumeClaim{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "dummy-claim-name",
						Namespace: ns,
						UID:       types.UID("01234567890"),
					},
				},
			})
			switch framework.TestContext.Provider {
			case "aws":
				pv.Spec.PersistentVolumeSource = v1.PersistentVolumeSource{
					AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
						VolumeID: diskName,
					},
				}
			case "gce", "gke":
				pv.Spec.PersistentVolumeSource = v1.PersistentVolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
						PDName: diskName,
					},
				}
			}
			pv, err = c.CoreV1().PersistentVolumes().Create(pv)
			framework.ExpectNoError(err)

			ginkgo.By("waiting for the PV to get Released")
			err = framework.WaitForPersistentVolumePhase(v1.VolumeReleased, c, pv.Name, 2*time.Second, framework.PVReclaimingTimeout)
			framework.ExpectNoError(err)

			ginkgo.By("deleting the PD")
			err = framework.DeletePVSource(&pv.Spec.PersistentVolumeSource)
			framework.ExpectNoError(err)

			ginkgo.By("changing the PV reclaim policy")
			pv, err = c.CoreV1().PersistentVolumes().Get(pv.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			pv.Spec.PersistentVolumeReclaimPolicy = v1.PersistentVolumeReclaimDelete
			pv, err = c.CoreV1().PersistentVolumes().Update(pv)
			framework.ExpectNoError(err)

			ginkgo.By("waiting for the PV to get deleted")
			err = framework.WaitForPersistentVolumeDeleted(c, pv.Name, 5*time.Second, framework.PVDeletingTimeout)
			framework.ExpectNoError(err)
		})
	})

	ginkgo.Describe("DynamicProvisioner External", func() {
		ginkgo.It("should let an external dynamic provisioner create and delete persistent volumes [Slow]", func() {
			// external dynamic provisioner pods need additional permissions provided by the
			// persistent-volume-provisioner clusterrole and a leader-locking role
			serviceAccountName := "default"
			subject := rbacv1.Subject{
				Kind:      rbacv1.ServiceAccountKind,
				Namespace: ns,
				Name:      serviceAccountName,
			}

			err := auth.BindClusterRole(c.RbacV1(), "system:persistent-volume-provisioner", ns, subject)
			framework.ExpectNoError(err)

			roleName := "leader-locking-nfs-provisioner"
			_, err = f.ClientSet.RbacV1().Roles(ns).Create(&rbacv1.Role{
				ObjectMeta: metav1.ObjectMeta{
					Name: roleName,
				},
				Rules: []rbacv1.PolicyRule{{
					APIGroups: []string{""},
					Resources: []string{"endpoints"},
					Verbs:     []string{"get", "list", "watch", "create", "update", "patch"},
				}},
			})
			framework.ExpectNoError(err, "Failed to create leader-locking role")

			err = auth.BindRoleInNamespace(c.RbacV1(), roleName, ns, subject)
			framework.ExpectNoError(err)

			err = auth.WaitForAuthorizationUpdate(c.AuthorizationV1(),
				serviceaccount.MakeUsername(ns, serviceAccountName),
				"", "get", schema.GroupResource{Group: "storage.k8s.io", Resource: "storageclasses"}, true)
			framework.ExpectNoError(err, "Failed to update authorization")

			ginkgo.By("creating an external dynamic provisioner pod")
			pod := utils.StartExternalProvisioner(c, ns, externalPluginName)
			defer e2epod.DeletePodOrFail(c, ns, pod.Name)

			ginkgo.By("creating a StorageClass")
			test := testsuites.StorageClassTest{
				Client:       c,
				Name:         "external provisioner test",
				Provisioner:  externalPluginName,
				ClaimSize:    "1500Mi",
				ExpectedSize: "1500Mi",
			}
			test.Class = newStorageClass(test, ns, "external")
			test.Claim = framework.MakePersistentVolumeClaim(framework.PersistentVolumeClaimConfig{
				ClaimSize:        test.ClaimSize,
				StorageClassName: &test.Class.Name,
				VolumeMode:       &test.VolumeMode,
			}, ns)

			ginkgo.By("creating a claim with a external provisioning annotation")
			test.TestDynamicProvisioning()
		})
	})

	ginkgo.Describe("DynamicProvisioner Default", func() {
		ginkgo.It("should create and delete default persistent volumes [Slow]", func() {
			framework.SkipUnlessProviderIs("openstack", "gce", "aws", "gke", "vsphere", "azure")

			ginkgo.By("creating a claim with no annotation")
			test := testsuites.StorageClassTest{
				Client:       c,
				Name:         "default",
				ClaimSize:    "2Gi",
				ExpectedSize: "2Gi",
			}

			test.Claim = framework.MakePersistentVolumeClaim(framework.PersistentVolumeClaimConfig{
				ClaimSize:  test.ClaimSize,
				VolumeMode: &test.VolumeMode,
			}, ns)
			test.TestDynamicProvisioning()
		})

		// Modifying the default storage class can be disruptive to other tests that depend on it
		ginkgo.It("should be disabled by changing the default annotation [Serial] [Disruptive]", func() {
			framework.SkipUnlessProviderIs("openstack", "gce", "aws", "gke", "vsphere", "azure")
			scName, scErr := framework.GetDefaultStorageClassName(c)
			if scErr != nil {
				e2elog.Failf(scErr.Error())
			}
			test := testsuites.StorageClassTest{
				Name:      "default",
				ClaimSize: "2Gi",
			}

			ginkgo.By("setting the is-default StorageClass annotation to false")
			verifyDefaultStorageClass(c, scName, true)
			defer updateDefaultStorageClass(c, scName, "true")
			updateDefaultStorageClass(c, scName, "false")

			ginkgo.By("creating a claim with default storageclass and expecting it to timeout")
			claim := framework.MakePersistentVolumeClaim(framework.PersistentVolumeClaimConfig{
				ClaimSize:  test.ClaimSize,
				VolumeMode: &test.VolumeMode,
			}, ns)
			claim, err := c.CoreV1().PersistentVolumeClaims(ns).Create(claim)
			framework.ExpectNoError(err)
			defer func() {
				framework.ExpectNoError(framework.DeletePersistentVolumeClaim(c, claim.Name, ns))
			}()

			// The claim should timeout phase:Pending
			err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, c, ns, claim.Name, 2*time.Second, framework.ClaimProvisionShortTimeout)
			framework.ExpectError(err)
			e2elog.Logf(err.Error())
			claim, err = c.CoreV1().PersistentVolumeClaims(ns).Get(claim.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			framework.ExpectEqual(claim.Status.Phase, v1.ClaimPending)
		})

		// Modifying the default storage class can be disruptive to other tests that depend on it
		ginkgo.It("should be disabled by removing the default annotation [Serial] [Disruptive]", func() {
			framework.SkipUnlessProviderIs("openstack", "gce", "aws", "gke", "vsphere", "azure")
			scName, scErr := framework.GetDefaultStorageClassName(c)
			if scErr != nil {
				e2elog.Failf(scErr.Error())
			}
			test := testsuites.StorageClassTest{
				Name:      "default",
				ClaimSize: "2Gi",
			}

			ginkgo.By("removing the is-default StorageClass annotation")
			verifyDefaultStorageClass(c, scName, true)
			defer updateDefaultStorageClass(c, scName, "true")
			updateDefaultStorageClass(c, scName, "")

			ginkgo.By("creating a claim with default storageclass and expecting it to timeout")
			claim := framework.MakePersistentVolumeClaim(framework.PersistentVolumeClaimConfig{
				ClaimSize:  test.ClaimSize,
				VolumeMode: &test.VolumeMode,
			}, ns)
			claim, err := c.CoreV1().PersistentVolumeClaims(ns).Create(claim)
			framework.ExpectNoError(err)
			defer func() {
				framework.ExpectNoError(framework.DeletePersistentVolumeClaim(c, claim.Name, ns))
			}()

			// The claim should timeout phase:Pending
			err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, c, ns, claim.Name, 2*time.Second, framework.ClaimProvisionShortTimeout)
			framework.ExpectError(err)
			e2elog.Logf(err.Error())
			claim, err = c.CoreV1().PersistentVolumeClaims(ns).Get(claim.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			framework.ExpectEqual(claim.Status.Phase, v1.ClaimPending)
		})
	})

	framework.KubeDescribe("GlusterDynamicProvisioner", func() {
		ginkgo.It("should create and delete persistent volumes [fast]", func() {
			framework.SkipIfProviderIs("gke")
			ginkgo.By("creating a Gluster DP server Pod")
			pod := startGlusterDpServerPod(c, ns)
			serverURL := "http://" + net.JoinHostPort(pod.Status.PodIP, "8081")
			ginkgo.By("creating a StorageClass")
			test := testsuites.StorageClassTest{
				Client:       c,
				Name:         "Gluster Dynamic provisioner test",
				Provisioner:  "kubernetes.io/glusterfs",
				ClaimSize:    "2Gi",
				ExpectedSize: "2Gi",
				Parameters:   map[string]string{"resturl": serverURL},
			}
			suffix := fmt.Sprintf("glusterdptest")
			test.Class = newStorageClass(test, ns, suffix)

			ginkgo.By("creating a claim object with a suffix for gluster dynamic provisioner")
			test.Claim = framework.MakePersistentVolumeClaim(framework.PersistentVolumeClaimConfig{
				ClaimSize:        test.ClaimSize,
				StorageClassName: &test.Class.Name,
				VolumeMode:       &test.VolumeMode,
			}, ns)

			test.TestDynamicProvisioning()
		})
	})

	ginkgo.Describe("Invalid AWS KMS key", func() {
		ginkgo.It("should report an error and create no PV", func() {
			framework.SkipUnlessProviderIs("aws")
			test := testsuites.StorageClassTest{
				Name:        "AWS EBS with invalid KMS key",
				Provisioner: "kubernetes.io/aws-ebs",
				ClaimSize:   "2Gi",
				Parameters:  map[string]string{"kmsKeyId": "arn:aws:kms:us-east-1:123456789012:key/55555555-5555-5555-5555-555555555555"},
			}

			ginkgo.By("creating a StorageClass")
			suffix := fmt.Sprintf("invalid-aws")
			class := newStorageClass(test, ns, suffix)
			class, err := c.StorageV1().StorageClasses().Create(class)
			framework.ExpectNoError(err)
			defer func() {
				e2elog.Logf("deleting storage class %s", class.Name)
				framework.ExpectNoError(c.StorageV1().StorageClasses().Delete(class.Name, nil))
			}()

			ginkgo.By("creating a claim object with a suffix for gluster dynamic provisioner")
			claim := framework.MakePersistentVolumeClaim(framework.PersistentVolumeClaimConfig{
				ClaimSize:        test.ClaimSize,
				StorageClassName: &class.Name,
				VolumeMode:       &test.VolumeMode,
			}, ns)
			claim, err = c.CoreV1().PersistentVolumeClaims(claim.Namespace).Create(claim)
			framework.ExpectNoError(err)
			defer func() {
				e2elog.Logf("deleting claim %q/%q", claim.Namespace, claim.Name)
				err = c.CoreV1().PersistentVolumeClaims(claim.Namespace).Delete(claim.Name, nil)
				if err != nil && !apierrs.IsNotFound(err) {
					e2elog.Failf("Error deleting claim %q. Error: %v", claim.Name, err)
				}
			}()

			// Watch events until the message about invalid key appears.
			// Event delivery is not reliable and it's used only as a quick way how to check if volume with wrong KMS
			// key was not provisioned. If the event is not delivered, we check that the volume is not Bound for whole
			// ClaimProvisionTimeout in the very same loop.
			err = wait.Poll(time.Second, framework.ClaimProvisionTimeout, func() (bool, error) {
				events, err := c.CoreV1().Events(claim.Namespace).List(metav1.ListOptions{})
				framework.ExpectNoError(err)
				for _, event := range events.Items {
					if strings.Contains(event.Message, "failed to create encrypted volume: the volume disappeared after creation, most likely due to inaccessible KMS encryption key") {
						return true, nil
					}
				}

				pvc, err := c.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(claim.Name, metav1.GetOptions{})
				if err != nil {
					return true, err
				}
				if pvc.Status.Phase != v1.ClaimPending {
					// The PVC was bound to something, i.e. PV was created for wrong KMS key. That's bad!
					return true, fmt.Errorf("PVC got unexpectedly %s (to PV %q)", pvc.Status.Phase, pvc.Spec.VolumeName)
				}

				return false, nil
			})
			if err == wait.ErrWaitTimeout {
				e2elog.Logf("The test missed event about failed provisioning, but checked that no volume was provisioned for %v", framework.ClaimProvisionTimeout)
				err = nil
			}
			framework.ExpectNoError(err)
		})
	})
	ginkgo.Describe("DynamicProvisioner delayed binding [Slow]", func() {
		ginkgo.It("should create persistent volumes in the same zone as node after a pod mounting the claims is started", func() {
			testZonalDelayedBinding(c, ns, false /*specifyAllowedTopology*/, 1 /*pvcCount*/)
			testZonalDelayedBinding(c, ns, false /*specifyAllowedTopology*/, 3 /*pvcCount*/)
		})
	})
	ginkgo.Describe("DynamicProvisioner allowedTopologies", func() {
		ginkgo.It("should create persistent volume in the zone specified in allowedTopologies of storageclass", func() {
			tests := []testsuites.StorageClassTest{
				{
					Name:           "AllowedTopologies EBS storage class test",
					CloudProviders: []string{"aws"},
					Provisioner:    "kubernetes.io/aws-ebs",
					ClaimSize:      "2Gi",
					ExpectedSize:   "2Gi",
				},
				{
					Name:           "AllowedTopologies GCE PD storage class test",
					CloudProviders: []string{"gce", "gke"},
					Provisioner:    "kubernetes.io/gce-pd",
					ClaimSize:      "2Gi",
					ExpectedSize:   "2Gi",
				},
			}
			for _, test := range tests {
				if !framework.ProviderIs(test.CloudProviders...) {
					e2elog.Logf("Skipping %q: cloud providers is not %v", test.Name, test.CloudProviders)
					continue
				}
				ginkgo.By("creating a claim with class with allowedTopologies set")
				suffix := "topology"
				test.Client = c
				test.Class = newStorageClass(test, ns, suffix)
				zone := getRandomClusterZone(c)
				addSingleZoneAllowedTopologyToStorageClass(c, test.Class, zone)
				test.Claim = framework.MakePersistentVolumeClaim(framework.PersistentVolumeClaimConfig{
					ClaimSize:        test.ClaimSize,
					StorageClassName: &test.Class.Name,
					VolumeMode:       &test.VolumeMode,
				}, ns)
				pv := test.TestDynamicProvisioning()
				checkZoneFromLabelAndAffinity(pv, zone, true)
			}
		})
	})
	ginkgo.Describe("DynamicProvisioner delayed binding with allowedTopologies [Slow]", func() {
		ginkgo.It("should create persistent volumes in the same zone as specified in allowedTopologies after a pod mounting the claims is started", func() {
			testZonalDelayedBinding(c, ns, true /*specifyAllowedTopology*/, 1 /*pvcCount*/)
			testZonalDelayedBinding(c, ns, true /*specifyAllowedTopology*/, 3 /*pvcCount*/)
		})
	})
})

func verifyDefaultStorageClass(c clientset.Interface, scName string, expectedDefault bool) {
	sc, err := c.StorageV1().StorageClasses().Get(scName, metav1.GetOptions{})
	framework.ExpectNoError(err)
	framework.ExpectEqual(storageutil.IsDefaultAnnotation(sc.ObjectMeta), expectedDefault)
}

func updateDefaultStorageClass(c clientset.Interface, scName string, defaultStr string) {
	sc, err := c.StorageV1().StorageClasses().Get(scName, metav1.GetOptions{})
	framework.ExpectNoError(err)

	if defaultStr == "" {
		delete(sc.Annotations, storageutil.BetaIsDefaultStorageClassAnnotation)
		delete(sc.Annotations, storageutil.IsDefaultStorageClassAnnotation)
	} else {
		if sc.Annotations == nil {
			sc.Annotations = make(map[string]string)
		}
		sc.Annotations[storageutil.BetaIsDefaultStorageClassAnnotation] = defaultStr
		sc.Annotations[storageutil.IsDefaultStorageClassAnnotation] = defaultStr
	}

	sc, err = c.StorageV1().StorageClasses().Update(sc)
	framework.ExpectNoError(err)

	expectedDefault := false
	if defaultStr == "true" {
		expectedDefault = true
	}
	verifyDefaultStorageClass(c, scName, expectedDefault)
}

func getDefaultPluginName() string {
	switch {
	case framework.ProviderIs("gke"), framework.ProviderIs("gce"):
		return "kubernetes.io/gce-pd"
	case framework.ProviderIs("aws"):
		return "kubernetes.io/aws-ebs"
	case framework.ProviderIs("openstack"):
		return "kubernetes.io/cinder"
	case framework.ProviderIs("vsphere"):
		return "kubernetes.io/vsphere-volume"
	case framework.ProviderIs("azure"):
		return "kubernetes.io/azure-disk"
	}
	return ""
}

func addSingleZoneAllowedTopologyToStorageClass(c clientset.Interface, sc *storagev1.StorageClass, zone string) {
	term := v1.TopologySelectorTerm{
		MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
			{
				Key:    v1.LabelZoneFailureDomain,
				Values: []string{zone},
			},
		},
	}
	sc.AllowedTopologies = append(sc.AllowedTopologies, term)
}

func newStorageClass(t testsuites.StorageClassTest, ns string, suffix string) *storagev1.StorageClass {
	pluginName := t.Provisioner
	if pluginName == "" {
		pluginName = getDefaultPluginName()
	}
	if suffix == "" {
		suffix = "sc"
	}
	bindingMode := storagev1.VolumeBindingImmediate
	if t.DelayBinding {
		bindingMode = storagev1.VolumeBindingWaitForFirstConsumer
	}
	sc := getStorageClass(pluginName, t.Parameters, &bindingMode, ns, suffix)
	if t.AllowVolumeExpansion {
		sc.AllowVolumeExpansion = &t.AllowVolumeExpansion
	}
	return sc
}

func getStorageClass(
	provisioner string,
	parameters map[string]string,
	bindingMode *storagev1.VolumeBindingMode,
	ns string,
	suffix string,
) *storagev1.StorageClass {
	if bindingMode == nil {
		defaultBindingMode := storagev1.VolumeBindingImmediate
		bindingMode = &defaultBindingMode
	}
	return &storagev1.StorageClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			// Name must be unique, so let's base it on namespace name
			Name: ns + "-" + suffix,
		},
		Provisioner:       provisioner,
		Parameters:        parameters,
		VolumeBindingMode: bindingMode,
	}
}

// TODO: remove when storage.k8s.io/v1beta1 is removed.
func newBetaStorageClass(t testsuites.StorageClassTest, suffix string) *storagev1beta1.StorageClass {
	pluginName := t.Provisioner

	if pluginName == "" {
		pluginName = getDefaultPluginName()
	}
	if suffix == "" {
		suffix = "default"
	}

	return &storagev1beta1.StorageClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: suffix + "-",
		},
		Provisioner: pluginName,
		Parameters:  t.Parameters,
	}
}

func startGlusterDpServerPod(c clientset.Interface, ns string) *v1.Pod {
	podClient := c.CoreV1().Pods(ns)

	provisionerPod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "glusterdynamic-provisioner-",
		},

		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "glusterdynamic-provisioner",
					Image: "docker.io/humblec/glusterdynamic-provisioner:v1.0",
					Args: []string{
						"-config=" + "/etc/heketi/heketi.json",
					},
					Ports: []v1.ContainerPort{
						{Name: "heketi", ContainerPort: 8081},
					},
					Env: []v1.EnvVar{
						{
							Name: "POD_IP",
							ValueFrom: &v1.EnvVarSource{
								FieldRef: &v1.ObjectFieldSelector{
									FieldPath: "status.podIP",
								},
							},
						},
					},
					ImagePullPolicy: v1.PullIfNotPresent,
				},
			},
		},
	}
	provisionerPod, err := podClient.Create(provisionerPod)
	framework.ExpectNoError(err, "Failed to create %s pod: %v", provisionerPod.Name, err)

	framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(c, provisionerPod))

	ginkgo.By("locating the provisioner pod")
	pod, err := podClient.Get(provisionerPod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "Cannot locate the provisioner pod %v: %v", provisionerPod.Name, err)
	return pod
}

// waitForProvisionedVolumesDelete is a polling wrapper to scan all PersistentVolumes for any associated to the test's
// StorageClass.  Returns either an error and nil values or the remaining PVs and their count.
func waitForProvisionedVolumesDeleted(c clientset.Interface, scName string) ([]*v1.PersistentVolume, error) {
	var remainingPVs []*v1.PersistentVolume

	err := wait.Poll(10*time.Second, 300*time.Second, func() (bool, error) {
		remainingPVs = []*v1.PersistentVolume{}

		allPVs, err := c.CoreV1().PersistentVolumes().List(metav1.ListOptions{})
		if err != nil {
			return true, err
		}
		for _, pv := range allPVs.Items {
			if pv.Spec.StorageClassName == scName {
				remainingPVs = append(remainingPVs, &pv)
			}
		}
		if len(remainingPVs) > 0 {
			return false, nil // Poll until no PVs remain
		}
		return true, nil // No PVs remain
	})
	return remainingPVs, err
}

// deleteStorageClass deletes the passed in StorageClass and catches errors other than "Not Found"
func deleteStorageClass(c clientset.Interface, className string) {
	err := c.StorageV1().StorageClasses().Delete(className, nil)
	if err != nil && !apierrs.IsNotFound(err) {
		framework.ExpectNoError(err)
	}
}

// deleteProvisionedVolumes [gce||gke only]  iteratively deletes persistent volumes and attached GCE PDs.
func deleteProvisionedVolumesAndDisks(c clientset.Interface, pvs []*v1.PersistentVolume) {
	for _, pv := range pvs {
		framework.ExpectNoError(framework.DeletePDWithRetry(pv.Spec.PersistentVolumeSource.GCEPersistentDisk.PDName))
		framework.ExpectNoError(framework.DeletePersistentVolume(c, pv.Name))
	}
}

func getRandomClusterZone(c clientset.Interface) string {
	zones, err := framework.GetClusterZones(c)
	framework.ExpectNoError(err)
	framework.ExpectNotEqual(len(zones), 0)

	zonesList := zones.UnsortedList()
	return zonesList[rand.Intn(zones.Len())]
}
