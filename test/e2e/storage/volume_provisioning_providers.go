// +build !providerless

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
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/framework/providers/gce"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

// checkAWSEBS checks properties of an AWS EBS. Test framework does not
// instantiate full AWS provider, therefore we need use ec2 API directly.
func checkAWSEBS(volume *v1.PersistentVolume, volumeType string, encrypted bool) error {
	diskName := volume.Spec.AWSElasticBlockStore.VolumeID

	var client *ec2.EC2

	tokens := strings.Split(diskName, "/")
	volumeID := tokens[len(tokens)-1]

	zone := framework.TestContext.CloudConfig.Zone

	awsSession, err := session.NewSession()
	if err != nil {
		return fmt.Errorf("error creating session: %v", err)
	}

	if len(zone) > 0 {
		region := zone[:len(zone)-1]
		cfg := aws.Config{Region: &region}
		framework.Logf("using region %s", region)
		client = ec2.New(awsSession, &cfg)
	} else {
		framework.Logf("no region configured")
		client = ec2.New(awsSession)
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

var _ = utils.SIGDescribe("Dynamic Provisioning with cloud providers", func() {
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
						volume := testsuites.PVWriteReadSingleNodeCheck(c, claim, e2epod.NodeSelection{})
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
						volume := testsuites.PVWriteReadSingleNodeCheck(c, claim, e2epod.NodeSelection{})
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
						volume := testsuites.PVWriteReadSingleNodeCheck(c, claim, e2epod.NodeSelection{})
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
						volume := testsuites.PVWriteReadSingleNodeCheck(c, claim, e2epod.NodeSelection{})
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
						volume := testsuites.PVWriteReadSingleNodeCheck(c, claim, e2epod.NodeSelection{})
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
						volume := testsuites.PVWriteReadSingleNodeCheck(c, claim, e2epod.NodeSelection{})
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
						volume := testsuites.PVWriteReadSingleNodeCheck(c, claim, e2epod.NodeSelection{})
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
						testsuites.PVWriteReadSingleNodeCheck(c, claim, e2epod.NodeSelection{})
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
						testsuites.PVWriteReadSingleNodeCheck(c, claim, e2epod.NodeSelection{})
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
						testsuites.PVWriteReadSingleNodeCheck(c, claim, e2epod.NodeSelection{})
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
						testsuites.PVWriteReadSingleNodeCheck(c, claim, e2epod.NodeSelection{})
					},
				},
			}

			var betaTest *testsuites.StorageClassTest
			for i, t := range tests {
				// Beware of clojure, use local variables instead of those from
				// outer scope
				test := t

				if !framework.ProviderIs(test.CloudProviders...) {
					framework.Logf("Skipping %q: cloud providers is not %v", test.Name, test.CloudProviders)
					continue
				}

				// Remember the last supported test for subsequent test of beta API
				betaTest = &test

				ginkgo.By("Testing " + test.Name)
				suffix := fmt.Sprintf("%d", i)
				test.Client = c
				test.Class = newStorageClass(test, ns, suffix)
				test.Claim = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
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
				class, err := c.StorageV1beta1().StorageClasses().Create(context.TODO(), class, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				defer deleteStorageClass(c, class.Name)

				betaTest.Client = c
				betaTest.Class = nil
				betaTest.Claim = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
					ClaimSize:        betaTest.ClaimSize,
					StorageClassName: &class.Name,
					VolumeMode:       &betaTest.VolumeMode,
				}, ns)
				betaTest.Claim.Spec.StorageClassName = &(class.Name)
				(*betaTest).TestDynamicProvisioning()
			}
		})

		ginkgo.It("should provision storage with non-default reclaim policy Retain", func() {
			e2eskipper.SkipUnlessProviderIs("gce", "gke")

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
					volume := testsuites.PVWriteReadSingleNodeCheck(c, claim, e2epod.NodeSelection{})
					gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")

					err := checkGCEPD(volume, "pd-standard")
					framework.ExpectNoError(err, "checkGCEPD")
				},
			}
			test.Class = newStorageClass(test, ns, "reclaimpolicy")
			retain := v1.PersistentVolumeReclaimRetain
			test.Class.ReclaimPolicy = &retain
			test.Claim = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
				ClaimSize:        test.ClaimSize,
				StorageClassName: &test.Class.Name,
				VolumeMode:       &test.VolumeMode,
			}, ns)
			pv := test.TestDynamicProvisioning()

			ginkgo.By(fmt.Sprintf("waiting for the provisioned PV %q to enter phase %s", pv.Name, v1.VolumeReleased))
			framework.ExpectNoError(e2epv.WaitForPersistentVolumePhase(v1.VolumeReleased, c, pv.Name, 1*time.Second, 30*time.Second))

			ginkgo.By(fmt.Sprintf("deleting the storage asset backing the PV %q", pv.Name))
			framework.ExpectNoError(e2epv.DeletePDWithRetry(pv.Spec.GCEPersistentDisk.PDName))

			ginkgo.By(fmt.Sprintf("deleting the PV %q", pv.Name))
			framework.ExpectNoError(e2epv.DeletePersistentVolume(c, pv.Name), "Failed to delete PV ", pv.Name)
			framework.ExpectNoError(e2epv.WaitForPersistentVolumeDeleted(c, pv.Name, 1*time.Second, 30*time.Second))
		})

		ginkgo.It("should not provision a volume in an unmanaged GCE zone.", func() {
			e2eskipper.SkipUnlessProviderIs("gce", "gke")
			var suffix string = "unmananged"

			ginkgo.By("Discovering an unmanaged zone")
			allZones := sets.NewString() // all zones in the project

			gceCloud, err := gce.GetGCECloud()
			framework.ExpectNoError(err)

			// Get all k8s managed zones (same as zones with nodes in them for test)
			managedZones, err := gceCloud.GetAllZonesFromCloudProvider()
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
				e2eskipper.Skipf("No unmanaged zones found.")
			}

			ginkgo.By("Creating a StorageClass for the unmanaged zone")
			test := testsuites.StorageClassTest{
				Name:        "unmanaged_zone",
				Provisioner: "kubernetes.io/gce-pd",
				Parameters:  map[string]string{"zone": unmanagedZone},
				ClaimSize:   "1Gi",
			}
			sc := newStorageClass(test, ns, suffix)
			sc, err = c.StorageV1().StorageClasses().Create(context.TODO(), sc, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			defer deleteStorageClass(c, sc.Name)

			ginkgo.By("Creating a claim and expecting it to timeout")
			pvc := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
				ClaimSize:        test.ClaimSize,
				StorageClassName: &sc.Name,
				VolumeMode:       &test.VolumeMode,
			}, ns)
			pvc, err = c.CoreV1().PersistentVolumeClaims(ns).Create(context.TODO(), pvc, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			defer func() {
				framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(c, pvc.Name, ns), "Failed to delete PVC ", pvc.Name)
			}()

			// The claim should timeout phase:Pending
			err = e2epv.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, c, ns, pvc.Name, 2*time.Second, framework.ClaimProvisionShortTimeout)
			framework.ExpectError(err)
			framework.Logf(err.Error())
		})

		ginkgo.It("should test that deleting a claim before the volume is provisioned deletes the volume.", func() {
			// This case tests for the regressions of a bug fixed by PR #21268
			// REGRESSION: Deleting the PVC before the PV is provisioned can result in the PV
			// not being deleted.
			// NOTE:  Polls until no PVs are detected, times out at 5 minutes.

			e2eskipper.SkipUnlessProviderIs("openstack", "gce", "aws", "gke", "vsphere", "azure")

			const raceAttempts int = 100
			var residualPVs []*v1.PersistentVolume
			ginkgo.By(fmt.Sprintf("Creating and deleting PersistentVolumeClaims %d times", raceAttempts))
			test := testsuites.StorageClassTest{
				Name:        "deletion race",
				Provisioner: "", // Use a native one based on current cloud provider
				ClaimSize:   "1Gi",
			}

			class := newStorageClass(test, ns, "race")
			class, err := c.StorageV1().StorageClasses().Create(context.TODO(), class, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			defer deleteStorageClass(c, class.Name)

			// To increase chance of detection, attempt multiple iterations
			for i := 0; i < raceAttempts; i++ {
				prefix := fmt.Sprintf("race-%d", i)
				claim := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
					NamePrefix:       prefix,
					ClaimSize:        test.ClaimSize,
					StorageClassName: &class.Name,
					VolumeMode:       &test.VolumeMode,
				}, ns)
				tmpClaim, err := e2epv.CreatePVC(c, ns, claim)
				framework.ExpectNoError(err)
				framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(c, tmpClaim.Name, ns))
			}

			ginkgo.By(fmt.Sprintf("Checking for residual PersistentVolumes associated with StorageClass %s", class.Name))
			residualPVs, err = waitForProvisionedVolumesDeleted(c, class.Name)
			// Cleanup the test resources before breaking
			defer deleteProvisionedVolumesAndDisks(c, residualPVs)
			framework.ExpectNoError(err, "PersistentVolumes were not deleted as expected. %d remain", len(residualPVs))

			framework.Logf("0 PersistentVolumes remain.")
		})

		ginkgo.It("deletion should be idempotent", func() {
			// This test ensures that deletion of a volume is idempotent.
			// It creates a PV with Retain policy, deletes underlying AWS / GCE
			// volume and changes the reclaim policy to Delete.
			// PV controller should delete the PV even though the underlying volume
			// is already deleted.
			e2eskipper.SkipUnlessProviderIs("gce", "gke", "aws")
			ginkgo.By("creating PD")
			diskName, err := e2epv.CreatePDWithRetry()
			framework.ExpectNoError(err)

			ginkgo.By("creating PV")
			pv := e2epv.MakePersistentVolume(e2epv.PersistentVolumeConfig{
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
			pv, err = c.CoreV1().PersistentVolumes().Create(context.TODO(), pv, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("waiting for the PV to get Released")
			err = e2epv.WaitForPersistentVolumePhase(v1.VolumeReleased, c, pv.Name, 2*time.Second, e2epv.PVReclaimingTimeout)
			framework.ExpectNoError(err)

			ginkgo.By("deleting the PD")
			err = e2epv.DeletePVSource(&pv.Spec.PersistentVolumeSource)
			framework.ExpectNoError(err)

			ginkgo.By("changing the PV reclaim policy")
			pv, err = c.CoreV1().PersistentVolumes().Get(context.TODO(), pv.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			pv.Spec.PersistentVolumeReclaimPolicy = v1.PersistentVolumeReclaimDelete
			pv, err = c.CoreV1().PersistentVolumes().Update(context.TODO(), pv, metav1.UpdateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("waiting for the PV to get deleted")
			err = e2epv.WaitForPersistentVolumeDeleted(c, pv.Name, 5*time.Second, e2epv.PVDeletingTimeout)
			framework.ExpectNoError(err)
		})
	})
})
