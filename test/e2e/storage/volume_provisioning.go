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
	"net"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ec2"

	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	clientset "k8s.io/client-go/kubernetes"
	storageutil "k8s.io/kubernetes/pkg/apis/storage/util"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eauth "k8s.io/kubernetes/test/e2e/framework/auth"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/framework/providers/gce"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	// Plugin name of the external provisioner
	externalPluginName = "example.com/nfs"
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

var _ = utils.SIGDescribe("Dynamic Provisioning", func() {
	f := framework.NewDefaultFramework("volume-provisioning")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	// filled in BeforeEach
	var c clientset.Interface
	var timeouts *framework.TimeoutContext
	var ns string

	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
		timeouts = f.Timeouts
	})

	ginkgo.Describe("DynamicProvisioner [Slow] [Feature:StorageProvider]", func() {
		ginkgo.It("should provision storage with different parameters", func() {

			// This test checks that dynamic provisioning can provision a volume
			// that can be used to persist data among pods.
			tests := []testsuites.StorageClassTest{
				// GCE/GKE
				{
					Name:           "SSD PD on GCE/GKE",
					CloudProviders: []string{"gce", "gke"},
					Timeouts:       f.Timeouts,
					Provisioner:    "kubernetes.io/gce-pd",
					Parameters: map[string]string{
						"type": "pd-ssd",
						"zone": getRandomClusterZone(c),
					},
					ClaimSize:    "1.5Gi",
					ExpectedSize: "2Gi",
					PvCheck: func(claim *v1.PersistentVolumeClaim) {
						volume := testsuites.PVWriteReadSingleNodeCheck(c, f.Timeouts, claim, e2epod.NodeSelection{})
						gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")

						err := checkGCEPD(volume, "pd-ssd")
						framework.ExpectNoError(err, "checkGCEPD pd-ssd")
					},
				},
				{
					Name:           "HDD PD on GCE/GKE",
					CloudProviders: []string{"gce", "gke"},
					Timeouts:       f.Timeouts,
					Provisioner:    "kubernetes.io/gce-pd",
					Parameters: map[string]string{
						"type": "pd-standard",
					},
					ClaimSize:    "1.5Gi",
					ExpectedSize: "2Gi",
					PvCheck: func(claim *v1.PersistentVolumeClaim) {
						volume := testsuites.PVWriteReadSingleNodeCheck(c, f.Timeouts, claim, e2epod.NodeSelection{})
						gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")

						err := checkGCEPD(volume, "pd-standard")
						framework.ExpectNoError(err, "checkGCEPD pd-standard")
					},
				},
				// AWS
				{
					Name:           "gp2 EBS on AWS",
					CloudProviders: []string{"aws"},
					Timeouts:       f.Timeouts,
					Provisioner:    "kubernetes.io/aws-ebs",
					Parameters: map[string]string{
						"type": "gp2",
						"zone": getRandomClusterZone(c),
					},
					ClaimSize:    "1.5Gi",
					ExpectedSize: "2Gi",
					PvCheck: func(claim *v1.PersistentVolumeClaim) {
						volume := testsuites.PVWriteReadSingleNodeCheck(c, f.Timeouts, claim, e2epod.NodeSelection{})
						gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")

						err := checkAWSEBS(volume, "gp2", false)
						framework.ExpectNoError(err, "checkAWSEBS gp2")
					},
				},
				{
					Name:           "io1 EBS on AWS",
					CloudProviders: []string{"aws"},
					Timeouts:       f.Timeouts,
					Provisioner:    "kubernetes.io/aws-ebs",
					Parameters: map[string]string{
						"type":      "io1",
						"iopsPerGB": "50",
					},
					ClaimSize:    "3.5Gi",
					ExpectedSize: "4Gi", // 4 GiB is minimum for io1
					PvCheck: func(claim *v1.PersistentVolumeClaim) {
						volume := testsuites.PVWriteReadSingleNodeCheck(c, f.Timeouts, claim, e2epod.NodeSelection{})
						gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")

						err := checkAWSEBS(volume, "io1", false)
						framework.ExpectNoError(err, "checkAWSEBS io1")
					},
				},
				{
					Name:           "sc1 EBS on AWS",
					CloudProviders: []string{"aws"},
					Timeouts:       f.Timeouts,
					Provisioner:    "kubernetes.io/aws-ebs",
					Parameters: map[string]string{
						"type": "sc1",
					},
					ClaimSize:    "500Gi", // minimum for sc1
					ExpectedSize: "500Gi",
					PvCheck: func(claim *v1.PersistentVolumeClaim) {
						volume := testsuites.PVWriteReadSingleNodeCheck(c, f.Timeouts, claim, e2epod.NodeSelection{})
						gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")

						err := checkAWSEBS(volume, "sc1", false)
						framework.ExpectNoError(err, "checkAWSEBS sc1")
					},
				},
				{
					Name:           "st1 EBS on AWS",
					CloudProviders: []string{"aws"},
					Timeouts:       f.Timeouts,
					Provisioner:    "kubernetes.io/aws-ebs",
					Parameters: map[string]string{
						"type": "st1",
					},
					ClaimSize:    "500Gi", // minimum for st1
					ExpectedSize: "500Gi",
					PvCheck: func(claim *v1.PersistentVolumeClaim) {
						volume := testsuites.PVWriteReadSingleNodeCheck(c, f.Timeouts, claim, e2epod.NodeSelection{})
						gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")

						err := checkAWSEBS(volume, "st1", false)
						framework.ExpectNoError(err, "checkAWSEBS st1")
					},
				},
				{
					Name:           "encrypted EBS on AWS",
					CloudProviders: []string{"aws"},
					Timeouts:       f.Timeouts,
					Provisioner:    "kubernetes.io/aws-ebs",
					Parameters: map[string]string{
						"encrypted": "true",
					},
					ClaimSize:    "1Gi",
					ExpectedSize: "1Gi",
					PvCheck: func(claim *v1.PersistentVolumeClaim) {
						volume := testsuites.PVWriteReadSingleNodeCheck(c, f.Timeouts, claim, e2epod.NodeSelection{})
						gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")

						err := checkAWSEBS(volume, "gp2", true)
						framework.ExpectNoError(err, "checkAWSEBS gp2 encrypted")
					},
				},
				// vSphere generic test
				{
					Name:           "generic vSphere volume",
					CloudProviders: []string{"vsphere"},
					Timeouts:       f.Timeouts,
					Provisioner:    "kubernetes.io/vsphere-volume",
					Parameters:     map[string]string{},
					ClaimSize:      "1.5Gi",
					ExpectedSize:   "1.5Gi",
					PvCheck: func(claim *v1.PersistentVolumeClaim) {
						testsuites.PVWriteReadSingleNodeCheck(c, f.Timeouts, claim, e2epod.NodeSelection{})
					},
				},
				// Azure
				{
					Name:           "Azure disk volume with empty sku and location",
					CloudProviders: []string{"azure"},
					Timeouts:       f.Timeouts,
					Provisioner:    "kubernetes.io/azure-disk",
					Parameters:     map[string]string{},
					ClaimSize:      "1Gi",
					ExpectedSize:   "1Gi",
					PvCheck: func(claim *v1.PersistentVolumeClaim) {
						testsuites.PVWriteReadSingleNodeCheck(c, f.Timeouts, claim, e2epod.NodeSelection{})
					},
				},
			}

			for i, t := range tests {
				// Beware of closure, use local variables instead of those from
				// outer scope
				test := t

				if !framework.ProviderIs(test.CloudProviders...) {
					framework.Logf("Skipping %q: cloud providers is not %v", test.Name, test.CloudProviders)
					continue
				}

				if zone, ok := test.Parameters["zone"]; ok {
					framework.ExpectNotEqual(len(zone), 0, "expect at least one zone")
				}

				ginkgo.By("Testing " + test.Name)
				suffix := fmt.Sprintf("%d", i)
				test.Client = c

				// overwrite StorageClass spec with provisioned StorageClass
				storageClass, clearStorageClass := testsuites.SetupStorageClass(test.Client, newStorageClass(test, ns, suffix))
				defer clearStorageClass()

				test.Class = storageClass
				test.Claim = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
					ClaimSize:        test.ClaimSize,
					StorageClassName: &test.Class.Name,
					VolumeMode:       &test.VolumeMode,
				}, ns)

				test.TestDynamicProvisioning()
			}
		})

		ginkgo.It("should provision storage with non-default reclaim policy Retain", func() {
			e2eskipper.SkipUnlessProviderIs("gce", "gke")

			test := testsuites.StorageClassTest{
				Client:         c,
				Name:           "HDD PD on GCE/GKE",
				CloudProviders: []string{"gce", "gke"},
				Provisioner:    "kubernetes.io/gce-pd",
				Timeouts:       f.Timeouts,
				Parameters: map[string]string{
					"type": "pd-standard",
				},
				ClaimSize:    "1Gi",
				ExpectedSize: "1Gi",
				PvCheck: func(claim *v1.PersistentVolumeClaim) {
					volume := testsuites.PVWriteReadSingleNodeCheck(c, f.Timeouts, claim, e2epod.NodeSelection{})
					gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")

					err := checkGCEPD(volume, "pd-standard")
					framework.ExpectNoError(err, "checkGCEPD")
				},
			}
			test.Class = newStorageClass(test, ns, "reclaimpolicy")
			retain := v1.PersistentVolumeReclaimRetain
			test.Class.ReclaimPolicy = &retain
			storageClass, clearStorageClass := testsuites.SetupStorageClass(test.Client, test.Class)
			defer clearStorageClass()
			test.Class = storageClass

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

		ginkgo.It("should test that deleting a claim before the volume is provisioned deletes the volume.", func() {
			// This case tests for the regressions of a bug fixed by PR #21268
			// REGRESSION: Deleting the PVC before the PV is provisioned can result in the PV
			// not being deleted.
			// NOTE:  Polls until no PVs are detected, times out at 5 minutes.

			e2eskipper.SkipUnlessProviderIs("gce", "aws", "gke", "vsphere", "azure")

			const raceAttempts int = 100
			var residualPVs []*v1.PersistentVolume
			ginkgo.By(fmt.Sprintf("Creating and deleting PersistentVolumeClaims %d times", raceAttempts))
			test := testsuites.StorageClassTest{
				Name:        "deletion race",
				Provisioner: "", // Use a native one based on current cloud provider
				Timeouts:    f.Timeouts,
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
			err = e2epv.WaitForPersistentVolumePhase(v1.VolumeReleased, c, pv.Name, 2*time.Second, timeouts.PVReclaim)
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
			err = e2epv.WaitForPersistentVolumeDeleted(c, pv.Name, 5*time.Second, timeouts.PVDelete)
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

			err := e2eauth.BindClusterRole(c.RbacV1(), "system:persistent-volume-provisioner", ns, subject)
			framework.ExpectNoError(err)

			roleName := "leader-locking-nfs-provisioner"
			_, err = f.ClientSet.RbacV1().Roles(ns).Create(context.TODO(), &rbacv1.Role{
				ObjectMeta: metav1.ObjectMeta{
					Name: roleName,
				},
				Rules: []rbacv1.PolicyRule{{
					APIGroups: []string{""},
					Resources: []string{"endpoints"},
					Verbs:     []string{"get", "list", "watch", "create", "update", "patch"},
				}},
			}, metav1.CreateOptions{})
			framework.ExpectNoError(err, "Failed to create leader-locking role")

			err = e2eauth.BindRoleInNamespace(c.RbacV1(), roleName, ns, subject)
			framework.ExpectNoError(err)

			err = e2eauth.WaitForAuthorizationUpdate(c.AuthorizationV1(),
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
				Timeouts:     f.Timeouts,
				ClaimSize:    "1500Mi",
				ExpectedSize: "1500Mi",
			}

			storageClass, clearStorageClass := testsuites.SetupStorageClass(test.Client, newStorageClass(test, ns, "external"))
			defer clearStorageClass()
			test.Class = storageClass

			test.Claim = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
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
			e2eskipper.SkipUnlessProviderIs("gce", "aws", "gke", "vsphere", "azure")
			e2epv.SkipIfNoDefaultStorageClass(c)

			ginkgo.By("creating a claim with no annotation")
			test := testsuites.StorageClassTest{
				Client:       c,
				Name:         "default",
				Timeouts:     f.Timeouts,
				ClaimSize:    "2Gi",
				ExpectedSize: "2Gi",
			}

			test.Claim = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
				ClaimSize:  test.ClaimSize,
				VolumeMode: &test.VolumeMode,
			}, ns)
			// NOTE: this test assumes that there's a default storageclass
			storageClass, clearStorageClass := testsuites.SetupStorageClass(test.Client, nil)
			test.Class = storageClass
			defer clearStorageClass()

			test.TestDynamicProvisioning()
		})

		// Modifying the default storage class can be disruptive to other tests that depend on it
		ginkgo.It("should be disabled by changing the default annotation [Serial] [Disruptive]", func() {
			e2eskipper.SkipUnlessProviderIs("gce", "aws", "gke", "vsphere", "azure")
			e2epv.SkipIfNoDefaultStorageClass(c)

			scName, scErr := e2epv.GetDefaultStorageClassName(c)
			framework.ExpectNoError(scErr)

			test := testsuites.StorageClassTest{
				Name:      "default",
				Timeouts:  f.Timeouts,
				ClaimSize: "2Gi",
			}

			ginkgo.By("setting the is-default StorageClass annotation to false")
			verifyDefaultStorageClass(c, scName, true)
			defer updateDefaultStorageClass(c, scName, "true")
			updateDefaultStorageClass(c, scName, "false")

			ginkgo.By("creating a claim with default storageclass and expecting it to timeout")
			claim := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
				ClaimSize:  test.ClaimSize,
				VolumeMode: &test.VolumeMode,
			}, ns)
			claim, err := c.CoreV1().PersistentVolumeClaims(ns).Create(context.TODO(), claim, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			defer func() {
				framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(c, claim.Name, ns))
			}()

			// The claim should timeout phase:Pending
			err = e2epv.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, c, ns, claim.Name, 2*time.Second, framework.ClaimProvisionShortTimeout)
			framework.ExpectError(err)
			framework.Logf(err.Error())
			claim, err = c.CoreV1().PersistentVolumeClaims(ns).Get(context.TODO(), claim.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			framework.ExpectEqual(claim.Status.Phase, v1.ClaimPending)
		})

		// Modifying the default storage class can be disruptive to other tests that depend on it
		ginkgo.It("should be disabled by removing the default annotation [Serial] [Disruptive]", func() {
			e2eskipper.SkipUnlessProviderIs("gce", "aws", "gke", "vsphere", "azure")
			e2epv.SkipIfNoDefaultStorageClass(c)

			scName, scErr := e2epv.GetDefaultStorageClassName(c)
			framework.ExpectNoError(scErr)

			test := testsuites.StorageClassTest{
				Name:      "default",
				Timeouts:  f.Timeouts,
				ClaimSize: "2Gi",
			}

			ginkgo.By("removing the is-default StorageClass annotation")
			verifyDefaultStorageClass(c, scName, true)
			defer updateDefaultStorageClass(c, scName, "true")
			updateDefaultStorageClass(c, scName, "")

			ginkgo.By("creating a claim with default storageclass and expecting it to timeout")
			claim := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
				ClaimSize:  test.ClaimSize,
				VolumeMode: &test.VolumeMode,
			}, ns)
			claim, err := c.CoreV1().PersistentVolumeClaims(ns).Create(context.TODO(), claim, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			defer func() {
				framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(c, claim.Name, ns))
			}()

			// The claim should timeout phase:Pending
			err = e2epv.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, c, ns, claim.Name, 2*time.Second, framework.ClaimProvisionShortTimeout)
			framework.ExpectError(err)
			framework.Logf(err.Error())
			claim, err = c.CoreV1().PersistentVolumeClaims(ns).Get(context.TODO(), claim.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			framework.ExpectEqual(claim.Status.Phase, v1.ClaimPending)
		})
	})

	ginkgo.Describe("GlusterDynamicProvisioner", func() {
		ginkgo.It("should create and delete persistent volumes [fast]", func() {
			e2eskipper.SkipIfProviderIs("gke")
			ginkgo.By("creating a Gluster DP server Pod")
			pod := startGlusterDpServerPod(c, ns)
			serverURL := "http://" + net.JoinHostPort(pod.Status.PodIP, "8081")
			ginkgo.By("creating a StorageClass")
			test := testsuites.StorageClassTest{
				Client:       c,
				Name:         "Gluster Dynamic provisioner test",
				Provisioner:  "kubernetes.io/glusterfs",
				Timeouts:     f.Timeouts,
				ClaimSize:    "2Gi",
				ExpectedSize: "2Gi",
				Parameters:   map[string]string{"resturl": serverURL},
			}
			storageClass, clearStorageClass := testsuites.SetupStorageClass(test.Client, newStorageClass(test, ns, "glusterdptest"))
			defer clearStorageClass()
			test.Class = storageClass

			ginkgo.By("creating a claim object with a suffix for gluster dynamic provisioner")
			test.Claim = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
				ClaimSize:        test.ClaimSize,
				StorageClassName: &test.Class.Name,
				VolumeMode:       &test.VolumeMode,
			}, ns)

			test.TestDynamicProvisioning()
		})
	})

	ginkgo.Describe("Invalid AWS KMS key", func() {
		ginkgo.It("should report an error and create no PV", func() {
			e2eskipper.SkipUnlessProviderIs("aws")
			test := testsuites.StorageClassTest{
				Client:      c,
				Name:        "AWS EBS with invalid KMS key",
				Provisioner: "kubernetes.io/aws-ebs",
				Timeouts:    f.Timeouts,
				ClaimSize:   "2Gi",
				Parameters:  map[string]string{"kmsKeyId": "arn:aws:kms:us-east-1:123456789012:key/55555555-5555-5555-5555-555555555555"},
			}

			ginkgo.By("creating a StorageClass")
			storageClass, clearStorageClass := testsuites.SetupStorageClass(test.Client, newStorageClass(test, ns, "invalid-aws"))
			defer clearStorageClass()
			test.Class = storageClass

			ginkgo.By("creating a claim object with a suffix for gluster dynamic provisioner")
			claim := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
				ClaimSize:        test.ClaimSize,
				StorageClassName: &test.Class.Name,
				VolumeMode:       &test.VolumeMode,
			}, ns)
			claim, err := c.CoreV1().PersistentVolumeClaims(claim.Namespace).Create(context.TODO(), claim, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			defer func() {
				framework.Logf("deleting claim %q/%q", claim.Namespace, claim.Name)
				err = c.CoreV1().PersistentVolumeClaims(claim.Namespace).Delete(context.TODO(), claim.Name, metav1.DeleteOptions{})
				if err != nil && !apierrors.IsNotFound(err) {
					framework.Failf("Error deleting claim %q. Error: %v", claim.Name, err)
				}
			}()

			// Watch events until the message about invalid key appears.
			// Event delivery is not reliable and it's used only as a quick way how to check if volume with wrong KMS
			// key was not provisioned. If the event is not delivered, we check that the volume is not Bound for whole
			// ClaimProvisionTimeout in the very same loop.
			err = wait.Poll(time.Second, framework.ClaimProvisionTimeout, func() (bool, error) {
				events, err := c.CoreV1().Events(claim.Namespace).List(context.TODO(), metav1.ListOptions{})
				if err != nil {
					return false, fmt.Errorf("could not list PVC events in %s: %v", claim.Namespace, err)
				}
				for _, event := range events.Items {
					if strings.Contains(event.Message, "failed to create encrypted volume: the volume disappeared after creation, most likely due to inaccessible KMS encryption key") {
						return true, nil
					}
				}

				pvc, err := c.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(context.TODO(), claim.Name, metav1.GetOptions{})
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
				framework.Logf("The test missed event about failed provisioning, but checked that no volume was provisioned for %v", framework.ClaimProvisionTimeout)
				err = nil
			}
			framework.ExpectNoError(err, "Error waiting for PVC to fail provisioning: %v", err)
		})
	})
})

func verifyDefaultStorageClass(c clientset.Interface, scName string, expectedDefault bool) {
	sc, err := c.StorageV1().StorageClasses().Get(context.TODO(), scName, metav1.GetOptions{})
	framework.ExpectNoError(err)
	framework.ExpectEqual(storageutil.IsDefaultAnnotation(sc.ObjectMeta), expectedDefault)
}

func updateDefaultStorageClass(c clientset.Interface, scName string, defaultStr string) {
	sc, err := c.StorageV1().StorageClasses().Get(context.TODO(), scName, metav1.GetOptions{})
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

	_, err = c.StorageV1().StorageClasses().Update(context.TODO(), sc, metav1.UpdateOptions{})
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
	case framework.ProviderIs("vsphere"):
		return "kubernetes.io/vsphere-volume"
	case framework.ProviderIs("azure"):
		return "kubernetes.io/azure-disk"
	}
	return ""
}

func newStorageClass(t testsuites.StorageClassTest, ns string, prefix string) *storagev1.StorageClass {
	pluginName := t.Provisioner
	if pluginName == "" {
		pluginName = getDefaultPluginName()
	}
	if prefix == "" {
		prefix = "sc"
	}
	bindingMode := storagev1.VolumeBindingImmediate
	if t.DelayBinding {
		bindingMode = storagev1.VolumeBindingWaitForFirstConsumer
	}
	if t.Parameters == nil {
		t.Parameters = make(map[string]string)
	}

	if framework.NodeOSDistroIs("windows") {
		// fstype might be forced from outside, in that case skip setting a default
		if _, exists := t.Parameters["fstype"]; !exists {
			t.Parameters["fstype"] = e2epv.GetDefaultFSType()
			framework.Logf("settings a default fsType=%s in the storage class", t.Parameters["fstype"])
		}
	}

	sc := getStorageClass(pluginName, t.Parameters, &bindingMode, ns, prefix)
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
	prefix string,
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
			// Name must be unique, so let's base it on namespace name and the prefix (the prefix is test specific)
			GenerateName: ns + "-" + prefix,
		},
		Provisioner:       provisioner,
		Parameters:        parameters,
		VolumeBindingMode: bindingMode,
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
					Image: imageutils.GetE2EImage(imageutils.GlusterDynamicProvisioner),
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
	provisionerPod, err := podClient.Create(context.TODO(), provisionerPod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "Failed to create %s pod: %v", provisionerPod.Name, err)

	framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(c, provisionerPod))

	ginkgo.By("locating the provisioner pod")
	pod, err := podClient.Get(context.TODO(), provisionerPod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "Cannot locate the provisioner pod %v: %v", provisionerPod.Name, err)
	return pod
}

// waitForProvisionedVolumesDelete is a polling wrapper to scan all PersistentVolumes for any associated to the test's
// StorageClass.  Returns either an error and nil values or the remaining PVs and their count.
func waitForProvisionedVolumesDeleted(c clientset.Interface, scName string) ([]*v1.PersistentVolume, error) {
	var remainingPVs []*v1.PersistentVolume

	err := wait.Poll(10*time.Second, 300*time.Second, func() (bool, error) {
		remainingPVs = []*v1.PersistentVolume{}

		allPVs, err := c.CoreV1().PersistentVolumes().List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			return true, err
		}
		for _, pv := range allPVs.Items {
			if pv.Spec.StorageClassName == scName {
				pv := pv
				remainingPVs = append(remainingPVs, &pv)
			}
		}
		if len(remainingPVs) > 0 {
			return false, nil // Poll until no PVs remain
		}
		return true, nil // No PVs remain
	})
	if err != nil {
		return remainingPVs, fmt.Errorf("Error waiting for PVs to be deleted: %v", err)
	}
	return nil, nil
}

// deleteStorageClass deletes the passed in StorageClass and catches errors other than "Not Found"
func deleteStorageClass(c clientset.Interface, className string) {
	err := c.StorageV1().StorageClasses().Delete(context.TODO(), className, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		framework.ExpectNoError(err)
	}
}

// deleteProvisionedVolumes [gce||gke only]  iteratively deletes persistent volumes and attached GCE PDs.
func deleteProvisionedVolumesAndDisks(c clientset.Interface, pvs []*v1.PersistentVolume) {
	framework.Logf("Remaining PersistentVolumes:")
	for i, pv := range pvs {
		framework.Logf("\t%d) %s", i+1, pv.Name)
	}
	for _, pv := range pvs {
		framework.ExpectNoError(e2epv.DeletePDWithRetry(pv.Spec.PersistentVolumeSource.GCEPersistentDisk.PDName))
		framework.ExpectNoError(e2epv.DeletePersistentVolume(c, pv.Name))
	}
}

func getRandomClusterZone(c clientset.Interface) string {
	zones, err := e2enode.GetClusterZones(c)
	zone := ""
	framework.ExpectNoError(err)
	if len(zones) != 0 {
		zonesList := zones.UnsortedList()
		zone = zonesList[rand.Intn(zones.Len())]
	}
	return zone
}
