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

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	clientset "k8s.io/client-go/kubernetes"
	storageutil "k8s.io/kubernetes/pkg/apis/storage/util"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eauth "k8s.io/kubernetes/test/e2e/framework/auth"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	// Plugin name of the external provisioner
	externalPluginName = "example.com/nfs"
)

var _ = utils.SIGDescribe("Dynamic Provisioning", func() {
	f := framework.NewDefaultFramework("volume-provisioning")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	// filled in BeforeEach
	var c clientset.Interface
	var ns string

	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
	})

	f.Describe("DynamicProvisioner", framework.WithSlow(), feature.StorageProvider, func() {
		ginkgo.It("should provision storage with different parameters", func(ctx context.Context) {

			// This test checks that dynamic provisioning can provision a volume
			// that can be used to persist data among pods.
			tests := []testsuites.StorageClassTest{
				// GCE/GKE
				{
					Name:           "SSD PD on GCE/GKE",
					CloudProviders: []string{"gce"},
					Timeouts:       f.Timeouts,
					Provisioner:    "kubernetes.io/gce-pd",
					Parameters: map[string]string{
						"type": "pd-ssd",
						"zone": getRandomClusterZone(ctx, c),
					},
					ClaimSize:    "1.5Gi",
					ExpectedSize: "2Gi",
					PvCheck: func(ctx context.Context, claim *v1.PersistentVolumeClaim) {
						volume := testsuites.PVWriteReadSingleNodeCheck(ctx, c, f.Timeouts, claim, e2epod.NodeSelection{})
						gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")
					},
				},
				{
					Name:           "HDD PD on GCE",
					CloudProviders: []string{"gce"},
					Timeouts:       f.Timeouts,
					Provisioner:    "kubernetes.io/gce-pd",
					Parameters: map[string]string{
						"type": "pd-standard",
					},
					ClaimSize:    "1.5Gi",
					ExpectedSize: "2Gi",
					PvCheck: func(ctx context.Context, claim *v1.PersistentVolumeClaim) {
						volume := testsuites.PVWriteReadSingleNodeCheck(ctx, c, f.Timeouts, claim, e2epod.NodeSelection{})
						gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")
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
						"zone": getRandomClusterZone(ctx, c),
					},
					ClaimSize:    "1.5Gi",
					ExpectedSize: "2Gi",
					PvCheck: func(ctx context.Context, claim *v1.PersistentVolumeClaim) {
						volume := testsuites.PVWriteReadSingleNodeCheck(ctx, c, f.Timeouts, claim, e2epod.NodeSelection{})
						gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")
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
					PvCheck: func(ctx context.Context, claim *v1.PersistentVolumeClaim) {
						volume := testsuites.PVWriteReadSingleNodeCheck(ctx, c, f.Timeouts, claim, e2epod.NodeSelection{})
						gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")
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
					PvCheck: func(ctx context.Context, claim *v1.PersistentVolumeClaim) {
						volume := testsuites.PVWriteReadSingleNodeCheck(ctx, c, f.Timeouts, claim, e2epod.NodeSelection{})
						gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")
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
					PvCheck: func(ctx context.Context, claim *v1.PersistentVolumeClaim) {
						volume := testsuites.PVWriteReadSingleNodeCheck(ctx, c, f.Timeouts, claim, e2epod.NodeSelection{})
						gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")
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
					PvCheck: func(ctx context.Context, claim *v1.PersistentVolumeClaim) {
						volume := testsuites.PVWriteReadSingleNodeCheck(ctx, c, f.Timeouts, claim, e2epod.NodeSelection{})
						gomega.Expect(volume).NotTo(gomega.BeNil(), "get bound PV")
					},
				},
				// OpenStack generic tests (works on all OpenStack deployments)
				{
					Name:           "generic Cinder volume on OpenStack",
					CloudProviders: []string{"openstack"},
					Timeouts:       f.Timeouts,
					Provisioner:    "kubernetes.io/cinder",
					Parameters:     map[string]string{},
					ClaimSize:      "1.5Gi",
					ExpectedSize:   "2Gi",
					PvCheck: func(ctx context.Context, claim *v1.PersistentVolumeClaim) {
						testsuites.PVWriteReadSingleNodeCheck(ctx, c, f.Timeouts, claim, e2epod.NodeSelection{})
					},
				},
				{
					Name:           "Cinder volume with empty volume type and zone on OpenStack",
					CloudProviders: []string{"openstack"},
					Timeouts:       f.Timeouts,
					Provisioner:    "kubernetes.io/cinder",
					Parameters: map[string]string{
						"type":         "",
						"availability": "",
					},
					ClaimSize:    "1.5Gi",
					ExpectedSize: "2Gi",
					PvCheck: func(ctx context.Context, claim *v1.PersistentVolumeClaim) {
						testsuites.PVWriteReadSingleNodeCheck(ctx, c, f.Timeouts, claim, e2epod.NodeSelection{})
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
					PvCheck: func(ctx context.Context, claim *v1.PersistentVolumeClaim) {
						testsuites.PVWriteReadSingleNodeCheck(ctx, c, f.Timeouts, claim, e2epod.NodeSelection{})
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
					PvCheck: func(ctx context.Context, claim *v1.PersistentVolumeClaim) {
						testsuites.PVWriteReadSingleNodeCheck(ctx, c, f.Timeouts, claim, e2epod.NodeSelection{})
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
					gomega.Expect(zone).ToNot(gomega.BeEmpty(), "expect at least one zone")
				}

				ginkgo.By("Testing " + test.Name)
				suffix := fmt.Sprintf("%d", i)
				test.Client = c

				// overwrite StorageClass spec with provisioned StorageClass
				storageClass := testsuites.SetupStorageClass(ctx, test.Client, newStorageClass(test, ns, suffix))

				test.Class = storageClass
				test.Claim = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
					ClaimSize:        test.ClaimSize,
					StorageClassName: &test.Class.Name,
					VolumeMode:       &test.VolumeMode,
				}, ns)

				test.TestDynamicProvisioning(ctx)
			}
		})
	})

	ginkgo.Describe("DynamicProvisioner External", func() {
		f.It("should let an external dynamic provisioner create and delete persistent volumes", f.WithSlow(), func(ctx context.Context) {
			// external dynamic provisioner pods need additional permissions provided by the
			// persistent-volume-provisioner clusterrole and a leader-locking role
			serviceAccountName := "default"
			subject := rbacv1.Subject{
				Kind:      rbacv1.ServiceAccountKind,
				Namespace: ns,
				Name:      serviceAccountName,
			}

			cleanupFunc, err := e2eauth.BindClusterRole(ctx, c.RbacV1(), "system:persistent-volume-provisioner", ns, subject)
			framework.ExpectNoError(err)
			defer cleanupFunc(ctx)

			roleName := "leader-locking-nfs-provisioner"
			_, err = f.ClientSet.RbacV1().Roles(ns).Create(ctx, &rbacv1.Role{
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

			err = e2eauth.BindRoleInNamespace(ctx, c.RbacV1(), roleName, ns, subject)
			framework.ExpectNoError(err)

			err = e2eauth.WaitForAuthorizationUpdate(ctx, c.AuthorizationV1(),
				serviceaccount.MakeUsername(ns, serviceAccountName),
				"", "get", schema.GroupResource{Group: "storage.k8s.io", Resource: "storageclasses"}, true)
			framework.ExpectNoError(err, "Failed to update authorization")

			ginkgo.By("creating an external dynamic provisioner pod")
			pod := utils.StartExternalProvisioner(ctx, c, ns, externalPluginName)
			ginkgo.DeferCleanup(e2epod.DeletePodOrFail, c, ns, pod.Name)

			ginkgo.By("creating a StorageClass")
			test := testsuites.StorageClassTest{
				Client:       c,
				Name:         "external provisioner test",
				Provisioner:  externalPluginName,
				Timeouts:     f.Timeouts,
				ClaimSize:    "1500Mi",
				ExpectedSize: "1500Mi",
			}

			storageClass := testsuites.SetupStorageClass(ctx, test.Client, newStorageClass(test, ns, "external"))
			test.Class = storageClass

			test.Claim = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
				ClaimSize:        test.ClaimSize,
				StorageClassName: &test.Class.Name,
				VolumeMode:       &test.VolumeMode,
			}, ns)

			ginkgo.By("creating a claim with a external provisioning annotation")

			test.TestDynamicProvisioning(ctx)
		})
	})

	ginkgo.Describe("DynamicProvisioner Default", func() {
		f.It("should create and delete default persistent volumes", f.WithSlow(), func(ctx context.Context) {
			e2eskipper.SkipUnlessProviderIs("openstack", "gce", "aws", "vsphere", "azure")
			e2epv.SkipIfNoDefaultStorageClass(ctx, c)

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
			test.Class = testsuites.SetupStorageClass(ctx, test.Client, nil)

			test.TestDynamicProvisioning(ctx)
		})

		// Modifying the default storage class can be disruptive to other tests that depend on it
		f.It("should be disabled by changing the default annotation", f.WithSerial(), f.WithDisruptive(), func(ctx context.Context) {
			e2eskipper.SkipUnlessProviderIs("openstack", "gce", "aws", "vsphere", "azure")
			e2epv.SkipIfNoDefaultStorageClass(ctx, c)

			scName, scErr := e2epv.GetDefaultStorageClassName(ctx, c)
			framework.ExpectNoError(scErr)

			test := testsuites.StorageClassTest{
				Name:      "default",
				Timeouts:  f.Timeouts,
				ClaimSize: "2Gi",
			}

			ginkgo.By("setting the is-default StorageClass annotation to false")
			verifyDefaultStorageClass(ctx, c, scName, true)
			ginkgo.DeferCleanup(updateDefaultStorageClass, c, scName, "true")
			updateDefaultStorageClass(ctx, c, scName, "false")

			ginkgo.By("creating a claim with default storageclass and expecting it to timeout")
			claim := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
				ClaimSize:  test.ClaimSize,
				VolumeMode: &test.VolumeMode,
			}, ns)
			claim, err := c.CoreV1().PersistentVolumeClaims(ns).Create(ctx, claim, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			ginkgo.DeferCleanup(e2epv.DeletePersistentVolumeClaim, c, claim.Name, ns)

			// The claim should timeout phase:Pending
			err = e2epv.WaitForPersistentVolumeClaimPhase(ctx, v1.ClaimBound, c, ns, claim.Name, 2*time.Second, framework.ClaimProvisionShortTimeout)
			gomega.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("not all in phase Bound")))
			framework.Logf("%s", err.Error())
			claim, err = c.CoreV1().PersistentVolumeClaims(ns).Get(ctx, claim.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(claim.Status.Phase).To(gomega.Equal(v1.ClaimPending))
		})

		// Modifying the default storage class can be disruptive to other tests that depend on it
		f.It("should be disabled by removing the default annotation", f.WithSerial(), f.WithDisruptive(), func(ctx context.Context) {
			e2eskipper.SkipUnlessProviderIs("openstack", "gce", "aws", "vsphere", "azure")
			e2epv.SkipIfNoDefaultStorageClass(ctx, c)

			scName, scErr := e2epv.GetDefaultStorageClassName(ctx, c)
			framework.ExpectNoError(scErr)

			test := testsuites.StorageClassTest{
				Name:      "default",
				Timeouts:  f.Timeouts,
				ClaimSize: "2Gi",
			}

			ginkgo.By("removing the is-default StorageClass annotation")
			verifyDefaultStorageClass(ctx, c, scName, true)
			ginkgo.DeferCleanup(updateDefaultStorageClass, c, scName, "true")
			updateDefaultStorageClass(ctx, c, scName, "")

			ginkgo.By("creating a claim with default storageclass and expecting it to timeout")
			claim := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
				ClaimSize:  test.ClaimSize,
				VolumeMode: &test.VolumeMode,
			}, ns)
			claim, err := c.CoreV1().PersistentVolumeClaims(ns).Create(ctx, claim, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			defer func() {
				framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(ctx, c, claim.Name, ns))
			}()

			// The claim should timeout phase:Pending
			err = e2epv.WaitForPersistentVolumeClaimPhase(ctx, v1.ClaimBound, c, ns, claim.Name, 2*time.Second, framework.ClaimProvisionShortTimeout)
			gomega.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("not all in phase Bound")))
			framework.Logf("%s", err.Error())
			claim, err = c.CoreV1().PersistentVolumeClaims(ns).Get(ctx, claim.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(claim.Status.Phase).To(gomega.Equal(v1.ClaimPending))
		})
	})

	ginkgo.Describe("Invalid AWS KMS key", func() {
		ginkgo.It("should report an error and create no PV", func(ctx context.Context) {
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
			test.Class = testsuites.SetupStorageClass(ctx, test.Client, newStorageClass(test, ns, "invalid-aws"))

			ginkgo.By("creating a claim object")
			claim := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
				ClaimSize:        test.ClaimSize,
				StorageClassName: &test.Class.Name,
				VolumeMode:       &test.VolumeMode,
			}, ns)
			claim, err := c.CoreV1().PersistentVolumeClaims(claim.Namespace).Create(ctx, claim, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			defer func() {
				framework.Logf("deleting claim %q/%q", claim.Namespace, claim.Name)
				err = c.CoreV1().PersistentVolumeClaims(claim.Namespace).Delete(ctx, claim.Name, metav1.DeleteOptions{})
				if err != nil && !apierrors.IsNotFound(err) {
					framework.Failf("Error deleting claim %q. Error: %v", claim.Name, err)
				}
			}()

			// Watch events until the message about invalid key appears.
			// Event delivery is not reliable and it's used only as a quick way how to check if volume with wrong KMS
			// key was not provisioned. If the event is not delivered, we check that the volume is not Bound for whole
			// ClaimProvisionTimeout in the very same loop.
			err = wait.Poll(time.Second, framework.ClaimProvisionTimeout, func() (bool, error) {
				events, err := c.CoreV1().Events(claim.Namespace).List(ctx, metav1.ListOptions{})
				if err != nil {
					return false, fmt.Errorf("could not list PVC events in %s: %w", claim.Namespace, err)
				}
				for _, event := range events.Items {
					if strings.Contains(event.Message, "failed to create encrypted volume: the volume disappeared after creation, most likely due to inaccessible KMS encryption key") {
						return true, nil
					}
				}

				pvc, err := c.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(ctx, claim.Name, metav1.GetOptions{})
				if err != nil {
					return true, err
				}
				if pvc.Status.Phase != v1.ClaimPending {
					// The PVC was bound to something, i.e. PV was created for wrong KMS key. That's bad!
					return true, fmt.Errorf("PVC got unexpectedly %s (to PV %q)", pvc.Status.Phase, pvc.Spec.VolumeName)
				}

				return false, nil
			})
			if wait.Interrupted(err) {
				framework.Logf("The test missed event about failed provisioning, but checked that no volume was provisioned for %v", framework.ClaimProvisionTimeout)
				err = nil
			}
			framework.ExpectNoError(err, "Error waiting for PVC to fail provisioning: %v", err)
		})
	})
})

func verifyDefaultStorageClass(ctx context.Context, c clientset.Interface, scName string, expectedDefault bool) {
	sc, err := c.StorageV1().StorageClasses().Get(ctx, scName, metav1.GetOptions{})
	framework.ExpectNoError(err)
	gomega.Expect(storageutil.IsDefaultAnnotation(sc.ObjectMeta)).To(gomega.Equal(expectedDefault))
}

func updateDefaultStorageClass(ctx context.Context, c clientset.Interface, scName string, defaultStr string) {
	sc, err := c.StorageV1().StorageClasses().Get(ctx, scName, metav1.GetOptions{})
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

	_, err = c.StorageV1().StorageClasses().Update(ctx, sc, metav1.UpdateOptions{})
	framework.ExpectNoError(err)

	expectedDefault := false
	if defaultStr == "true" {
		expectedDefault = true
	}
	verifyDefaultStorageClass(ctx, c, scName, expectedDefault)
}

// deleteStorageClass deletes the passed in StorageClass and catches errors other than "Not Found"
func deleteStorageClass(ctx context.Context, c clientset.Interface, className string) {
	err := c.StorageV1().StorageClasses().Delete(ctx, className, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		framework.ExpectNoError(err)
	}
}

func getRandomClusterZone(ctx context.Context, c clientset.Interface) string {
	zones, err := e2enode.GetClusterZones(ctx, c)
	zone := ""
	framework.ExpectNoError(err)
	if len(zones) != 0 {
		zonesList := zones.UnsortedList()
		zone = zonesList[rand.Intn(zones.Len())]
	}
	return zone
}
