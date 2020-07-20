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

	"github.com/onsi/ginkgo"

	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	storagev1 "k8s.io/api/storage/v1"
	storagev1beta1 "k8s.io/api/storage/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	clientset "k8s.io/client-go/kubernetes"
	storageutil "k8s.io/kubernetes/pkg/apis/storage/v1/util"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eauth "k8s.io/kubernetes/test/e2e/framework/auth"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	// Plugin name of the external provisioner
	externalPluginName = "example.com/nfs"
)

var _ = utils.SIGDescribe("Dynamic Provisioning", func() {
	f := framework.NewDefaultFramework("volume-provisioning")

	// filled in BeforeEach
	var c clientset.Interface
	var ns string

	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
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
				ClaimSize:    "1500Mi",
				ExpectedSize: "1500Mi",
			}
			test.Class = newStorageClass(test, ns, "external")
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
			e2eskipper.SkipUnlessProviderIs("openstack", "gce", "aws", "gke", "vsphere", "azure")

			ginkgo.By("creating a claim with no annotation")
			test := testsuites.StorageClassTest{
				Client:       c,
				Name:         "default",
				ClaimSize:    "2Gi",
				ExpectedSize: "2Gi",
			}

			test.Claim = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
				ClaimSize:  test.ClaimSize,
				VolumeMode: &test.VolumeMode,
			}, ns)
			test.TestDynamicProvisioning()
		})

		// Modifying the default storage class can be disruptive to other tests that depend on it
		ginkgo.It("should be disabled by changing the default annotation [Serial] [Disruptive]", func() {
			e2eskipper.SkipUnlessProviderIs("openstack", "gce", "aws", "gke", "vsphere", "azure")

			scName, scErr := e2epv.GetDefaultStorageClassName(c)
			framework.ExpectNoError(scErr)

			test := testsuites.StorageClassTest{
				Name:      "default",
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
			e2eskipper.SkipUnlessProviderIs("openstack", "gce", "aws", "gke", "vsphere", "azure")

			scName, scErr := e2epv.GetDefaultStorageClassName(c)
			framework.ExpectNoError(scErr)

			test := testsuites.StorageClassTest{
				Name:      "default",
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

	framework.KubeDescribe("GlusterDynamicProvisioner", func() {
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
				ClaimSize:    "2Gi",
				ExpectedSize: "2Gi",
				Parameters:   map[string]string{"resturl": serverURL},
			}
			suffix := fmt.Sprintf("glusterdptest")
			test.Class = newStorageClass(test, ns, suffix)

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
				Name:        "AWS EBS with invalid KMS key",
				Provisioner: "kubernetes.io/aws-ebs",
				ClaimSize:   "2Gi",
				Parameters:  map[string]string{"kmsKeyId": "arn:aws:kms:us-east-1:123456789012:key/55555555-5555-5555-5555-555555555555"},
			}

			ginkgo.By("creating a StorageClass")
			suffix := fmt.Sprintf("invalid-aws")
			class := newStorageClass(test, ns, suffix)
			class, err := c.StorageV1().StorageClasses().Create(context.TODO(), class, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			defer func() {
				framework.Logf("deleting storage class %s", class.Name)
				framework.ExpectNoError(c.StorageV1().StorageClasses().Delete(context.TODO(), class.Name, metav1.DeleteOptions{}))
			}()

			ginkgo.By("creating a claim object with a suffix for gluster dynamic provisioner")
			claim := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
				ClaimSize:        test.ClaimSize,
				StorageClassName: &class.Name,
				VolumeMode:       &test.VolumeMode,
			}, ns)
			claim, err = c.CoreV1().PersistentVolumeClaims(claim.Namespace).Create(context.TODO(), claim, metav1.CreateOptions{})
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
	case framework.ProviderIs("openstack"):
		return "kubernetes.io/cinder"
	case framework.ProviderIs("vsphere"):
		return "kubernetes.io/vsphere-volume"
	case framework.ProviderIs("azure"):
		return "kubernetes.io/azure-disk"
	}
	return ""
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
	framework.ExpectNoError(err)
	framework.ExpectNotEqual(len(zones), 0)

	zonesList := zones.UnsortedList()
	return zonesList[rand.Intn(zones.Len())]
}
