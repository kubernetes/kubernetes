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

package e2e

import (
	"time"

	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	storage "k8s.io/kubernetes/pkg/apis/storage/v1beta1"
	storageutil "k8s.io/kubernetes/pkg/apis/storage/v1beta1/util"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	// Requested size of the volume
	requestedSize = "1500Mi"
	// Expected size of the volume is 2GiB, because all three supported cloud
	// providers allocate volumes in 1GiB chunks.
	expectedSize = "2Gi"
)

func testDynamicProvisioning(client clientset.Interface, claim *v1.PersistentVolumeClaim) {
	err := framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, client, claim.Namespace, claim.Name, framework.Poll, framework.ClaimProvisionTimeout)
	Expect(err).NotTo(HaveOccurred())

	By("checking the claim")
	// Get new copy of the claim
	claim, err = client.Core().PersistentVolumeClaims(claim.Namespace).Get(claim.Name)
	Expect(err).NotTo(HaveOccurred())

	// Get the bound PV
	pv, err := client.Core().PersistentVolumes().Get(claim.Spec.VolumeName)
	Expect(err).NotTo(HaveOccurred())

	// Check sizes
	expectedCapacity := resource.MustParse(expectedSize)
	pvCapacity := pv.Spec.Capacity[v1.ResourceName(v1.ResourceStorage)]
	Expect(pvCapacity.Value()).To(Equal(expectedCapacity.Value()))

	requestedCapacity := resource.MustParse(requestedSize)
	claimCapacity := claim.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	Expect(claimCapacity.Value()).To(Equal(requestedCapacity.Value()))

	// Check PV properties
	Expect(pv.Spec.PersistentVolumeReclaimPolicy).To(Equal(v1.PersistentVolumeReclaimDelete))
	expectedAccessModes := []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
	Expect(pv.Spec.AccessModes).To(Equal(expectedAccessModes))
	Expect(pv.Spec.ClaimRef.Name).To(Equal(claim.ObjectMeta.Name))
	Expect(pv.Spec.ClaimRef.Namespace).To(Equal(claim.ObjectMeta.Namespace))

	// We start two pods:
	// - The first writes 'hello word' to the /mnt/test (= the volume).
	// - The second one runs grep 'hello world' on /mnt/test.
	// If both succeed, Kubernetes actually allocated something that is
	// persistent across pods.
	By("checking the created volume is writable")
	runInPodWithVolume(client, claim.Namespace, claim.Name, "echo 'hello world' > /mnt/test/data")

	By("checking the created volume is readable and retains data")
	runInPodWithVolume(client, claim.Namespace, claim.Name, "grep 'hello world' /mnt/test/data")

	// Ugly hack: if we delete the AWS/GCE/OpenStack volume here, it will
	// probably collide with destruction of the pods above - the pods
	// still have the volume attached (kubelet is slow...) and deletion
	// of attached volume is not allowed by AWS/GCE/OpenStack.
	// Kubernetes *will* retry deletion several times in
	// pvclaimbinder-sync-period.
	// So, technically, this sleep is not needed. On the other hand,
	// the sync perion is 10 minutes and we really don't want to wait
	// 10 minutes here. There is no way how to see if kubelet is
	// finished with cleaning volumes. A small sleep here actually
	// speeds up the test!
	// Three minutes should be enough to clean up the pods properly.
	// We've seen GCE PD detach to take more than 1 minute.
	By("Sleeping to let kubelet destroy all pods")
	time.Sleep(3 * time.Minute)

	By("deleting the claim")
	framework.ExpectNoError(client.Core().PersistentVolumeClaims(claim.Namespace).Delete(claim.Name, nil))

	// Wait for the PV to get deleted too.
	framework.ExpectNoError(framework.WaitForPersistentVolumeDeleted(client, pv.Name, 5*time.Second, 20*time.Minute))
}

var _ = framework.KubeDescribe("Dynamic provisioning", func() {
	f := framework.NewDefaultFramework("volume-provisioning")

	// filled in BeforeEach
	var c clientset.Interface
	var ns string

	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
	})

	framework.KubeDescribe("DynamicProvisioner", func() {
		It("should create and delete persistent volumes [Slow]", func() {
			framework.SkipUnlessProviderIs("openstack", "gce", "aws", "gke")

			By("creating a StorageClass")
			class := newStorageClass()
			_, err := c.Storage().StorageClasses().Create(class)
			defer c.Storage().StorageClasses().Delete(class.Name, nil)
			Expect(err).NotTo(HaveOccurred())

			By("creating a claim with a dynamic provisioning annotation")
			claim := newClaim(ns, false)
			defer func() {
				c.Core().PersistentVolumeClaims(ns).Delete(claim.Name, nil)
			}()
			claim, err = c.Core().PersistentVolumeClaims(ns).Create(claim)
			Expect(err).NotTo(HaveOccurred())

			testDynamicProvisioning(c, claim)
		})
	})

	framework.KubeDescribe("DynamicProvisioner Alpha", func() {
		It("should create and delete alpha persistent volumes [Slow]", func() {
			framework.SkipUnlessProviderIs("openstack", "gce", "aws", "gke")

			By("creating a claim with an alpha dynamic provisioning annotation")
			claim := newClaim(ns, true)
			defer func() {
				c.Core().PersistentVolumeClaims(ns).Delete(claim.Name, nil)
			}()
			claim, err := c.Core().PersistentVolumeClaims(ns).Create(claim)
			Expect(err).NotTo(HaveOccurred())

			testDynamicProvisioning(c, claim)
		})
	})
})

func newClaim(ns string, alpha bool) *v1.PersistentVolumeClaim {
	claim := v1.PersistentVolumeClaim{
		ObjectMeta: v1.ObjectMeta{
			GenerateName: "pvc-",
			Namespace:    ns,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse(requestedSize),
				},
			},
		},
	}

	if alpha {
		claim.Annotations = map[string]string{
			storageutil.AlphaStorageClassAnnotation: "",
		}
	} else {
		claim.Annotations = map[string]string{
			storageutil.StorageClassAnnotation: "fast",
		}

	}

	return &claim
}

// runInPodWithVolume runs a command in a pod with given claim mounted to /mnt directory.
func runInPodWithVolume(c clientset.Interface, ns, claimName, command string) {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: v1.ObjectMeta{
			GenerateName: "pvc-volume-tester-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "volume-tester",
					Image:   "gcr.io/google_containers/busybox:1.24",
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
	pod, err := c.Core().Pods(ns).Create(pod)
	defer func() {
		framework.ExpectNoError(c.Core().Pods(ns).Delete(pod.Name, nil))
	}()
	framework.ExpectNoError(err, "Failed to create pod: %v", err)
	framework.ExpectNoError(framework.WaitForPodSuccessInNamespaceSlow(c, pod.Name, pod.Namespace))
}

func newStorageClass() *storage.StorageClass {
	var pluginName string

	switch {
	case framework.ProviderIs("gke"), framework.ProviderIs("gce"):
		pluginName = "kubernetes.io/gce-pd"
	case framework.ProviderIs("aws"):
		pluginName = "kubernetes.io/aws-ebs"
	case framework.ProviderIs("openstack"):
		pluginName = "kubernetes.io/cinder"
	}

	return &storage.StorageClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: v1.ObjectMeta{
			Name: "fast",
		},
		Provisioner: pluginName,
	}
}
