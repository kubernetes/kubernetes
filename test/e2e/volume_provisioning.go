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

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/kubernetes/pkg/api/v1"
	rbacv1beta1 "k8s.io/kubernetes/pkg/apis/rbac/v1beta1"
	storage "k8s.io/kubernetes/pkg/apis/storage/v1beta1"
	storageutil "k8s.io/kubernetes/pkg/apis/storage/v1beta1/util"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	// Requested size of the volume
	requestedSize = "1500Mi"
	// Plugin name of the external provisioner
	externalPluginName = "example.com/nfs"
)

func testDynamicProvisioning(client clientset.Interface, claim *v1.PersistentVolumeClaim, expectedSize string) {
	err := framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, client, claim.Namespace, claim.Name, framework.Poll, framework.ClaimProvisionTimeout)
	Expect(err).NotTo(HaveOccurred())

	By("checking the claim")
	// Get new copy of the claim
	claim, err = client.Core().PersistentVolumeClaims(claim.Namespace).Get(claim.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	// Get the bound PV
	pv, err := client.Core().PersistentVolumes().Get(claim.Spec.VolumeName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	// Check sizes
	expectedCapacity := resource.MustParse(expectedSize)
	pvCapacity := pv.Spec.Capacity[v1.ResourceName(v1.ResourceStorage)]
	Expect(pvCapacity.Value()).To(Equal(expectedCapacity.Value()), "pvCapacity is not equal to expectedCapacity")

	requestedCapacity := resource.MustParse(requestedSize)
	claimCapacity := claim.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	Expect(claimCapacity.Value()).To(Equal(requestedCapacity.Value()), "claimCapacity is not equal to requestedCapacity")

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

	By("deleting the claim")
	framework.ExpectNoError(client.Core().PersistentVolumeClaims(claim.Namespace).Delete(claim.Name, nil))

	// Wait for the PV to get deleted. Technically, the first few delete
	// attempts may fail, as the volume is still attached to a node because
	// kubelet is slowly cleaning up a pod, however it should succeed in a
	// couple of minutes. Wait 20 minutes to recover from random cloud hiccups.
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
		It("should create and delete persistent volumes [Slow] [Volume]", func() {
			framework.SkipUnlessProviderIs("openstack", "gce", "aws", "gke", "vsphere")

			By("creating a StorageClass")
			class := newStorageClass("", "internal")
			_, err := c.Storage().StorageClasses().Create(class)
			defer c.Storage().StorageClasses().Delete(class.Name, nil)
			Expect(err).NotTo(HaveOccurred())

			By("creating a claim with a dynamic provisioning annotation")
			claim := newClaim(ns, "internal", false)
			defer func() {
				c.Core().PersistentVolumeClaims(ns).Delete(claim.Name, nil)
			}()
			claim, err = c.Core().PersistentVolumeClaims(ns).Create(claim)
			Expect(err).NotTo(HaveOccurred())

			if framework.ProviderIs("vsphere") {
				// vsphere provider does not allocate volumes in 1GiB chunks, so setting expected size
				// equal to requestedSize
				testDynamicProvisioning(c, claim, requestedSize)
			} else {
				// Expected size of the volume is 2GiB, because the other three supported cloud
				// providers allocate volumes in 1GiB chunks.
				testDynamicProvisioning(c, claim, "2Gi")
			}
		})
	})

	framework.KubeDescribe("DynamicProvisioner Alpha", func() {
		It("should create and delete alpha persistent volumes [Slow] [Volume]", func() {
			framework.SkipUnlessProviderIs("openstack", "gce", "aws", "gke", "vsphere")

			By("creating a claim with an alpha dynamic provisioning annotation")
			claim := newClaim(ns, "", true)
			defer func() {
				c.Core().PersistentVolumeClaims(ns).Delete(claim.Name, nil)
			}()
			claim, err := c.Core().PersistentVolumeClaims(ns).Create(claim)
			Expect(err).NotTo(HaveOccurred())

			if framework.ProviderIs("vsphere") {
				testDynamicProvisioning(c, claim, requestedSize)
			} else {
				testDynamicProvisioning(c, claim, "2Gi")
			}
		})
	})

	framework.KubeDescribe("DynamicProvisioner External", func() {
		It("should let an external dynamic provisioner create and delete persistent volumes [Slow]", func() {
			// external dynamic provisioner pods need additional permissions provided by the
			// persistent-volume-provisioner role
			framework.BindClusterRole(c.Rbac(), "system:persistent-volume-provisioner", ns,
				rbacv1beta1.Subject{Kind: rbacv1beta1.ServiceAccountKind, Namespace: ns, Name: "default"})

			err := framework.WaitForAuthorizationUpdate(c.AuthorizationV1beta1(),
				serviceaccount.MakeUsername(ns, "default"),
				"", "get", schema.GroupResource{Group: "storage.k8s.io", Resource: "storageclasses"}, true)
			framework.ExpectNoError(err, "Failed to update authorization: %v", err)

			By("creating an external dynamic provisioner pod")
			pod := startExternalProvisioner(c, ns)
			defer c.Core().Pods(ns).Delete(pod.Name, nil)

			By("creating a StorageClass")
			class := newStorageClass(externalPluginName, "external")
			_, err = c.Storage().StorageClasses().Create(class)
			defer c.Storage().StorageClasses().Delete(class.Name, nil)
			Expect(err).NotTo(HaveOccurred())

			By("creating a claim with a dynamic provisioning annotation")
			claim := newClaim(ns, "external", false)
			defer func() {
				c.Core().PersistentVolumeClaims(ns).Delete(claim.Name, nil)
			}()
			claim, err = c.Core().PersistentVolumeClaims(ns).Create(claim)
			Expect(err).NotTo(HaveOccurred())

			// Expected size of the externally provisioned volume depends on the external
			// provisioner: for nfs-provisioner used here, it's equal to requested
			testDynamicProvisioning(c, claim, requestedSize)
		})
	})
})

func newClaim(ns, suffix string, alpha bool) *v1.PersistentVolumeClaim {
	claim := v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
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
			storageutil.StorageClassAnnotation: "myclass-" + suffix,
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
		ObjectMeta: metav1.ObjectMeta{
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

func newStorageClass(pluginName, suffix string) *storage.StorageClass {
	if pluginName == "" {
		switch {
		case framework.ProviderIs("gke"), framework.ProviderIs("gce"):
			pluginName = "kubernetes.io/gce-pd"
		case framework.ProviderIs("aws"):
			pluginName = "kubernetes.io/aws-ebs"
		case framework.ProviderIs("openstack"):
			pluginName = "kubernetes.io/cinder"
		case framework.ProviderIs("vsphere"):
			pluginName = "kubernetes.io/vsphere-volume"
		}
	}

	return &storage.StorageClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "myclass-" + suffix,
		},
		Provisioner: pluginName,
	}
}

func startExternalProvisioner(c clientset.Interface, ns string) *v1.Pod {
	podClient := c.Core().Pods(ns)

	provisionerPod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "external-provisioner-",
		},

		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "nfs-provisioner",
					Image: "quay.io/kubernetes_incubator/nfs-provisioner:v1.0.3",
					SecurityContext: &v1.SecurityContext{
						Capabilities: &v1.Capabilities{
							Add: []v1.Capability{"DAC_READ_SEARCH"},
						},
					},
					Args: []string{
						"-provisioner=" + externalPluginName,
						"-grace-period=0",
					},
					Ports: []v1.ContainerPort{
						{Name: "nfs", ContainerPort: 2049},
						{Name: "mountd", ContainerPort: 20048},
						{Name: "rpcbind", ContainerPort: 111},
						{Name: "rpcbind-udp", ContainerPort: 111, Protocol: v1.ProtocolUDP},
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
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "export-volume",
							MountPath: "/export",
						},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "export-volume",
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{},
					},
				},
			},
		},
	}
	provisionerPod, err := podClient.Create(provisionerPod)
	framework.ExpectNoError(err, "Failed to create %s pod: %v", provisionerPod.Name, err)

	framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, provisionerPod))

	By("locating the provisioner pod")
	pod, err := podClient.Get(provisionerPod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "Cannot locate the provisioner pod %v: %v", provisionerPod.Name, err)

	return pod
}
