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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	client "k8s.io/kubernetes/pkg/client/unversioned"
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

var _ = framework.KubeDescribe("Dynamic provisioning", func() {
	f := framework.NewDefaultFramework("volume-provisioning")

	// filled in BeforeEach
	var c *client.Client
	var ns string

	BeforeEach(func() {
		c = f.Client
		ns = f.Namespace.Name
	})

	framework.KubeDescribe("DynamicProvisioner", func() {
		It("should create and delete persistent volumes", func() {
			framework.SkipUnlessProviderIs("openstack", "gce", "aws", "gke")
			By("creating a claim with a dynamic provisioning annotation")
			claim := createClaim(ns)
			defer func() {
				c.PersistentVolumeClaims(ns).Delete(claim.Name)
			}()
			claim, err := c.PersistentVolumeClaims(ns).Create(claim)
			Expect(err).NotTo(HaveOccurred())

			err = framework.WaitForPersistentVolumeClaimPhase(api.ClaimBound, c, ns, claim.Name, framework.Poll, framework.ClaimProvisionTimeout)
			Expect(err).NotTo(HaveOccurred())

			By("checking the claim")
			// Get new copy of the claim
			claim, err = c.PersistentVolumeClaims(ns).Get(claim.Name)
			Expect(err).NotTo(HaveOccurred())

			// Get the bound PV
			pv, err := c.PersistentVolumes().Get(claim.Spec.VolumeName)
			Expect(err).NotTo(HaveOccurred())

			// Check sizes
			expectedCapacity := resource.MustParse(expectedSize)
			pvCapacity := pv.Spec.Capacity[api.ResourceName(api.ResourceStorage)]
			Expect(pvCapacity.Value()).To(Equal(expectedCapacity.Value()))

			requestedCapacity := resource.MustParse(requestedSize)
			claimCapacity := claim.Spec.Resources.Requests[api.ResourceName(api.ResourceStorage)]
			Expect(claimCapacity.Value()).To(Equal(requestedCapacity.Value()))

			// Check PV properties
			Expect(pv.Spec.PersistentVolumeReclaimPolicy).To(Equal(api.PersistentVolumeReclaimDelete))
			expectedAccessModes := []api.PersistentVolumeAccessMode{api.ReadWriteOnce}
			Expect(pv.Spec.AccessModes).To(Equal(expectedAccessModes))
			Expect(pv.Spec.ClaimRef.Name).To(Equal(claim.ObjectMeta.Name))
			Expect(pv.Spec.ClaimRef.Namespace).To(Equal(claim.ObjectMeta.Namespace))

			// We start two pods:
			// - The first writes 'hello word' to the /mnt/test (= the volume).
			// - The second one runs grep 'hello world' on /mnt/test.
			// If both succeed, Kubernetes actually allocated something that is
			// persistent across pods.
			By("checking the created volume is writable")
			runInPodWithVolume(c, ns, claim.Name, "echo 'hello world' > /mnt/test/data")

			By("checking the created volume is readable and retains data")
			runInPodWithVolume(c, ns, claim.Name, "grep 'hello world' /mnt/test/data")

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
			framework.ExpectNoError(c.PersistentVolumeClaims(ns).Delete(claim.Name))

			// Wait for the PV to get deleted too.
			framework.ExpectNoError(framework.WaitForPersistentVolumeDeleted(c, pv.Name, 5*time.Second, 20*time.Minute))
		})
	})
})

func createClaim(ns string) *api.PersistentVolumeClaim {
	return &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "pvc-",
			Namespace:    ns,
			Annotations: map[string]string{
				"volume.alpha.kubernetes.io/storage-class": "",
			},
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
			},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse(requestedSize),
				},
			},
		},
	}
}

// runInPodWithVolume runs a command in a pod with given claim mounted to /mnt directory.
func runInPodWithVolume(c *client.Client, ns, claimName, command string) {
	pod := &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: api.ObjectMeta{
			GenerateName: "pvc-volume-tester-",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:    "volume-tester",
					Image:   "gcr.io/google_containers/busybox:1.24",
					Command: []string{"/bin/sh"},
					Args:    []string{"-c", command},
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "my-volume",
							MountPath: "/mnt/test",
						},
					},
				},
			},
			RestartPolicy: api.RestartPolicyNever,
			Volumes: []api.Volume{
				{
					Name: "my-volume",
					VolumeSource: api.VolumeSource{
						PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{
							ClaimName: claimName,
							ReadOnly:  false,
						},
					},
				},
			},
		},
	}
	pod, err := c.Pods(ns).Create(pod)
	defer func() {
		framework.ExpectNoError(c.Pods(ns).Delete(pod.Name, nil))
	}()
	framework.ExpectNoError(err, "Failed to create pod: %v", err)
	framework.ExpectNoError(framework.WaitForPodSuccessInNamespaceSlow(c, pod.Name, pod.Spec.Containers[0].Name, pod.Namespace))
}
