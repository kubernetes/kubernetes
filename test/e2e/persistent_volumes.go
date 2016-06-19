/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"encoding/json"
	"time"

	. "github.com/onsi/ginkgo"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/test/e2e/framework"
)

// Clean both server and client pods.
func persistentVolumeTestCleanup(client *client.Client, config VolumeTestConfig) {
	defer GinkgoRecover()

	podClient := client.Pods(config.namespace)

	if config.serverImage != "" {
		err := podClient.Delete(config.prefix+"-server", nil)
		if err != nil {
			framework.Failf("Failed to delete the server pod: %v", err)
		}
	}
}

func deletePersistentVolume(c *client.Client, pv *api.PersistentVolume) {
	// Delete the PersistentVolume
	framework.Logf("Deleting PersistentVolume")
	err := c.PersistentVolumes().Delete(pv.Name)
	if err != nil {
		framework.Failf("Delete PersistentVolume failed: %v", err)
	}
	// Wait for PersistentVolume to Delete
	framework.WaitForPersistentVolumeDeleted(c, pv.Name, 3*time.Second, 30*time.Second)
}

var _ = framework.KubeDescribe("PersistentVolumes", func() {
	f := framework.NewDefaultFramework("pv")
	var c *client.Client
	var ns string
	BeforeEach(func() {
		c = f.Client
		ns = f.Namespace.Name
	})

	It("should create a PersistentVolume, Claim, and a client Pod that will test the read/write access of the volume[Flaky]", func() {
		config := VolumeTestConfig{
			namespace:   ns,
			prefix:      "nfs",
			serverImage: "gcr.io/google_containers/volume-nfs:0.6",
			serverPorts: []int{2049},
		}

		defer func() {
			persistentVolumeTestCleanup(c, config)
		}()

		// Create the nfs server pod
		pod := startVolumeServer(c, config)
		serverIP := pod.Status.PodIP
		framework.Logf("NFS server IP address: %v", serverIP)

		// Define the PersistentVolume and PersistentVolumeClaim
		pv := makePersistentVolume(serverIP)
		pvc := makePersistentVolumeClaim(ns)

		// Create the PersistentVolume and wait for PersistentVolume.Status.Phase to be Available
		// defer deletion to clean up the PV should the test fail post-creation.
		framework.Logf("Creating PersistentVolume")
		pv, err := c.PersistentVolumes().Create(pv)
		if err != nil {
			framework.Failf("Create PersistentVolume failed: %v", err)
		}
		defer deletePersistentVolume(c, pv)
		framework.WaitForPersistentVolumePhase(api.VolumeAvailable, c, pv.Name, 1*time.Second, 20*time.Second)

		// Create the PersistentVolumeClaim and wait for Bound phase
		framework.Logf("Creating PersistentVolumeClaim")
		pvc, err = c.PersistentVolumeClaims(ns).Create(pvc)
		if err != nil {
			framework.Failf("Create PersistentVolumeClaim failed: %v", err)
		}
		framework.WaitForPersistentVolumeClaimPhase(api.ClaimBound, c, ns, pvc.Name, 3*time.Second, 300*time.Second)

		// Wait for PersistentVolume.Status.Phase to be Bound. Can take several minutes.
		err = framework.WaitForPersistentVolumePhase(api.VolumeBound, c, pv.Name, 3*time.Second, 300*time.Second)
		if err != nil {
			framework.Failf("PersistentVolume failed to enter a bound state: %+v", err)
		}
		// Check the PersistentVolume.ClaimRef.UID for non-nil value as confirmation of the bound state.
		framework.Logf("Checking PersistentVolume ClaimRef is non-nil")
		pv, err = c.PersistentVolumes().Get(pv.Name)
		if pv.Spec.ClaimRef == nil || len(pv.Spec.ClaimRef.UID) == 0 {
			pvJson, _ := json.MarshalIndent(pv, "", "  ")
			framework.Failf("Expected PersistentVolume to be bound, but got nil ClaimRef or UID: %+v", string(pvJson))
		}

		// Check the PersistentVolumeClaim.Status.Phase for Bound state
		framework.Logf("Checking PersistentVolumeClaim status is Bound")
		pvc, err = c.PersistentVolumeClaims(ns).Get(pvc.Name)
		if pvcPhase := pvc.Status.Phase; pvcPhase != "Bound" {
			framework.Failf("Expected PersistentVolumeClaim status Bound. Actual:  %+v.  Error: %+v", pvcPhase, err)
		}

		// Check that the PersistentVolume's ClaimRef contains the UID of the PersistendVolumeClaim
		if pvc.ObjectMeta.UID != pv.Spec.ClaimRef.UID {
			framework.Failf("Binding failed: PersistentVolumeClaim UID does not match PersistentVolume's ClaimRef UID. ")
		}

		// writePod writes to the nfs volume
		framework.Logf("Creating writePod")
		pvc, _ = c.PersistentVolumeClaims(ns).Get(pvc.Name)
		writePod := makeWritePod(ns, pvc.Name)
		writePod, err = c.Pods(ns).Create(writePod)
		if err != nil {
			framework.Failf("Create writePod failed: %+v", err)
		}

		// Wait for the writePod to complete it's lifecycle
		err = framework.WaitForPodSuccessInNamespace(c, writePod.Name, writePod.Spec.Containers[0].Name, writePod.Namespace)
		if err != nil {
			framework.Failf("WritePod exited with error: %+v", err)
		} else {
			framework.Logf("WritePod exited without error.")
		}

		// Delete the PersistentVolumeClaim
		framework.Logf("Deleting PersistentVolumeClaim to trigger PV Recycling")
		err = c.PersistentVolumeClaims(ns).Delete(pvc.Name)
		if err != nil {
			framework.Failf("Delete PersistentVolumeClaim failed: %v", err)
		}

		// Wait for the PersistentVolume phase to return to Available
		framework.Logf("Waiting for recycling process to complete.")
		err = framework.WaitForPersistentVolumePhase(api.VolumeAvailable, c, pv.Name, 3*time.Second, 300*time.Second)
		if err != nil {
			framework.Failf("Recycling failed: %v", err)
		}

		// Examine the PersistentVolume.ClaimRef and UID.  Expect nil values.
		pv, err = c.PersistentVolumes().Get(pv.Name)
		if pv.Spec.ClaimRef != nil && len(pv.Spec.ClaimRef.UID) > 0 {
			crjson, _ := json.MarshalIndent(pv.Spec.ClaimRef, "", "  ")
			framework.Failf("Expected a nil ClaimRef or UID. Found: ", string(crjson))
		}

	})
})

func makePersistentVolume(serverIP string) *api.PersistentVolume {
	// Takes an NFS server IP address and returns a PersistentVolume object for instantiation.
	// Specs are expected to match this test's PersistentVolumeClaim
	return &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "nfs-",
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimRecycle,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("2Gi"),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				NFS: &api.NFSVolumeSource{
					Server:   serverIP,
					Path:     "/exports",
					ReadOnly: false,
				},
			},
			AccessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
				api.ReadOnlyMany,
				api.ReadWriteMany,
			},
		},
	}
}

func makePersistentVolumeClaim(ns string) *api.PersistentVolumeClaim {
	// Takes a namespace and returns a PersistentVolumeClaim object for instantiation.
	// Specs are expected to match this test's PersistentVolume
	return &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "pvc-",
			Namespace:    ns,
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
				api.ReadOnlyMany,
				api.ReadWriteMany,
			},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("1Gi"),
				},
			},
		},
	}
}

func makeWritePod(ns string, pvcName string) *api.Pod {
	// Prepare pod that mounts the NFS volume again and
	// checks that /mnt/index.html was scrubbed there

	var isPrivileged bool = true
	return &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: testapi.Default.GroupVersion().String(),
		},
		ObjectMeta: api.ObjectMeta{
			GenerateName: "write-pod-",
			Namespace:    ns,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:    "write-pod",
					Image:   "gcr.io/google_containers/busybox:1.24",
					Command: []string{"/bin/sh"},
					Args:    []string{"-c", "touch /mnt/SUCCESS && exit 0 || exit 1"},
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "nfs-pvc",
							MountPath: "/mnt",
						},
					},
					SecurityContext: &api.SecurityContext{
						Privileged: &isPrivileged,
					},
				},
			},
			Volumes: []api.Volume{
				{
					Name: "nfs-pvc",
					VolumeSource: api.VolumeSource{
						PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{
							ClaimName: pvcName,
						},
					},
				},
			},
		},
	}
}
