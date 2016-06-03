/*
Copyright 2015 The Kubernetes Authors.

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
	apierrs "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/test/e2e/framework"
)

// Delete the nfs-server pod.
func nfsServerPodCleanup(c *client.Client, config VolumeTestConfig) {
	defer GinkgoRecover()

	podClient := c.Pods(config.namespace)

	if config.serverImage != "" {
		podName := config.prefix+"-server"
		err := podClient.Delete(podName, nil)
		if err != nil {
			framework.Failf("Delete of %v pod failed: %v", podName, err)
		}
	}
}

// Delete the PV. Fail test if delete fails.
func deletePersistentVolume(c *client.Client, pv *api.PersistentVolume) {
	// Delete the PersistentVolume
	framework.Logf("Deleting PersistentVolume")
	err := c.PersistentVolumes().Delete(pv.Name)
	if err != nil {
		framework.Failf("Delete PersistentVolume %v failed: %v", pv.Name, err)
	}
	// Wait for PersistentVolume to Delete
	framework.WaitForPersistentVolumeDeleted(c, pv.Name, 3*time.Second, 30*time.Second)
}

// Test the pod's exitcode to be zero, delete the pod, wait for it to be deleted,
// and fail if these steps return an error.
func testPodSuccessOrFail(f *framework.Framework, c *client.Client, ns string, pod *api.Pod) {

        By("Pod should terminate with exitcode 0 (success)")

        err := framework.WaitForPodSuccessInNamespace(c, pod.Name, pod.Spec.Containers[0].Name, ns)
        if err != nil {
                framework.Failf("Pod %v returned non-zero exitcode: %+v", pod.Name, err)
        }

        framework.Logf("Deleting pod %v after it exited successfully", pod.Name)
        err = c.Pods(ns).Delete(pod.Name, nil)
        if err != nil {
                framework.Failf("Pod %v exited successfully but failed to delete: %+v", pod.Name, err)
        }

        // Wait for pod to terminate
        err = f.WaitForPodTerminated(pod.Name, "")
        if err != nil && !apierrs.IsNotFound(err) {
                framework.Failf("Pod %v has exitcode 0 but will not teminate: %v", pod.Name, err)
        }
        framework.Logf("Pod %v exited SUCCESSFULLY and was deleted", pod.Name)
}


var _ = framework.KubeDescribe("PersistentVolumes", func() {
	f := framework.NewDefaultFramework("pv")
	var c *client.Client
	var ns string
	var NFSconfig VolumeTestConfig
	var serverIP string
	var nfsServerPod *api.Pod
	var checkPod *api.Pod
	var pv *api.PersistentVolume
	var pvc *api.PersistentVolumeClaim
	var err error

	// config for the nfs-server pod in the default namespace
	NFSconfig = VolumeTestConfig{
		namespace:   api.NamespaceDefault,
		prefix:      "nfs",
		serverImage: "gcr.io/google_containers/volume-nfs:0.6",
		serverPorts: []int{2049},
	}

	BeforeEach(func() {
		c = f.Client
		ns = f.Namespace.Name

		// If it doesn't exist, create the nfs server pod in "default" ns
		if nfsServerPod == nil {
			nfsServerPod = startVolumeServer(c, NFSconfig)
			serverIP = nfsServerPod.Status.PodIP
			framework.Logf("NFS server IP address: %v", serverIP)
		}
	})

	AfterEach(func() {
		if c != nil && len(ns) > 0 {
			if checkPod != nil {
				// Wait for checkpod to complete termination
				err = c.Pods(ns).Delete(checkPod.Name, nil)
				if err != nil {
					framework.Failf("AfterEach: pod %v delete ierror: %v", checkPod.Name, err)
				}
				checkPod = nil
			}

			if pvc != nil {
				// Delete the PersistentVolumeClaim
				err = c.PersistentVolumeClaims(ns).Delete(pvc.Name)
				if err != nil && !apierrs.IsNotFound(err) {
					framework.Failf("AfterEach: delete of PersistentVolumeClaim %v experienced an unexpected error: %v", pvc.Name, err)
				}
				pvc = nil
			}
			if pv != nil {
				deletePersistentVolume(c, pv)
				pv = nil
			}
		}
	})

	// Execute after *all* the tests have run
	AddCleanupAction(func() {
		if nfsServerPod != nil && c != nil {
			nfsServerPodCleanup(c, NFSconfig)
			nfsServerPod = nil
		}
	})


	// Individual tests follow:
	It("should create a PersistentVolume, Claim, and a client Pod that will test the read/write access of the volume", func() {

		// Define the PersistentVolume and PersistentVolumeClaim
		pv := makePersistentVolume(serverIP)
		pvc := makePersistentVolumeClaim(ns)

		// Create the PersistentVolume and wait for PersistentVolume.Status.Phase to be Available
		By("Creating PV and PVC and waiting for Bound status")
		framework.Logf("Creating PersistentVolume")
		pv, err := c.PersistentVolumes().Create(pv)
		if err != nil {
			framework.Failf("Create PersistentVolume failed: %v", err)
		}
		// Wait for PV to become Available.
		framework.WaitForPersistentVolumePhase(api.VolumeAvailable, c, pv.Name, 1*time.Second, 20*time.Second)

		// Create the PersistentVolumeClaim and wait for Bound phase, can take several minutes.
		framework.Logf("Creating PersistentVolumeClaim")
		pvc, err = c.PersistentVolumeClaims(ns).Create(pvc)
		if err != nil {
			framework.Failf("Create PersistentVolumeClaim failed: %v", err)
		}
		framework.WaitForPersistentVolumeClaimPhase(api.ClaimBound, c, ns, pvc.Name, 3*time.Second, 300*time.Second)

		// Wait for PersistentVolume.Status.Phase to be Bound, which it should already be since the PVC is bound.
		err = framework.WaitForPersistentVolumePhase(api.VolumeBound, c, pv.Name, 3*time.Second, 300*time.Second)
		if err != nil {
			framework.Failf("PersistentVolume failed to enter a bound state even though PVC is Bound: %+v", err)
		}

		// Check the PersistentVolume.ClaimRef is valid and matches the PVC
		framework.Logf("Checking PersistentVolume ClaimRef is non-nil")
		pv, err = c.PersistentVolumes().Get(pv.Name)
		if err != nil {
			framework.Failf("Cannot re-get PersistentVolume %v:", pv.Name, err)
		}
		pvc, err = c.PersistentVolumeClaims(ns).Get(pvc.Name)
		if err != nil {
			framework.Failf("Cannot re-get PersistentVolumeClaim %v:", pvc.Name, err)
		}
		if pv.Spec.ClaimRef == nil || pv.Spec.ClaimRef.UID != pvc.UID {
			pvJson, _ := json.MarshalIndent(pv.Spec.ClaimRef, "", "  ")
			framework.Failf("Expected Bound PersistentVolume %v to have valid ClaimRef: %+v", pv.Name, string(pvJson))
		}

		// checkPod writes to the nfs volume
		By("Checking pod has write access to PersistentVolume")
		framework.Logf("Creating checkPod")
		checkPod := makeWritePod(ns, pvc.Name)
		checkPod, err = c.Pods(ns).Create(checkPod)
		if err != nil {
			framework.Failf("Create checkPod failed: %+v", err)
		}
		// Wait for the checkPod to complete its lifecycle
		testPodSuccessOrFail(f, c, ns, checkPod)
		checkPod = nil

		// Delete the PersistentVolumeClaim
		By("Deleting PersistentVolumeClaim to trigger PV Recycling")
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

		// Examine the PersistentVolume.ClaimRef and UID. Expect nil values.
		pv, err = c.PersistentVolumes().Get(pv.Name)
		if pv.Spec.ClaimRef != nil && len(pv.Spec.ClaimRef.UID) > 0 {
			crjson, _ := json.MarshalIndent(pv.Spec.ClaimRef, "", "  ")
			framework.Failf("Expected a nil pv.ClaimRef or empty UID. Found: ", string(crjson))
		}

		// Delete the PersistentVolume
		By("Deleting PersistentVolume")
		deletePersistentVolume(c, pv)
	})


	It("should create another pod.... testing...", func() {
		checkPod = makeTestPod(ns, serverIP)
		checkPod, err = c.Pods(ns).Create(checkPod)
		if err != nil {
			framework.Failf("Error during testpod create: %v", err)
		}

		// Wait for checkpod to complete it's lifecycle
		testPodSuccessOrFail(f, c, ns, checkPod)
		checkPod = nil // for AfterEach above

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

func makeTestPod(ns string, nfsserver string) *api.Pod {
	// Prepare pod that mounts the NFS volume again and
	// checks that the volume can be written to via the mount.

	var isPrivileged bool = true
	return &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: testapi.Default.GroupVersion().String(),
		},
		ObjectMeta: api.ObjectMeta{
			GenerateName: "test-pod-",
			Namespace:    ns,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:    "test-pod",
					Image:   "gcr.io/google_containers/busybox:1.24",
					Command: []string{"/bin/sh"},
					Args:    []string{"-c", "touch /mnt/FOO && exit 0 || exit 1"},
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "nfs-vol",
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
					Name: "nfs-vol",
					VolumeSource: api.VolumeSource{
						NFS: &api.NFSVolumeSource{
							Server: nfsserver,
							Path: "/",
						},
					},
				},
			},
		},
	}
}

