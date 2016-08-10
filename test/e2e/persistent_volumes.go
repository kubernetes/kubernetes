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
	"fmt"
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
		podName := config.prefix + "-server"
		err := podClient.Delete(podName, nil)
		if err != nil {
			framework.Logf("Delete of %v pod failed: %v", podName, err)
		}
	}
}

// Delete the PV. Fail test if delete fails.
func deletePersistentVolume(c *client.Client, pv *api.PersistentVolume) (*api.PersistentVolume, error) {

	if pv == nil {
		return nil, fmt.Errorf("PV to be deleted is nil")
	}

	By("Deleting PersistentVolume")

	framework.Logf("Deleting PersistentVolume %v", pv.Name)
	err := c.PersistentVolumes().Delete(pv.Name)
	if err != nil {
		return pv, fmt.Errorf("Delete() PersistentVolume %v failed: %v", pv.Name, err)
	}

	// Wait for PersistentVolume to delete
	deleteDuration := 90 * time.Second
	err = framework.WaitForPersistentVolumeDeleted(c, pv.Name, 3*time.Second, deleteDuration)
	if err != nil {
		return pv, fmt.Errorf("Unable to delete PersistentVolume %s after waiting for %v: %v", pv.Name, deleteDuration, err)
	}

	return nil, nil // success
}

// Delete the PVC and wait for the PV to become Available again.
// Validate that the PV has recycled (assumption here about reclaimPolicy).
func deletePVCandValidatePV(c *client.Client, ns string, pvc *api.PersistentVolumeClaim, pv *api.PersistentVolume) (*api.PersistentVolume, *api.PersistentVolumeClaim, error) {

	By("Deleting PersistentVolumeClaim to trigger PV Recycling")

	framework.Logf("Deleting PersistentVolumeClaim %v to trigger PV Recycling", pvc.Name)
	err := c.PersistentVolumeClaims(ns).Delete(pvc.Name)
	if err != nil {
		return pv, pvc, fmt.Errorf("Delete of PVC %v failed: %v", pvc.Name, err)
	}

	// Check that the PVC is really deleted.
	pvc, err = c.PersistentVolumeClaims(ns).Get(pvc.Name)
	if err == nil {
		return pv, pvc, fmt.Errorf("PVC %v deleted yet still exists", pvc.Name)
	}
	if !apierrs.IsNotFound(err) {
		return pv, pvc, fmt.Errorf("Get on deleted PVC %v failed with error other than \"not found\": %v", pvc.Name, err)
	}

	// Wait for the PV's phase to return to Available
	framework.Logf("Waiting for recycling process to complete.")
	err = framework.WaitForPersistentVolumePhase(api.VolumeAvailable, c, pv.Name, 3*time.Second, 300*time.Second)
	if err != nil {
		return pv, pvc, fmt.Errorf("Recycling failed: %v", err)
	}

	// Examine the pv.ClaimRef and UID. Expect nil values.
	pv, err = c.PersistentVolumes().Get(pv.Name)
	if err != nil {
		return pv, pvc, fmt.Errorf("Cannot re-get PersistentVolume %v:", pv.Name)
	}
	if pv.Spec.ClaimRef != nil && len(pv.Spec.ClaimRef.UID) > 0 {
		crJSON, _ := json.Marshal(pv.Spec.ClaimRef)
		return pv, pvc, fmt.Errorf("Expected PV %v's ClaimRef to be nil, or the claimRef's UID to be blank. Instead claimRef is: %v", pv.Name, string(crJSON))
	}

	return pv, pvc, nil
}

// create the PV resource. Fails test on error.
func createPV(c *client.Client, pv *api.PersistentVolume) (*api.PersistentVolume, error) {

	pv, err := c.PersistentVolumes().Create(pv)
	if err != nil {
		return pv, fmt.Errorf("Create PersistentVolume %v failed: %v", pv.Name, err)
	}

	return pv, nil
}

// create the PVC resource. Fails test on error.
func createPVC(c *client.Client, ns string, pvc *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {

	pvc, err := c.PersistentVolumeClaims(ns).Create(pvc)
	if err != nil {
		return pvc, fmt.Errorf("Create PersistentVolumeClaim %v failed: %v", pvc.Name, err)
	}

	return pvc, nil
}

// Create a PV and PVC based on the passed in nfs-server ip and namespace.
// There are 4 combinations, 3 of which are supported here:
//   1) prebind and create pvc first
//   2) no prebind and create pvc first
//   3) no prebind and create pv first
// The case of prebinding and creating the pv first is not possible due to using a
// *generated* name in the pvc, and thus not knowing the claim's name until after
// it has been created.
// **Note: this function complements makePersistentVolume() and fills in the remaining
// name field in the pv's ClaimRef.
func createPVandPVC(c *client.Client, serverIP, ns string, pvFirst, preBind bool) (*api.PersistentVolume, *api.PersistentVolumeClaim, error) {

	var bindTo *api.PersistentVolumeClaim
	var err error

	pvc := makePersistentVolumeClaim(ns) // pvc.Name not known yet

	bindTo = nil
	if preBind { // implies pvc *must* be created before the pv
		pvFirst = false
		bindTo = pvc
	}
	pv := makePersistentVolume(serverIP, bindTo)

	if pvFirst {
		By("Creating the PV followed by the PVC")
		pv, err = createPV(c, pv)
	} else {
		By("Creating the PVC followed by the PV")
		pvc, err = createPVC(c, ns, pvc)
	}
	if err != nil {
		return nil, nil, err
	}

	if pvFirst {
		pvc, err = createPVC(c, ns, pvc)
		if err != nil {
			return pv, nil, err
		}
	} else {
		// need to fill-in claimRef with pvc.Name
		pv.Spec.ClaimRef.Name = pvc.Name
		pv, err = createPV(c, pv)
		if err != nil {
			return nil, pvc, err
		}
	}

	return pv, pvc, nil
}

// Wait for the pv and pvc to bind to each other. Fail test on errors.
func waitOnPVandPVC(c *client.Client, ns string, pv *api.PersistentVolume, pvc *api.PersistentVolumeClaim) error {

	// Wait for newly created PVC to bind to the PV
	framework.Logf("Waiting for PV %v to bind to PVC %v", pv.Name, pvc.Name)
	err := framework.WaitForPersistentVolumeClaimPhase(api.ClaimBound, c, ns, pvc.Name, 3*time.Second, 300*time.Second)
	if err != nil {
		return fmt.Errorf("PersistentVolumeClaim failed to enter a bound state: %+v", err)
	}

	// Wait for PersistentVolume.Status.Phase to be Bound, which it should be
	// since the PVC is already bound.
	err = framework.WaitForPersistentVolumePhase(api.VolumeBound, c, pv.Name, 3*time.Second, 300*time.Second)
	if err != nil {
		return fmt.Errorf("PersistentVolume failed to enter a bound state even though PVC is Bound: %+v", err)
	}

	return nil
}

// Waits for the pv and pvc to be bound to each other, then checks that the pv's
// claimRef matches the pvc. Fails test on errors.
func waitAndValidatePVandPVC(c *client.Client, ns string, pv *api.PersistentVolume, pvc *api.PersistentVolumeClaim) (*api.PersistentVolume, *api.PersistentVolumeClaim, error) {

	var err error

	// Wait for pv and pvc to bind to each other
	if err = waitOnPVandPVC(c, ns, pv, pvc); err != nil {
		return pv, pvc, err
	}

	// Check that the PersistentVolume.ClaimRef is valid and matches the PVC
	framework.Logf("Checking PersistentVolume ClaimRef is non-nil")
	pv, err = c.PersistentVolumes().Get(pv.Name)
	if err != nil {
		return pv, pvc, fmt.Errorf("Cannot re-get PersistentVolume %v:", pv.Name)
	}

	pvc, err = c.PersistentVolumeClaims(ns).Get(pvc.Name)
	if err != nil {
		return pv, pvc, fmt.Errorf("Cannot re-get PersistentVolumeClaim %v:", pvc.Name)
	}

	if pv.Spec.ClaimRef == nil || pv.Spec.ClaimRef.UID != pvc.UID {
		pvJSON, _ := json.Marshal(pv.Spec.ClaimRef)
		return pv, pvc, fmt.Errorf("Expected Bound PersistentVolume %v to have valid ClaimRef: %+v", pv.Name, string(pvJSON))
	}

	return pv, pvc, nil
}

// Test the pod's exitcode to be zero.
func testPodSuccessOrFail(f *framework.Framework, c *client.Client, ns string, pod *api.Pod) error {

	By("Pod should terminate with exitcode 0 (success)")

	err := framework.WaitForPodSuccessInNamespace(c, pod.Name, pod.Spec.Containers[0].Name, ns)
	if err != nil {
		return fmt.Errorf("Pod %v returned non-zero exitcode: %+v", pod.Name, err)
	}

	framework.Logf("pod %v exited successfully", pod.Name)
	return nil
}

// Delete the passed in pod.
func deletePod(f *framework.Framework, c *client.Client, ns string, pod *api.Pod) error {

	framework.Logf("Deleting pod %v", pod.Name)
	err := c.Pods(ns).Delete(pod.Name, nil)
	if err != nil {
		return fmt.Errorf("Pod %v encountered a delete error: %v", pod.Name, err)
	}

	// Wait for pod to terminate
	err = f.WaitForPodTerminated(pod.Name, "")
	if err != nil && !apierrs.IsNotFound(err) {
		return fmt.Errorf("Pod %v will not teminate: %v", pod.Name, err)
	}

	// Re-get the pod to double check that it has been deleted; expect err
	// Note: Get() writes a log error if the pod is not found
	_, err = c.Pods(ns).Get(pod.Name)
	if err == nil {
		return fmt.Errorf("Pod %v has been deleted but able to re-Get the deleted pod", pod.Name)
	}
	if !apierrs.IsNotFound(err) {
		return fmt.Errorf("Pod %v has been deleted but still exists: %v", pod.Name, err)
	}

	framework.Logf("Ignore \"not found\" error above. Pod %v successfully deleted", pod.Name)
	return nil
}

// Create the test pod, wait for (hopefully) success, and then delete the pod.
func createWaitAndDeletePod(f *framework.Framework, c *client.Client, ns string, claimName string) error {

	var errmsg string

	framework.Logf("Creating nfs test pod")

	// Make pod spec
	pod := makeWritePod(ns, claimName)

	// Instantiate pod (Create)
	runPod, err := c.Pods(ns).Create(pod)
	if err != nil || runPod == nil {
		name := ""
		if runPod != nil {
			name = runPod.Name
		}
		return fmt.Errorf("Create test pod %v failed: %v", name, err)
	}

	// Wait for the test pod to complete its lifecycle
	podErr := testPodSuccessOrFail(f, c, ns, runPod)

	// Regardless of podErr above, delete the pod if it exists
	if runPod != nil {
		err = deletePod(f, c, ns, runPod)
	}

	// Check results of pod success and pod delete
	if podErr != nil {
		errmsg = fmt.Sprintf("Pod %v exited with non-zero exitcode: %v", runPod.Name, podErr)
	}
	if err != nil { // Delete error
		if len(errmsg) > 0 {
			errmsg += "; and "
		}
		errmsg += fmt.Sprintf("Delete error on pod %v: %v", runPod.Name, err)
	}

	if len(errmsg) > 0 {
		return fmt.Errorf(errmsg)
	}

	return nil
}

var _ = framework.KubeDescribe("PersistentVolumes", func() {

	// global vars for the It() tests below
	f := framework.NewDefaultFramework("pv")
	var c *client.Client
	var ns string
	var NFSconfig VolumeTestConfig
	var serverIP string
	var nfsServerPod *api.Pod
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
		// The "default" ns is used so that individual tests can delete
		// their ns without impacting the nfs-server pod.
		if nfsServerPod == nil {
			nfsServerPod = startVolumeServer(c, NFSconfig)
			serverIP = nfsServerPod.Status.PodIP
			framework.Logf("NFS server IP address: %v", serverIP)
		}
	})

	AfterEach(func() {
		if c != nil && len(ns) > 0 { // still have client and namespace
			if pvc != nil && len(pvc.Name) > 0 {
				// Delete the PersistentVolumeClaim
				framework.Logf("AfterEach: PVC %v is non-nil, deleting claim", pvc.Name)
				err := c.PersistentVolumeClaims(ns).Delete(pvc.Name)
				if err != nil && !apierrs.IsNotFound(err) {
					framework.Logf("AfterEach: delete of PersistentVolumeClaim %v error: %v", pvc.Name, err)
				}
				pvc = nil
			}
			if pv != nil && len(pv.Name) > 0 {
				framework.Logf("AfterEach: PV %v is non-nil, deleting pv", pv.Name)
				err := c.PersistentVolumes().Delete(pv.Name)
				if err != nil && !apierrs.IsNotFound(err) {
					framework.Logf("AfterEach: delete of PersistentVolume %v error: %v", pv.Name, err)
				}
				pv = nil
			}
		}
	})

	// Execute after *all* the tests have run
	AddCleanupAction(func() {
		if nfsServerPod != nil && c != nil {
			framework.Logf("AfterSuite: nfs-server pod %v is non-nil, deleting pod", nfsServerPod.Name)
			nfsServerPodCleanup(c, NFSconfig)
			nfsServerPod = nil
		}
	})

	// Individual tests follow:
	//
	// Create an nfs PV, a claim that matches the PV, a pod that contains the
	// claim. Verify that the PV and PVC bind correctly and that the pod can
	// write to the nfs volume.
	It("should create a PersistentVolume, Claim, and Pod that will test write access of the volume [Flaky]", func() {

		pv, pvc, err = createPVandPVC(c, serverIP, ns, true /*pv first*/, false)
		if err != nil {
			framework.Failf("%v", err)
		}

		pv, pvc, err = waitAndValidatePVandPVC(c, ns, pv, pvc)
		if err != nil {
			framework.Failf("%v", err)
		}

		By("Checking pod has write access to PersistentVolume")

		if err = createWaitAndDeletePod(f, c, ns, pvc.Name); err != nil {
			framework.Failf("%v", err)
		}

		// Delete the PVC before deleting PV, wait for PV to be Available
		pv, pvc, err = deletePVCandValidatePV(c, ns, pvc, pv)
		if err != nil {
			framework.Failf("%v", err)
		}

		// Last cleanup step is to delete the pv
		pv, err = deletePersistentVolume(c, pv)
		if err != nil {
			framework.Failf("%v", err)
		}
	})

	// Create an nfs PV that is *pre-bound* to a claim. Create a pod that
	// contains the claim. Verify that the PV and PVC bind correctly and that
	// the pod can write to the nfs volume.
	It("should create a pre-bound PersistentVolume, Claim, and Pod that will test write access of the volume [Flaky]", func() {

		pv, pvc, err = createPVandPVC(c, serverIP, ns, false /*pvc first*/, true /*prebind*/)
		if err != nil {
			framework.Failf("%v", err)
		}

		pv, pvc, err = waitAndValidatePVandPVC(c, ns, pv, pvc)
		if err != nil {
			framework.Failf("%v", err)
		}

		// checkPod writes to the nfs volume
		By("Checking pod has write access to pre-bound PersistentVolume")
		// Instantiate pod, wait for it to exit, then delete it
		if err = createWaitAndDeletePod(f, c, ns, pvc.Name); err != nil {
			framework.Failf("%v", err)
		}

		// Delete the PVC before deleting PV, wait for PV to be Available
		pv, pvc, err = deletePVCandValidatePV(c, ns, pvc, pv)
		if err != nil {
			framework.Failf("%v", err)
		}

		// Last cleanup step is to delete the pv
		pv, err = deletePersistentVolume(c, pv)
		if err != nil {
			framework.Failf("%v", err)
		}
	})
})

// Returns a PV definition based on the nfs server IP. If the PVC is not nil then the
// PV is defined with a ClaimRef which includes the PVC's namespace. If the PVC is
// nil then the PV is not defined with a ClaimRef.
// **Note: the passed-in claim does not have a name until it is created (instantiated)
// and thus the PV's ClaimRef cannot be completely filled-in in this func. Therefore,
// the ClaimRef's name is added later in createPVandPVC().
func makePersistentVolume(serverIP string, pvc *api.PersistentVolumeClaim) *api.PersistentVolume {
	// Specs are expected to match this test's PersistentVolumeClaim

	var claimRef *api.ObjectReference

	claimRef = nil
	if pvc != nil {
		claimRef = &api.ObjectReference{
			Name:      pvc.Name,
			Namespace: pvc.Namespace,
		}
	}

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
			ClaimRef: claimRef,
		},
	}
}

// Returns a PVC definition based on the namespace.
func makePersistentVolumeClaim(ns string) *api.PersistentVolumeClaim {
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

// Returns a pod definition based on the namespace. The pod references the PVC's
// name.
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
