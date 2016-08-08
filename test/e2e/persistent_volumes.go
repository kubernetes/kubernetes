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
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
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

// Delete the PV. Fail test if delete fails. If success the returned PV should
// be nil, which prevents the AfterEach from attempting to delete it.
func deletePersistentVolume(c *client.Client, pv *api.PersistentVolume) (*api.PersistentVolume, error) {

	if pv == nil {
		return nil, fmt.Errorf("PV to be deleted is nil")
	}

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

// Delete the PVC and wait for the PV to become Available again. Validate that
// the PV has recycled (assumption here about reclaimPolicy). Return the pv and
// pvc to reflect that these resources have been retrieved again (Get). If the
// delete is successful the returned pvc should be nil and the pv non-nil.
// Note: the pv and pvc are returned back to the It() caller so that the
//   AfterEach func can delete these objects if they are not nil.
func deletePVCandValidatePV(c *client.Client, ns string, pvc *api.PersistentVolumeClaim, pv *api.PersistentVolume) (*api.PersistentVolume, *api.PersistentVolumeClaim, error) {

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

// Create a PVC followed by the PV based on the passed in nfs-server ip and
// namespace. If the "preBind" bool is true then pre-bind the PV to the PVC
// via the PV's ClaimRef. Return the pv and pvc to reflect the created objects.
// Note: the pv and pvc are returned back to the It() caller so that the
//   AfterEach func can delete these objects if they are not nil.
// Note: in the pre-bind case the real PVC name, which is generated, is not
//   known until after the PVC is instantiated. This is why the pvc is created
//   before the pv.
func createPVCPV(c *client.Client, serverIP, ns string, preBind bool) (*api.PersistentVolume, *api.PersistentVolumeClaim, error) {

	var bindTo *api.PersistentVolumeClaim
	var preBindMsg string

	// make the pvc definition first
	pvc := makePersistentVolumeClaim(ns)
	if preBind {
		preBindMsg = " pre-bound"
		bindTo = pvc
	}
	// make the pv spec
	pv := makePersistentVolume(serverIP, bindTo)

	By(fmt.Sprintf("Creating a PVC followed by a%s PV", preBindMsg))

	// instantiate the pvc
	pvc, err := createPVC(c, ns, pvc)
	if err != nil {
		return nil, nil, err
	}

	// instantiate the pvc, handle pre-binding by ClaimRef if needed
	if preBind {
		pv.Spec.ClaimRef.Name = pvc.Name
	}
	pv, err = createPV(c, pv)
	if err != nil {
		return nil, pvc, err
	}

	return pv, pvc, nil
}

// Create a PV followed by the PVC based on the passed in nfs-server ip and
// namespace. If the "preBind" bool is true then pre-bind the PVC to the PV
// via the PVC's VolumeName. Return the pv and pvc to reflect the created
// objects.
// Note: the pv and pvc are returned back to the It() caller so that the
//   AfterEach func can delete these objects if they are not nil.
// Note: in the pre-bind case the real PV name, which is generated, is not
//   known until after the PV is instantiated. This is why the pv is created
//   before the pvc.
func createPVPVC(c *client.Client, serverIP, ns string, preBind bool) (*api.PersistentVolume, *api.PersistentVolumeClaim, error) {

	preBindMsg := ""
	if preBind {
		preBindMsg = " pre-bound"
	}

	By(fmt.Sprintf("Creating a PV followed by a%s PVC", preBindMsg))

	// make the pv and pvc definitions
	pv := makePersistentVolume(serverIP, nil)
	pvc := makePersistentVolumeClaim(ns)

	// instantiate the pv
	pv, err := createPV(c, pv)
	if err != nil {
		return nil, nil, err
	}

	// instantiate the pvc, handle pre-binding by VolumeName if needed
	if preBind {
		pvc.Spec.VolumeName = pv.Name
	}
	pvc, err = createPVC(c, ns, pvc)
	if err != nil {
		return pv, nil, err
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
// claimRef matches the pvc. Fails test on errors. Return the pv and pvc to
// reflect that these resources have been retrieved again (Get).
// Note: the pv and pvc are returned back to the It() caller so that the
//   AfterEach func can delete these objects if they are not nil.
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

// validate PV/PVC, create and verify writer pod, delete PVC and PV. Ensure
// all of these steps were successful. Return the pv and pvc to reflect that
// these resources have been retrieved again (Get).
// Note: the pv and pvc are returned back to the It() caller so that the
//   AfterEach func can delete these objects if they are not nil.
func completeTest(f *framework.Framework, c *client.Client, ns string, pv *api.PersistentVolume, pvc *api.PersistentVolumeClaim) (*api.PersistentVolume, *api.PersistentVolumeClaim, error) {

	// 1. verify that the PV and PVC have binded correctly
	By("Validating the PV-PVC binding")
	pv, pvc, err := waitAndValidatePVandPVC(c, ns, pv, pvc)
	if err != nil {
		return pv, pvc, err
	}

	// 2. create the nfs writer pod, test if the write was successful,
	//    then delete the pod and verify that it was deleted
	By("Checking pod has write access to PersistentVolume")
	if err = createWaitAndDeletePod(f, c, ns, pvc.Name); err != nil {
		return pv, pvc, err
	}

	// 3. delete the PVC before deleting PV, wait for PV to be "Available"
	By("Deleting the PVC to invoke the recycler")
	pv, pvc, err = deletePVCandValidatePV(c, ns, pvc, pv)
	if err != nil {
		return pv, pvc, err
	}

	// 4. cleanup by deleting the pv
	By("Deleting the PV")
	if pv, err = deletePersistentVolume(c, pv); err != nil {
		return pv, pvc, err
	}

	return pv, pvc, nil
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
		serverArgs:  []string{"-G", "777", "/exports"},
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
	// Create an nfs PV, then a claim that matches the PV, and a pod that
	// contains the claim. Verify that the PV and PVC bind correctly, and
	// that the pod can write to the nfs volume.
	It("should create a non-pre-bound PV and PVC: test write access [Flaky]", func() {

		pv, pvc, err = createPVPVC(c, serverIP, ns, false)
		if err != nil {
			framework.Failf("%v", err)
		}

		// validate PV-PVC, create and verify writer pod, delete PVC
		// and PV
		pv, pvc, err = completeTest(f, c, ns, pv, pvc)
		if err != nil {
			framework.Failf("%v", err)
		}
	})

	// Create a claim first, then a nfs PV that matches the claim, and a
	// pod that contains the claim. Verify that the PV and PVC bind
	// correctly, and that the pod can write to the nfs volume.
	It("create a PVC and non-pre-bound PV: test write access [Flaky]", func() {

		pv, pvc, err = createPVCPV(c, serverIP, ns, false)
		if err != nil {
			framework.Failf("%v", err)
		}

		// validate PV-PVC, create and verify writer pod, delete PVC
		// and PV
		pv, pvc, err = completeTest(f, c, ns, pv, pvc)
		if err != nil {
			framework.Failf("%v", err)
		}
	})

	// Create a claim first, then a pre-bound nfs PV that matches the claim,
	// and a pod that contains the claim. Verify that the PV and PVC bind
	// correctly, and that the pod can write to the nfs volume.
	It("create a PVC and a pre-bound PV: test write access [Flaky]", func() {

		pv, pvc, err = createPVCPV(c, serverIP, ns, true)
		if err != nil {
			framework.Failf("%v", err)
		}

		// validate PV-PVC, create and verify writer pod, delete PVC
		// and PV
		pv, pvc, err = completeTest(f, c, ns, pv, pvc)
		if err != nil {
			framework.Failf("%v", err)
		}
	})

	// Create a nfs PV first, then a pre-bound PVC that matches the PV,
	// and a pod that contains the claim. Verify that the PV and PVC bind
	// correctly, and that the pod can write to the nfs volume.
	It("create a PV and a pre-bound PVC: test write access [Flaky]", func() {

		pv, pvc, err = createPVPVC(c, serverIP, ns, true)
		if err != nil {
			framework.Failf("%v", err)
		}

		// validate PV-PVC, create and verify writer pod, delete PVC
		// and PV
		pv, pvc, err = completeTest(f, c, ns, pv, pvc)
		if err != nil {
			framework.Failf("%v", err)
		}
	})
})

// Returns a PV definition based on the nfs server IP. If the PVC is not nil
// then the PV is defined with a ClaimRef which includes the PVC's namespace.
// If the PVC is nil then the PV is not defined with a ClaimRef.
// Note: the passed-in claim does not have a name until it is created
//   (instantiated) and thus the PV's ClaimRef cannot be completely filled-in in
//   this func. Therefore, the ClaimRef's name is added later in
//   createPVCPV.
func makePersistentVolume(serverIP string, pvc *api.PersistentVolumeClaim) *api.PersistentVolume {
	// Specs are expected to match this test's PersistentVolumeClaim

	var claimRef *api.ObjectReference

	if pvc != nil {
		claimRef = &api.ObjectReference{
			Name:      pvc.Name,
			Namespace: pvc.Namespace,
		}
	}

	return &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "nfs-",
			Annotations: map[string]string{
				volumehelper.VolumeGidAnnotationKey: "777",
			},
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
// Note: if this PVC is intended to be pre-bound to a PV, whose name is not
//   known until the PV is instantiated, then the func createPVPVC will add
//   pvc.Spec.VolumeName to this claim.
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
					Args:    []string{"-c", "touch /mnt/SUCCESS && (id -G | grep -E '\\b777\\b')"},
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
