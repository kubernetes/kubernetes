/*
Copyright 2017 The Kubernetes Authors.

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
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8stype "k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	storage "k8s.io/kubernetes/pkg/apis/storage/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	vsphere "k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
	"k8s.io/kubernetes/test/e2e/framework"
)

/*
	Test to perform Disk Ops storm.

	Steps
    	1. Create storage class for thin Provisioning.
    	2. Create 30 PVCs using above storage class in annotation, requesting 2 GB files.
    	3. Wait until all disks are ready and all PVs and PVCs get bind. (CreateVolume storm)
    	4. Create pod to mount volumes using PVCs created in step 2. (AttachDisk storm)
    	5. Wait for pod status to be running.
    	6. Verify all volumes accessible and available in the pod.
    	7. Delete pod.
    	8. wait until volumes gets detached. (DetachDisk storm)
    	9. Delete all PVCs. This should delete all Disks. (DeleteVolume storm)
		10. Delete storage class.
*/

var _ = framework.KubeDescribe("vsphere volume operations storm [Volume]", func() {
	f := framework.NewDefaultFramework("volume-ops-storm")
	const DEFAULT_VOLUME_OPS_SCALE = 30
	var (
		client            clientset.Interface
		namespace         string
		storageclass      *storage.StorageClass
		pvclaims          []*v1.PersistentVolumeClaim
		persistentvolumes []*v1.PersistentVolume
		err               error
		volume_ops_scale  int
		vsp               *vsphere.VSphere
	)
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("vsphere")
		client = f.ClientSet
		namespace = f.Namespace.Name
		nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		if len(nodeList.Items) == 0 {
			framework.Failf("Unable to find ready and schedulable Node")
		}
		if os.Getenv("VOLUME_OPS_SCALE") != "" {
			volume_ops_scale, err = strconv.Atoi(os.Getenv("VOLUME_OPS_SCALE"))
			Expect(err).NotTo(HaveOccurred())
		} else {
			volume_ops_scale = DEFAULT_VOLUME_OPS_SCALE
		}
		pvclaims = make([]*v1.PersistentVolumeClaim, volume_ops_scale)
		vsp, err = vsphere.GetVSphere()
		Expect(err).NotTo(HaveOccurred())
	})
	AfterEach(func() {
		By("Deleting PVCs")
		for _, claim := range pvclaims {
			framework.DeletePersistentVolumeClaim(client, claim.Name, namespace)
		}
		By("Deleting StorageClass")
		err = client.StorageV1beta1().StorageClasses().Delete(storageclass.Name, nil)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should create pod with many volumes and verify no attach call fails", func() {
		By(fmt.Sprintf("Running test with VOLUME_OPS_SCALE: %v", volume_ops_scale))
		By("Creating Storage Class")
		scParameters := make(map[string]string)
		scParameters["diskformat"] = "thin"
		storageclass, err = client.StorageV1beta1().StorageClasses().Create(getVSphereStorageClassSpec("thinsc", scParameters))
		Expect(err).NotTo(HaveOccurred())

		By("Creating PVCs using the Storage Class")
		count := 0
		for count < volume_ops_scale {
			pvclaims[count] = framework.CreatePVC(client, namespace, getVSphereClaimSpecWithStorageClassAnnotation(namespace, storageclass))
			count++
		}

		By("Waiting for all claims to be in bound phase")
		persistentvolumes = waitForPVClaimBoundPhase(client, pvclaims)

		By("Creating pod to attach PVs to the node")
		pod := createPod(client, namespace, pvclaims)

		By("Verify all volumes are accessible and available in the pod")
		verifyVolumesAccessible(pod, persistentvolumes, vsp)

		By("Deleting pod")
		framework.DeletePodWithWait(f, client, pod)

		By("Waiting for volumes to be detached from the node")
		for _, pv := range persistentvolumes {
			waitForVSphereDiskToDetach(vsp, pv.Spec.VsphereVolume.VolumePath, k8stype.NodeName(pod.Spec.NodeName))
		}
	})
})

// create pod with given claims
func createPod(client clientset.Interface, namespace string, pvclaims []*v1.PersistentVolumeClaim) *v1.Pod {
	podSpec := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pod-many-volumes-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "volume-tester",
					Image:   "gcr.io/google_containers/busybox:1.24",
					Command: []string{"/bin/sh"},
					Args:    []string{"-c", "while true ; do sleep 2 ; done"},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
	var volumeMounts = make([]v1.VolumeMount, len(pvclaims))
	var volumes = make([]v1.Volume, len(pvclaims))
	for index, pvclaim := range pvclaims {
		volumename := fmt.Sprintf("volume%v", index+1)
		volumeMounts[index] = v1.VolumeMount{Name: volumename, MountPath: "/mnt/" + volumename}
		volumes[index] = v1.Volume{Name: volumename, VolumeSource: v1.VolumeSource{PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{ClaimName: pvclaim.Name, ReadOnly: false}}}
	}
	podSpec.Spec.Containers[0].VolumeMounts = volumeMounts
	podSpec.Spec.Volumes = volumes

	pod, err := client.CoreV1().Pods(namespace).Create(podSpec)
	Expect(err).NotTo(HaveOccurred())

	// Waiting for pod to be running
	Expect(framework.WaitForPodNameRunningInNamespace(client, pod.Name, namespace)).To(Succeed())

	// get fresh pod info
	pod, err = client.CoreV1().Pods(namespace).Get(pod.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	return pod
}

// verify volumes are attached to the node and are accessible in pod
func verifyVolumesAccessible(pod *v1.Pod, persistentvolumes []*v1.PersistentVolume, vsp *vsphere.VSphere) {
	nodeName := pod.Spec.NodeName
	namespace := pod.Namespace
	for index, pv := range persistentvolumes {
		// Verify disks are attached to the node
		isAttached, err := verifyVSphereDiskAttached(vsp, pv.Spec.VsphereVolume.VolumePath, k8stype.NodeName(nodeName))
		Expect(err).NotTo(HaveOccurred())
		Expect(isAttached).To(BeTrue(), fmt.Sprintf("disk %v is not attached with the node", pv.Spec.VsphereVolume.VolumePath))
		// Verify Volumes are accessible
		filepath := filepath.Join("/mnt/", fmt.Sprintf("volume%v", index+1), "/emptyFile.txt")
		_, err = framework.LookForStringInPodExec(namespace, pod.Name, []string{"/bin/touch", filepath}, "", time.Minute)
		Expect(err).NotTo(HaveOccurred())
	}
}

// wait until all pvcs phase set to bound
func waitForPVClaimBoundPhase(client clientset.Interface, pvclaims []*v1.PersistentVolumeClaim) []*v1.PersistentVolume {
	var persistentvolumes = make([]*v1.PersistentVolume, len(pvclaims))
	for index, claim := range pvclaims {
		err := framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, client, claim.Namespace, claim.Name, framework.Poll, framework.ClaimProvisionTimeout)
		Expect(err).NotTo(HaveOccurred())
		// Get new copy of the claim
		claim, err := client.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(claim.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())

		// Get the bounded PV
		persistentvolumes[index], err = client.CoreV1().PersistentVolumes().Get(claim.Spec.VolumeName, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
	}
	return persistentvolumes
}
