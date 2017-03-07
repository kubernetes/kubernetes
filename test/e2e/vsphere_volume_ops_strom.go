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
    	3. Wait until all disks are ready and all PVs and PVCs get bind. (CreateVolume strom)
    	4. Create pod to mount volumes using PVCs created in step 2. (AttachDisk strom)
    	5. Wait for pod status to be running.
    	6. Verify all volumes accessible and available in the pod.
    	7. Delete pod.
    	8. wait until volumes gets detached. (DetachDisk strom)
    	9. Delete all PVCs. This should delete all Disks. (DeleteVolume strom)
	10. Delete storage class.
*/

var _ = framework.KubeDescribe("volume operations strom [Volume]", func() {
	f := framework.NewDefaultFramework("volume-ops-strom")
	var (
		client            clientset.Interface
		namespace         string
		storageclass      *storage.StorageClass
		pvclaims          []*v1.PersistentVolumeClaim
		persistentvolumes []*v1.PersistentVolume
		err               error
		volume_ops_scale  int
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
			volume_ops_scale = 30
		}
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

	It("should create pod with many volumes and verify none attach call fails", func() {
		By(fmt.Sprintf("Running test with VOLUME_OPS_SCALE: %v", volume_ops_scale))
		By("Creating Storage Class")
		scParameters := make(map[string]string)
		scParameters["diskformat"] = "thin"
		storageClassSpec := getVSphereStorageClassSpec("thinsc", scParameters)
		storageclass, err = client.StorageV1beta1().StorageClasses().Create(storageClassSpec)
		Expect(err).NotTo(HaveOccurred())

		By("Creating PVCs using the Storage Class")
		count := 0
		for count < volume_ops_scale {
			pvclaimSpec := getVSphereClaimSpecWithStorageClassAnnotation(namespace, storageclass)
			claim, err := client.CoreV1().PersistentVolumeClaims(namespace).Create(pvclaimSpec)
			Expect(err).NotTo(HaveOccurred())
			pvclaims = append(pvclaims, claim)
			count++
		}

		By("Waiting for all claims to be in bound phase")
		for _, claim := range pvclaims {
			err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, client, namespace, claim.Name, framework.Poll, framework.ClaimProvisionTimeout)
			Expect(err).NotTo(HaveOccurred())
			// Get new copy of the claim
			claim, err := client.CoreV1().PersistentVolumeClaims(namespace).Get(claim.Name, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred())

			// Get the bounded PV
			persistentvolume, err := client.CoreV1().PersistentVolumes().Get(claim.Spec.VolumeName, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred())
			persistentvolumes = append(persistentvolumes, persistentvolume)
		}

		By("Creating pod to attach PV to the node")
		podSpec := &v1.Pod{
			TypeMeta: metav1.TypeMeta{
				Kind:       "Pod",
				APIVersion: "v1",
			},
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "pod-30-volumes-",
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
		var volumeMounts []v1.VolumeMount
		var volumes []v1.Volume
		for index, pvclaims := range pvclaims {
			volumename := fmt.Sprintf("volume%v", index+1)
			volumeMounts = append(volumeMounts, v1.VolumeMount{Name: volumename, MountPath: "/mnt/" + volumename})
			volumes = append(volumes, v1.Volume{Name: volumename, VolumeSource: v1.VolumeSource{PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{ClaimName: pvclaims.Name, ReadOnly: false}}})
		}
		podSpec.Spec.Containers[0].VolumeMounts = volumeMounts
		podSpec.Spec.Volumes = volumes

		pod, err := client.CoreV1().Pods(namespace).Create(podSpec)
		Expect(err).NotTo(HaveOccurred())

		By("Waiting for pod to be running")
		Expect(framework.WaitForPodNameRunningInNamespace(client, pod.Name, namespace)).To(Succeed())

		// get fresh pod info
		pod, err = client.CoreV1().Pods(namespace).Get(pod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())

		nodeName := pod.Spec.NodeName
		vsp, err := vsphere.GetVSphere()
		Expect(err).NotTo(HaveOccurred())

		By("Verify all volumes are accessible and available in the pod")
		for index, pv := range persistentvolumes {
			// Verify disks are attached to the node
			isAttached, err := verifyVSphereDiskAttached(vsp, pv.Spec.VsphereVolume.VolumePath, k8stype.NodeName(nodeName))
			Expect(err).NotTo(HaveOccurred())
			Expect(isAttached).To(BeTrue(), fmt.Sprintf("disk %v is not attached with the node", pv.Spec.VsphereVolume.VolumePath))
			// Verify Volumes are accessible
			filePath := "/mnt/" + fmt.Sprintf("volume%v", index+1) + "/emptyFile.txt"
			_, err = framework.LookForStringInPodExec(namespace, pod.Name, []string{"/bin/touch", filePath}, "", time.Minute)
			Expect(err).NotTo(HaveOccurred())
		}

		By("Deleting pod")
		err = client.CoreV1().Pods(namespace).Delete(pod.Name, nil)
		Expect(err).NotTo(HaveOccurred())

		By("Waiting for volumes to be detached from the node")
		for _, pv := range persistentvolumes {
			By(fmt.Sprintf("Waiting for volume:%v to be detached from node:%v", pv.Spec.VsphereVolume.VolumePath, nodeName))
			waitForVSphereDiskToDetach(vsp, pv.Spec.VsphereVolume.VolumePath, k8stype.NodeName(nodeName))
		}
	})
})
