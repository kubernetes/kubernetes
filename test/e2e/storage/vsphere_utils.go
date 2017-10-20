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

package storage

import (
	"fmt"
	"path/filepath"
	"strconv"
	"time"

	. "github.com/onsi/gomega"
	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	k8stype "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	vsphere "k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere/vclib"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
	"k8s.io/kubernetes/test/e2e/framework"
)

// Sanity check for vSphere testing.  Verify the persistent disk attached to the node.
func verifyVSphereDiskAttached(vsp *vsphere.VSphere, volumePath string, nodeName types.NodeName) (bool, error) {
	var (
		isAttached bool
		err        error
	)
	if vsp == nil {
		vsp, err = vsphere.GetVSphere()
		Expect(err).NotTo(HaveOccurred())
	}
	isAttached, err = vsp.DiskIsAttached(volumePath, nodeName)
	Expect(err).NotTo(HaveOccurred())
	return isAttached, err
}

// Wait until vsphere vmdk is deteched from the given node or time out after 5 minutes
func waitForVSphereDiskToDetach(vsp *vsphere.VSphere, volumePath string, nodeName types.NodeName) error {
	var (
		err            error
		diskAttached   = true
		detachTimeout  = 5 * time.Minute
		detachPollTime = 10 * time.Second
	)
	if vsp == nil {
		vsp, err = vsphere.GetVSphere()
		if err != nil {
			return err
		}
	}
	err = wait.Poll(detachPollTime, detachTimeout, func() (bool, error) {
		diskAttached, err = verifyVSphereDiskAttached(vsp, volumePath, nodeName)
		if err != nil {
			return true, err
		}
		if !diskAttached {
			framework.Logf("Volume %q appears to have successfully detached from %q.",
				volumePath, nodeName)
			return true, nil
		}
		framework.Logf("Waiting for Volume %q to detach from %q.", volumePath, nodeName)
		return false, nil
	})
	if err != nil {
		return err
	}
	if diskAttached {
		return fmt.Errorf("Gave up waiting for Volume %q to detach from %q after %v", volumePath, nodeName, detachTimeout)
	}
	return nil
}

// function to create vsphere volume spec with given VMDK volume path, Reclaim Policy and labels
func getVSpherePersistentVolumeSpec(volumePath string, persistentVolumeReclaimPolicy v1.PersistentVolumeReclaimPolicy, labels map[string]string) *v1.PersistentVolume {
	var (
		pvConfig framework.PersistentVolumeConfig
		pv       *v1.PersistentVolume
		claimRef *v1.ObjectReference
	)
	pvConfig = framework.PersistentVolumeConfig{
		NamePrefix: "vspherepv-",
		PVSource: v1.PersistentVolumeSource{
			VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
				VolumePath: volumePath,
				FSType:     "ext4",
			},
		},
		Prebind: nil,
	}

	pv = &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: pvConfig.NamePrefix,
			Annotations: map[string]string{
				volumehelper.VolumeGidAnnotationKey: "777",
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: persistentVolumeReclaimPolicy,
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("2Gi"),
			},
			PersistentVolumeSource: pvConfig.PVSource,
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			ClaimRef: claimRef,
		},
	}
	if labels != nil {
		pv.Labels = labels
	}
	return pv
}

// function to get vsphere persistent volume spec with given selector labels.
func getVSpherePersistentVolumeClaimSpec(namespace string, labels map[string]string) *v1.PersistentVolumeClaim {
	var (
		pvc *v1.PersistentVolumeClaim
	)
	pvc = &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pvc-",
			Namespace:    namespace,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("2Gi"),
				},
			},
		},
	}
	if labels != nil {
		pvc.Spec.Selector = &metav1.LabelSelector{MatchLabels: labels}
	}

	return pvc
}

// function to create vmdk volume
func createVSphereVolume(vsp *vsphere.VSphere, volumeOptions *vclib.VolumeOptions) (string, error) {
	var (
		volumePath string
		err        error
	)
	if volumeOptions == nil {
		volumeOptions = new(vclib.VolumeOptions)
		volumeOptions.CapacityKB = 2097152
		volumeOptions.Name = "e2e-vmdk-" + strconv.FormatInt(time.Now().UnixNano(), 10)
	}
	volumePath, err = vsp.CreateVolume(volumeOptions)
	Expect(err).NotTo(HaveOccurred())
	return volumePath, nil
}

// function to write content to the volume backed by given PVC
func writeContentToVSpherePV(client clientset.Interface, pvc *v1.PersistentVolumeClaim, expectedContent string) {
	runInPodWithVolume(client, pvc.Namespace, pvc.Name, "echo "+expectedContent+" > /mnt/test/data")
	framework.Logf("Done with writing content to volume")
}

// function to verify content is matching on the volume backed for given PVC
func verifyContentOfVSpherePV(client clientset.Interface, pvc *v1.PersistentVolumeClaim, expectedContent string) {
	runInPodWithVolume(client, pvc.Namespace, pvc.Name, "grep '"+expectedContent+"' /mnt/test/data")
	framework.Logf("Successfully verified content of the volume")
}

func getVSphereStorageClassSpec(name string, scParameters map[string]string) *storage.StorageClass {
	var sc *storage.StorageClass

	sc = &storage.StorageClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Provisioner: "kubernetes.io/vsphere-volume",
	}
	if scParameters != nil {
		sc.Parameters = scParameters
	}
	return sc
}

func getVSphereClaimSpecWithStorageClassAnnotation(ns string, diskSize string, storageclass *storage.StorageClass) *v1.PersistentVolumeClaim {
	scAnnotation := make(map[string]string)
	scAnnotation[v1.BetaStorageClassAnnotation] = storageclass.Name

	claim := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pvc-",
			Namespace:    ns,
			Annotations:  scAnnotation,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse(diskSize),
				},
			},
		},
	}
	return claim
}

// func to get pod spec with given volume claim, node selector labels and command
func getVSpherePodSpecWithClaim(claimName string, nodeSelectorKV map[string]string, command string) *v1.Pod {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pod-pvc-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "volume-tester",
					Image:   "busybox",
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
	if nodeSelectorKV != nil {
		pod.Spec.NodeSelector = nodeSelectorKV
	}
	return pod
}

// func to get pod spec with given volume paths, node selector lables and container commands
func getVSpherePodSpecWithVolumePaths(volumePaths []string, keyValuelabel map[string]string, commands []string) *v1.Pod {
	var volumeMounts []v1.VolumeMount
	var volumes []v1.Volume

	for index, volumePath := range volumePaths {
		name := fmt.Sprintf("volume%v", index+1)
		volumeMounts = append(volumeMounts, v1.VolumeMount{Name: name, MountPath: "/mnt/" + name})
		vsphereVolume := new(v1.VsphereVirtualDiskVolumeSource)
		vsphereVolume.VolumePath = volumePath
		vsphereVolume.FSType = "ext4"
		volumes = append(volumes, v1.Volume{Name: name})
		volumes[index].VolumeSource.VsphereVolume = vsphereVolume
	}

	if commands == nil || len(commands) == 0 {
		commands = []string{
			"/bin/sh",
			"-c",
			"while true; do sleep 2; done",
		}
	}
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "vsphere-e2e-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:         "vsphere-e2e-container-" + string(uuid.NewUUID()),
					Image:        "busybox",
					Command:      commands,
					VolumeMounts: volumeMounts,
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes:       volumes,
		},
	}

	if keyValuelabel != nil {
		pod.Spec.NodeSelector = keyValuelabel
	}
	return pod
}

func verifyFilesExistOnVSphereVolume(namespace string, podName string, filePaths []string) {
	for _, filePath := range filePaths {
		_, err := framework.RunKubectl("exec", fmt.Sprintf("--namespace=%s", namespace), podName, "--", "/bin/ls", filePath)
		Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to verify file: %q on the pod: %q", filePath, podName))
	}
}

func createEmptyFilesOnVSphereVolume(namespace string, podName string, filePaths []string) {
	for _, filePath := range filePaths {
		err := framework.CreateEmptyFileOnPod(namespace, podName, filePath)
		Expect(err).NotTo(HaveOccurred())
	}
}

// verify volumes are attached to the node and are accessible in pod
func verifyVSphereVolumesAccessible(pod *v1.Pod, persistentvolumes []*v1.PersistentVolume, vsp *vsphere.VSphere) {
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

// Get vSphere Volume Path from PVC
func getvSphereVolumePathFromClaim(client clientset.Interface, namespace string, claimName string) string {
	pvclaim, err := client.CoreV1().PersistentVolumeClaims(namespace).Get(claimName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	pv, err := client.CoreV1().PersistentVolumes().Get(pvclaim.Spec.VolumeName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	return pv.Spec.VsphereVolume.VolumePath
}
