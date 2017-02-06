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
	"strconv"
	"time"

	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	vsphere "k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
	"k8s.io/kubernetes/test/e2e/framework"
)

// Sanity check for vSphere testing.  Verify the persistent disk attached to the node.
func verifyVSphereDiskAttached(vsp *vsphere.VSphere, volumePath string, nodeName types.NodeName) (isAttached bool, err error) {
	if vsp == nil {
		vsp, err = vsphere.GetVSphere()
		Expect(err).NotTo(HaveOccurred())
	}
	isAttached, err = vsp.DiskIsAttached(volumePath, nodeName)
	Expect(err).NotTo(HaveOccurred())
	return isAttached, err
}

// Wait until vsphere vmdk is deteched from the given node or time out after 5 minutes
func waitForVSphereDiskToDetach(vsp *vsphere.VSphere, volumePath string, nodeName types.NodeName) {
	var err error
	if vsp == nil {
		vsp, err = vsphere.GetVSphere()
		Expect(err).NotTo(HaveOccurred())
	}
	var diskAttached = true
	var detachTimeout = 5 * time.Minute
	var detachPollTime = 10 * time.Second
	for start := time.Now(); time.Since(start) < detachTimeout; time.Sleep(detachPollTime) {
		diskAttached, err = verifyVSphereDiskAttached(vsp, volumePath, nodeName)
		Expect(err).NotTo(HaveOccurred())
		if !diskAttached {
			// Specified disk does not appear to be attached to specified node
			framework.Logf("Volume %q appears to have successfully detached from %q.",
				volumePath, nodeName)
			break
		}
		framework.Logf("Waiting for Volume %q to detach from %q.", volumePath, nodeName)
	}
	if diskAttached {
		_ = fmt.Errorf("Gave up waiting for Volume %q to detach from %q after %v",
			volumePath, nodeName, detachTimeout)
	}
}

// function to create vsphere volume spec with given VMDK volume path, Reclaim Policy and labels
func getVSpherePersistentVolumeSpec(volumePath string, persistentVolumeReclaimPolicy v1.PersistentVolumeReclaimPolicy, labels map[string]string) *v1.PersistentVolume {
	var (
		pvConfig persistentVolumeConfig
		pv       *v1.PersistentVolume
		claimRef *v1.ObjectReference
	)
	pvConfig = persistentVolumeConfig{
		namePrefix: "vspherepv-",
		pvSource: v1.PersistentVolumeSource{
			VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
				VolumePath: volumePath,
				FSType:     "ext4",
			},
		},
		prebind: nil,
	}

	pv = &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: pvConfig.namePrefix,
			Annotations: map[string]string{
				volumehelper.VolumeGidAnnotationKey: "777",
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: persistentVolumeReclaimPolicy,
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("2Gi"),
			},
			PersistentVolumeSource: pvConfig.pvSource,
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
func createVSphereVolume(vsp *vsphere.VSphere, volumeOptions *vsphere.VolumeOptions) (volumePath string, err error) {
	if volumeOptions == nil {
		volumeOptions = new(vsphere.VolumeOptions)
		volumeOptions.CapacityKB = 2097152
		volumeOptions.Name = "e2e-vmdk-" + strconv.FormatInt(time.Now().UnixNano(), 10)
	}
	volumePath, err = vsp.CreateVolume(volumeOptions)
	Expect(err).NotTo(HaveOccurred())
	return volumePath, nil
}

// function to write content to the volume backed by given PVC
func writeContentToVSpherePV(client clientset.Interface, ns string, pvc *v1.PersistentVolumeClaim, content string) {
	podClient := client.CoreV1().Pods(ns)
	name := "writerpod" + strconv.FormatInt(time.Now().UnixNano(), 10)
	framework.Logf("Creating POD: %s to write content to volume", name)

	writerPod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    name,
					Image:   "gcr.io/google_containers/busybox:1.24",
					Command: []string{"/bin/sh"},
					Args:    []string{"-c", "echo '" + content + "' > /mnt/file.txt && chmod o+rX /mnt /mnt/file.txt"},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      pvc.Name,
							MountPath: "/mnt",
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name: pvc.Name,
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: pvc.Name,
						},
					},
				},
			},
		},
	}

	defer func() {
		podClient.Delete(name, nil)
	}()

	writerPod, err := podClient.Create(writerPod)
	framework.ExpectNoError(err, "Failed to create write pod: %v", err)
	err = framework.WaitForPodSuccessInNamespace(client, writerPod.Name, writerPod.Namespace)
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("Done with writing content to volume")
}

// function to verify content is matching on the volume backed for given PVC
func verifyContentOfVSpherePV(client clientset.Interface, ns string, pvc *v1.PersistentVolumeClaim, expectedContent string) {
	framework.Logf("Creating POD to read content of the volume")
	podClient := client.CoreV1().Pods(ns)
	name := "readerpod" + strconv.FormatInt(time.Now().UnixNano(), 10)
	readerPod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  name,
					Image: "gcr.io/google_containers/busybox:1.24",
					Command: []string{
						"/bin/sh",
						"-c",
						"while true ; do cat /mnt/file.txt ; sleep 2 ; done ",
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      pvc.Name,
							MountPath: "/mnt",
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name: pvc.Name,
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: pvc.Name,
						},
					},
				},
			},
		},
	}

	defer func() {
		podClient.Delete(name, nil)
	}()

	readerPod, err := podClient.Create(readerPod)
	framework.ExpectNoError(err, "Failed to create reader pod: %v", err)
	framework.ExpectNoError(framework.WaitForPodRunningInNamespace(client, readerPod))

	_, err = framework.LookForStringInPodExec(ns, readerPod.Name, []string{"cat", "/mnt/file.txt"}, expectedContent, time.Minute)
	Expect(err).NotTo(HaveOccurred(), "failed to find expected content in the volume")
	framework.Logf("Sucessfully verified content of the volume")
}
