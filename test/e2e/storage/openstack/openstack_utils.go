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

package openstack

import (
	"fmt"
	"path/filepath"
	"time"

	. "github.com/onsi/gomega"

	"k8s.io/apimachinery/pkg/util/rand"
	clientset "k8s.io/client-go/kubernetes"
	openstack "k8s.io/kubernetes/pkg/cloudprovider/providers/openstack"

	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	k8stype "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

const (
	volumesPerNode = 55
	storageclass1  = "sc-default"
	storageclass2  = "sc-vsan"
	storageclass3  = "sc-spbm"
	storageclass4  = "sc-user-specified-ds"
)

// volumeState represents the state of a volume.
type volumeState int32

const (
	volumeStateDetached volumeState = 1
	volumeStateAttached volumeState = 2
)

// volume constants.
const (
	VmfsDatastore                              = "sharedVmfs-0"
	VsanDatastore                              = "vsanDatastore"
	Datastore                                  = "datastore"
	PolicyDiskStripes                          = "diskStripes"
	PolicyHostFailuresToTolerate               = "hostFailuresToTolerate"
	PolicyCacheReservation                     = "cacheReservation"
	PolicyObjectSpaceReservation               = "objectSpaceReservation"
	PolicyIopsLimit                            = "iopsLimit"
	DiskFormat                                 = "diskformat"
	ThinDisk                                   = "thin"
	SpbmStoragePolicy                          = "storagepolicyname"
	BronzeStoragePolicy                        = "bronze"
	HostFailuresToTolerateCapabilityVal        = "0"
	CacheReservationCapabilityVal              = "20"
	DiskStripesCapabilityVal                   = "1"
	ObjectSpaceReservationCapabilityVal        = "30"
	IopsLimitCapabilityVal                     = "100"
	StripeWidthCapabilityVal                   = "2"
	DiskStripesCapabilityInvalidVal            = "14"
	HostFailuresToTolerateCapabilityInvalidVal = "4"
	DummyVMPrefixName                          = "openstack-k8s"
	DiskStripesCapabilityMaxVal                = "11"
)

// volume status constants
const (
	VolumeAvailableStatus = "available"
	volumeInUseStatus     = "in-use"
	testClusterName       = "testCluster"

	volumeStatusTimeoutSeconds = 30
	// volumeStatus* is configuration of exponential backoff for
	// waiting for specified volume status. Starting with 1
	// seconds, multiplying by 1.2 with each step and taking 13 steps at maximum
	// it will time out after 32s, which roughly corresponds to 30s
	volumeStatusInitDealy = 1 * time.Second
	volumeStatusFactor    = 1.2
	volumeStatusSteps     = 13
)

func verifyOpenstackDiskAttached(c clientset.Interface, os *openstack.OpenStack, instanceID string, volumeID string, nodeName types.NodeName) (bool, error) {
	var (
		isAttached bool
		err        error
	)

	isAttached, err = os.DiskIsAttached(instanceID, volumeID)
	return isAttached, err
}

// Wait until openstack volumes are detached from the list of nodes or time out after 5 minutes
func waitForOpenstackDisksToDetach(c clientset.Interface, instanceId string, osp *openstack.OpenStack, nodeVolumes map[k8stype.NodeName][]string) error {
	var (
		err            error
		disksAttached  = true
		detachTimeout  = 5 * time.Minute
		detachPollTime = 10 * time.Second
	)
	if osp == nil {
		osp, _, err = getOpenstack(c)
		if err != nil {
			return err
		}
	}
	err = wait.Poll(detachPollTime, detachTimeout, func() (bool, error) {
		for _, volumeids := range nodeVolumes {
			attachedResult, err := osp.DisksAreAttached(instanceId, volumeids)
			if err != nil {
				return false, err
			}
			for volumeID, attached := range attachedResult {
				if attached {
					framework.Logf("Waiting for volumes %q to detach from %v.", volumeID, attached)
					return false, nil
				}
			}
		}

		disksAttached = false
		framework.Logf("Volume are successfully detached from all the nodes: %+v", nodeVolumes)
		return true, nil
	})
	if err != nil {
		return err
	}
	if disksAttached {
		return fmt.Errorf("Gave up waiting for volumes to detach after %v", detachTimeout)
	}
	return nil
}

func createOpenstackVolume(os *openstack.OpenStack) (string, error) {

	tags := map[string]string{
		"test": "value",
	}
	vol, _, _, err := os.CreateVolume("kubernetes-test-volume-"+rand.String(10), 1, "", "", &tags)

	return vol, err
}

func getTestStorageClasses(client clientset.Interface, policyName, datastoreName string) []*storage.StorageClass {
	const (
		storageclass1 = "sc-default"
		storageclass2 = "sc-vsan"
		storageclass3 = "sc-spbm"
		storageclass4 = "sc-user-specified-ds"
	)
	scNames := []string{storageclass1, storageclass2, storageclass3, storageclass4}
	scArrays := make([]*storage.StorageClass, len(scNames))
	for index, scname := range scNames {
		// Create openstack Storage Class
		var sc *storage.StorageClass
		var err error
		switch scname {
		case storageclass1:
			sc, err = client.StorageV1().StorageClasses().Create(getOpenstackStorageClassSpec(storageclass1, nil))
		case storageclass2:
			var scVSanParameters map[string]string
			scVSanParameters = make(map[string]string)
			scVSanParameters[PolicyHostFailuresToTolerate] = "1"
			sc, err = client.StorageV1().StorageClasses().Create(getOpenstackStorageClassSpec(storageclass2, scVSanParameters))
		case storageclass3:
			var scSPBMPolicyParameters map[string]string
			scSPBMPolicyParameters = make(map[string]string)
			scSPBMPolicyParameters[SpbmStoragePolicy] = policyName
			sc, err = client.StorageV1().StorageClasses().Create(getOpenstackStorageClassSpec(storageclass3, scSPBMPolicyParameters))
		case storageclass4:
			var scWithDSParameters map[string]string
			scWithDSParameters = make(map[string]string)
			scWithDSParameters[Datastore] = datastoreName
			scWithDatastoreSpec := getOpenstackStorageClassSpec(storageclass4, scWithDSParameters)
			sc, err = client.StorageV1().StorageClasses().Create(scWithDatastoreSpec)
		}
		Expect(sc).NotTo(BeNil())
		Expect(err).NotTo(HaveOccurred())
		scArrays[index] = sc
	}
	return scArrays
}

func getOpenstackStorageClassSpec(name string, scParameters map[string]string) *storage.StorageClass {
	var sc *storage.StorageClass

	sc = &storage.StorageClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Provisioner: "kubernetes.io/openstack-volume",
	}
	if scParameters != nil {
		sc.Parameters = scParameters
	}
	return sc
}

func getOpenstackClaimSpecWithStorageClassAnnotation(ns string, diskSize string, storageclass *storage.StorageClass) *v1.PersistentVolumeClaim {
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

func getOpenstack(client clientset.Interface) (*openstack.OpenStack, string, error) {
	cfg, _ := openstack.ConfigFromEnv()
	osp, err := openstack.NewOpenStack(cfg)

	id, err := osp.InstanceID()
	if err != nil {
		return nil, id, err
	}
	return osp, id, err
}

// Get openstack Volume ID from PVC
func getopenStackVolumeIDFromClaim(client clientset.Interface, namespace string, claimName string) string {
	pvclaim, err := client.CoreV1().PersistentVolumeClaims(namespace).Get(claimName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	pv, err := client.CoreV1().PersistentVolumes().Get(pvclaim.Spec.VolumeName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	return pv.Spec.Cinder.VolumeID
}

func getOpenstackPodSpecWithVolumeIDs(volumeIDs []string, keyValuelabel map[string]string, commands []string) *v1.Pod {
	var volumeMounts []v1.VolumeMount
	var volumes []v1.Volume

	for index, VolumeID := range volumeIDs {
		name := fmt.Sprintf("volume%v", index+1)
		volumeMounts = append(volumeMounts, v1.VolumeMount{Name: name, MountPath: "/mnt/" + name})
		openstackVolume := new(v1.CinderVolumeSource)
		openstackVolume.VolumeID = VolumeID
		openstackVolume.FSType = "ext4"
		volumes = append(volumes, v1.Volume{Name: name})
		volumes[index].VolumeSource.Cinder = openstackVolume
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
			GenerateName: "openstack-e2e-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:         "openstack-e2e-container-" + string(uuid.NewUUID()),
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

// func to wait for volume status
func WaitForVolumeStatus(os *openstack.OpenStack, volumeName string, status string) {
	backoff := wait.Backoff{
		Duration: volumeStatusInitDealy,
		Factor:   volumeStatusFactor,
		Steps:    volumeStatusSteps,
	}
	err := wait.ExponentialBackoff(backoff, func() (bool, error) {
		getVol, err := os.GetVolume(volumeName)
		if err != nil {
			return false, err
		}
		if getVol.Status == status {
			fmt.Printf("Volume (%s) status changed to %s after %v seconds\n",
				volumeName,
				status,
				volumeStatusTimeoutSeconds)
			return true, nil
		} else {
			return false, nil
		}
	})
	if err == wait.ErrWaitTimeout {
		fmt.Printf("Volume (%s) status did not change to %s after %v seconds\n",
			volumeName,
			status,
			volumeStatusTimeoutSeconds)
		return
	}
	if err != nil {
		fmt.Printf("Cannot get existing Cinder volume (%s): %v", volumeName, err)
		return
	}
}

// func to verify openstack volume accesibility
func verifyOpenstackVolumesAccessible(c clientset.Interface, pod *v1.Pod, persistentvolumes []*v1.PersistentVolume, instanceID string, os *openstack.OpenStack) {
	nodeName := pod.Spec.NodeName
	namespace := pod.Namespace
	for index, pv := range persistentvolumes {
		isAttached, err := verifyOpenstackDiskAttached(c, os, instanceID, pv.Spec.Cinder.VolumeID, k8stype.NodeName(nodeName))
		Expect(err).NotTo(HaveOccurred())
		Expect(isAttached).To(BeTrue(), fmt.Sprintf("disk %v is not attached with the node", pv.Spec.Cinder.VolumeID))
		filepath := filepath.Join("/mnt/", fmt.Sprintf("volume%v", index+1), "/emptyFile.txt")
		_, err = framework.LookForStringInPodExec(namespace, pod.Name, []string{"/bin/touch", filepath}, "", time.Minute)
		Expect(err).NotTo(HaveOccurred())
	}
}

// func to get pod spec with given volume claim, node selector labels and command
func getOpenstackPodSpecWithClaim(claimName string, nodeSelectorKV map[string]string, command string) *v1.Pod {
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

// function to get openstack persistent volume spec with given selector labels.
func getOpenstackPersistentVolumeClaimSpec(namespace string, labels map[string]string) *v1.PersistentVolumeClaim {
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

// function to create openstack volume spec with given VMDK volume path, Reclaim Policy and labels
func getOpenstackPersistentVolumeSpec(volumeID string, persistentVolumeReclaimPolicy v1.PersistentVolumeReclaimPolicy, labels map[string]string) *v1.PersistentVolume {
	var (
		pvConfig framework.PersistentVolumeConfig
		pv       *v1.PersistentVolume
		claimRef *v1.ObjectReference
	)
	pvConfig = framework.PersistentVolumeConfig{
		NamePrefix: "openstackpv-",
		PVSource: v1.PersistentVolumeSource{
			Cinder: &v1.CinderVolumeSource{
				VolumeID: volumeID,
				FSType:   "ext4",
			},
		},
		Prebind: nil,
	}

	pv = &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: pvConfig.NamePrefix,
			Annotations: map[string]string{
				util.VolumeGidAnnotationKey: "777",
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

func waitForOpenstackDiskStatus(c clientset.Interface, instanceID string, osp *openstack.OpenStack, volumeID string, nodeName types.NodeName, expectedState volumeState) error {
	var (
		err          error
		diskAttached bool
		currentState volumeState
		timeout      = 6 * time.Minute
		pollTime     = 10 * time.Second
	)

	var attachedState = map[bool]volumeState{
		true:  volumeStateAttached,
		false: volumeStateDetached,
	}

	var attachedStateMsg = map[volumeState]string{
		volumeStateAttached: "attached to",
		volumeStateDetached: "detached from",
	}

	err = wait.Poll(pollTime, timeout, func() (bool, error) {
		diskAttached, err = verifyOpenstackDiskAttached(c, osp, instanceID, volumeID, nodeName)
		if err != nil {
			return true, err
		}

		currentState = attachedState[diskAttached]
		if currentState == expectedState {
			framework.Logf("Volume %q has successfully %s %q", volumeID, attachedStateMsg[currentState], nodeName)
			return true, nil
		}
		framework.Logf("Waiting for Volume %q to be %s %q.", volumeID, attachedStateMsg[expectedState], nodeName)
		return false, nil
	})
	if err != nil {
		return err
	}

	if currentState != expectedState {
		err = fmt.Errorf("Gave up waiting for Volume %q to be %s %q after %v", volumeID, attachedStateMsg[expectedState], nodeName, timeout)
	}
	return err
}

// Wait until openstack vmdk is attached from the given node or time out after 6 minutes
func waitForOpenstackDiskToAttach(c clientset.Interface, instanceID string, osp *openstack.OpenStack, volumeID string, nodeName types.NodeName) error {
	return waitForOpenstackDiskStatus(c, instanceID, osp, volumeID, nodeName, volumeStateAttached)
}

func waitForOpenstackDiskToDetach(c clientset.Interface, instanceID string, osp *openstack.OpenStack, volumeID string, nodeName types.NodeName) error {
	return waitForOpenstackDiskStatus(c, instanceID, osp, volumeID, nodeName, volumeStateDetached)
}

func getOSTestStorageClasses(client clientset.Interface, policyName, datastoreName string) []*storage.StorageClass {
	const (
		storageclass1 = "sc-default"
		storageclass2 = "sc-osan"
		storageclass3 = "sc-spbm"
		storageclass4 = "sc-user-specified-ds"
	)
	scNames := []string{storageclass1, storageclass2, storageclass3, storageclass4}
	scArrays := make([]*storage.StorageClass, len(scNames))
	for index, scname := range scNames {
		// Create openstack Storage Class
		var sc *storage.StorageClass
		var err error
		switch scname {
		case storageclass1:
			sc, err = client.StorageV1().StorageClasses().Create(getOpenstackStorageClassSpec(storageclass1, nil))
		case storageclass2:
			var scOSanParameters map[string]string
			scOSanParameters = make(map[string]string)
			scOSanParameters[PolicyHostFailuresToTolerate] = "1"
			sc, err = client.StorageV1().StorageClasses().Create(getOpenstackStorageClassSpec(storageclass2, scOSanParameters))
		case storageclass3:
			var scSPBMPolicyParameters map[string]string
			scSPBMPolicyParameters = make(map[string]string)
			scSPBMPolicyParameters[SpbmStoragePolicy] = policyName
			sc, err = client.StorageV1().StorageClasses().Create(getOpenstackStorageClassSpec(storageclass3, scSPBMPolicyParameters))
		case storageclass4:
			var scWithDSParameters map[string]string
			scWithDSParameters = make(map[string]string)
			scWithDSParameters[Datastore] = datastoreName
			scWithDatastoreSpec := getOpenstackStorageClassSpec(storageclass4, scWithDSParameters)
			sc, err = client.StorageV1().StorageClasses().Create(scWithDatastoreSpec)
		}
		Expect(sc).NotTo(BeNil())
		Expect(err).NotTo(HaveOccurred())
		scArrays[index] = sc
	}
	return scArrays
}

// function to write content to the volume backed by given PVC
func writeContentToOpenstackPV(client clientset.Interface, pvc *v1.PersistentVolumeClaim, expectedContent string) {
	utils.RunInPodWithVolume(client, pvc.Namespace, pvc.Name, "echo "+expectedContent+" > /mnt/test/data")
	framework.Logf("Done with writing content to volume")
}

func verifyContentOfOpenstackPV(client clientset.Interface, pvc *v1.PersistentVolumeClaim, expectedContent string) {
	utils.RunInPodWithVolume(client, pvc.Namespace, pvc.Name, "grep '"+expectedContent+"' /mnt/test/data")
	framework.Logf("Successfully verified content of the volume")
}

func verifyFilesExistOnOpenstackVolume(namespace string, podName string, filePaths []string) {
	for _, filePath := range filePaths {
		_, err := framework.RunKubectl("exec", fmt.Sprintf("--namespace=%s", namespace), podName, "--", "/bin/ls", filePath)
		Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to verify file: %q on the pod: %q", filePath, podName))
	}
}

func createEmptyFilesOnOpenstackVolume(namespace string, podName string, filePaths []string) {
	for _, filePath := range filePaths {
		err := framework.CreateEmptyFileOnPod(namespace, podName, filePath)
		Expect(err).NotTo(HaveOccurred())
	}
}
