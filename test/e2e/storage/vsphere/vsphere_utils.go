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

package vsphere

import (
	"context"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/mo"
	vim25types "github.com/vmware/govmomi/vim25/types"
	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	volumesPerNode = 55
	storageclass1  = "sc-default"
	storageclass2  = "sc-vsan"
	storageclass3  = "sc-spbm"
	storageclass4  = "sc-user-specified-ds"
	dummyDiskName  = "kube-dummyDisk.vmdk"
	providerPrefix = "vsphere://"
)

// volumeState represents the state of a volume.
type volumeState int32

const (
	volumeStateDetached volumeState = 1
	volumeStateAttached volumeState = 2
)

// Wait until vsphere volumes are detached from the list of nodes or time out after 5 minutes
func waitForVSphereDisksToDetach(nodeVolumes map[string][]string) error {
	var (
		detachTimeout  = 5 * time.Minute
		detachPollTime = 10 * time.Second
	)
	waitErr := wait.Poll(detachPollTime, detachTimeout, func() (bool, error) {
		attachedResult, err := disksAreAttached(nodeVolumes)
		if err != nil {
			return false, err
		}
		for nodeName, nodeVolumes := range attachedResult {
			for volumePath, attached := range nodeVolumes {
				if attached {
					framework.Logf("Volume %q is still attached to %q.", volumePath, string(nodeName))
					return false, nil
				}
			}
		}
		framework.Logf("Volume are successfully detached from all the nodes: %+v", nodeVolumes)
		return true, nil
	})
	if waitErr != nil {
		if waitErr == wait.ErrWaitTimeout {
			return fmt.Errorf("volumes have not detached after %v: %v", detachTimeout, waitErr)
		}
		return fmt.Errorf("error waiting for volumes to detach: %v", waitErr)
	}
	return nil
}

// Wait until vsphere vmdk moves to expected state on the given node, or time out after 6 minutes
func waitForVSphereDiskStatus(volumePath string, nodeName string, expectedState volumeState) error {
	var (
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

	waitErr := wait.Poll(pollTime, timeout, func() (bool, error) {
		diskAttached, err := diskIsAttached(volumePath, nodeName)
		if err != nil {
			return true, err
		}

		currentState = attachedState[diskAttached]
		if currentState == expectedState {
			framework.Logf("Volume %q has successfully %s %q", volumePath, attachedStateMsg[currentState], nodeName)
			return true, nil
		}
		framework.Logf("Waiting for Volume %q to be %s %q.", volumePath, attachedStateMsg[expectedState], nodeName)
		return false, nil
	})
	if waitErr != nil {
		if waitErr == wait.ErrWaitTimeout {
			return fmt.Errorf("volume %q is not %s %q after %v: %v", volumePath, attachedStateMsg[expectedState], nodeName, timeout, waitErr)
		}
		return fmt.Errorf("error waiting for volume %q to be %s %q: %v", volumePath, attachedStateMsg[expectedState], nodeName, waitErr)
	}
	return nil
}

// Wait until vsphere vmdk is attached from the given node or time out after 6 minutes
func waitForVSphereDiskToAttach(volumePath string, nodeName string) error {
	return waitForVSphereDiskStatus(volumePath, nodeName, volumeStateAttached)
}

// Wait until vsphere vmdk is detached from the given node or time out after 6 minutes
func waitForVSphereDiskToDetach(volumePath string, nodeName string) error {
	return waitForVSphereDiskStatus(volumePath, nodeName, volumeStateDetached)
}

// function to create vsphere volume spec with given VMDK volume path, Reclaim Policy and labels
func getVSpherePersistentVolumeSpec(volumePath string, persistentVolumeReclaimPolicy v1.PersistentVolumeReclaimPolicy, labels map[string]string) *v1.PersistentVolume {
	return e2epv.MakePersistentVolume(e2epv.PersistentVolumeConfig{
		NamePrefix: "vspherepv-",
		PVSource: v1.PersistentVolumeSource{
			VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
				VolumePath: volumePath,
				FSType:     "ext4",
			},
		},
		ReclaimPolicy: persistentVolumeReclaimPolicy,
		Capacity:      "2Gi",
		AccessModes: []v1.PersistentVolumeAccessMode{
			v1.ReadWriteOnce,
		},
		Labels: labels,
	})
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

// function to write content to the volume backed by given PVC
func writeContentToVSpherePV(client clientset.Interface, pvc *v1.PersistentVolumeClaim, expectedContent string) {
	utils.RunInPodWithVolume(client, pvc.Namespace, pvc.Name, "echo "+expectedContent+" > /mnt/test/data")
	framework.Logf("Done with writing content to volume")
}

// function to verify content is matching on the volume backed for given PVC
func verifyContentOfVSpherePV(client clientset.Interface, pvc *v1.PersistentVolumeClaim, expectedContent string) {
	utils.RunInPodWithVolume(client, pvc.Namespace, pvc.Name, "grep '"+expectedContent+"' /mnt/test/data")
	framework.Logf("Successfully verified content of the volume")
}

func getVSphereStorageClassSpec(name string, scParameters map[string]string, zones []string, volumeBindingMode storagev1.VolumeBindingMode) *storagev1.StorageClass {
	var sc *storagev1.StorageClass

	sc = &storagev1.StorageClass{
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
	if zones != nil {
		term := v1.TopologySelectorTerm{
			MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
				{
					Key:    v1.LabelFailureDomainBetaZone,
					Values: zones,
				},
			},
		}
		sc.AllowedTopologies = append(sc.AllowedTopologies, term)
	}
	if volumeBindingMode != "" {
		mode := storagev1.VolumeBindingMode(string(volumeBindingMode))
		sc.VolumeBindingMode = &mode
	}
	return sc
}

func getVSphereClaimSpecWithStorageClass(ns string, diskSize string, storageclass *storagev1.StorageClass) *v1.PersistentVolumeClaim {
	claim := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pvc-",
			Namespace:    ns,
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
			StorageClassName: &(storageclass.Name),
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
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
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
					Image:        imageutils.GetE2EImage(imageutils.BusyBox),
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

func verifyFilesExistOnVSphereVolume(namespace string, podName string, filePaths ...string) {
	for _, filePath := range filePaths {
		_, err := framework.RunKubectl(namespace, "exec", podName, "--", "/bin/ls", filePath)
		framework.ExpectNoError(err, fmt.Sprintf("failed to verify file: %q on the pod: %q", filePath, podName))
	}
}

func createEmptyFilesOnVSphereVolume(namespace string, podName string, filePaths []string) {
	for _, filePath := range filePaths {
		err := framework.CreateEmptyFileOnPod(namespace, podName, filePath)
		framework.ExpectNoError(err)
	}
}

// verify volumes are attached to the node and are accessible in pod
func verifyVSphereVolumesAccessible(c clientset.Interface, pod *v1.Pod, persistentvolumes []*v1.PersistentVolume) {
	nodeName := pod.Spec.NodeName
	namespace := pod.Namespace
	for index, pv := range persistentvolumes {
		// Verify disks are attached to the node
		isAttached, err := diskIsAttached(pv.Spec.VsphereVolume.VolumePath, nodeName)
		framework.ExpectNoError(err)
		framework.ExpectEqual(isAttached, true, fmt.Sprintf("disk %v is not attached with the node", pv.Spec.VsphereVolume.VolumePath))
		// Verify Volumes are accessible
		filepath := filepath.Join("/mnt/", fmt.Sprintf("volume%v", index+1), "/emptyFile.txt")
		_, err = framework.LookForStringInPodExec(namespace, pod.Name, []string{"/bin/touch", filepath}, "", time.Minute)
		framework.ExpectNoError(err)
	}
}

// verify volumes are created on one of the specified zones
func verifyVolumeCreationOnRightZone(persistentvolumes []*v1.PersistentVolume, nodeName string, zones []string) {
	for _, pv := range persistentvolumes {
		volumePath := pv.Spec.VsphereVolume.VolumePath
		// Extract datastoreName from the volume path in the pv spec
		// For example : "vsanDatastore" is extracted from "[vsanDatastore] 25d8b159-948c-4b73-e499-02001ad1b044/volume.vmdk"
		datastorePathObj, _ := getDatastorePathObjFromVMDiskPath(volumePath)
		datastoreName := datastorePathObj.Datastore
		nodeInfo := TestContext.NodeMapper.GetNodeInfo(nodeName)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		// Get the datastore object reference from the datastore name
		datastoreRef, err := nodeInfo.VSphere.GetDatastoreRefFromName(ctx, nodeInfo.DataCenterRef, datastoreName)
		if err != nil {
			framework.ExpectNoError(err)
		}
		// Find common datastores among the specified zones
		var datastoreCountMap = make(map[string]int)
		numZones := len(zones)
		var commonDatastores []string
		for _, zone := range zones {
			datastoreInZone := TestContext.NodeMapper.GetDatastoresInZone(nodeInfo.VSphere.Config.Hostname, zone)
			for _, datastore := range datastoreInZone {
				datastoreCountMap[datastore] = datastoreCountMap[datastore] + 1
				if datastoreCountMap[datastore] == numZones {
					commonDatastores = append(commonDatastores, datastore)
				}
			}
		}
		gomega.Expect(commonDatastores).To(gomega.ContainElement(datastoreRef.Value), "PV was created in an unsupported zone.")
	}
}

// Get vSphere Volume Path from PVC
func getvSphereVolumePathFromClaim(client clientset.Interface, namespace string, claimName string) string {
	pvclaim, err := client.CoreV1().PersistentVolumeClaims(namespace).Get(context.TODO(), claimName, metav1.GetOptions{})
	framework.ExpectNoError(err)
	pv, err := client.CoreV1().PersistentVolumes().Get(context.TODO(), pvclaim.Spec.VolumeName, metav1.GetOptions{})
	framework.ExpectNoError(err)
	return pv.Spec.VsphereVolume.VolumePath
}

// Get canonical volume path for volume Path.
// Example1: The canonical path for volume path - [vsanDatastore] kubevols/volume.vmdk will be [vsanDatastore] 25d8b159-948c-4b73-e499-02001ad1b044/volume.vmdk
// Example2: The canonical path for volume path - [vsanDatastore] 25d8b159-948c-4b73-e499-02001ad1b044/volume.vmdk will be same as volume Path.
func getCanonicalVolumePath(ctx context.Context, dc *object.Datacenter, volumePath string) (string, error) {
	var folderID string
	canonicalVolumePath := volumePath
	dsPathObj, err := getDatastorePathObjFromVMDiskPath(volumePath)
	if err != nil {
		return "", err
	}
	dsPath := strings.Split(strings.TrimSpace(dsPathObj.Path), "/")
	if len(dsPath) <= 1 {
		return canonicalVolumePath, nil
	}
	datastore := dsPathObj.Datastore
	dsFolder := dsPath[0]
	// Get the datastore folder ID if datastore or folder doesn't exist in datastoreFolderIDMap
	if !isValidUUID(dsFolder) {
		dummyDiskVolPath := "[" + datastore + "] " + dsFolder + "/" + dummyDiskName
		// Querying a non-existent dummy disk on the datastore folder.
		// It would fail and return an folder ID in the error message.
		_, err := getVirtualDiskPage83Data(ctx, dc, dummyDiskVolPath)
		if err != nil {
			re := regexp.MustCompile("File (.*?) was not found")
			match := re.FindStringSubmatch(err.Error())
			canonicalVolumePath = match[1]
		}
	}
	diskPath := getPathFromVMDiskPath(canonicalVolumePath)
	if diskPath == "" {
		return "", fmt.Errorf("Failed to parse canonicalVolumePath: %s in getcanonicalVolumePath method", canonicalVolumePath)
	}
	folderID = strings.Split(strings.TrimSpace(diskPath), "/")[0]
	canonicalVolumePath = strings.Replace(volumePath, dsFolder, folderID, 1)
	return canonicalVolumePath, nil
}

// getPathFromVMDiskPath retrieves the path from VM Disk Path.
// Example: For vmDiskPath - [vsanDatastore] kubevols/volume.vmdk, the path is kubevols/volume.vmdk
func getPathFromVMDiskPath(vmDiskPath string) string {
	datastorePathObj := new(object.DatastorePath)
	isSuccess := datastorePathObj.FromString(vmDiskPath)
	if !isSuccess {
		framework.Logf("Failed to parse vmDiskPath: %s", vmDiskPath)
		return ""
	}
	return datastorePathObj.Path
}

//getDatastorePathObjFromVMDiskPath gets the datastorePathObj from VM disk path.
func getDatastorePathObjFromVMDiskPath(vmDiskPath string) (*object.DatastorePath, error) {
	datastorePathObj := new(object.DatastorePath)
	isSuccess := datastorePathObj.FromString(vmDiskPath)
	if !isSuccess {
		framework.Logf("Failed to parse volPath: %s", vmDiskPath)
		return nil, fmt.Errorf("Failed to parse volPath: %s", vmDiskPath)
	}
	return datastorePathObj, nil
}

// getVirtualDiskPage83Data gets the virtual disk UUID by diskPath
func getVirtualDiskPage83Data(ctx context.Context, dc *object.Datacenter, diskPath string) (string, error) {
	if len(diskPath) > 0 && filepath.Ext(diskPath) != ".vmdk" {
		diskPath += ".vmdk"
	}
	vdm := object.NewVirtualDiskManager(dc.Client())
	// Returns uuid of vmdk virtual disk
	diskUUID, err := vdm.QueryVirtualDiskUuid(ctx, diskPath, dc)

	if err != nil {
		klog.Warningf("QueryVirtualDiskUuid failed for diskPath: %q. err: %+v", diskPath, err)
		return "", err
	}
	diskUUID = formatVirtualDiskUUID(diskUUID)
	return diskUUID, nil
}

// formatVirtualDiskUUID removes any spaces and hyphens in UUID
// Example UUID input is 42375390-71f9-43a3-a770-56803bcd7baa and output after format is 4237539071f943a3a77056803bcd7baa
func formatVirtualDiskUUID(uuid string) string {
	uuidwithNoSpace := strings.Replace(uuid, " ", "", -1)
	uuidWithNoHypens := strings.Replace(uuidwithNoSpace, "-", "", -1)
	return strings.ToLower(uuidWithNoHypens)
}

//isValidUUID checks if the string is a valid UUID.
func isValidUUID(uuid string) bool {
	r := regexp.MustCompile("^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$")
	return r.MatchString(uuid)
}

// removeStorageClusterORFolderNameFromVDiskPath removes the cluster or folder path from the vDiskPath
// for vDiskPath [DatastoreCluster/sharedVmfs-0] kubevols/e2e-vmdk-1234.vmdk, return value is [sharedVmfs-0] kubevols/e2e-vmdk-1234.vmdk
// for vDiskPath [sharedVmfs-0] kubevols/e2e-vmdk-1234.vmdk, return value remains same [sharedVmfs-0] kubevols/e2e-vmdk-1234.vmdk
func removeStorageClusterORFolderNameFromVDiskPath(vDiskPath string) string {
	datastore := regexp.MustCompile("\\[(.*?)\\]").FindStringSubmatch(vDiskPath)[1]
	if filepath.Base(datastore) != datastore {
		vDiskPath = strings.Replace(vDiskPath, datastore, filepath.Base(datastore), 1)
	}
	return vDiskPath
}

// getVirtualDeviceByPath gets the virtual device by path
func getVirtualDeviceByPath(ctx context.Context, vm *object.VirtualMachine, diskPath string) (vim25types.BaseVirtualDevice, error) {
	vmDevices, err := vm.Device(ctx)
	if err != nil {
		framework.Logf("Failed to get the devices for VM: %q. err: %+v", vm.InventoryPath, err)
		return nil, err
	}

	// filter vm devices to retrieve device for the given vmdk file identified by disk path
	for _, device := range vmDevices {
		if vmDevices.TypeName(device) == "VirtualDisk" {
			virtualDevice := device.GetVirtualDevice()
			if backing, ok := virtualDevice.Backing.(*vim25types.VirtualDiskFlatVer2BackingInfo); ok {
				if matchVirtualDiskAndVolPath(backing.FileName, diskPath) {
					framework.Logf("Found VirtualDisk backing with filename %q for diskPath %q", backing.FileName, diskPath)
					return device, nil
				}
				framework.Logf("VirtualDisk backing filename %q does not match with diskPath %q", backing.FileName, diskPath)
			}
		}
	}
	return nil, nil
}

func matchVirtualDiskAndVolPath(diskPath, volPath string) bool {
	fileExt := ".vmdk"
	diskPath = strings.TrimSuffix(diskPath, fileExt)
	volPath = strings.TrimSuffix(volPath, fileExt)
	return diskPath == volPath
}

// convertVolPathsToDevicePaths removes cluster or folder path from volPaths and convert to canonicalPath
func convertVolPathsToDevicePaths(ctx context.Context, nodeVolumes map[string][]string) (map[string][]string, error) {
	vmVolumes := make(map[string][]string)
	for nodeName, volPaths := range nodeVolumes {
		nodeInfo := TestContext.NodeMapper.GetNodeInfo(nodeName)
		datacenter := nodeInfo.VSphere.GetDatacenterFromObjectReference(ctx, nodeInfo.DataCenterRef)
		for i, volPath := range volPaths {
			deviceVolPath, err := convertVolPathToDevicePath(ctx, datacenter, volPath)
			if err != nil {
				framework.Logf("Failed to convert vsphere volume path %s to device path for volume %s. err: %+v", volPath, deviceVolPath, err)
				return nil, err
			}
			volPaths[i] = deviceVolPath
		}
		vmVolumes[nodeName] = volPaths
	}
	return vmVolumes, nil
}

// convertVolPathToDevicePath takes volPath and returns canonical volume path
func convertVolPathToDevicePath(ctx context.Context, dc *object.Datacenter, volPath string) (string, error) {
	volPath = removeStorageClusterORFolderNameFromVDiskPath(volPath)
	// Get the canonical volume path for volPath.
	canonicalVolumePath, err := getCanonicalVolumePath(ctx, dc, volPath)
	if err != nil {
		framework.Logf("Failed to get canonical vsphere volume path for volume: %s. err: %+v", volPath, err)
		return "", err
	}
	// Check if the volume path contains .vmdk extension. If not, add the extension and update the nodeVolumes Map
	if len(canonicalVolumePath) > 0 && filepath.Ext(canonicalVolumePath) != ".vmdk" {
		canonicalVolumePath += ".vmdk"
	}
	return canonicalVolumePath, nil
}

// get .vmx file path for a virtual machine
func getVMXFilePath(vmObject *object.VirtualMachine) (vmxPath string) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var nodeVM mo.VirtualMachine
	err := vmObject.Properties(ctx, vmObject.Reference(), []string{"config.files"}, &nodeVM)
	framework.ExpectNoError(err)
	gomega.Expect(nodeVM.Config).NotTo(gomega.BeNil())

	vmxPath = nodeVM.Config.Files.VmPathName
	framework.Logf("vmx file path is %s", vmxPath)
	return vmxPath
}

// verify ready node count. Try upto 3 minutes. Return true if count is expected count
func verifyReadyNodeCount(client clientset.Interface, expectedNodes int) bool {
	numNodes := 0
	for i := 0; i < 36; i++ {
		nodeList, err := e2enode.GetReadySchedulableNodes(client)
		framework.ExpectNoError(err)

		numNodes = len(nodeList.Items)
		if numNodes == expectedNodes {
			break
		}
		time.Sleep(5 * time.Second)
	}
	return (numNodes == expectedNodes)
}

// poweroff nodeVM and confirm the poweroff state
func poweroffNodeVM(nodeName string, vm *object.VirtualMachine) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	framework.Logf("Powering off node VM %s", nodeName)

	_, err := vm.PowerOff(ctx)
	framework.ExpectNoError(err)
	err = vm.WaitForPowerState(ctx, vim25types.VirtualMachinePowerStatePoweredOff)
	framework.ExpectNoError(err, "Unable to power off the node")
}

// poweron nodeVM and confirm the poweron state
func poweronNodeVM(nodeName string, vm *object.VirtualMachine) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	framework.Logf("Powering on node VM %s", nodeName)

	vm.PowerOn(ctx)
	err := vm.WaitForPowerState(ctx, vim25types.VirtualMachinePowerStatePoweredOn)
	framework.ExpectNoError(err, "Unable to power on the node")
}

// unregister a nodeVM from VC
func unregisterNodeVM(nodeName string, vm *object.VirtualMachine) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	poweroffNodeVM(nodeName, vm)

	framework.Logf("Unregistering node VM %s", nodeName)
	err := vm.Unregister(ctx)
	framework.ExpectNoError(err, "Unable to unregister the node")
}

// register a nodeVM into a VC
func registerNodeVM(nodeName, workingDir, vmxFilePath string, rpool *object.ResourcePool, host *object.HostSystem) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	framework.Logf("Registering node VM %s with vmx file path %s", nodeName, vmxFilePath)

	nodeInfo := TestContext.NodeMapper.GetNodeInfo(nodeName)
	finder := find.NewFinder(nodeInfo.VSphere.Client.Client, false)

	vmFolder, err := finder.FolderOrDefault(ctx, workingDir)
	framework.ExpectNoError(err)

	registerTask, err := vmFolder.RegisterVM(ctx, vmxFilePath, nodeName, false, rpool, host)
	framework.ExpectNoError(err)
	err = registerTask.Wait(ctx)
	framework.ExpectNoError(err)

	vmPath := filepath.Join(workingDir, nodeName)
	vm, err := finder.VirtualMachine(ctx, vmPath)
	framework.ExpectNoError(err)

	poweronNodeVM(nodeName, vm)
}

// disksAreAttached takes map of node and it's volumes and returns map of node, its volumes and attachment state
func disksAreAttached(nodeVolumes map[string][]string) (map[string]map[string]bool, error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	disksAttached := make(map[string]map[string]bool)
	if len(nodeVolumes) == 0 {
		return disksAttached, nil
	}
	// Convert VolPaths into canonical form so that it can be compared with the VM device path.
	vmVolumes, err := convertVolPathsToDevicePaths(ctx, nodeVolumes)
	if err != nil {
		framework.Logf("Failed to convert volPaths to devicePaths: %+v. err: %+v", nodeVolumes, err)
		return nil, err
	}
	for vm, volumes := range vmVolumes {
		volumeAttachedMap := make(map[string]bool)
		for _, volume := range volumes {
			attached, err := diskIsAttached(volume, vm)
			if err != nil {
				return nil, err
			}
			volumeAttachedMap[volume] = attached
		}
		disksAttached[vm] = volumeAttachedMap
	}
	return disksAttached, nil
}

// diskIsAttached returns if disk is attached to the VM using controllers supported by the plugin.
func diskIsAttached(volPath string, nodeName string) (bool, error) {
	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	nodeInfo := TestContext.NodeMapper.GetNodeInfo(nodeName)
	Connect(ctx, nodeInfo.VSphere)
	vm := object.NewVirtualMachine(nodeInfo.VSphere.Client.Client, nodeInfo.VirtualMachineRef)
	volPath = removeStorageClusterORFolderNameFromVDiskPath(volPath)
	device, err := getVirtualDeviceByPath(ctx, vm, volPath)
	if err != nil {
		framework.Logf("diskIsAttached failed to determine whether disk %q is still attached on node %q",
			volPath,
			nodeName)
		return false, err
	}
	if device == nil {
		return false, nil
	}
	framework.Logf("diskIsAttached found the disk %q attached on node %q", volPath, nodeName)
	return true, nil
}

// getUUIDFromProviderID strips ProviderPrefix - "vsphere://" from the providerID
// this gives the VM UUID which can be used to find Node VM from vCenter
func getUUIDFromProviderID(providerID string) string {
	return strings.TrimPrefix(providerID, providerPrefix)
}

// GetReadySchedulableNodeInfos returns NodeInfo objects for all nodes with Ready and schedulable state
func GetReadySchedulableNodeInfos() []*NodeInfo {
	nodeList, err := e2enode.GetReadySchedulableNodes(f.ClientSet)
	framework.ExpectNoError(err)
	var nodesInfo []*NodeInfo
	for _, node := range nodeList.Items {
		nodeInfo := TestContext.NodeMapper.GetNodeInfo(node.Name)
		if nodeInfo != nil {
			nodesInfo = append(nodesInfo, nodeInfo)
		}
	}
	return nodesInfo
}

// GetReadySchedulableRandomNodeInfo returns NodeInfo object for one of the Ready and Schedulable Node.
// if multiple nodes are present with Ready and Scheduable state then one of the Node is selected randomly
// and it's associated NodeInfo object is returned.
func GetReadySchedulableRandomNodeInfo() *NodeInfo {
	nodesInfo := GetReadySchedulableNodeInfos()
	gomega.Expect(nodesInfo).NotTo(gomega.BeEmpty())
	return nodesInfo[rand.Int()%len(nodesInfo)]
}

// invokeVCenterServiceControl invokes the given command for the given service
// via service-control on the given vCenter host over SSH.
func invokeVCenterServiceControl(command, service, host string) error {
	sshCmd := fmt.Sprintf("service-control --%s %s", command, service)
	framework.Logf("Invoking command %v on vCenter host %v", sshCmd, host)
	result, err := e2essh.SSH(sshCmd, host, framework.TestContext.Provider)
	if err != nil || result.Code != 0 {
		e2essh.LogResult(result)
		return fmt.Errorf("couldn't execute command: %s on vCenter host: %v", sshCmd, err)
	}
	return nil
}

// expectVolumeToBeAttached checks if the given Volume is attached to the given
// Node, else fails.
func expectVolumeToBeAttached(nodeName, volumePath string) {
	isAttached, err := diskIsAttached(volumePath, nodeName)
	framework.ExpectNoError(err)
	framework.ExpectEqual(isAttached, true, fmt.Sprintf("disk: %s is not attached with the node", volumePath))
}

// expectVolumesToBeAttached checks if the given Volumes are attached to the
// corresponding set of Nodes, else fails.
func expectVolumesToBeAttached(pods []*v1.Pod, volumePaths []string) {
	for i, pod := range pods {
		nodeName := pod.Spec.NodeName
		volumePath := volumePaths[i]
		ginkgo.By(fmt.Sprintf("Verifying that volume %v is attached to node %v", volumePath, nodeName))
		expectVolumeToBeAttached(nodeName, volumePath)
	}
}

// expectFilesToBeAccessible checks if the given files are accessible on the
// corresponding set of Nodes, else fails.
func expectFilesToBeAccessible(namespace string, pods []*v1.Pod, filePaths []string) {
	for i, pod := range pods {
		podName := pod.Name
		filePath := filePaths[i]
		ginkgo.By(fmt.Sprintf("Verifying that file %v is accessible on pod %v", filePath, podName))
		verifyFilesExistOnVSphereVolume(namespace, podName, filePath)
	}
}

// writeContentToPodFile writes the given content to the specified file.
func writeContentToPodFile(namespace, podName, filePath, content string) error {
	_, err := framework.RunKubectl(namespace, "exec", podName,
		"--", "/bin/sh", "-c", fmt.Sprintf("echo '%s' > %s", content, filePath))
	return err
}

// expectFileContentToMatch checks if a given file contains the specified
// content, else fails.
func expectFileContentToMatch(namespace, podName, filePath, content string) {
	_, err := framework.RunKubectl(namespace, "exec", podName,
		"--", "/bin/sh", "-c", fmt.Sprintf("grep '%s' %s", content, filePath))
	framework.ExpectNoError(err, fmt.Sprintf("failed to match content of file: %q on the pod: %q", filePath, podName))
}

// expectFileContentsToMatch checks if the given contents match the ones present
// in corresponding files on respective Pods, else fails.
func expectFileContentsToMatch(namespace string, pods []*v1.Pod, filePaths []string, contents []string) {
	for i, pod := range pods {
		podName := pod.Name
		filePath := filePaths[i]
		ginkgo.By(fmt.Sprintf("Matching file content for %v on pod %v", filePath, podName))
		expectFileContentToMatch(namespace, podName, filePath, contents[i])
	}
}
