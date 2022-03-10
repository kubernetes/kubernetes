/*
Copyright 2022 The Kubernetes Authors.

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

package reconciler

import (
	"context"
	"fmt"
	"path/filepath"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
	volumepkg "k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/nestedpendingoperations"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
)

// sync process tries to observe the real world by scanning all pods' volume directories from the disk.
// If the actual and desired state of worlds are not consistent with the observed world, it means that some
// mounted volumes are left out probably during kubelet restart. This process will reconstruct
// the volumes and update the actual and desired states. For the volumes that cannot support reconstruction,
// it will try to clean up the mount paths with operation executor.
func (rc *reconciler) sync() {
	defer rc.updateLastSyncTime()
	rc.syncStates()
}

// syncStates scans the volume directories under the given pod directory.
// If the volume is not in desired state of world, this function will reconstruct
// the volume related information and put it in both the actual and desired state of worlds.
// For some volume plugins that cannot support reconstruction, it will clean up the existing
// mount points since the volume is no long needed (removed from desired state)
func (rc *reconciler) syncStates() {
	// Get volumes information by reading the pod's directory
	podVolumes, err := getVolumesFromPodDir(rc.kubeletPodsDir)
	if err != nil {
		klog.ErrorS(err, "Cannot get volumes from disk, skip sync states for volume reconstruction")
		return
	}
	volumesNeedUpdate := make(map[v1.UniqueVolumeName]*reconstructedVolume)
	volumeNeedReport := []v1.UniqueVolumeName{}
	for _, volume := range podVolumes {
		if rc.actualStateOfWorld.VolumeExistsWithSpecName(volume.podName, volume.volumeSpecName) {
			klog.V(4).InfoS("Volume exists in actual state, skip cleaning up mounts", "podName", volume.podName, "volumeSpecName", volume.volumeSpecName)
			// There is nothing to reconstruct
			continue
		}
		volumeInDSW := rc.desiredStateOfWorld.VolumeExistsWithSpecName(volume.podName, volume.volumeSpecName)

		reconstructedVolume, err := rc.reconstructVolume(volume)
		if err != nil {
			if volumeInDSW {
				// Some pod needs the volume, don't clean it up and hope that
				// reconcile() calls SetUp and reconstructs the volume in ASW.
				klog.V(4).InfoS("Volume exists in desired state, skip cleaning up mounts", "podName", volume.podName, "volumeSpecName", volume.volumeSpecName)
				continue
			}
			// No pod needs the volume.
			klog.InfoS("Could not construct volume information, cleaning up mounts", "podName", volume.podName, "volumeSpecName", volume.volumeSpecName, "err", err)
			rc.cleanupMounts(volume)
			continue
		}
		if volumeInDSW {
			// Some pod needs the volume. And it exists on disk. Some previous
			// kubelet must have created the directory, therefore it must have
			// reported the volume as in use. Mark the volume as in use also in
			// this new kubelet so reconcile() calls SetUp and re-mounts the
			// volume if it's necessary.
			volumeNeedReport = append(volumeNeedReport, reconstructedVolume.volumeName)
			klog.V(4).InfoS("Volume exists in desired state, marking as InUse", "podName", volume.podName, "volumeSpecName", volume.volumeSpecName)
			continue
		}
		// There is no pod that uses the volume.
		if rc.operationExecutor.IsOperationPending(reconstructedVolume.volumeName, nestedpendingoperations.EmptyUniquePodName, nestedpendingoperations.EmptyNodeName) {
			klog.InfoS("Volume is in pending operation, skip cleaning up mounts")
		}
		klog.V(2).InfoS("Reconciler sync states: could not find pod information in desired state, update it in actual state", "reconstructedVolume", reconstructedVolume)
		volumesNeedUpdate[reconstructedVolume.volumeName] = reconstructedVolume
	}

	if len(volumesNeedUpdate) > 0 {
		if err = rc.updateStates(volumesNeedUpdate); err != nil {
			klog.ErrorS(err, "Error occurred during reconstruct volume from disk")
		}
	}
	if len(volumeNeedReport) > 0 {
		rc.desiredStateOfWorld.MarkVolumesReportedInUse(volumeNeedReport)
	}
}

// Reconstruct volume data structure by reading the pod's volume directories
func (rc *reconciler) reconstructVolume(volume podVolume) (*reconstructedVolume, error) {
	// plugin initializations
	plugin, err := rc.volumePluginMgr.FindPluginByName(volume.pluginName)
	if err != nil {
		return nil, err
	}
	attachablePlugin, err := rc.volumePluginMgr.FindAttachablePluginByName(volume.pluginName)
	if err != nil {
		return nil, err
	}
	deviceMountablePlugin, err := rc.volumePluginMgr.FindDeviceMountablePluginByName(volume.pluginName)
	if err != nil {
		return nil, err
	}

	// Create pod object
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: types.UID(volume.podName),
		},
	}
	mapperPlugin, err := rc.volumePluginMgr.FindMapperPluginByName(volume.pluginName)
	if err != nil {
		return nil, err
	}
	if volume.volumeMode == v1.PersistentVolumeBlock && mapperPlugin == nil {
		return nil, fmt.Errorf("could not find block volume plugin %q (spec.Name: %q) pod %q (UID: %q)", volume.pluginName, volume.volumeSpecName, volume.podName, pod.UID)
	}

	volumeSpec, err := rc.operationExecutor.ReconstructVolumeOperation(
		volume.volumeMode,
		plugin,
		mapperPlugin,
		pod.UID,
		volume.podName,
		volume.volumeSpecName,
		volume.volumePath,
		volume.pluginName)
	if err != nil {
		return nil, err
	}

	var uniqueVolumeName v1.UniqueVolumeName
	if attachablePlugin != nil || deviceMountablePlugin != nil {
		uniqueVolumeName, err = util.GetUniqueVolumeNameFromSpec(plugin, volumeSpec)
		if err != nil {
			return nil, err
		}
	} else {
		uniqueVolumeName = util.GetUniqueVolumeNameFromSpecWithPod(volume.podName, plugin, volumeSpec)
	}

	var volumeMapper volumepkg.BlockVolumeMapper
	var volumeMounter volumepkg.Mounter
	var deviceMounter volumepkg.DeviceMounter
	// Path to the mount or block device to check
	var checkPath string

	if volume.volumeMode == v1.PersistentVolumeBlock {
		var newMapperErr error
		volumeMapper, newMapperErr = mapperPlugin.NewBlockVolumeMapper(
			volumeSpec,
			pod,
			volumepkg.VolumeOptions{})
		if newMapperErr != nil {
			return nil, fmt.Errorf(
				"reconstructVolume.NewBlockVolumeMapper failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
				uniqueVolumeName,
				volumeSpec.Name(),
				volume.podName,
				pod.UID,
				newMapperErr)
		}
		mapDir, linkName := volumeMapper.GetPodDeviceMapPath()
		checkPath = filepath.Join(mapDir, linkName)
	} else {
		var err error
		volumeMounter, err = plugin.NewMounter(
			volumeSpec,
			pod,
			volumepkg.VolumeOptions{})
		if err != nil {
			return nil, fmt.Errorf(
				"reconstructVolume.NewMounter failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
				uniqueVolumeName,
				volumeSpec.Name(),
				volume.podName,
				pod.UID,
				err)
		}
		checkPath = volumeMounter.GetPath()
		if deviceMountablePlugin != nil {
			deviceMounter, err = deviceMountablePlugin.NewDeviceMounter()
			if err != nil {
				return nil, fmt.Errorf("reconstructVolume.NewDeviceMounter failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
					uniqueVolumeName,
					volumeSpec.Name(),
					volume.podName,
					pod.UID,
					err)
			}
		}
	}

	// Check existence of mount point for filesystem volume or symbolic link for block volume
	isExist, checkErr := rc.operationExecutor.CheckVolumeExistenceOperation(volumeSpec, checkPath, volumeSpec.Name(), rc.mounter, uniqueVolumeName, volume.podName, pod.UID, attachablePlugin)
	if checkErr != nil {
		return nil, checkErr
	}
	// If mount or symlink doesn't exist, volume reconstruction should be failed
	if !isExist {
		return nil, fmt.Errorf("volume: %q is not mounted", uniqueVolumeName)
	}

	reconstructedVolume := &reconstructedVolume{
		volumeName: uniqueVolumeName,
		podName:    volume.podName,
		volumeSpec: volumeSpec,
		// volume.volumeSpecName is actually InnerVolumeSpecName. It will not be used
		// for volume cleanup.
		// in case pod is added back to desired state, outerVolumeSpecName will be updated from dsw information.
		// See issue #103143 and its fix for details.
		outerVolumeSpecName: volume.volumeSpecName,
		pod:                 pod,
		deviceMounter:       deviceMounter,
		volumeGidValue:      "",
		// devicePath is updated during updateStates() by checking node status's VolumesAttached data.
		// TODO: get device path directly from the volume mount path.
		devicePath:        "",
		mounter:           volumeMounter,
		blockVolumeMapper: volumeMapper,
	}
	return reconstructedVolume, nil
}

// updateDevicePath gets the node status to retrieve volume device path information.
func (rc *reconciler) updateDevicePath(volumesNeedUpdate map[v1.UniqueVolumeName]*reconstructedVolume) {
	node, fetchErr := rc.kubeClient.CoreV1().Nodes().Get(context.TODO(), string(rc.nodeName), metav1.GetOptions{})
	if fetchErr != nil {
		klog.ErrorS(fetchErr, "UpdateStates in reconciler: could not get node status with error")
	} else {
		for _, attachedVolume := range node.Status.VolumesAttached {
			if volume, exists := volumesNeedUpdate[attachedVolume.Name]; exists {
				volume.devicePath = attachedVolume.DevicePath
				volumesNeedUpdate[attachedVolume.Name] = volume
				klog.V(4).InfoS("Update devicePath from node status for volume", "volumeName", attachedVolume.Name, "path", volume.devicePath)
			}
		}
	}
}

func (rc *reconciler) updateStates(volumesNeedUpdate map[v1.UniqueVolumeName]*reconstructedVolume) error {
	// Get the node status to retrieve volume device path information.
	rc.updateDevicePath(volumesNeedUpdate)

	for _, volume := range volumesNeedUpdate {
		err := rc.actualStateOfWorld.MarkVolumeAsAttached(
			//TODO: the devicePath might not be correct for some volume plugins: see issue #54108
			volume.volumeName, volume.volumeSpec, "" /* nodeName */, volume.devicePath)
		if err != nil {
			klog.ErrorS(err, "Could not add volume information to actual state of world", "pod", klog.KObj(volume.pod))
			continue
		}
		markVolumeOpts := operationexecutor.MarkVolumeOpts{
			PodName:             volume.podName,
			PodUID:              types.UID(volume.podName),
			VolumeName:          volume.volumeName,
			Mounter:             volume.mounter,
			BlockVolumeMapper:   volume.blockVolumeMapper,
			OuterVolumeSpecName: volume.outerVolumeSpecName,
			VolumeGidVolume:     volume.volumeGidValue,
			VolumeSpec:          volume.volumeSpec,
			VolumeMountState:    operationexecutor.VolumeMounted,
		}
		err = rc.actualStateOfWorld.MarkVolumeAsMounted(markVolumeOpts)
		if err != nil {
			klog.ErrorS(err, "Could not add pod to volume information to actual state of world", "pod", klog.KObj(volume.pod))
			continue
		}
		klog.V(4).InfoS("Volume is marked as mounted and added into the actual state", "pod", klog.KObj(volume.pod), "podName", volume.podName, "volumeName", volume.volumeName)
		// If the volume has device to mount, we mark its device as mounted.
		if volume.deviceMounter != nil || volume.blockVolumeMapper != nil {
			deviceMountPath, err := getDeviceMountPath(volume)
			if err != nil {
				klog.ErrorS(err, "Could not find device mount path for volume", "volumeName", volume.volumeName, "pod", klog.KObj(volume.pod))
				continue
			}
			err = rc.actualStateOfWorld.MarkDeviceAsMounted(volume.volumeName, volume.devicePath, deviceMountPath)
			if err != nil {
				klog.ErrorS(err, "Could not mark device is mounted to actual state of world", "pod", klog.KObj(volume.pod))
				continue
			}
			klog.V(4).InfoS("Volume is marked device as mounted and added into the actual state", "pod", klog.KObj(volume.pod), "podName", volume.podName, "volumeName", volume.volumeName)
		}
	}
	return nil
}
