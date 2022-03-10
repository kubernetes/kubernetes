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
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/config"
	volumepkg "k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
	utilpath "k8s.io/utils/path"
	utilstrings "k8s.io/utils/strings"
)

type podVolume struct {
	podName        volumetypes.UniquePodName
	volumeSpecName string
	volumePath     string
	pluginName     string
	volumeMode     v1.PersistentVolumeMode
}

type reconstructedVolume struct {
	volumeName          v1.UniqueVolumeName
	podName             volumetypes.UniquePodName
	volumeSpec          *volumepkg.Spec
	outerVolumeSpecName string
	pod                 *v1.Pod
	volumeGidValue      string
	devicePath          string
	mounter             volumepkg.Mounter
	deviceMounter       volumepkg.DeviceMounter
	blockVolumeMapper   volumepkg.BlockVolumeMapper
}

func (rc *reconciler) updateLastSyncTime() {
	rc.timeOfLastSync = time.Now()
}

func (rc *reconciler) StatesHasBeenSynced() bool {
	return !rc.timeOfLastSync.IsZero()
}

func (rc *reconciler) cleanupMounts(volume podVolume) {
	klog.V(2).InfoS("Reconciler sync states: could not find volume information in desired state, clean up the mount points", "podName", volume.podName, "volumeSpecName", volume.volumeSpecName)
	mountedVolume := operationexecutor.MountedVolume{
		PodName:             volume.podName,
		VolumeName:          v1.UniqueVolumeName(volume.volumeSpecName),
		InnerVolumeSpecName: volume.volumeSpecName,
		PluginName:          volume.pluginName,
		PodUID:              types.UID(volume.podName),
	}
	// TODO: Currently cleanupMounts only includes UnmountVolume operation. In the next PR, we will add
	// to unmount both volume and device in the same routine.
	err := rc.operationExecutor.UnmountVolume(mountedVolume, rc.actualStateOfWorld, rc.kubeletPodsDir)
	if err != nil {
		klog.ErrorS(err, mountedVolume.GenerateErrorDetailed("volumeHandler.UnmountVolumeHandler for UnmountVolume failed", err).Error())
		return
	}
}

// getDeviceMountPath returns device mount path for block volume which
// implements BlockVolumeMapper or filesystem volume which implements
// DeviceMounter
func getDeviceMountPath(volume *reconstructedVolume) (string, error) {
	if volume.blockVolumeMapper != nil {
		// for block volume, we return its global map path
		return volume.blockVolumeMapper.GetGlobalMapPath(volume.volumeSpec)
	} else if volume.deviceMounter != nil {
		// for filesystem volume, we return its device mount path if the plugin implements DeviceMounter
		return volume.deviceMounter.GetDeviceMountPath(volume.volumeSpec)
	} else {
		return "", fmt.Errorf("blockVolumeMapper or deviceMounter required")
	}
}

// getVolumesFromPodDir scans through the volumes directories under the given pod directory.
// It returns a list of pod volume information including pod's uid, volume's plugin name, mount path,
// and volume spec name.
func getVolumesFromPodDir(podDir string) ([]podVolume, error) {
	podsDirInfo, err := ioutil.ReadDir(podDir)
	if err != nil {
		return nil, err
	}
	volumes := []podVolume{}
	for i := range podsDirInfo {
		if !podsDirInfo[i].IsDir() {
			continue
		}
		podName := podsDirInfo[i].Name()
		podDir := path.Join(podDir, podName)

		// Find filesystem volume information
		// ex. filesystem volume: /pods/{podUid}/volume/{escapeQualifiedPluginName}/{volumeName}
		volumesDirs := map[v1.PersistentVolumeMode]string{
			v1.PersistentVolumeFilesystem: path.Join(podDir, config.DefaultKubeletVolumesDirName),
		}
		// Find block volume information
		// ex. block volume: /pods/{podUid}/volumeDevices/{escapeQualifiedPluginName}/{volumeName}
		volumesDirs[v1.PersistentVolumeBlock] = path.Join(podDir, config.DefaultKubeletVolumeDevicesDirName)

		for volumeMode, volumesDir := range volumesDirs {
			var volumesDirInfo []os.FileInfo
			if volumesDirInfo, err = ioutil.ReadDir(volumesDir); err != nil {
				// Just skip the loop because given volumesDir doesn't exist depending on volumeMode
				continue
			}
			for _, volumeDir := range volumesDirInfo {
				pluginName := volumeDir.Name()
				volumePluginPath := path.Join(volumesDir, pluginName)
				volumePluginDirs, err := utilpath.ReadDirNoStat(volumePluginPath)
				if err != nil {
					klog.ErrorS(err, "Could not read volume plugin directory", "volumePluginPath", volumePluginPath)
					continue
				}
				unescapePluginName := utilstrings.UnescapeQualifiedName(pluginName)
				for _, volumeName := range volumePluginDirs {
					volumePath := path.Join(volumePluginPath, volumeName)
					klog.V(5).InfoS("Volume path from volume plugin directory", "podName", podName, "volumePath", volumePath)
					volumes = append(volumes, podVolume{
						podName:        volumetypes.UniquePodName(podName),
						volumeSpecName: volumeName,
						volumePath:     volumePath,
						pluginName:     unescapePluginName,
						volumeMode:     volumeMode,
					})
				}
			}
		}
	}
	klog.V(4).InfoS("Get volumes from pod directory", "path", podDir, "volumes", volumes)
	return volumes, nil
}
