/*
Copyright 2018 The Kubernetes Authors.

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

package vsphere_volume

import (
	"fmt"
	"path/filepath"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/volumepathhandler"
	utilstrings "k8s.io/utils/strings"
)

var _ volume.BlockVolumePlugin = &vsphereVolumePlugin{}

func (plugin *vsphereVolumePlugin) ConstructBlockVolumeSpec(podUID types.UID, volumeName, mapPath string) (*volume.Spec, error) {

	pluginDir := plugin.host.GetPluginDir(plugin.GetPluginName())
	blkUtil := volumepathhandler.NewBlockVolumePathHandler()
	globalMapPathUUID, err := blkUtil.FindGlobalMapPathUUIDFromPod(pluginDir, mapPath, podUID)
	if err != nil {
		klog.Errorf("Failed to find GlobalMapPathUUID from Pod: %s with error: %+v", podUID, err)
		return nil, err
	}
	klog.V(5).Infof("globalMapPathUUID: %v", globalMapPathUUID)
	globalMapPath := filepath.Dir(globalMapPathUUID)
	if len(globalMapPath) <= 1 {
		return nil, fmt.Errorf("failed to get volume plugin information from globalMapPathUUID: %v", globalMapPathUUID)
	}
	return getVolumeSpecFromGlobalMapPath(globalMapPath)
}

func getVolumeSpecFromGlobalMapPath(globalMapPath string) (*volume.Spec, error) {
	// Construct volume spec from globalMapPath
	// globalMapPath example:
	//   plugins/kubernetes.io/{PluginName}/{DefaultKubeletVolumeDevicesDirName}/{volumeID}
	//   plugins/kubernetes.io/vsphere-volume/volumeDevices/[datastore1]\\040volumes/myDisk
	volPath := filepath.Base(globalMapPath)
	volPath = strings.Replace(volPath, "\\040", "", -1)
	if len(volPath) <= 1 {
		return nil, fmt.Errorf("failed to get volume path from global path=%s", globalMapPath)
	}
	block := v1.PersistentVolumeBlock
	vsphereVolume := &v1.PersistentVolume{
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
					VolumePath: volPath,
				},
			},
			VolumeMode: &block,
		},
	}
	return volume.NewSpecFromPersistentVolume(vsphereVolume, true), nil
}

func (plugin *vsphereVolumePlugin) NewBlockVolumeMapper(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.BlockVolumeMapper, error) {
	// If this called via GenerateUnmapDeviceFunc(), pod is nil.
	// Pass empty string as dummy uid since uid isn't used in the case.
	var uid types.UID
	if pod != nil {
		uid = pod.UID
	}
	return plugin.newBlockVolumeMapperInternal(spec, uid, &VsphereDiskUtil{}, plugin.host.GetMounter(plugin.GetPluginName()))
}

func (plugin *vsphereVolumePlugin) newBlockVolumeMapperInternal(spec *volume.Spec, podUID types.UID, manager vdManager, mounter mount.Interface) (volume.BlockVolumeMapper, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		klog.Errorf("Failed to get Volume source from volume Spec: %+v with error: %+v", *spec, err)
		return nil, err
	}
	volPath := volumeSource.VolumePath
	return &vsphereBlockVolumeMapper{
		vsphereVolume: &vsphereVolume{
			volName:         spec.Name(),
			podUID:          podUID,
			volPath:         volPath,
			manager:         manager,
			mounter:         mounter,
			plugin:          plugin,
			MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, spec.Name(), plugin.host)),
		},
	}, nil

}

func (plugin *vsphereVolumePlugin) NewBlockVolumeUnmapper(volName string, podUID types.UID) (volume.BlockVolumeUnmapper, error) {
	return plugin.newUnmapperInternal(volName, podUID, &VsphereDiskUtil{})
}

func (plugin *vsphereVolumePlugin) newUnmapperInternal(volName string, podUID types.UID, manager vdManager) (volume.BlockVolumeUnmapper, error) {
	return &vsphereBlockVolumeUnmapper{
		vsphereVolume: &vsphereVolume{
			volName: volName,
			podUID:  podUID,
			volPath: volName,
			manager: manager,
			plugin:  plugin,
		},
	}, nil
}

var _ volume.BlockVolumeMapper = &vsphereBlockVolumeMapper{}

type vsphereBlockVolumeMapper struct {
	*vsphereVolume
}

func (v vsphereBlockVolumeMapper) SetUpDevice() (string, error) {
	return "", nil
}

func (v vsphereBlockVolumeMapper) MapDevice(devicePath, globalMapPath, volumeMapPath, volumeMapName string, podUID types.UID) error {
	return util.MapBlockVolume(devicePath, globalMapPath, volumeMapPath, volumeMapName, podUID)
}

var _ volume.BlockVolumeUnmapper = &vsphereBlockVolumeUnmapper{}

type vsphereBlockVolumeUnmapper struct {
	*vsphereVolume
}

func (v *vsphereBlockVolumeUnmapper) TearDownDevice(mapPath, devicePath string) error {
	return nil
}

// GetGlobalMapPath returns global map path and error
// path: plugins/kubernetes.io/{PluginName}/volumeDevices/volumePath
func (v *vsphereVolume) GetGlobalMapPath(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}
	return filepath.Join(v.plugin.host.GetVolumeDevicePluginDir(vsphereVolumePluginName), string(volumeSource.VolumePath)), nil
}

func (v *vsphereVolume) GetPodDeviceMapPath() (string, string) {
	return v.plugin.host.GetPodVolumeDeviceDir(v.podUID, utilstrings.EscapeQualifiedName(vsphereVolumePluginName)), v.volName
}
