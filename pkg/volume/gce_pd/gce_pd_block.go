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

package gce_pd

import (
	"fmt"
	"path"
	"path/filepath"
	"strconv"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	kstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/volumepathhandler"
)

var _ volume.VolumePlugin = &gcePersistentDiskPlugin{}
var _ volume.PersistentVolumePlugin = &gcePersistentDiskPlugin{}
var _ volume.BlockVolumePlugin = &gcePersistentDiskPlugin{}
var _ volume.DeletableVolumePlugin = &gcePersistentDiskPlugin{}
var _ volume.ProvisionableVolumePlugin = &gcePersistentDiskPlugin{}
var _ volume.ExpandableVolumePlugin = &gcePersistentDiskPlugin{}

func (plugin *gcePersistentDiskPlugin) ConstructBlockVolumeSpec(podUID types.UID, volumeName, mapPath string) (*volume.Spec, error) {
	pluginDir := plugin.host.GetVolumeDevicePluginDir(gcePersistentDiskPluginName)
	blkutil := volumepathhandler.NewBlockVolumePathHandler()
	globalMapPathUUID, err := blkutil.FindGlobalMapPathUUIDFromPod(pluginDir, mapPath, podUID)
	if err != nil {
		return nil, err
	}
	glog.V(5).Infof("globalMapPathUUID: %v, err: %v", globalMapPathUUID, err)

	globalMapPath := filepath.Dir(globalMapPathUUID)
	if len(globalMapPath) <= 1 {
		return nil, fmt.Errorf("failed to get volume plugin information from globalMapPathUUID: %v", globalMapPathUUID)
	}

	return getVolumeSpecFromGlobalMapPath(globalMapPath)
}

func getVolumeSpecFromGlobalMapPath(globalMapPath string) (*volume.Spec, error) {
	// Get volume spec information from globalMapPath
	// globalMapPath example:
	//   plugins/kubernetes.io/{PluginName}/{DefaultKubeletVolumeDevicesDirName}/{volumeID}
	//   plugins/kubernetes.io/gce-pd/volumeDevices/vol-XXXXXX
	pdName := filepath.Base(globalMapPath)
	if len(pdName) <= 1 {
		return nil, fmt.Errorf("failed to get pd name from global path=%s", globalMapPath)
	}
	block := v1.PersistentVolumeBlock
	gceVolume := &v1.PersistentVolume{
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
					PDName: pdName,
				},
			},
			VolumeMode: &block,
		},
	}

	return volume.NewSpecFromPersistentVolume(gceVolume, true), nil
}

// NewBlockVolumeMapper creates a new volume.BlockVolumeMapper from an API specification.
func (plugin *gcePersistentDiskPlugin) NewBlockVolumeMapper(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.BlockVolumeMapper, error) {
	// If this is called via GenerateUnmapDeviceFunc(), pod is nil.
	// Pass empty string as dummy uid since uid isn't used in the case.
	var uid types.UID
	if pod != nil {
		uid = pod.UID
	}

	return plugin.newBlockVolumeMapperInternal(spec, uid, &GCEDiskUtil{}, plugin.host.GetMounter(plugin.GetPluginName()))
}

func (plugin *gcePersistentDiskPlugin) newBlockVolumeMapperInternal(spec *volume.Spec, podUID types.UID, manager pdManager, mounter mount.Interface) (volume.BlockVolumeMapper, error) {
	volumeSource, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}
	pdName := volumeSource.PDName
	partition := ""
	if volumeSource.Partition != 0 {
		partition = strconv.Itoa(int(volumeSource.Partition))
	}

	return &gcePersistentDiskMapper{
		gcePersistentDisk: &gcePersistentDisk{
			volName:   spec.Name(),
			podUID:    podUID,
			pdName:    pdName,
			partition: partition,
			manager:   manager,
			mounter:   mounter,
			plugin:    plugin,
		},
		readOnly: readOnly}, nil
}

func (plugin *gcePersistentDiskPlugin) NewBlockVolumeUnmapper(volName string, podUID types.UID) (volume.BlockVolumeUnmapper, error) {
	return plugin.newUnmapperInternal(volName, podUID, &GCEDiskUtil{})
}

func (plugin *gcePersistentDiskPlugin) newUnmapperInternal(volName string, podUID types.UID, manager pdManager) (volume.BlockVolumeUnmapper, error) {
	return &gcePersistentDiskUnmapper{
		gcePersistentDisk: &gcePersistentDisk{
			volName: volName,
			podUID:  podUID,
			pdName:  volName,
			manager: manager,
			plugin:  plugin,
		}}, nil
}

func (c *gcePersistentDiskUnmapper) TearDownDevice(mapPath, devicePath string) error {
	return nil
}

type gcePersistentDiskUnmapper struct {
	*gcePersistentDisk
}

var _ volume.BlockVolumeUnmapper = &gcePersistentDiskUnmapper{}

type gcePersistentDiskMapper struct {
	*gcePersistentDisk
	readOnly bool
}

var _ volume.BlockVolumeMapper = &gcePersistentDiskMapper{}

func (b *gcePersistentDiskMapper) SetUpDevice() (string, error) {
	return "", nil
}

func (b *gcePersistentDiskMapper) MapDevice(devicePath, globalMapPath, volumeMapPath, volumeMapName string, podUID types.UID) error {
	return util.MapBlockVolume(devicePath, globalMapPath, volumeMapPath, volumeMapName, podUID)
}

// GetGlobalMapPath returns global map path and error
// path: plugins/kubernetes.io/{PluginName}/volumeDevices/pdName
func (pd *gcePersistentDisk) GetGlobalMapPath(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}
	return path.Join(pd.plugin.host.GetVolumeDevicePluginDir(gcePersistentDiskPluginName), string(volumeSource.PDName)), nil
}

// GetPodDeviceMapPath returns pod device map path and volume name
// path: pods/{podUid}/volumeDevices/kubernetes.io~aws
func (pd *gcePersistentDisk) GetPodDeviceMapPath() (string, string) {
	name := gcePersistentDiskPluginName
	return pd.plugin.host.GetPodVolumeDeviceDir(pd.podUID, kstrings.EscapeQualifiedNameForDisk(name)), pd.volName
}
