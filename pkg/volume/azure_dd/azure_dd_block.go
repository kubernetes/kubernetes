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

package azure_dd

import (
	"fmt"
	"path/filepath"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	kstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/volumepathhandler"
)

var _ volume.VolumePlugin = &azureDataDiskPlugin{}
var _ volume.PersistentVolumePlugin = &azureDataDiskPlugin{}
var _ volume.BlockVolumePlugin = &azureDataDiskPlugin{}
var _ volume.DeletableVolumePlugin = &azureDataDiskPlugin{}
var _ volume.ProvisionableVolumePlugin = &azureDataDiskPlugin{}

func (plugin *azureDataDiskPlugin) ConstructBlockVolumeSpec(podUID types.UID, volumeName, mapPath string) (*volume.Spec, error) {
	pluginDir := plugin.host.GetVolumeDevicePluginDir(azureDataDiskPluginName)
	blkutil := volumepathhandler.NewBlockVolumePathHandler()
	globalMapPathUUID, err := blkutil.FindGlobalMapPathUUIDFromPod(pluginDir, mapPath, podUID)
	if err != nil {
		return nil, err
	}
	glog.V(5).Infof("constructing block volume spec from globalMapPathUUID: %s", globalMapPathUUID)

	globalMapPath := filepath.Dir(globalMapPathUUID)
	if len(globalMapPath) <= 1 {
		return nil, fmt.Errorf("failed to get volume plugin information from globalMapPathUUID: %v", globalMapPathUUID)
	}

	return getVolumeSpecFromGlobalMapPath(globalMapPath, volumeName)
}

func getVolumeSpecFromGlobalMapPath(globalMapPath, volumeName string) (*volume.Spec, error) {
	// Get volume spec information from globalMapPath
	// globalMapPath example:
	//   plugins/kubernetes.io/{PluginName}/{DefaultKubeletVolumeDevicesDirName}/{volumeID}
	//   plugins/kubernetes.io/azure-disk/volumeDevices/vol-XXXXXX
	diskName := filepath.Base(globalMapPath)
	if len(diskName) <= 1 {
		return nil, fmt.Errorf("failed to get diskName from global path=%s", globalMapPath)
	}
	glog.V(5).Infof("got diskName(%s) from globalMapPath: %s", globalMapPath, diskName)
	block := v1.PersistentVolumeBlock
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: volumeName,
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				AzureDisk: &v1.AzureDiskVolumeSource{
					DiskName: diskName,
				},
			},
			VolumeMode: &block,
		},
	}

	return volume.NewSpecFromPersistentVolume(pv, true), nil
}

// NewBlockVolumeMapper creates a new volume.BlockVolumeMapper from an API specification.
func (plugin *azureDataDiskPlugin) NewBlockVolumeMapper(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.BlockVolumeMapper, error) {
	// If this is called via GenerateUnmapDeviceFunc(), pod is nil.
	// Pass empty string as dummy uid since uid isn't used in the case.
	var uid types.UID
	if pod != nil {
		uid = pod.UID
	}

	return plugin.newBlockVolumeMapperInternal(spec, uid, plugin.host.GetMounter(plugin.GetPluginName()))
}

func (plugin *azureDataDiskPlugin) newBlockVolumeMapperInternal(spec *volume.Spec, podUID types.UID, mounter mount.Interface) (volume.BlockVolumeMapper, error) {
	volumeSource, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	disk := makeDataDisk(spec.Name(), podUID, volumeSource.DiskName, plugin.host, plugin)

	return &azureDataDiskMapper{
		dataDisk: disk,
		readOnly: readOnly,
	}, nil
}

func (plugin *azureDataDiskPlugin) NewBlockVolumeUnmapper(volName string, podUID types.UID) (volume.BlockVolumeUnmapper, error) {
	return plugin.newUnmapperInternal(volName, podUID, plugin.host.GetMounter(plugin.GetPluginName()))
}

func (plugin *azureDataDiskPlugin) newUnmapperInternal(volName string, podUID types.UID, mounter mount.Interface) (volume.BlockVolumeUnmapper, error) {
	disk := makeDataDisk(volName, podUID, "", plugin.host, plugin)
	return &azureDataDiskUnmapper{dataDisk: disk}, nil
}

func (c *azureDataDiskUnmapper) TearDownDevice(mapPath, devicePath string) error {
	return nil
}

type azureDataDiskUnmapper struct {
	*dataDisk
}

var _ volume.BlockVolumeUnmapper = &azureDataDiskUnmapper{}

type azureDataDiskMapper struct {
	*dataDisk
	readOnly bool
}

var _ volume.BlockVolumeMapper = &azureDataDiskMapper{}

func (b *azureDataDiskMapper) SetUpDevice() (string, error) {
	return "", nil
}

func (b *azureDataDiskMapper) MapDevice(devicePath, globalMapPath, volumeMapPath, volumeMapName string, podUID types.UID) error {
	return util.MapBlockVolume(devicePath, globalMapPath, volumeMapPath, volumeMapName, podUID)
}

// GetGlobalMapPath returns global map path and error
// path: plugins/kubernetes.io/{PluginName}/volumeDevices/volumeID
//       plugins/kubernetes.io/azure-disk/volumeDevices/vol-XXXXXX
func (disk *dataDisk) GetGlobalMapPath(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}
	return filepath.Join(disk.plugin.host.GetVolumeDevicePluginDir(azureDataDiskPluginName), string(volumeSource.DiskName)), nil
}

// GetPodDeviceMapPath returns pod device map path and volume name
// path: pods/{podUid}/volumeDevices/kubernetes.io~azure
func (disk *dataDisk) GetPodDeviceMapPath() (string, string) {
	name := azureDataDiskPluginName
	return disk.plugin.host.GetPodVolumeDeviceDir(disk.podUID, kstrings.EscapeQualifiedNameForDisk(name)), disk.volumeName
}
