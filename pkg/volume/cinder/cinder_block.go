// +build !providerless

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

package cinder

import (
	"fmt"
	"path/filepath"

	"k8s.io/klog"
	"k8s.io/utils/mount"
	utilstrings "k8s.io/utils/strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/volumepathhandler"
)

var _ volume.VolumePlugin = &cinderPlugin{}
var _ volume.PersistentVolumePlugin = &cinderPlugin{}
var _ volume.BlockVolumePlugin = &cinderPlugin{}
var _ volume.DeletableVolumePlugin = &cinderPlugin{}
var _ volume.ProvisionableVolumePlugin = &cinderPlugin{}
var _ volume.ExpandableVolumePlugin = &cinderPlugin{}

func (plugin *cinderPlugin) ConstructBlockVolumeSpec(podUID types.UID, volumeName, mapPath string) (*volume.Spec, error) {
	pluginDir := plugin.host.GetVolumeDevicePluginDir(cinderVolumePluginName)
	blkutil := volumepathhandler.NewBlockVolumePathHandler()
	globalMapPathUUID, err := blkutil.FindGlobalMapPathUUIDFromPod(pluginDir, mapPath, podUID)
	if err != nil {
		return nil, err
	}
	klog.V(5).Infof("globalMapPathUUID: %v, err: %v", globalMapPathUUID, err)

	globalMapPath := filepath.Dir(globalMapPathUUID)
	if len(globalMapPath) <= 1 {
		return nil, fmt.Errorf("failed to get volume plugin information from globalMapPathUUID: %v", globalMapPathUUID)
	}

	return getVolumeSpecFromGlobalMapPath(volumeName, globalMapPath)
}

func getVolumeSpecFromGlobalMapPath(volumeName, globalMapPath string) (*volume.Spec, error) {
	// Get volume spec information from globalMapPath
	// globalMapPath example:
	//   plugins/kubernetes.io/{PluginName}/{DefaultKubeletVolumeDevicesDirName}/{volumeID}
	//   plugins/kubernetes.io/cinder/volumeDevices/vol-XXXXXX
	vID := filepath.Base(globalMapPath)
	if len(vID) <= 1 {
		return nil, fmt.Errorf("failed to get volumeID from global path=%s", globalMapPath)
	}
	block := v1.PersistentVolumeBlock
	cinderVolume := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: volumeName,
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				Cinder: &v1.CinderPersistentVolumeSource{
					VolumeID: vID,
				},
			},
			VolumeMode: &block,
		},
	}
	return volume.NewSpecFromPersistentVolume(cinderVolume, true), nil
}

// NewBlockVolumeMapper creates a new volume.BlockVolumeMapper from an API specification.
func (plugin *cinderPlugin) NewBlockVolumeMapper(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.BlockVolumeMapper, error) {
	// If this is called via GenerateUnmapDeviceFunc(), pod is nil.
	// Pass empty string as dummy uid since uid isn't used in the case.
	var uid types.UID
	if pod != nil {
		uid = pod.UID
	}

	return plugin.newBlockVolumeMapperInternal(spec, uid, &DiskUtil{}, plugin.host.GetMounter(plugin.GetPluginName()))
}

func (plugin *cinderPlugin) newBlockVolumeMapperInternal(spec *volume.Spec, podUID types.UID, manager cdManager, mounter mount.Interface) (volume.BlockVolumeMapper, error) {
	pdName, fsType, readOnly, err := getVolumeInfo(spec)
	if err != nil {
		return nil, err
	}

	return &cinderVolumeMapper{
		cinderVolume: &cinderVolume{
			podUID:  podUID,
			volName: spec.Name(),
			pdName:  pdName,
			fsType:  fsType,
			manager: manager,
			mounter: mounter,
			plugin:  plugin,
		},
		readOnly: readOnly}, nil
}

func (plugin *cinderPlugin) NewBlockVolumeUnmapper(volName string, podUID types.UID) (volume.BlockVolumeUnmapper, error) {
	return plugin.newUnmapperInternal(volName, podUID, &DiskUtil{}, plugin.host.GetMounter(plugin.GetPluginName()))
}

func (plugin *cinderPlugin) newUnmapperInternal(volName string, podUID types.UID, manager cdManager, mounter mount.Interface) (volume.BlockVolumeUnmapper, error) {
	return &cinderPluginUnmapper{
		cinderVolume: &cinderVolume{
			podUID:  podUID,
			volName: volName,
			manager: manager,
			mounter: mounter,
			plugin:  plugin,
		}}, nil
}

type cinderPluginUnmapper struct {
	*cinderVolume
}

var _ volume.BlockVolumeUnmapper = &cinderPluginUnmapper{}

type cinderVolumeMapper struct {
	*cinderVolume
	readOnly bool
}

var _ volume.BlockVolumeMapper = &cinderVolumeMapper{}

// GetGlobalMapPath returns global map path and error
// path: plugins/kubernetes.io/{PluginName}/volumeDevices/volumeID
//       plugins/kubernetes.io/cinder/volumeDevices/vol-XXXXXX
func (cd *cinderVolume) GetGlobalMapPath(spec *volume.Spec) (string, error) {
	pdName, _, _, err := getVolumeInfo(spec)
	if err != nil {
		return "", err
	}
	return filepath.Join(cd.plugin.host.GetVolumeDevicePluginDir(cinderVolumePluginName), pdName), nil
}

// GetPodDeviceMapPath returns pod device map path and volume name
// path: pods/{podUid}/volumeDevices/kubernetes.io~cinder
func (cd *cinderVolume) GetPodDeviceMapPath() (string, string) {
	name := cinderVolumePluginName
	return cd.plugin.host.GetPodVolumeDeviceDir(cd.podUID, utilstrings.EscapeQualifiedName(name)), cd.volName
}
