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

package awsebs

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"k8s.io/klog/v2"
	"k8s.io/utils/mount"
	utilstrings "k8s.io/utils/strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/volumepathhandler"
	"k8s.io/legacy-cloud-providers/aws"
)

var _ volume.BlockVolumePlugin = &awsElasticBlockStorePlugin{}

func (plugin *awsElasticBlockStorePlugin) ConstructBlockVolumeSpec(podUID types.UID, volumeName, mapPath string) (*volume.Spec, error) {
	pluginDir := plugin.host.GetVolumeDevicePluginDir(awsElasticBlockStorePluginName)
	blkutil := volumepathhandler.NewBlockVolumePathHandler()
	globalMapPathUUID, err := blkutil.FindGlobalMapPathUUIDFromPod(pluginDir, mapPath, podUID)
	if err != nil {
		return nil, err
	}
	klog.V(5).Infof("globalMapPathUUID: %s", globalMapPathUUID)

	globalMapPath := filepath.Dir(globalMapPathUUID)
	if len(globalMapPath) <= 1 {
		return nil, fmt.Errorf("failed to get volume plugin information from globalMapPathUUID: %v", globalMapPathUUID)
	}

	return plugin.getVolumeSpecFromGlobalMapPath(volumeName, globalMapPath)
}

func (plugin *awsElasticBlockStorePlugin) getVolumeSpecFromGlobalMapPath(volumeName string, globalMapPath string) (*volume.Spec, error) {
	// Get volume spec information from globalMapPath
	// globalMapPath example:
	//   plugins/kubernetes.io/{PluginName}/{DefaultKubeletVolumeDevicesDirName}/{volumeID}
	//   plugins/kubernetes.io/aws-ebs/volumeDevices/vol-XXXXXX
	pluginDir := plugin.host.GetVolumeDevicePluginDir(awsElasticBlockStorePluginName)
	if !strings.HasPrefix(globalMapPath, pluginDir) {
		return nil, fmt.Errorf("volume symlink %s is not in global plugin directory", globalMapPath)
	}
	fullVolumeID := strings.TrimPrefix(globalMapPath, pluginDir) // /vol-XXXXXX
	fullVolumeID = strings.TrimLeft(fullVolumeID, "/")           // vol-XXXXXX
	vID, err := formatVolumeID(fullVolumeID)
	if err != nil {
		return nil, fmt.Errorf("failed to get AWS volume id from map path %q: %v", globalMapPath, err)
	}

	block := v1.PersistentVolumeBlock
	return newAWSVolumeSpec(volumeName, vID, block), nil
}

// NewBlockVolumeMapper creates a new volume.BlockVolumeMapper from an API specification.
func (plugin *awsElasticBlockStorePlugin) NewBlockVolumeMapper(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.BlockVolumeMapper, error) {
	// If this is called via GenerateUnmapDeviceFunc(), pod is nil.
	// Pass empty string as dummy uid since uid isn't used in the case.
	var uid types.UID
	if pod != nil {
		uid = pod.UID
	}

	return plugin.newBlockVolumeMapperInternal(spec, uid, &AWSDiskUtil{}, plugin.host.GetMounter(plugin.GetPluginName()))
}

func (plugin *awsElasticBlockStorePlugin) newBlockVolumeMapperInternal(spec *volume.Spec, podUID types.UID, manager ebsManager, mounter mount.Interface) (volume.BlockVolumeMapper, error) {
	ebs, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	volumeID := aws.KubernetesVolumeID(ebs.VolumeID)
	partition := ""
	if ebs.Partition != 0 {
		partition = strconv.Itoa(int(ebs.Partition))
	}

	return &awsElasticBlockStoreMapper{
		awsElasticBlockStore: &awsElasticBlockStore{
			podUID:    podUID,
			volName:   spec.Name(),
			volumeID:  volumeID,
			partition: partition,
			manager:   manager,
			mounter:   mounter,
			plugin:    plugin,
		},
		readOnly: readOnly}, nil
}

func (plugin *awsElasticBlockStorePlugin) NewBlockVolumeUnmapper(volName string, podUID types.UID) (volume.BlockVolumeUnmapper, error) {
	return plugin.newUnmapperInternal(volName, podUID, &AWSDiskUtil{}, plugin.host.GetMounter(plugin.GetPluginName()))
}

func (plugin *awsElasticBlockStorePlugin) newUnmapperInternal(volName string, podUID types.UID, manager ebsManager, mounter mount.Interface) (volume.BlockVolumeUnmapper, error) {
	return &awsElasticBlockStoreUnmapper{
		awsElasticBlockStore: &awsElasticBlockStore{
			podUID:  podUID,
			volName: volName,
			manager: manager,
			mounter: mounter,
			plugin:  plugin,
		}}, nil
}

type awsElasticBlockStoreUnmapper struct {
	*awsElasticBlockStore
}

var _ volume.BlockVolumeUnmapper = &awsElasticBlockStoreUnmapper{}

type awsElasticBlockStoreMapper struct {
	*awsElasticBlockStore
	readOnly bool
}

var _ volume.BlockVolumeMapper = &awsElasticBlockStoreMapper{}

// GetGlobalMapPath returns global map path and error
// path: plugins/kubernetes.io/{PluginName}/volumeDevices/volumeID
//       plugins/kubernetes.io/aws-ebs/volumeDevices/vol-XXXXXX
func (ebs *awsElasticBlockStore) GetGlobalMapPath(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}
	return filepath.Join(ebs.plugin.host.GetVolumeDevicePluginDir(awsElasticBlockStorePluginName), string(volumeSource.VolumeID)), nil
}

// GetPodDeviceMapPath returns pod device map path and volume name
// path: pods/{podUid}/volumeDevices/kubernetes.io~aws
func (ebs *awsElasticBlockStore) GetPodDeviceMapPath() (string, string) {
	name := awsElasticBlockStorePluginName
	return ebs.plugin.host.GetPodVolumeDeviceDir(ebs.podUID, utilstrings.EscapeQualifiedName(name)), ebs.volName
}
