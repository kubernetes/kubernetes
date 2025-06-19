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

package portworx

import (
	"fmt"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
)

// ProbeVolumePlugins is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&portworxVolumePlugin{}}
}

type portworxVolumePlugin struct {
}

var _ volume.VolumePlugin = &portworxVolumePlugin{}
var _ volume.PersistentVolumePlugin = &portworxVolumePlugin{}
var _ volume.DeletableVolumePlugin = &portworxVolumePlugin{}
var _ volume.ProvisionableVolumePlugin = &portworxVolumePlugin{}
var _ volume.ExpandableVolumePlugin = &portworxVolumePlugin{}

const (
	portworxVolumePluginName = "kubernetes.io/portworx-volume"
)

func (plugin *portworxVolumePlugin) IsMigratedToCSI() bool {
	return utilfeature.DefaultFeatureGate.Enabled(features.CSIMigrationPortworx)
}

func (plugin *portworxVolumePlugin) Init(host volume.VolumeHost) error {
	return fmt.Errorf("%s plugin Init() is not supported in-tree", portworxVolumePluginName)
}

func (plugin *portworxVolumePlugin) GetPluginName() string {
	return portworxVolumePluginName
}

func (plugin *portworxVolumePlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	return "", fmt.Errorf("%s plugin GetVolumeName() is not supported in-tree", portworxVolumePluginName)
}

func (plugin *portworxVolumePlugin) CanSupport(spec *volume.Spec) bool {
	return false
}

func (plugin *portworxVolumePlugin) RequiresRemount(spec *volume.Spec) bool {
	return false
}

func (plugin *portworxVolumePlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
		v1.ReadWriteMany,
	}
}

func (plugin *portworxVolumePlugin) NewMounter(spec *volume.Spec, pod *v1.Pod) (volume.Mounter, error) {
	return nil, fmt.Errorf("%s plugin NewMounter() is not supported in-tree", portworxVolumePluginName)
}

func (plugin *portworxVolumePlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return nil, fmt.Errorf("%s plugin NewUnmounter() is not supported in-tree", portworxVolumePluginName)
}

func (plugin *portworxVolumePlugin) NewDeleter(logger klog.Logger, spec *volume.Spec) (volume.Deleter, error) {
	return nil, fmt.Errorf("%s plugin NewDeleter() is not supported in-tree", portworxVolumePluginName)
}

func (plugin *portworxVolumePlugin) NewProvisioner(logger klog.Logger, options volume.VolumeOptions) (volume.Provisioner, error) {
	return nil, fmt.Errorf("%s plugin NewProvisioner() is not supported in-tree", portworxVolumePluginName)
}

func (plugin *portworxVolumePlugin) RequiresFSResize() bool {
	return false
}

func (plugin *portworxVolumePlugin) ExpandVolumeDevice(
	_ *volume.Spec,
	_ resource.Quantity,
	_ resource.Quantity) (resource.Quantity, error) {
	return resource.Quantity{}, fmt.Errorf("%s plugin ExpandVolumeDevice() is not supported in-tree", portworxVolumePluginName)
}

func (plugin *portworxVolumePlugin) ConstructVolumeSpec(volumeName, mountPath string) (volume.ReconstructedVolume, error) {
	return volume.ReconstructedVolume{}, fmt.Errorf("%s plugin ConstructVolumeSpec() is not supported in-tree", portworxVolumePluginName)
}

func (plugin *portworxVolumePlugin) SupportsMountOption() bool {
	return false
}

func (plugin *portworxVolumePlugin) SupportsSELinuxContextMount(spec *volume.Spec) (bool, error) {
	return false, nil
}
