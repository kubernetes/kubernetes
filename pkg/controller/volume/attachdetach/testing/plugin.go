/*
Copyright 2024 The Kubernetes Authors.

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

package testing

import (
	"fmt"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
)

const TestPluginName = "kubernetes.io/testPlugin"

type TestPlugin struct {
	// SupportCSIVolume allows the plugin to support CSI volumes.
	// It does not mock the actual CSI volume operations.
	SupportCSIVolume  bool
	ErrorEncountered  bool
	attachedVolumeMap map[string][]string
	detachedVolumeMap map[string][]string
	pluginLock        *sync.RWMutex
}

func (plugin *TestPlugin) Init(host volume.VolumeHost) error {
	return nil
}

func (plugin *TestPlugin) GetPluginName() string {
	return TestPluginName
}

func (plugin *TestPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	plugin.pluginLock.Lock()
	defer plugin.pluginLock.Unlock()
	if spec == nil {
		plugin.ErrorEncountered = true
		return "", fmt.Errorf("GetVolumeName called with nil volume spec")
	}
	if spec.Volume != nil {
		return spec.Name(), nil
	} else if spec.PersistentVolume != nil {
		if spec.PersistentVolume.Spec.PersistentVolumeSource.GCEPersistentDisk != nil {
			return spec.PersistentVolume.Spec.PersistentVolumeSource.GCEPersistentDisk.PDName, nil
		} else if spec.PersistentVolume.Spec.PersistentVolumeSource.NFS != nil {
			return spec.PersistentVolume.Spec.PersistentVolumeSource.NFS.Server, nil
		} else if spec.PersistentVolume.Spec.PersistentVolumeSource.RBD != nil {
			return spec.PersistentVolume.Spec.PersistentVolumeSource.RBD.RBDImage, nil
		} else if spec.PersistentVolume.Spec.PersistentVolumeSource.CSI != nil {
			csi := spec.PersistentVolume.Spec.PersistentVolumeSource.CSI
			return fmt.Sprintf("%s^%s", csi.Driver, csi.VolumeHandle), nil
		}
		return "", fmt.Errorf("GetVolumeName called with unexpected PersistentVolume: %v", spec)
	} else {
		return "", nil
	}
}

func (plugin *TestPlugin) CanSupport(spec *volume.Spec) bool {
	plugin.pluginLock.Lock()
	defer plugin.pluginLock.Unlock()
	if spec == nil {
		plugin.ErrorEncountered = true
	} else {
		if spec.Volume != nil && spec.Volume.CSI != nil {
			return plugin.SupportCSIVolume
		}
		if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.CSI != nil {
			return plugin.SupportCSIVolume
		}
	}
	return true
}

func (plugin *TestPlugin) RequiresRemount(spec *volume.Spec) bool {
	return false
}

func (plugin *TestPlugin) NewMounter(spec *volume.Spec, podRef *v1.Pod) (volume.Mounter, error) {
	plugin.pluginLock.Lock()
	defer plugin.pluginLock.Unlock()
	if spec == nil {
		plugin.ErrorEncountered = true
	}
	return nil, nil
}

func (plugin *TestPlugin) NewUnmounter(name string, podUID types.UID) (volume.Unmounter, error) {
	return nil, nil
}

func (plugin *TestPlugin) ConstructVolumeSpec(volumeName, mountPath string) (volume.ReconstructedVolume, error) {
	fakeVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
				PDName:   "pdName",
				FSType:   "ext4",
				ReadOnly: false,
			},
		},
	}
	return volume.ReconstructedVolume{
		Spec: volume.NewSpecFromVolume(fakeVolume),
	}, nil
}

func (plugin *TestPlugin) NewAttacher() (volume.Attacher, error) {
	attacher := testPluginAttacher{
		ErrorEncountered:  &plugin.ErrorEncountered,
		attachedVolumeMap: plugin.attachedVolumeMap,
		pluginLock:        plugin.pluginLock,
	}
	return &attacher, nil
}

func (plugin *TestPlugin) NewDeviceMounter() (volume.DeviceMounter, error) {
	return plugin.NewAttacher()
}

func (plugin *TestPlugin) NewDetacher() (volume.Detacher, error) {
	detacher := testPluginDetacher{
		detachedVolumeMap: plugin.detachedVolumeMap,
		pluginLock:        plugin.pluginLock,
	}
	return &detacher, nil
}

func (plugin *TestPlugin) CanAttach(spec *volume.Spec) (bool, error) {
	return true, nil
}

func (plugin *TestPlugin) CanDeviceMount(spec *volume.Spec) (bool, error) {
	return true, nil
}

func (plugin *TestPlugin) NewDeviceUnmounter() (volume.DeviceUnmounter, error) {
	return plugin.NewDetacher()
}

func (plugin *TestPlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	return []string{}, nil
}

func (plugin *TestPlugin) SupportsMountOption() bool {
	return false
}

func (plugin *TestPlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func (plugin *TestPlugin) SupportsSELinuxContextMount(spec *volume.Spec) (bool, error) {
	return false, nil
}

func (plugin *TestPlugin) GetErrorEncountered() bool {
	plugin.pluginLock.RLock()
	defer plugin.pluginLock.RUnlock()
	return plugin.ErrorEncountered
}

func (plugin *TestPlugin) GetAttachedVolumes() map[string][]string {
	plugin.pluginLock.RLock()
	defer plugin.pluginLock.RUnlock()
	ret := make(map[string][]string)
	for nodeName, volumeList := range plugin.attachedVolumeMap {
		ret[nodeName] = make([]string, len(volumeList))
		copy(ret[nodeName], volumeList)
	}
	return ret
}

func (plugin *TestPlugin) GetDetachedVolumes() map[string][]string {
	plugin.pluginLock.RLock()
	defer plugin.pluginLock.RUnlock()
	ret := make(map[string][]string)
	for nodeName, volumeList := range plugin.detachedVolumeMap {
		ret[nodeName] = make([]string, len(volumeList))
		copy(ret[nodeName], volumeList)
	}
	return ret
}

func CreateTestPlugin(supportCSIVolume bool) []volume.VolumePlugin {
	attachedVolumes := make(map[string][]string)
	detachedVolumes := make(map[string][]string)
	return []volume.VolumePlugin{&TestPlugin{
		SupportCSIVolume:  supportCSIVolume,
		ErrorEncountered:  false,
		attachedVolumeMap: attachedVolumes,
		detachedVolumeMap: detachedVolumes,
		pluginLock:        &sync.RWMutex{},
	}}
}

// Attacher
type testPluginAttacher struct {
	ErrorEncountered  *bool
	attachedVolumeMap map[string][]string
	pluginLock        *sync.RWMutex
}

func (attacher *testPluginAttacher) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	attacher.pluginLock.Lock()
	defer attacher.pluginLock.Unlock()
	if spec == nil {
		*attacher.ErrorEncountered = true
		return "", fmt.Errorf("Attach called with nil volume spec")
	}
	attacher.attachedVolumeMap[string(nodeName)] = append(attacher.attachedVolumeMap[string(nodeName)], spec.Name())
	return spec.Name(), nil
}

func (attacher *testPluginAttacher) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	return nil, nil
}

func (attacher *testPluginAttacher) WaitForAttach(spec *volume.Spec, devicePath string, pod *v1.Pod, timeout time.Duration) (string, error) {
	attacher.pluginLock.Lock()
	defer attacher.pluginLock.Unlock()
	if spec == nil {
		*attacher.ErrorEncountered = true
		return "", fmt.Errorf("WaitForAttach called with nil volume spec")
	}
	fakePath := fmt.Sprintf("%s/%s", devicePath, spec.Name())
	return fakePath, nil
}

func (attacher *testPluginAttacher) GetDeviceMountPath(spec *volume.Spec) (string, error) {
	attacher.pluginLock.Lock()
	defer attacher.pluginLock.Unlock()
	if spec == nil {
		*attacher.ErrorEncountered = true
		return "", fmt.Errorf("GetDeviceMountPath called with nil volume spec")
	}
	return "", nil
}

func (attacher *testPluginAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string, _ volume.DeviceMounterArgs) error {
	attacher.pluginLock.Lock()
	defer attacher.pluginLock.Unlock()
	if spec == nil {
		*attacher.ErrorEncountered = true
		return fmt.Errorf("MountDevice called with nil volume spec")
	}
	return nil
}

// Detacher
type testPluginDetacher struct {
	detachedVolumeMap map[string][]string
	pluginLock        *sync.RWMutex
}

func (detacher *testPluginDetacher) Detach(volumeName string, nodeName types.NodeName) error {
	detacher.pluginLock.Lock()
	defer detacher.pluginLock.Unlock()
	detacher.detachedVolumeMap[string(nodeName)] = append(detacher.detachedVolumeMap[string(nodeName)], volumeName)
	return nil
}

func (detacher *testPluginDetacher) UnmountDevice(deviceMountPath string) error {
	return nil
}
