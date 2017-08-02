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

package digitalocean

import (
	"encoding/base64"
	"fmt"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/utils/exec"
)

const (
	doVolumePluginName = "kubernetes.io/do-volume"
	secretNamespace    = "kube-system"
	secretName         = "digitalocean"
	secretToken        = "token"
	secretRegion       = "region"
)

// ProbeVolumePlugins is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&doVolumePlugin{}}
}

type doVolumePlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &doVolumePlugin{}
var _ volume.PersistentVolumePlugin = &doVolumePlugin{}
var _ volume.DeletableVolumePlugin = &doVolumePlugin{}
var _ volume.ProvisionableVolumePlugin = &doVolumePlugin{}
var _ volume.AttachableVolumePlugin = &doVolumePlugin{}

// Init initializes the plugin
func (plugin *doVolumePlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

// GetPluginName returns the plugin name for Digital Ocean volumes
func (plugin *doVolumePlugin) GetPluginName() string {
	return doVolumePluginName
}

// GetVolumeName returns the ID to uniquely identifying the volume spec.
func (plugin *doVolumePlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return volumeSource.VolumeID, nil
}

// CanSupport returns a boolean that indicates if the volume is supported by this plugin
func (plugin *doVolumePlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.DOVolume != nil) ||
		(spec.Volume != nil && spec.Volume.DOVolume != nil)
}

// RequiresRemount returns false if this plugin doesn't need re-mount
func (plugin *doVolumePlugin) RequiresRemount() bool {
	return false
}

// SupportsMountOption returns false if volume plugin don't supports Mount options
func (plugin *doVolumePlugin) SupportsMountOption() bool {
	return false
}

// SupportsBulkVolumeVerification checks if volume plugin allows bulk volume verification
func (plugin *doVolumePlugin) SupportsBulkVolumeVerification() bool {
	return false
}

// GetAccessModes return access modes supported by the plugin
func (plugin *doVolumePlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
	}
}

// ConstructVolumeSpec constructs a volume spec based on name and path
func (plugin *doVolumePlugin) ConstructVolumeSpec(volName, mountPath string) (*volume.Spec, error) {
	mounter := plugin.host.GetMounter()
	pluginDir := plugin.host.GetPluginDir(plugin.GetPluginName())
	volumeID, err := mounter.GetDeviceNameFromMount(mountPath, pluginDir)
	if err != nil {
		return nil, err
	}
	doVolume := &v1.Volume{
		Name: volName,
		VolumeSource: v1.VolumeSource{
			DOVolume: &v1.DOVolumeSource{
				VolumeID: volumeID,
			},
		},
	}
	return volume.NewSpecFromVolume(doVolume), nil
}

// NewMounter creates a new volume.Mounter from an API specification
func (plugin *doVolumePlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	return plugin.newMounterInternal(spec, pod.UID, plugin.host.GetMounter())
}

func (plugin *doVolumePlugin) newMounterInternal(spec *volume.Spec, podUID types.UID, mounter mount.Interface) (volume.Mounter, error) {
	vol, err := getVolumeSource(spec)
	if err != nil {
		glog.Errorf("failed to extract a Digital Volume source from spec: %v", err)
		return nil, err
	}

	return &doVolumeMounter{
		doVolume: &doVolume{
			volName:         spec.Name(),
			podUID:          podUID,
			volumeID:        vol.VolumeID,
			mounter:         mounter,
			plugin:          plugin,
			MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, spec.Name(), plugin.host)),
		},
		fsType:      vol.FSType,
		readOnly:    vol.ReadOnly,
		diskMounter: &mount.SafeFormatAndMount{Interface: plugin.host.GetMounter(), Runner: exec.New()},
	}, nil
}

// NewUnmounter creates a new volume.Unmounter from recoverable state.
func (plugin *doVolumePlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, plugin.host.GetMounter()), nil
}

func (plugin *doVolumePlugin) newUnmounterInternal(volName string, podUID types.UID, mounter mount.Interface) volume.Unmounter {
	return &doVolumeUnmounter{
		&doVolume{
			volName:         volName,
			podUID:          podUID,
			mounter:         mounter,
			plugin:          plugin,
			MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, volName, plugin.host)),
		},
	}
}

// NewDeleter creates a new volume.Deleter which knows how to delete this resource
func (plugin *doVolumePlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	manager, err := plugin.createManager()
	if err != nil {
		glog.Errorf("deleter failed to create Digital Ocean manager: %v", err)
		return nil, err
	}
	return plugin.newDeleterInternal(spec, manager)
}

func (plugin *doVolumePlugin) newDeleterInternal(spec *volume.Spec, manager volManager) (volume.Deleter, error) {
	vol, err := getVolumeSource(spec)
	if err != nil {
		glog.Errorf("deleter failed to extract source from spec: %v", err)
		return nil, err
	}

	return &doVolumeDeleter{
		doVolume: &doVolume{
			volumeID: vol.VolumeID,
			plugin:   plugin,
			manager:  manager,
		},
	}, nil
}

func (plugin *doVolumePlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	manager, err := plugin.createManager()
	if err != nil {
		glog.Errorf("provisioner failed to create Digital Ocean manager: %v", err)
		return nil, err
	}
	return plugin.newProvisionerInternal(options, manager), nil
}

func (plugin *doVolumePlugin) newProvisionerInternal(options volume.VolumeOptions, manager volManager) volume.Provisioner {
	return &doVolumeProvisioner{
		doVolume: &doVolume{
			plugin:  plugin,
			manager: manager,
		},
		options: options,
	}
}

func (plugin *doVolumePlugin) newAttacherInternal(manager volManager) (volume.Attacher, error) {
	return &doVolumeAttacher{
		host:    plugin.host,
		manager: manager,
	}, nil
}

func (plugin *doVolumePlugin) NewAttacher() (volume.Attacher, error) {
	manager, err := plugin.createManager()
	if err != nil {
		return nil, err
	}
	return plugin.newAttacherInternal(manager)
}

func (plugin *doVolumePlugin) NewDetacher() (volume.Detacher, error) {
	manager, err := plugin.createManager()
	if err != nil {
		return nil, err
	}
	return plugin.newDetacherInternal(manager)
}

func (plugin *doVolumePlugin) newDetacherInternal(manager volManager) (volume.Detacher, error) {
	return &doVolumeDetacher{
		host:    plugin.host,
		mounter: plugin.host.GetMounter(),
		manager: manager,
	}, nil
}

func (plugin *doVolumePlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	mounter := plugin.host.GetMounter()
	return mount.GetMountRefs(mounter, deviceMountPath)
}

func (plugin *doVolumePlugin) getDOToken() (*doManagerConfig, error) {
	secretMap, err := util.GetSecretForPV(secretNamespace, secretName, doVolumePluginName, plugin.host.GetKubeClient())
	if err != nil {
		return nil, err
	}

	token, ok := secretMap[secretToken]
	if !ok {
		return nil, fmt.Errorf("Missing \"%s\" in secret %s/%s", secretToken, secretNamespace, secretName)
	}

	region, ok := secretMap[secretRegion]
	if !ok {
		return nil, fmt.Errorf("Missing \"%s\" in secret %s/%s", secretRegion, secretNamespace, secretName)
	}

	return &doManagerConfig{
		token:  base64.StdEncoding.EncodeToString([]byte(token)),
		region: base64.StdEncoding.EncodeToString([]byte(region)),
	}, nil
}

func (plugin *doVolumePlugin) createManager() (volManager, error) {
	config, err := plugin.getDOToken()
	if err != nil {
		return nil, err
	}
	return newDOManager(config)
}
