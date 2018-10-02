/*
Copyright 2015 The Kubernetes Authors.

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

package configmap

import (
	"fmt"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

// ProbeVolumePlugin is the entry point for plugin detection in a package.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&configMapPlugin{}}
}

const (
	configMapPluginName = "kubernetes.io/configmap"
)

// configMapPlugin implements the VolumePlugin interface.
type configMapPlugin struct {
	host         volume.VolumeHost
	getConfigMap func(namespace, name string) (*v1.ConfigMap, error)
}

var _ volume.VolumePlugin = &configMapPlugin{}

func (plugin *configMapPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	plugin.getConfigMap = host.GetConfigMapFunc()
	return nil
}

func (plugin *configMapPlugin) GetPluginName() string {
	return configMapPluginName
}

func (plugin *configMapPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _ := getVolumeSource(spec)
	if volumeSource == nil {
		return "", fmt.Errorf("Spec does not reference a ConfigMap volume type")
	}

	return fmt.Sprintf(
		"%v/%v",
		spec.Name(),
		volumeSource.Name), nil
}

func (plugin *configMapPlugin) CanSupport(spec *volume.Spec) bool {
	return spec.Volume != nil && spec.Volume.ConfigMap != nil
}

func (plugin *configMapPlugin) RequiresRemount() bool {
	return true
}

func (plugin *configMapPlugin) SupportsMountOption() bool {
	return false
}

func (plugin *configMapPlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func (plugin *configMapPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	return &configMapVolumeMounter{
		configMapVolume: &configMapVolume{
			spec.Name(),
			pod.UID,
			plugin,
			plugin.host.GetMounter(plugin.GetPluginName()),
			volume.MetricsNil{},
		},
		source:       *spec.Volume.ConfigMap,
		pod:          *pod,
		opts:         &opts,
		getConfigMap: plugin.getConfigMap,
	}, nil
}

func (plugin *configMapPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return &configMapVolumeUnmounter{
		&configMapVolume{
			volName,
			podUID,
			plugin,
			plugin.host.GetMounter(plugin.GetPluginName()),
			volume.MetricsNil{},
		},
	}, nil
}

func (plugin *configMapPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	configMapVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			ConfigMap: &v1.ConfigMapVolumeSource{},
		},
	}
	return volume.NewSpecFromVolume(configMapVolume), nil
}

type configMapVolume struct {
	volName string
	podUID  types.UID
	plugin  *configMapPlugin
	mounter mount.Interface
	volume.MetricsNil
}

var _ volume.Volume = &configMapVolume{}

func (sv *configMapVolume) GetPath() string {
	return sv.plugin.host.GetPodVolumeDir(sv.podUID, strings.EscapeQualifiedNameForDisk(configMapPluginName), sv.volName)
}

// configMapVolumeMounter handles retrieving secrets from the API server
// and placing them into the volume on the host.
type configMapVolumeMounter struct {
	*configMapVolume

	source       v1.ConfigMapVolumeSource
	pod          v1.Pod
	opts         *volume.VolumeOptions
	getConfigMap func(namespace, name string) (*v1.ConfigMap, error)
}

var _ volume.Mounter = &configMapVolumeMounter{}

func (sv *configMapVolume) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        true,
		Managed:         true,
		SupportsSELinux: true,
	}
}

func wrappedVolumeSpec() volume.Spec {
	// This is the spec for the volume that this plugin wraps.
	return volume.Spec{
		// This should be on a tmpfs instead of the local disk; the problem is
		// charging the memory for the tmpfs to the right cgroup.  We should make
		// this a tmpfs when we can do the accounting correctly.
		Volume: &v1.Volume{VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}},
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (b *configMapVolumeMounter) CanMount() error {
	return nil
}

func (b *configMapVolumeMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

func (b *configMapVolumeMounter) SetUpAt(dir string, fsGroup *int64) error {
	glog.V(3).Infof("Setting up volume %v for pod %v at %v", b.volName, b.pod.UID, dir)

	// Wrap EmptyDir, let it do the setup.
	wrapped, err := b.plugin.host.NewWrapperMounter(b.volName, wrappedVolumeSpec(), &b.pod, *b.opts)
	if err != nil {
		return err
	}

	optional := b.source.Optional != nil && *b.source.Optional
	configMap, err := b.getConfigMap(b.pod.Namespace, b.source.Name)
	if err != nil {
		if !(errors.IsNotFound(err) && optional) {
			glog.Errorf("Couldn't get configMap %v/%v: %v", b.pod.Namespace, b.source.Name, err)
			return err
		}
		configMap = &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: b.pod.Namespace,
				Name:      b.source.Name,
			},
		}
	}

	totalBytes := totalBytes(configMap)
	glog.V(3).Infof("Received configMap %v/%v containing (%v) pieces of data, %v total bytes",
		b.pod.Namespace,
		b.source.Name,
		len(configMap.Data)+len(configMap.BinaryData),
		totalBytes)

	payload, err := MakePayload(b.source.Items, configMap, b.source.DefaultMode, optional)
	if err != nil {
		return err
	}

	setupSuccess := false
	if err := wrapped.SetUpAt(dir, fsGroup); err != nil {
		return err
	}
	if err := volumeutil.MakeNestedMountpoints(b.volName, dir, b.pod); err != nil {
		return err
	}

	defer func() {
		// Clean up directories if setup fails
		if !setupSuccess {
			unmounter, unmountCreateErr := b.plugin.NewUnmounter(b.volName, b.podUID)
			if unmountCreateErr != nil {
				glog.Errorf("error cleaning up mount %s after failure. Create unmounter failed with %v", b.volName, unmountCreateErr)
				return
			}
			tearDownErr := unmounter.TearDown()
			if tearDownErr != nil {
				glog.Errorf("Error tearing down volume %s with : %v", b.volName, tearDownErr)
			}
		}
	}()

	writerContext := fmt.Sprintf("pod %v/%v volume %v", b.pod.Namespace, b.pod.Name, b.volName)
	writer, err := volumeutil.NewAtomicWriter(dir, writerContext)
	if err != nil {
		glog.Errorf("Error creating atomic writer: %v", err)
		return err
	}

	err = writer.Write(payload)
	if err != nil {
		glog.Errorf("Error writing payload to dir: %v", err)
		return err
	}

	err = volume.SetVolumeOwnership(b, fsGroup)
	if err != nil {
		glog.Errorf("Error applying volume ownership settings for group: %v", fsGroup)
		return err
	}
	setupSuccess = true
	return nil
}

// Note: this function is exported so that it can be called from the projection volume driver
func MakePayload(mappings []v1.KeyToPath, configMap *v1.ConfigMap, defaultMode *int32, optional bool) (map[string]volumeutil.FileProjection, error) {
	if defaultMode == nil {
		return nil, fmt.Errorf("No defaultMode used, not even the default value for it")
	}

	payload := make(map[string]volumeutil.FileProjection, (len(configMap.Data) + len(configMap.BinaryData)))
	var fileProjection volumeutil.FileProjection

	if len(mappings) == 0 {
		for name, data := range configMap.Data {
			fileProjection.Data = []byte(data)
			fileProjection.Mode = *defaultMode
			payload[name] = fileProjection
		}
		for name, data := range configMap.BinaryData {
			fileProjection.Data = data
			fileProjection.Mode = *defaultMode
			payload[name] = fileProjection
		}
	} else {
		for _, ktp := range mappings {
			if stringData, ok := configMap.Data[ktp.Key]; ok {
				fileProjection.Data = []byte(stringData)
			} else if binaryData, ok := configMap.BinaryData[ktp.Key]; ok {
				fileProjection.Data = binaryData
			} else {
				if optional {
					continue
				}
				return nil, fmt.Errorf("configmap references non-existent config key: %s", ktp.Key)
			}

			if ktp.Mode != nil {
				fileProjection.Mode = *ktp.Mode
			} else {
				fileProjection.Mode = *defaultMode
			}
			payload[ktp.Path] = fileProjection
		}
	}

	return payload, nil
}

func totalBytes(configMap *v1.ConfigMap) int {
	totalSize := 0
	for _, value := range configMap.Data {
		totalSize += len(value)
	}
	for _, value := range configMap.BinaryData {
		totalSize += len(value)
	}

	return totalSize
}

// configMapVolumeUnmounter handles cleaning up configMap volumes.
type configMapVolumeUnmounter struct {
	*configMapVolume
}

var _ volume.Unmounter = &configMapVolumeUnmounter{}

func (c *configMapVolumeUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

func (c *configMapVolumeUnmounter) TearDownAt(dir string) error {
	return volumeutil.UnmountViaEmptyDir(dir, c.plugin.host, c.volName, wrappedVolumeSpec(), c.podUID)
}

func getVolumeSource(spec *volume.Spec) (*v1.ConfigMapVolumeSource, bool) {
	var readOnly bool
	var volumeSource *v1.ConfigMapVolumeSource

	if spec.Volume != nil && spec.Volume.ConfigMap != nil {
		volumeSource = spec.Volume.ConfigMap
		readOnly = spec.ReadOnly
	}

	return volumeSource, readOnly
}
