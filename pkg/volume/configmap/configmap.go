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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	ioutil "k8s.io/kubernetes/pkg/util/io"
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
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &configMapPlugin{}

func (plugin *configMapPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
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

func (plugin *configMapPlugin) NewMounter(spec *volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	return &configMapVolumeMounter{
		configMapVolume: &configMapVolume{spec.Name(), pod.UID, plugin, plugin.host.GetMounter(), plugin.host.GetWriter(), volume.MetricsNil{}},
		source:          *spec.Volume.ConfigMap,
		pod:             *pod,
		opts:            &opts}, nil
}

func (plugin *configMapPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return &configMapVolumeUnmounter{&configMapVolume{volName, podUID, plugin, plugin.host.GetMounter(), plugin.host.GetWriter(), volume.MetricsNil{}}}, nil
}

func (plugin *configMapPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	configMapVolume := &api.Volume{
		Name: volumeName,
		VolumeSource: api.VolumeSource{
			ConfigMap: &api.ConfigMapVolumeSource{},
		},
	}
	return volume.NewSpecFromVolume(configMapVolume), nil
}

type configMapVolume struct {
	volName string
	podUID  types.UID
	plugin  *configMapPlugin
	mounter mount.Interface
	writer  ioutil.Writer
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

	source api.ConfigMapVolumeSource
	pod    api.Pod
	opts   *volume.VolumeOptions
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
		Volume: &api.Volume{VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
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
	if err := wrapped.SetUpAt(dir, fsGroup); err != nil {
		return err
	}

	kubeClient := b.plugin.host.GetKubeClient()
	if kubeClient == nil {
		return fmt.Errorf("Cannot setup configMap volume %v because kube client is not configured", b.volName)
	}

	configMap, err := kubeClient.Core().ConfigMaps(b.pod.Namespace).Get(b.source.Name)
	if err != nil {
		glog.Errorf("Couldn't get configMap %v/%v: %v", b.pod.Namespace, b.source.Name, err)
		return err
	}

	totalBytes := totalBytes(configMap)
	glog.V(3).Infof("Received configMap %v/%v containing (%v) pieces of data, %v total bytes",
		b.pod.Namespace,
		b.source.Name,
		len(configMap.Data),
		totalBytes)

	payload, err := makePayload(b.source.Items, configMap, b.source.DefaultMode)
	if err != nil {
		return err
	}

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

	return nil
}

func makePayload(mappings []api.KeyToPath, configMap *api.ConfigMap, defaultMode *int32) (map[string]volumeutil.FileProjection, error) {
	if defaultMode == nil {
		return nil, fmt.Errorf("No defaultMode used, not even the default value for it")
	}

	payload := make(map[string]volumeutil.FileProjection, len(configMap.Data))
	var fileProjection volumeutil.FileProjection

	if len(mappings) == 0 {
		for name, data := range configMap.Data {
			fileProjection.Data = []byte(data)
			fileProjection.Mode = *defaultMode
			payload[name] = fileProjection
		}
	} else {
		for _, ktp := range mappings {
			content, ok := configMap.Data[ktp.Key]
			if !ok {
				err_msg := "references non-existent config key"
				glog.Errorf(err_msg)
				return nil, fmt.Errorf(err_msg)
			}

			fileProjection.Data = []byte(content)
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

func totalBytes(configMap *api.ConfigMap) int {
	totalSize := 0
	for _, value := range configMap.Data {
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
	return volume.UnmountViaEmptyDir(dir, c.plugin.host, c.volName, wrappedVolumeSpec(), c.podUID)
}

func getVolumeSource(spec *volume.Spec) (*api.ConfigMapVolumeSource, bool) {
	var readOnly bool
	var volumeSource *api.ConfigMapVolumeSource

	if spec.Volume != nil && spec.Volume.ConfigMap != nil {
		volumeSource = spec.Volume.ConfigMap
		readOnly = spec.ReadOnly
	}

	return volumeSource, readOnly
}
