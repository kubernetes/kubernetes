/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package secret

import (
	"fmt"
	"os"
	"path"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	ioutil "k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

// ProbeVolumePlugin is the entry point for plugin detection in a package.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&secretPlugin{}}
}

const (
	secretPluginName = "kubernetes.io/secret"
)

// secretPlugin implements the VolumePlugin interface.
type secretPlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &secretPlugin{}

func (plugin *secretPlugin) Init(host volume.VolumeHost) {
	plugin.host = host
}

func (plugin *secretPlugin) Name() string {
	return secretPluginName
}

func (plugin *secretPlugin) CanSupport(spec *volume.Spec) bool {
	return spec.Volume != nil && spec.Volume.Secret != nil
}

func (plugin *secretPlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Builder, error) {
	return &secretVolumeBuilder{
		secretVolume: &secretVolume{spec.Name(), pod.UID, plugin, plugin.host.GetMounter(), plugin.host.GetWriter()},
		secretName:   spec.Volume.Secret.SecretName,
		pod:          *pod,
		opts:         &opts}, nil
}

func (plugin *secretPlugin) NewCleaner(volName string, podUID types.UID) (volume.Cleaner, error) {
	return &secretVolumeCleaner{&secretVolume{volName, podUID, plugin, plugin.host.GetMounter(), plugin.host.GetWriter()}}, nil
}

type secretVolume struct {
	volName string
	podUID  types.UID
	plugin  *secretPlugin
	mounter mount.Interface
	writer  ioutil.Writer
}

var _ volume.Volume = &secretVolume{}

func (sv *secretVolume) GetPath() string {
	return sv.plugin.host.GetPodVolumeDir(sv.podUID, util.EscapeQualifiedNameForDisk(secretPluginName), sv.volName)
}

// secretVolumeBuilder handles retrieving secrets from the API server
// and placing them into the volume on the host.
type secretVolumeBuilder struct {
	*secretVolume

	secretName string
	pod        api.Pod
	opts       *volume.VolumeOptions
}

var _ volume.Builder = &secretVolumeBuilder{}

func (b *secretVolumeBuilder) SetUp() error {
	return b.SetUpAt(b.GetPath())
}

// This is the spec for the volume that this plugin wraps.
var wrappedVolumeSpec = &volume.Spec{
	Volume: &api.Volume{VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{Medium: api.StorageMediumMemory}}},
}

func (b *secretVolumeBuilder) getMetaDir() string {
	return path.Join(b.plugin.host.GetPodPluginDir(b.podUID, util.EscapeQualifiedNameForDisk(secretPluginName)), b.volName)
}

func (b *secretVolumeBuilder) SetUpAt(dir string) error {
	notMnt, err := b.mounter.IsLikelyNotMountPoint(dir)
	// Getting an os.IsNotExist err from is a contingency; the directory
	// may not exist yet, in which case, setup should run.
	if err != nil && !os.IsNotExist(err) {
		return err
	}

	// If the plugin readiness file is present for this volume and
	// the setup dir is a mountpoint, this volume is already ready.
	if volumeutil.IsReady(b.getMetaDir()) && !notMnt {
		return nil
	}

	glog.V(3).Infof("Setting up volume %v for pod %v at %v", b.volName, b.pod.UID, dir)

	// Wrap EmptyDir, let it do the setup.
	wrapped, err := b.plugin.host.NewWrapperBuilder(wrappedVolumeSpec, &b.pod, *b.opts)
	if err != nil {
		return err
	}
	if err := wrapped.SetUpAt(dir); err != nil {
		return err
	}

	kubeClient := b.plugin.host.GetKubeClient()
	if kubeClient == nil {
		return fmt.Errorf("Cannot setup secret volume %v because kube client is not configured", b.volName)
	}

	secret, err := kubeClient.Secrets(b.pod.Namespace).Get(b.secretName)
	if err != nil {
		glog.Errorf("Couldn't get secret %v/%v", b.pod.Namespace, b.secretName)
		return err
	} else {
		totalBytes := totalSecretBytes(secret)
		glog.V(3).Infof("Received secret %v/%v containing (%v) pieces of data, %v total bytes",
			b.pod.Namespace,
			b.secretName,
			len(secret.Data),
			totalBytes)
	}

	for name, data := range secret.Data {
		hostFilePath := path.Join(dir, name)
		glog.V(3).Infof("Writing secret data %v/%v/%v (%v bytes) to host file %v", b.pod.Namespace, b.secretName, name, len(data), hostFilePath)
		err := b.writer.WriteFile(hostFilePath, data, 0444)
		if err != nil {
			glog.Errorf("Error writing secret data to host path: %v, %v", hostFilePath, err)
			return err
		}
	}

	volumeutil.SetReady(b.getMetaDir())

	return nil
}

func (sv *secretVolume) IsReadOnly() bool {
	return false
}

func totalSecretBytes(secret *api.Secret) int {
	totalSize := 0
	for _, bytes := range secret.Data {
		totalSize += len(bytes)
	}

	return totalSize
}

// secretVolumeCleaner handles cleaning up secret volumes.
type secretVolumeCleaner struct {
	*secretVolume
}

var _ volume.Cleaner = &secretVolumeCleaner{}

func (c *secretVolumeCleaner) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

func (c *secretVolumeCleaner) TearDownAt(dir string) error {
	glog.V(3).Infof("Tearing down volume %v for pod %v at %v", c.volName, c.podUID, dir)

	// Wrap EmptyDir, let it do the teardown.
	wrapped, err := c.plugin.host.NewWrapperCleaner(wrappedVolumeSpec, c.podUID)
	if err != nil {
		return err
	}
	return wrapped.TearDownAt(dir)
}
