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
	ioutil "k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
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

func (plugin *secretPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *secretPlugin) Name() string {
	return secretPluginName
}

func (plugin *secretPlugin) CanSupport(spec *volume.Spec) bool {
	return spec.Volume != nil && spec.Volume.Secret != nil
}

func (plugin *secretPlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Builder, error) {
	return &secretVolumeBuilder{
		secretVolume: &secretVolume{spec.Name(), pod.UID, plugin, plugin.host.GetMounter(), plugin.host.GetWriter(), volume.MetricsNil{}},
		source:       spec.Volume.Secret,
		pod:          *pod,
		opts:         &opts}, nil
}

func (plugin *secretPlugin) NewCleaner(volName string, podUID types.UID) (volume.Cleaner, error) {
	return &secretVolumeCleaner{&secretVolume{volName, podUID, plugin, plugin.host.GetMounter(), plugin.host.GetWriter(), volume.MetricsNil{}}}, nil
}

type secretVolume struct {
	volName string
	podUID  types.UID
	plugin  *secretPlugin
	mounter mount.Interface
	writer  ioutil.Writer
	volume.MetricsNil
}

var _ volume.Volume = &secretVolume{}

func (sv *secretVolume) GetPath() string {
	return sv.plugin.host.GetPodVolumeDir(sv.podUID, strings.EscapeQualifiedNameForDisk(secretPluginName), sv.volName)
}

// secretVolumeBuilder handles retrieving secrets from the API server
// and placing them into the volume on the host.
type secretVolumeBuilder struct {
	*secretVolume

	source *api.SecretVolumeSource
	pod    api.Pod
	opts   *volume.VolumeOptions
}

var _ volume.Builder = &secretVolumeBuilder{}

func (sv *secretVolume) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:                    true,
		Managed:                     true,
		SupportsOwnershipManagement: true,
		SupportsSELinux:             true,
	}
}
func (b *secretVolumeBuilder) SetUp() error {
	return b.SetUpAt(b.GetPath())
}

// This is the spec for the volume that this plugin wraps.
var wrappedVolumeSpec = &volume.Spec{
	Volume: &api.Volume{VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{Medium: api.StorageMediumMemory}}},
}

func (b *secretVolumeBuilder) getMetaDir() string {
	return path.Join(b.plugin.host.GetPodPluginDir(b.podUID, strings.EscapeQualifiedNameForDisk(secretPluginName)), b.volName)
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

	if len(b.source.SecretName) != 0 {
		err := b.projectSingleSecretIntoVolume(dir)
		if err != nil {
			return err
		}
	} else {
		err := b.projectSecretFilesIntoVolume(dir)
		if err != nil {
			return err
		}
	}

	volumeutil.SetReady(b.getMetaDir())

	return nil
}

func (b *secretVolumeBuilder) getSecret(secretName string) (*api.Secret, error) {
	kubeClient := b.plugin.host.GetKubeClient()
	if kubeClient == nil {
		return nil, fmt.Errorf("Cannot setup secret volume %v because kube client is not configured", b.volName)
	}

	secret, err := kubeClient.Secrets(b.pod.Namespace).Get(secretName)
	if err != nil {
		return nil, err
	} else {
		totalBytes := totalSecretBytes(secret)
		glog.V(3).Infof("Received secret %v/%v containing (%v) pieces of data, %v total bytes",
			b.pod.Namespace,
			b.source.SecretName,
			len(secret.Data),
			totalBytes)
	}

	return secret, nil
}

func (b *secretVolumeBuilder) projectSingleSecretIntoVolume(dir string) error {
	secret, err := b.getSecret(b.source.SecretName)

	for name := range secret.Data {
		hostFilePath := path.Join(dir, name)
		err = b.projectSecretIntoFile(secret, name, hostFilePath)
		if err != nil {
			glog.Errorf("Error writing secret data to host path: %v, %v", hostFilePath, err)
			return err
		}
	}

	return nil
}

func (b *secretVolumeBuilder) projectSecretFilesIntoVolume(dir string) error {
	fetchedSecrets := map[string]*api.Secret{}
	var err error

	for _, secretFile := range b.source.Items {
		secretName := secretFile.Name
		secret, ok := fetchedSecrets[secretName]
		if !ok {
			secret2, err := b.getSecret(secretName)
			if err != nil {
				glog.Errorf("Couldn't get secret %v/%v: %v", b.pod.Namespace, secretName, err)
				return err
			}
			fetchedSecrets[secretName] = secret2
			secret = secret2
		}

		filePath := path.Join(dir, secretFile.Path)
		err = b.projectSecretIntoFile(secret, secretFile.Key, filePath)
		if err != nil {
			return err
		}
	}

	return nil
}

func (b *secretVolumeBuilder) projectSecretIntoFile(secret *api.Secret, key string, filePath string) error {
	data, ok := secret.Data[key]
	if !ok {
		return fmt.Errorf("Secret key %v/%v/data[%v] does not exist.", b.pod.Namespace, secret.Name, key)
	}

	targetDir := path.Dir(filePath)
	err := os.MkdirAll(targetDir, 0660)
	if err != nil {
		return fmt.Errorf("Couldn't construct path to file %v: %v", filePath, err)
	}

	glog.V(3).Infof("Writing secret data %v/%v/%v/data[%v] (%v bytes) to host file %v", b.pod.Namespace, secret.Name, key, len(data), filePath)
	err = b.writer.WriteFile(filePath, data, 0444)
	if err != nil {
		return err
	}

	return nil
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
