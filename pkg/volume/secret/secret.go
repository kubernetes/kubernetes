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

package secret

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

func wrappedVolumeSpec() volume.Spec {
	return volume.Spec{
		Volume: &api.Volume{VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{Medium: api.StorageMediumMemory}}},
	}
}

func getPath(uid types.UID, volName string, host volume.VolumeHost) string {
	return host.GetPodVolumeDir(uid, strings.EscapeQualifiedNameForDisk(secretPluginName), volName)
}

func (plugin *secretPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *secretPlugin) GetPluginName() string {
	return secretPluginName
}

func (plugin *secretPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _ := getVolumeSource(spec)
	if volumeSource == nil {
		return "", fmt.Errorf("Spec does not reference a Secret volume type")
	}

	return volumeSource.SecretName, nil
}

func (plugin *secretPlugin) CanSupport(spec *volume.Spec) bool {
	return spec.Volume != nil && spec.Volume.Secret != nil
}

func (plugin *secretPlugin) RequiresRemount() bool {
	return true
}

func (plugin *secretPlugin) NewMounter(spec *volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	return &secretVolumeMounter{
		secretVolume: &secretVolume{
			spec.Name(),
			pod.UID,
			plugin,
			plugin.host.GetMounter(),
			plugin.host.GetWriter(),
			volume.NewCachedMetrics(volume.NewMetricsDu(getPath(pod.UID, spec.Name(), plugin.host))),
		},
		source: *spec.Volume.Secret,
		pod:    *pod,
		opts:   &opts,
	}, nil
}

func (plugin *secretPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return &secretVolumeUnmounter{
		&secretVolume{
			volName,
			podUID,
			plugin,
			plugin.host.GetMounter(),
			plugin.host.GetWriter(),
			volume.NewCachedMetrics(volume.NewMetricsDu(getPath(podUID, volName, plugin.host))),
		},
	}, nil
}

func (plugin *secretPlugin) ConstructVolumeSpec(volName, mountPath string) (*volume.Spec, error) {
	secretVolume := &api.Volume{
		Name: volName,
		VolumeSource: api.VolumeSource{
			Secret: &api.SecretVolumeSource{
				SecretName: volName,
			},
		},
	}
	return volume.NewSpecFromVolume(secretVolume), nil
}

type secretVolume struct {
	volName string
	podUID  types.UID
	plugin  *secretPlugin
	mounter mount.Interface
	writer  ioutil.Writer
	volume.MetricsProvider
}

var _ volume.Volume = &secretVolume{}

func (sv *secretVolume) GetPath() string {
	return getPath(sv.podUID, sv.volName, sv.plugin.host)
}

// secretVolumeMounter handles retrieving secrets from the API server
// and placing them into the volume on the host.
type secretVolumeMounter struct {
	*secretVolume

	source api.SecretVolumeSource
	pod    api.Pod
	opts   *volume.VolumeOptions
}

var _ volume.Mounter = &secretVolumeMounter{}

func (sv *secretVolume) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        true,
		Managed:         true,
		SupportsSELinux: true,
	}
}
func (b *secretVolumeMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

func (b *secretVolumeMounter) SetUpAt(dir string, fsGroup *int64) error {
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
		return fmt.Errorf("Cannot setup secret volume %v because kube client is not configured", b.volName)
	}

	secret, err := kubeClient.Core().Secrets(b.pod.Namespace).Get(b.source.SecretName)
	if err != nil {
		glog.Errorf("Couldn't get secret %v/%v", b.pod.Namespace, b.source.SecretName)
		return err
	}

	totalBytes := totalSecretBytes(secret)
	glog.V(3).Infof("Received secret %v/%v containing (%v) pieces of data, %v total bytes",
		b.pod.Namespace,
		b.source.SecretName,
		len(secret.Data),
		totalBytes)

	payload, err := makePayload(b.source.Items, secret)
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

func makePayload(mappings []api.KeyToPath, secret *api.Secret) (map[string][]byte, error) {
	payload := make(map[string][]byte, len(secret.Data))

	if len(mappings) == 0 {
		for name, data := range secret.Data {
			payload[name] = []byte(data)
		}
	} else {
		for _, ktp := range mappings {
			content, ok := secret.Data[ktp.Key]
			if !ok {
				glog.Errorf("references non-existent secret key")
				return nil, fmt.Errorf("references non-existent secret key")
			}

			payload[ktp.Path] = []byte(content)
		}
	}

	return payload, nil
}

func totalSecretBytes(secret *api.Secret) int {
	totalSize := 0
	for _, bytes := range secret.Data {
		totalSize += len(bytes)
	}

	return totalSize
}

// secretVolumeUnmounter handles cleaning up secret volumes.
type secretVolumeUnmounter struct {
	*secretVolume
}

var _ volume.Unmounter = &secretVolumeUnmounter{}

func (c *secretVolumeUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

func (c *secretVolumeUnmounter) TearDownAt(dir string) error {
	glog.V(3).Infof("Tearing down volume %v for pod %v at %v", c.volName, c.podUID, dir)

	// Wrap EmptyDir, let it do the teardown.
	wrapped, err := c.plugin.host.NewWrapperUnmounter(c.volName, wrappedVolumeSpec(), c.podUID)
	if err != nil {
		return err
	}
	return wrapped.TearDownAt(dir)
}

func getVolumeSource(spec *volume.Spec) (*api.SecretVolumeSource, bool) {
	var readOnly bool
	var volumeSource *api.SecretVolumeSource

	if spec.Volume != nil && spec.Volume.Secret != nil {
		volumeSource = spec.Volume.Secret
		readOnly = spec.ReadOnly
	}

	return volumeSource, readOnly
}
