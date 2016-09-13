/*
Copyright 2014 The Kubernetes Authors.

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

package remote_tar

import (
	"fmt"
	"path"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&remoteTarPlugin{nil}}
}

type remoteTarPlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &remoteTarPlugin{}

func wrappedVolumeSpec() volume.Spec {
	return volume.Spec{
		Volume: &api.Volume{VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
	}
}

const (
	remoteTarPluginName = "kubernetes.io/remote-tar"
)

func (plugin *remoteTarPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *remoteTarPlugin) GetPluginName() string {
	return remoteTarPluginName
}

func (plugin *remoteTarPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _ := getVolumeSource(spec)
	if volumeSource == nil {
		return "", fmt.Errorf("Spec does not reference a Remote tar volume type")
	}

	return fmt.Sprintf(
		"%v:%v",
		volumeSource.Source,
		volumeSource.Directory), nil
}

func (plugin *remoteTarPlugin) CanSupport(spec *volume.Spec) bool {
	return spec.Volume != nil && spec.Volume.RemoteTar != nil
}

func (plugin *remoteTarPlugin) RequiresRemount() bool {
	return false
}

func (plugin *remoteTarPlugin) NewMounter(spec *volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	return &remoteTarVolumeMounter{
		remoteTarVolume: &remoteTarVolume{
			volName: spec.Name(),
			podUID:  pod.UID,
			plugin:  plugin,
		},
		pod:    *pod,
		source: spec.Volume.RemoteTar.Source,
		target: spec.Volume.RemoteTar.Directory,
		exec:   exec.New(),
		opts:   opts,
	}, nil
}

func (plugin *remoteTarPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return &remoteTarVolumeUnmounter{
		&remoteTarVolume{
			volName: volName,
			podUID:  podUID,
			plugin:  plugin,
		},
	}, nil
}

func (plugin *remoteTarPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	remoteTarVolume := &api.Volume{
		Name: volumeName,
		VolumeSource: api.VolumeSource{
			RemoteTar: &api.RemoteTarVolumeSource{},
		},
	}
	return volume.NewSpecFromVolume(remoteTarVolume), nil
}

// remoteTar volumes are directories which are pre-filled from a tar archive.
// These do not persist beyond the lifetime of a pod.
type remoteTarVolume struct {
	volName string
	podUID  types.UID
	plugin  *remoteTarPlugin
	volume.MetricsNil
}

var _ volume.Volume = &remoteTarVolume{}

func (gr *remoteTarVolume) GetPath() string {
	name := remoteTarPluginName
	return gr.plugin.host.GetPodVolumeDir(gr.podUID, utilstrings.EscapeQualifiedNameForDisk(name), gr.volName)
}

// remoteTarVolumeMounter builds remote tar repo volumes.
type remoteTarVolumeMounter struct {
	*remoteTarVolume

	pod    api.Pod
	source string
	target string
	exec   exec.Interface
	opts   volume.VolumeOptions
}

var _ volume.Mounter = &remoteTarVolumeMounter{}

func (b *remoteTarVolumeMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        false,
		Managed:         true,
		SupportsSELinux: true, // xattr change should be okay, TODO: double check
	}
}

// SetUp creates new directory, download the tar then uncompress it.
func (b *remoteTarVolumeMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

// SetUp creates new directory, download the tar then uncompress it.
func (b *remoteTarVolumeMounter) SetUpAt(dir string, fsGroup *int64) error {
	if volumeutil.IsReady(b.getMetaDir()) {
		return nil
	}

	// Wrap EmptyDir, let it do the setup.
	wrapped, err := b.plugin.host.NewWrapperMounter(b.volName, wrappedVolumeSpec(), &b.pod, b.opts)
	if err != nil {
		return err
	}
	if err := wrapped.SetUpAt(dir, fsGroup); err != nil {
		return err
	}

	args := []string{"-O-", b.source, "|", "tar", "-xzC", b.target}

	if output, err := b.execCommand("wget", args, dir); err != nil {
		return fmt.Errorf("failed to exec '%s': %s : %v",
			strings.Join(args, " "), output, err)
	}

	volume.SetVolumeOwnership(b, fsGroup)

	volumeutil.SetReady(b.getMetaDir())
	return nil
}

func (b *remoteTarVolumeMounter) getMetaDir() string {
	return path.Join(b.plugin.host.GetPodPluginDir(b.podUID, utilstrings.EscapeQualifiedNameForDisk(remoteTarPluginName)), b.volName)
}

func (b *remoteTarVolumeMounter) execCommand(command string, args []string, dir string) ([]byte, error) {
	cmd := b.exec.Command(command, args...)
	cmd.SetDir(dir)
	return cmd.CombinedOutput()
}

// remoteTarVolumeUnmounter cleans tar volumes.
type remoteTarVolumeUnmounter struct {
	*remoteTarVolume
}

var _ volume.Unmounter = &remoteTarVolumeUnmounter{}

// TearDown simply deletes everything in the directory.
func (c *remoteTarVolumeUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

// TearDownAt simply deletes everything in the directory.
func (c *remoteTarVolumeUnmounter) TearDownAt(dir string) error {

	// Wrap EmptyDir, let it do the teardown.
	wrapped, err := c.plugin.host.NewWrapperUnmounter(c.volName, wrappedVolumeSpec(), c.podUID)
	if err != nil {
		return err
	}
	return wrapped.TearDownAt(dir)
}

func getVolumeSource(spec *volume.Spec) (*api.RemoteTarVolumeSource, bool) {
	var readOnly bool
	var volumeSource *api.RemoteTarVolumeSource

	if spec.Volume != nil && spec.Volume.RemoteTar != nil {
		volumeSource = spec.Volume.RemoteTar
		readOnly = spec.ReadOnly
	}

	return volumeSource, readOnly
}
