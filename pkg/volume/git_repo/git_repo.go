/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package git_repo

import (
	"fmt"
	"io/ioutil"
	"path"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&gitRepoPlugin{nil}}
}

type gitRepoPlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &gitRepoPlugin{}

const (
	gitRepoPluginName = "kubernetes.io/git-repo"
)

func (plugin *gitRepoPlugin) Init(host volume.VolumeHost) {
	plugin.host = host
}

func (plugin *gitRepoPlugin) Name() string {
	return gitRepoPluginName
}

func (plugin *gitRepoPlugin) CanSupport(spec *volume.Spec) bool {
	return spec.Volume != nil && spec.Volume.GitRepo != nil
}

func (plugin *gitRepoPlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Builder, error) {
	return &gitRepoVolumeBuilder{
		gitRepoVolume: &gitRepoVolume{
			volName: spec.Name(),
			podUID:  pod.UID,
			plugin:  plugin,
		},
		pod:      *pod,
		source:   spec.Volume.GitRepo.Repository,
		revision: spec.Volume.GitRepo.Revision,
		exec:     exec.New(),
		opts:     opts,
	}, nil
}

func (plugin *gitRepoPlugin) NewCleaner(volName string, podUID types.UID) (volume.Cleaner, error) {
	return &gitRepoVolumeCleaner{
		&gitRepoVolume{
			volName: volName,
			podUID:  podUID,
			plugin:  plugin,
		},
	}, nil
}

// gitRepo volumes are directories which are pre-filled from a git repository.
// These do not persist beyond the lifetime of a pod.
type gitRepoVolume struct {
	volName string
	podUID  types.UID
	plugin  *gitRepoPlugin
}

var _ volume.Volume = &gitRepoVolume{}

func (gr *gitRepoVolume) GetPath() string {
	name := gitRepoPluginName
	return gr.plugin.host.GetPodVolumeDir(gr.podUID, util.EscapeQualifiedNameForDisk(name), gr.volName)
}

// gitRepoVolumeBuilder builds git repo volumes.
type gitRepoVolumeBuilder struct {
	*gitRepoVolume

	pod      api.Pod
	source   string
	revision string
	exec     exec.Interface
	opts     volume.VolumeOptions
}

var _ volume.Builder = &gitRepoVolumeBuilder{}

// SetUp creates new directory and clones a git repo.
func (b *gitRepoVolumeBuilder) SetUp() error {
	return b.SetUpAt(b.GetPath())
}

func (b *gitRepoVolumeBuilder) IsReadOnly() bool {
	return false
}

// This is the spec for the volume that this plugin wraps.
var wrappedVolumeSpec = &volume.Spec{
	Volume: &api.Volume{VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
}

// SetUpAt creates new directory and clones a git repo.
func (b *gitRepoVolumeBuilder) SetUpAt(dir string) error {
	if volumeutil.IsReady(b.getMetaDir()) {
		return nil
	}

	// Wrap EmptyDir, let it do the setup.
	wrapped, err := b.plugin.host.NewWrapperBuilder(wrappedVolumeSpec, &b.pod, b.opts)
	if err != nil {
		return err
	}
	if err := wrapped.SetUpAt(dir); err != nil {
		return err
	}

	if output, err := b.execCommand("git", []string{"clone", b.source}, dir); err != nil {
		return fmt.Errorf("failed to exec 'git clone %s': %s: %v", b.source, output, err)
	}

	files, err := ioutil.ReadDir(dir)
	if err != nil {
		return err
	}
	if len(files) != 1 {
		return fmt.Errorf("unexpected directory contents: %v", files)
	}
	if len(b.revision) == 0 {
		// Done!
		volumeutil.SetReady(b.getMetaDir())
		return nil
	}

	subdir := path.Join(dir, files[0].Name())
	if output, err := b.execCommand("git", []string{"checkout", b.revision}, subdir); err != nil {
		return fmt.Errorf("failed to exec 'git checkout %s': %s: %v", b.revision, output, err)
	}
	if output, err := b.execCommand("git", []string{"reset", "--hard"}, subdir); err != nil {
		return fmt.Errorf("failed to exec 'git reset --hard': %s: %v", output, err)
	}

	volumeutil.SetReady(b.getMetaDir())
	return nil
}

func (b *gitRepoVolumeBuilder) getMetaDir() string {
	return path.Join(b.plugin.host.GetPodPluginDir(b.podUID, util.EscapeQualifiedNameForDisk(gitRepoPluginName)), b.volName)
}

func (b *gitRepoVolumeBuilder) execCommand(command string, args []string, dir string) ([]byte, error) {
	cmd := b.exec.Command(command, args...)
	cmd.SetDir(dir)
	return cmd.CombinedOutput()
}

// gitRepoVolumeCleaner cleans git repo volumes.
type gitRepoVolumeCleaner struct {
	*gitRepoVolume
}

var _ volume.Cleaner = &gitRepoVolumeCleaner{}

// TearDown simply deletes everything in the directory.
func (c *gitRepoVolumeCleaner) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

// TearDownAt simply deletes everything in the directory.
func (c *gitRepoVolumeCleaner) TearDownAt(dir string) error {
	// Wrap EmptyDir, let it do the teardown.
	wrapped, err := c.plugin.host.NewWrapperCleaner(wrappedVolumeSpec, c.podUID)
	if err != nil {
		return err
	}
	return wrapped.TearDownAt(dir)
}
