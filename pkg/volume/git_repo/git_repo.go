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

package git_repo

import (
	"fmt"
	"io/ioutil"
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
	return []volume.VolumePlugin{&gitRepoPlugin{nil}}
}

type gitRepoPlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &gitRepoPlugin{}

func wrappedVolumeSpec() volume.Spec {
	return volume.Spec{
		Volume: &api.Volume{VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
	}
}

const (
	gitRepoPluginName = "kubernetes.io/git-repo"
)

func (plugin *gitRepoPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *gitRepoPlugin) GetPluginName() string {
	return gitRepoPluginName
}

func (plugin *gitRepoPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _ := getVolumeSource(spec)
	if volumeSource == nil {
		return "", fmt.Errorf("Spec does not reference a Git repo volume type")
	}

	return fmt.Sprintf(
		"%v:%v:%v",
		volumeSource.Repository,
		volumeSource.Revision,
		volumeSource.Directory), nil
}

func (plugin *gitRepoPlugin) CanSupport(spec *volume.Spec) bool {
	return spec.Volume != nil && spec.Volume.GitRepo != nil
}

func (plugin *gitRepoPlugin) RequiresRemount() bool {
	return false
}

func (plugin *gitRepoPlugin) NewMounter(spec *volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	return &gitRepoVolumeMounter{
		gitRepoVolume: &gitRepoVolume{
			volName: spec.Name(),
			podUID:  pod.UID,
			plugin:  plugin,
		},
		pod:      *pod,
		source:   spec.Volume.GitRepo.Repository,
		revision: spec.Volume.GitRepo.Revision,
		target:   spec.Volume.GitRepo.Directory,
		exec:     exec.New(),
		opts:     opts,
	}, nil
}

func (plugin *gitRepoPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return &gitRepoVolumeUnmounter{
		&gitRepoVolume{
			volName: volName,
			podUID:  podUID,
			plugin:  plugin,
		},
	}, nil
}

func (plugin *gitRepoPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	gitVolume := &api.Volume{
		Name: volumeName,
		VolumeSource: api.VolumeSource{
			GitRepo: &api.GitRepoVolumeSource{},
		},
	}
	return volume.NewSpecFromVolume(gitVolume), nil
}

// gitRepo volumes are directories which are pre-filled from a git repository.
// These do not persist beyond the lifetime of a pod.
type gitRepoVolume struct {
	volName string
	podUID  types.UID
	plugin  *gitRepoPlugin
	volume.MetricsNil
}

var _ volume.Volume = &gitRepoVolume{}

func (gr *gitRepoVolume) GetPath() string {
	name := gitRepoPluginName
	return gr.plugin.host.GetPodVolumeDir(gr.podUID, utilstrings.EscapeQualifiedNameForDisk(name), gr.volName)
}

// gitRepoVolumeMounter builds git repo volumes.
type gitRepoVolumeMounter struct {
	*gitRepoVolume

	pod      api.Pod
	source   string
	revision string
	target   string
	exec     exec.Interface
	opts     volume.VolumeOptions
}

var _ volume.Mounter = &gitRepoVolumeMounter{}

func (b *gitRepoVolumeMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        false,
		Managed:         true,
		SupportsSELinux: true, // xattr change should be okay, TODO: double check
	}
}

// SetUp creates new directory and clones a git repo.
func (b *gitRepoVolumeMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

// SetUpAt creates new directory and clones a git repo.
func (b *gitRepoVolumeMounter) SetUpAt(dir string, fsGroup *int64) error {
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

	args := []string{"clone", b.source}

	if len(b.target) != 0 {
		args = append(args, b.target)
	}
	if output, err := b.execCommand("git", args, dir); err != nil {
		return fmt.Errorf("failed to exec 'git %s': %s: %v",
			strings.Join(args, " "), output, err)
	}

	files, err := ioutil.ReadDir(dir)
	if err != nil {
		return err
	}

	if len(b.revision) == 0 {
		// Done!
		volumeutil.SetReady(b.getMetaDir())
		return nil
	}

	var subdir string

	switch {
	case b.target == ".":
		// if target dir is '.', use the current dir
		subdir = path.Join(dir)
	case len(files) == 1:
		// if target is not '.', use the generated folder
		subdir = path.Join(dir, files[0].Name())
	default:
		// if target is not '.', but generated many files, it's wrong
		return fmt.Errorf("unexpected directory contents: %v", files)
	}

	if output, err := b.execCommand("git", []string{"checkout", b.revision}, subdir); err != nil {
		return fmt.Errorf("failed to exec 'git checkout %s': %s: %v", b.revision, output, err)
	}
	if output, err := b.execCommand("git", []string{"reset", "--hard"}, subdir); err != nil {
		return fmt.Errorf("failed to exec 'git reset --hard': %s: %v", output, err)
	}

	volume.SetVolumeOwnership(b, fsGroup)

	volumeutil.SetReady(b.getMetaDir())
	return nil
}

func (b *gitRepoVolumeMounter) getMetaDir() string {
	return path.Join(b.plugin.host.GetPodPluginDir(b.podUID, utilstrings.EscapeQualifiedNameForDisk(gitRepoPluginName)), b.volName)
}

func (b *gitRepoVolumeMounter) execCommand(command string, args []string, dir string) ([]byte, error) {
	cmd := b.exec.Command(command, args...)
	cmd.SetDir(dir)
	return cmd.CombinedOutput()
}

// gitRepoVolumeUnmounter cleans git repo volumes.
type gitRepoVolumeUnmounter struct {
	*gitRepoVolume
}

var _ volume.Unmounter = &gitRepoVolumeUnmounter{}

// TearDown simply deletes everything in the directory.
func (c *gitRepoVolumeUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

// TearDownAt simply deletes everything in the directory.
func (c *gitRepoVolumeUnmounter) TearDownAt(dir string) error {

	// Wrap EmptyDir, let it do the teardown.
	wrapped, err := c.plugin.host.NewWrapperUnmounter(c.volName, wrappedVolumeSpec(), c.podUID)
	if err != nil {
		return err
	}
	return wrapped.TearDownAt(dir)
}

func getVolumeSource(spec *volume.Spec) (*api.GitRepoVolumeSource, bool) {
	var readOnly bool
	var volumeSource *api.GitRepoVolumeSource

	if spec.Volume != nil && spec.Volume.GitRepo != nil {
		volumeSource = spec.Volume.GitRepo
		readOnly = spec.ReadOnly
	}

	return volumeSource, readOnly
}
