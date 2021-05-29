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
	"path/filepath"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/utils/exec"
	utilstrings "k8s.io/utils/strings"
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
		Volume: &v1.Volume{VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}},
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

func (plugin *gitRepoPlugin) RequiresRemount(spec *volume.Spec) bool {
	return false
}

func (plugin *gitRepoPlugin) SupportsMountOption() bool {
	return false
}

func (plugin *gitRepoPlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func (plugin *gitRepoPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	if err := validateVolume(spec.Volume.GitRepo); err != nil {
		return nil, err
	}

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
	gitVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			GitRepo: &v1.GitRepoVolumeSource{},
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
	return gr.plugin.host.GetPodVolumeDir(gr.podUID, utilstrings.EscapeQualifiedName(name), gr.volName)
}

// gitRepoVolumeMounter builds git repo volumes.
type gitRepoVolumeMounter struct {
	*gitRepoVolume

	pod      v1.Pod
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

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (b *gitRepoVolumeMounter) CanMount() error {
	return nil
}

// SetUp creates new directory and clones a git repo.
func (b *gitRepoVolumeMounter) SetUp(mounterArgs volume.MounterArgs) error {
	return b.SetUpAt(b.GetPath(), mounterArgs)
}

// SetUpAt creates new directory and clones a git repo.
func (b *gitRepoVolumeMounter) SetUpAt(dir string, mounterArgs volume.MounterArgs) error {
	if volumeutil.IsReady(b.getMetaDir()) {
		return nil
	}

	// Wrap EmptyDir, let it do the setup.
	wrapped, err := b.plugin.host.NewWrapperMounter(b.volName, wrappedVolumeSpec(), &b.pod, b.opts)
	if err != nil {
		return err
	}
	if err := wrapped.SetUpAt(dir, mounterArgs); err != nil {
		return err
	}

	args := []string{"clone", "--", b.source}

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
	case len(b.target) != 0 && filepath.Clean(b.target) == ".":
		// if target dir is '.', use the current dir
		subdir = filepath.Join(dir)
	case len(files) == 1:
		// if target is not '.', use the generated folder
		subdir = filepath.Join(dir, files[0].Name())
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

	volume.SetVolumeOwnership(b, mounterArgs.FsGroup, nil /*fsGroupChangePolicy*/, volumeutil.FSGroupCompleteHook(b.plugin, nil))

	volumeutil.SetReady(b.getMetaDir())
	return nil
}

func (b *gitRepoVolumeMounter) getMetaDir() string {
	return filepath.Join(b.plugin.host.GetPodPluginDir(b.podUID, utilstrings.EscapeQualifiedName(gitRepoPluginName)), b.volName)
}

func (b *gitRepoVolumeMounter) execCommand(command string, args []string, dir string) ([]byte, error) {
	cmd := b.exec.Command(command, args...)
	cmd.SetDir(dir)
	return cmd.CombinedOutput()
}

func validateVolume(src *v1.GitRepoVolumeSource) error {
	if err := validateNonFlagArgument(src.Repository, "repository"); err != nil {
		return err
	}
	if err := validateNonFlagArgument(src.Revision, "revision"); err != nil {
		return err
	}
	if err := validateNonFlagArgument(src.Directory, "directory"); err != nil {
		return err
	}
	return nil
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
	return volumeutil.UnmountViaEmptyDir(dir, c.plugin.host, c.volName, wrappedVolumeSpec(), c.podUID)
}

func getVolumeSource(spec *volume.Spec) (*v1.GitRepoVolumeSource, bool) {
	var readOnly bool
	var volumeSource *v1.GitRepoVolumeSource

	if spec.Volume != nil && spec.Volume.GitRepo != nil {
		volumeSource = spec.Volume.GitRepo
		readOnly = spec.ReadOnly
	}

	return volumeSource, readOnly
}

func validateNonFlagArgument(arg, argName string) error {
	if len(arg) > 0 && arg[0] == '-' {
		return fmt.Errorf("%q is an invalid value for %s", arg, argName)
	}
	return nil
}
