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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	volumeutil "github.com/GoogleCloudPlatform/kubernetes/pkg/volume/util"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&gitRepoPlugin{nil, false}, &gitRepoPlugin{nil, true}}
}

type gitRepoPlugin struct {
	host       volume.VolumeHost
	legacyMode bool // if set, plugin answers to the legacy name
}

var _ volume.VolumePlugin = &gitRepoPlugin{}

const (
	gitRepoPluginName       = "kubernetes.io/git-repo"
	gitRepoPluginLegacyName = "git"
)

func (plugin *gitRepoPlugin) Init(host volume.VolumeHost) {
	plugin.host = host
}

func (plugin *gitRepoPlugin) Name() string {
	if plugin.legacyMode {
		return gitRepoPluginLegacyName
	}
	return gitRepoPluginName
}

func (plugin *gitRepoPlugin) CanSupport(spec *volume.Spec) bool {
	if plugin.legacyMode {
		// Legacy mode instances can be cleaned up but not created anew.
		return false
	}

	return spec.VolumeSource.GitRepo != nil
}

func (plugin *gitRepoPlugin) NewBuilder(spec *volume.Spec, podRef *api.ObjectReference, opts volume.VolumeOptions, mounter mount.Interface) (volume.Builder, error) {
	if plugin.legacyMode {
		// Legacy mode instances can be cleaned up but not created anew.
		return nil, fmt.Errorf("legacy mode: can not create new instances")
	}
	return &gitRepo{
		podRef:     *podRef,
		volName:    spec.Name,
		source:     spec.VolumeSource.GitRepo.Repository,
		revision:   spec.VolumeSource.GitRepo.Revision,
		exec:       exec.New(),
		plugin:     plugin,
		legacyMode: false,
		opts:       opts,
		mounter:    mounter,
	}, nil
}

func (plugin *gitRepoPlugin) NewCleaner(volName string, podUID types.UID, mounter mount.Interface) (volume.Cleaner, error) {
	legacy := false
	if plugin.legacyMode {
		legacy = true
	}
	return &gitRepo{
		podRef:     api.ObjectReference{UID: podUID},
		volName:    volName,
		plugin:     plugin,
		legacyMode: legacy,
		mounter:    mounter,
	}, nil
}

// gitRepo volumes are directories which are pre-filled from a git repository.
// These do not persist beyond the lifetime of a pod.
type gitRepo struct {
	volName    string
	podRef     api.ObjectReference
	source     string
	revision   string
	exec       exec.Interface
	plugin     *gitRepoPlugin
	legacyMode bool
	opts       volume.VolumeOptions
	mounter    mount.Interface
}

// SetUp creates new directory and clones a git repo.
func (gr *gitRepo) SetUp() error {
	return gr.SetUpAt(gr.GetPath())
}

// This is the spec for the volume that this plugin wraps.
var wrappedVolumeSpec = &volume.Spec{
	Name:         "not-used",
	VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}},
}

// SetUpAt creates new directory and clones a git repo.
func (gr *gitRepo) SetUpAt(dir string) error {
	if volumeutil.IsReady(gr.getMetaDir()) {
		return nil
	}
	if gr.legacyMode {
		return fmt.Errorf("legacy mode: can not create new instances")
	}

	// Wrap EmptyDir, let it do the setup.
	wrapped, err := gr.plugin.host.NewWrapperBuilder(wrappedVolumeSpec, &gr.podRef, gr.opts, gr.mounter)
	if err != nil {
		return err
	}
	if err := wrapped.SetUpAt(dir); err != nil {
		return err
	}

	if output, err := gr.execCommand("git", []string{"clone", gr.source}, dir); err != nil {
		return fmt.Errorf("failed to exec 'git clone %s': %s: %v", gr.source, output, err)
	}

	files, err := ioutil.ReadDir(dir)
	if err != nil {
		return err
	}
	if len(files) != 1 {
		return fmt.Errorf("unexpected directory contents: %v", files)
	}
	if len(gr.revision) == 0 {
		// Done!
		volumeutil.SetReady(gr.getMetaDir())
		return nil
	}

	subdir := path.Join(dir, files[0].Name())
	if output, err := gr.execCommand("git", []string{"checkout", gr.revision}, subdir); err != nil {
		return fmt.Errorf("failed to exec 'git checkout %s': %s: %v", gr.revision, output, err)
	}
	if output, err := gr.execCommand("git", []string{"reset", "--hard"}, subdir); err != nil {
		return fmt.Errorf("failed to exec 'git reset --hard': %s: %v", output, err)
	}

	volumeutil.SetReady(gr.getMetaDir())
	return nil
}

func (gr *gitRepo) getMetaDir() string {
	return path.Join(gr.plugin.host.GetPodPluginDir(gr.podRef.UID, util.EscapeQualifiedNameForDisk(gitRepoPluginName)), gr.volName)
}

func (gr *gitRepo) execCommand(command string, args []string, dir string) ([]byte, error) {
	cmd := gr.exec.Command(command, args...)
	cmd.SetDir(dir)
	return cmd.CombinedOutput()
}

func (gr *gitRepo) GetPath() string {
	name := gitRepoPluginName
	if gr.legacyMode {
		name = gitRepoPluginLegacyName
	}
	return gr.plugin.host.GetPodVolumeDir(gr.podRef.UID, util.EscapeQualifiedNameForDisk(name), gr.volName)
}

// TearDown simply deletes everything in the directory.
func (gr *gitRepo) TearDown() error {
	return gr.TearDownAt(gr.GetPath())
}

// TearDownAt simply deletes everything in the directory.
func (gr *gitRepo) TearDownAt(dir string) error {
	// Wrap EmptyDir, let it do the teardown.
	wrapped, err := gr.plugin.host.NewWrapperCleaner(wrappedVolumeSpec, gr.podRef.UID, gr.mounter)
	if err != nil {
		return err
	}
	return wrapped.TearDownAt(dir)
}
