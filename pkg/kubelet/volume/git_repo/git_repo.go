/*
Copyright 2014 Google Inc. All rights reserved.

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
	"os"
	"path"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
	"github.com/golang/glog"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.Plugin {
	return []volume.Plugin{&gitRepoPlugin{nil, false}, &gitRepoPlugin{nil, true}}
}

type gitRepoPlugin struct {
	host       volume.Host
	legacyMode bool // if set, plugin answers to the legacy name
}

var _ volume.Plugin = &gitRepoPlugin{}

const (
	gitRepoPluginName       = "kubernetes.io/git-repo"
	gitRepoPluginLegacyName = "git"
)

func (plugin *gitRepoPlugin) Init(host volume.Host) {
	plugin.host = host
}

func (plugin *gitRepoPlugin) Name() string {
	if plugin.legacyMode {
		return gitRepoPluginLegacyName
	}
	return gitRepoPluginName
}

func (plugin *gitRepoPlugin) CanSupport(spec *api.Volume) bool {
	if plugin.legacyMode {
		// Legacy mode instances can be cleaned up but not created anew.
		return false
	}

	if spec.Source.GitRepo != nil {
		return true
	}
	return false
}

func (plugin *gitRepoPlugin) NewBuilder(spec *api.Volume, podUID types.UID) (volume.Builder, error) {
	if plugin.legacyMode {
		// Legacy mode instances can be cleaned up but not created anew.
		return nil, fmt.Errorf("legacy mode: can not create new instances")
	}
	return &gitRepo{
		podUID:     podUID,
		volName:    spec.Name,
		source:     spec.Source.GitRepo.Repository,
		revision:   spec.Source.GitRepo.Revision,
		exec:       exec.New(),
		plugin:     plugin,
		legacyMode: false,
	}, nil
}

func (plugin *gitRepoPlugin) NewCleaner(volName string, podUID types.UID) (volume.Cleaner, error) {
	legacy := false
	if plugin.legacyMode {
		legacy = true
	}
	return &gitRepo{
		podUID:     podUID,
		volName:    volName,
		plugin:     plugin,
		legacyMode: legacy,
	}, nil
}

// gitRepo volumes are directories which are pre-filled from a git repository.
// These do not persist beyond the lifetime of a pod.
type gitRepo struct {
	volName    string
	podUID     types.UID
	source     string
	revision   string
	exec       exec.Interface
	plugin     *gitRepoPlugin
	legacyMode bool
}

// SetUp creates new directory and clones a git repo.
func (gr *gitRepo) SetUp() error {
	if gr.isReady() {
		return nil
	}
	if gr.legacyMode {
		return fmt.Errorf("legacy mode: can not create new instances")
	}

	volPath := gr.GetPath()
	if err := os.MkdirAll(volPath, 0750); err != nil {
		return err
	}

	if output, err := gr.execCommand("git", []string{"clone", gr.source}, gr.GetPath()); err != nil {
		return fmt.Errorf("failed to exec 'git clone %s': %s: %v", gr.source, output, err)
	}

	files, err := ioutil.ReadDir(gr.GetPath())
	if err != nil {
		return err
	}
	if len(files) != 1 {
		return fmt.Errorf("unexpected directory contents: %v", files)
	}
	if len(gr.revision) == 0 {
		// Done!
		gr.setReady()
		return nil
	}

	dir := path.Join(gr.GetPath(), files[0].Name())
	if output, err := gr.execCommand("git", []string{"checkout", gr.revision}, dir); err != nil {
		return fmt.Errorf("failed to exec 'git checkout %s': %s: %v", gr.revision, output, err)
	}
	if output, err := gr.execCommand("git", []string{"reset", "--hard"}, dir); err != nil {
		return fmt.Errorf("failed to exec 'git reset --hard': %s: %v", output, err)
	}

	gr.setReady()
	return nil
}

func (gr *gitRepo) getMetaDir() string {
	return path.Join(gr.plugin.host.GetPodPluginDir(gr.podUID, volume.EscapePluginName(gitRepoPluginName)), gr.volName)
}

func (gr *gitRepo) isReady() bool {
	metaDir := gr.getMetaDir()
	readyFile := path.Join(metaDir, "ready")
	s, err := os.Stat(readyFile)
	if err != nil {
		return false
	}
	if !s.Mode().IsRegular() {
		glog.Errorf("GitRepo ready-file is not a file: %s", readyFile)
		return false
	}
	return true
}

func (gr *gitRepo) setReady() {
	metaDir := gr.getMetaDir()
	if err := os.MkdirAll(metaDir, 0750); err != nil && !os.IsExist(err) {
		glog.Errorf("Can't mkdir %s: %v", metaDir, err)
		return
	}
	readyFile := path.Join(metaDir, "ready")
	file, err := os.Create(readyFile)
	if err != nil {
		glog.Errorf("Can't touch %s: %v", readyFile, err)
		return
	}
	file.Close()
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
	return gr.plugin.host.GetPodVolumeDir(gr.podUID, volume.EscapePluginName(name), gr.volName)
}

// TearDown simply deletes everything in the directory.
func (gr *gitRepo) TearDown() error {
	tmpDir, err := volume.RenameDirectory(gr.GetPath(), gr.volName+".deleting~")
	if err != nil {
		return err
	}
	err = os.RemoveAll(tmpDir)
	if err != nil {
		return err
	}
	return nil
}
