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
	"io/ioutil"
	"os"
	"path"
	"reflect"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/empty_dir"
)

func newTestHost(t *testing.T) volume.VolumeHost {
	tempDir, err := ioutil.TempDir("/tmp", "git_repo_test.")
	if err != nil {
		t.Fatalf("can't make a temp rootdir: %v", err)
	}
	return volume.NewFakeVolumeHost(tempDir, nil, empty_dir.ProbeVolumePlugins())
}

func TestCanSupport(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), newTestHost(t))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/git-repo")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.Name() != "kubernetes.io/git-repo" {
		t.Errorf("Wrong name: %s", plug.Name())
	}
	if !plug.CanSupport(&volume.Spec{Name: "foo", VolumeSource: api.VolumeSource{GitRepo: &api.GitRepoVolumeSource{}}}) {
		t.Errorf("Expected true")
	}
}

func testSetUp(plug volume.VolumePlugin, builder volume.Builder, t *testing.T) {
	var fcmd exec.FakeCmd
	fcmd = exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// git clone
			func() ([]byte, error) {
				os.MkdirAll(path.Join(fcmd.Dirs[0], "kubernetes"), 0750)
				return []byte{}, nil
			},
			// git checkout
			func() ([]byte, error) { return []byte{}, nil },
			// git reset
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fake := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	g := builder.(*gitRepoVolumeBuilder)
	g.exec = &fake

	err := g.SetUp()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	expectedCmds := [][]string{
		{"git", "clone", g.source},
		{"git", "checkout", g.revision},
		{"git", "reset", "--hard"},
	}
	if fake.CommandCalls != len(expectedCmds) {
		t.Errorf("unexpected command calls: expected 3, saw: %d", fake.CommandCalls)
	}
	if !reflect.DeepEqual(expectedCmds, fcmd.CombinedOutputLog) {
		t.Errorf("unexpected commands: %v, expected: %v", fcmd.CombinedOutputLog, expectedCmds)
	}
	expectedDirs := []string{g.GetPath(), g.GetPath() + "/kubernetes", g.GetPath() + "/kubernetes"}
	if len(fcmd.Dirs) != 3 || !reflect.DeepEqual(expectedDirs, fcmd.Dirs) {
		t.Errorf("unexpected directories: %v, expected: %v", fcmd.Dirs, expectedDirs)
	}
}

func TestPlugin(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), newTestHost(t))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/git-repo")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &api.Volume{
		Name: "vol1",
		VolumeSource: api.VolumeSource{
			GitRepo: &api.GitRepoVolumeSource{
				Repository: "https://github.com/GoogleCloudPlatform/kubernetes.git",
				Revision:   "2a30ce65c5ab586b98916d83385c5983edd353a1",
			},
		},
	}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	builder, err := plug.NewBuilder(volume.NewSpecFromVolume(spec), pod, volume.VolumeOptions{""}, mount.New())
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}

	path := builder.GetPath()
	if !strings.HasSuffix(path, "pods/poduid/volumes/kubernetes.io~git-repo/vol1") {
		t.Errorf("Got unexpected path: %s", path)
	}

	testSetUp(plug, builder, t)
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", path)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}

	cleaner, err := plug.NewCleaner("vol1", types.UID("poduid"), mount.New())
	if err != nil {
		t.Errorf("Failed to make a new Cleaner: %v", err)
	}
	if cleaner == nil {
		t.Errorf("Got a nil Cleaner")
	}

	if err := cleaner.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", path)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}
}
