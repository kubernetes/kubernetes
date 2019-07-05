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
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/emptydir"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
)

func newTestHost(t *testing.T) (string, volume.VolumeHost) {
	tempDir, err := ioutil.TempDir("/tmp", "git_repo_test.")
	if err != nil {
		t.Fatalf("can't make a temp rootdir: %v", err)
	}
	return tempDir, volumetest.NewFakeVolumeHost(tempDir, nil, emptydir.ProbeVolumePlugins())
}

func TestCanSupport(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	tempDir, host := newTestHost(t)
	defer os.RemoveAll(tempDir)
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, host)

	plug, err := plugMgr.FindPluginByName("kubernetes.io/git-repo")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != "kubernetes.io/git-repo" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if !plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{GitRepo: &v1.GitRepoVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}
}

// Expected command
type expectedCommand struct {
	// The git command
	cmd []string
	// The dir of git command is executed
	dir string
}

func TestPlugin(t *testing.T) {
	gitUrl := "https://github.com/kubernetes/kubernetes.git"
	revision := "2a30ce65c5ab586b98916d83385c5983edd353a1"

	scenarios := []struct {
		name              string
		vol               *v1.Volume
		expecteds         []expectedCommand
		isExpectedFailure bool
	}{
		{
			name: "target-dir",
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitUrl,
						Revision:   revision,
						Directory:  "target_dir",
					},
				},
			},
			expecteds: []expectedCommand{
				{
					cmd: []string{"git", "clone", "--", gitUrl, "target_dir"},
					dir: "",
				},
				{
					cmd: []string{"git", "checkout", revision},
					dir: "/target_dir",
				},
				{
					cmd: []string{"git", "reset", "--hard"},
					dir: "/target_dir",
				},
			},
			isExpectedFailure: false,
		},
		{
			name: "target-dir-no-revision",
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitUrl,
						Directory:  "target_dir",
					},
				},
			},
			expecteds: []expectedCommand{
				{
					cmd: []string{"git", "clone", "--", gitUrl, "target_dir"},
					dir: "",
				},
			},
			isExpectedFailure: false,
		},
		{
			name: "only-git-clone",
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitUrl,
					},
				},
			},
			expecteds: []expectedCommand{
				{
					cmd: []string{"git", "clone", "--", gitUrl},
					dir: "",
				},
			},
			isExpectedFailure: false,
		},
		{
			name: "no-target-dir",
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitUrl,
						Revision:   revision,
						Directory:  "",
					},
				},
			},
			expecteds: []expectedCommand{
				{
					cmd: []string{"git", "clone", "--", gitUrl},
					dir: "",
				},
				{
					cmd: []string{"git", "checkout", revision},
					dir: "/kubernetes",
				},
				{
					cmd: []string{"git", "reset", "--hard"},
					dir: "/kubernetes",
				},
			},
			isExpectedFailure: false,
		},
		{
			name: "current-dir",
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitUrl,
						Revision:   revision,
						Directory:  ".",
					},
				},
			},
			expecteds: []expectedCommand{
				{
					cmd: []string{"git", "clone", "--", gitUrl, "."},
					dir: "",
				},
				{
					cmd: []string{"git", "checkout", revision},
					dir: "",
				},
				{
					cmd: []string{"git", "reset", "--hard"},
					dir: "",
				},
			},
			isExpectedFailure: false,
		},
		{
			name: "current-dir-mess",
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitUrl,
						Revision:   revision,
						Directory:  "./.",
					},
				},
			},
			expecteds: []expectedCommand{
				{
					cmd: []string{"git", "clone", "--", gitUrl, "./."},
					dir: "",
				},
				{
					cmd: []string{"git", "checkout", revision},
					dir: "",
				},
				{
					cmd: []string{"git", "reset", "--hard"},
					dir: "",
				},
			},
			isExpectedFailure: false,
		},
		{
			name: "invalid-repository",
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: "--foo",
					},
				},
			},
			isExpectedFailure: true,
		},
		{
			name: "invalid-revision",
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitUrl,
						Revision:   "--bar",
					},
				},
			},
			isExpectedFailure: true,
		},
		{
			name: "invalid-directory",
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitUrl,
						Directory:  "-b",
					},
				},
			},
			isExpectedFailure: true,
		},
	}

	for _, scenario := range scenarios {
		allErrs := doTestPlugin(scenario, t)
		if len(allErrs) == 0 && scenario.isExpectedFailure {
			t.Errorf("Unexpected success for scenario: %s", scenario.name)
		}
		if len(allErrs) > 0 && !scenario.isExpectedFailure {
			t.Errorf("Unexpected failure for scenario: %s - %+v", scenario.name, allErrs)
		}
	}

}

func doTestPlugin(scenario struct {
	name              string
	vol               *v1.Volume
	expecteds         []expectedCommand
	isExpectedFailure bool
}, t *testing.T) []error {
	allErrs := []error{}

	plugMgr := volume.VolumePluginMgr{}
	rootDir, host := newTestHost(t)
	defer os.RemoveAll(rootDir)
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, host)

	plug, err := plugMgr.FindPluginByName("kubernetes.io/git-repo")
	if err != nil {
		allErrs = append(allErrs,
			fmt.Errorf("Can't find the plugin by name"))
		return allErrs
	}
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, err := plug.NewMounter(volume.NewSpecFromVolume(scenario.vol), pod, volume.VolumeOptions{})

	if err != nil {
		allErrs = append(allErrs,
			fmt.Errorf("Failed to make a new Mounter: %v", err))
		return allErrs
	}
	if mounter == nil {
		allErrs = append(allErrs,
			fmt.Errorf("Got a nil Mounter"))
		return allErrs
	}

	path := mounter.GetPath()
	suffix := fmt.Sprintf("pods/poduid/volumes/kubernetes.io~git-repo/%v", scenario.vol.Name)
	if !strings.HasSuffix(path, suffix) {
		allErrs = append(allErrs,
			fmt.Errorf("Got unexpected path: %s", path))
		return allErrs
	}

	// Test setUp()
	setUpErrs := doTestSetUp(scenario, mounter)
	allErrs = append(allErrs, setUpErrs...)

	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			allErrs = append(allErrs,
				fmt.Errorf("SetUp() failed, volume path not created: %s", path))
			return allErrs
		} else {
			allErrs = append(allErrs,
				fmt.Errorf("SetUp() failed: %v", err))
			return allErrs
		}
	}

	// gitRepo volume should create its own empty wrapper path
	podWrapperMetadataDir := fmt.Sprintf("%v/pods/poduid/plugins/kubernetes.io~empty-dir/wrapped_%v", rootDir, scenario.vol.Name)

	if _, err := os.Stat(podWrapperMetadataDir); err != nil {
		if os.IsNotExist(err) {
			allErrs = append(allErrs,
				fmt.Errorf("SetUp() failed, empty-dir wrapper path is not created: %s", podWrapperMetadataDir))
		} else {
			allErrs = append(allErrs,
				fmt.Errorf("SetUp() failed: %v", err))
		}
	}

	unmounter, err := plug.NewUnmounter("vol1", types.UID("poduid"))
	if err != nil {
		allErrs = append(allErrs,
			fmt.Errorf("Failed to make a new Unmounter: %v", err))
		return allErrs
	}
	if unmounter == nil {
		allErrs = append(allErrs,
			fmt.Errorf("Got a nil Unmounter"))
		return allErrs
	}

	if err := unmounter.TearDown(); err != nil {
		allErrs = append(allErrs,
			fmt.Errorf("Expected success, got: %v", err))
		return allErrs
	}
	if _, err := os.Stat(path); err == nil {
		allErrs = append(allErrs,
			fmt.Errorf("TearDown() failed, volume path still exists: %s", path))
	} else if !os.IsNotExist(err) {
		allErrs = append(allErrs,
			fmt.Errorf("TearDown() failed: %v", err))
	}
	return allErrs
}

func doTestSetUp(scenario struct {
	name              string
	vol               *v1.Volume
	expecteds         []expectedCommand
	isExpectedFailure bool
}, mounter volume.Mounter) []error {
	expecteds := scenario.expecteds
	allErrs := []error{}

	// Construct combined outputs from expected commands
	var fakeOutputs []fakeexec.FakeCombinedOutputAction
	var fcmd fakeexec.FakeCmd
	for _, expected := range expecteds {
		if expected.cmd[1] == "clone" {
			fakeOutputs = append(fakeOutputs, func() ([]byte, error) {
				// git clone, it creates new dir/files
				os.MkdirAll(filepath.Join(fcmd.Dirs[0], expected.dir), 0750)
				return []byte{}, nil
			})
		} else {
			// git checkout || git reset, they create nothing
			fakeOutputs = append(fakeOutputs, func() ([]byte, error) {
				return []byte{}, nil
			})
		}
	}
	fcmd = fakeexec.FakeCmd{
		CombinedOutputScript: fakeOutputs,
	}

	// Construct fake exec outputs from fcmd
	var fakeAction []fakeexec.FakeCommandAction
	for i := 0; i < len(expecteds); i++ {
		fakeAction = append(fakeAction, func(cmd string, args ...string) exec.Cmd {
			return fakeexec.InitFakeCmd(&fcmd, cmd, args...)
		})

	}
	fake := fakeexec.FakeExec{
		CommandScript: fakeAction,
	}

	g := mounter.(*gitRepoVolumeMounter)
	g.exec = &fake

	g.SetUp(volume.MounterArgs{})

	if fake.CommandCalls != len(expecteds) {
		allErrs = append(allErrs,
			fmt.Errorf("unexpected command calls in scenario: expected %d, saw: %d", len(expecteds), fake.CommandCalls))
	}
	var expectedCmds [][]string
	for _, expected := range expecteds {
		expectedCmds = append(expectedCmds, expected.cmd)
	}
	if !reflect.DeepEqual(expectedCmds, fcmd.CombinedOutputLog) {
		allErrs = append(allErrs,
			fmt.Errorf("unexpected commands: %v, expected: %v", fcmd.CombinedOutputLog, expectedCmds))
	}

	var expectedPaths []string
	for _, expected := range expecteds {
		expectedPaths = append(expectedPaths, g.GetPath()+expected.dir)
	}
	if len(fcmd.Dirs) != len(expectedPaths) || !reflect.DeepEqual(expectedPaths, fcmd.Dirs) {
		allErrs = append(allErrs,
			fmt.Errorf("unexpected directories: %v, expected: %v", fcmd.Dirs, expectedPaths))
	}

	return allErrs
}
