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
	"path"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/emptydir"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
)

func newTestHost(t *testing.T) (string, volume.VolumeHost) {
	tempDir, err := ioutil.TempDir("", "git_repo_test.")
	if err != nil {
		t.Fatalf("can't make a temp rootdir: %v", err)
	}
	return tempDir, volumetest.NewFakeVolumeHost(t, tempDir, nil, emptydir.ProbeVolumePlugins())
}

func TestCanSupport(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	tempDir, host := newTestHost(t)
	defer os.RemoveAll(tempDir)
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, host)

	plug, err := plugMgr.FindPluginByName("kubernetes.io/git-repo")
	if err != nil {
		t.Fatal("Can't find the plugin by name")
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

type scenario struct {
	name                  string
	vol                   *v1.Volume
	expecteds             []expectedCommand
	isExpectedFailure     bool
	gitRepoPluginDisabled bool
}

func TestPlugin(t *testing.T) {
	gitURL := "https://github.com/kubernetes/kubernetes.git"
	revision := "2a30ce65c5ab586b98916d83385c5983edd353a1"

	scenarios := []scenario{
		{
			name: "target-dir",
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitURL,
						Revision:   revision,
						Directory:  "target_dir",
					},
				},
			},
			expecteds: []expectedCommand{
				{
					cmd: []string{"git", "clone", "--", gitURL, "target_dir"},
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
		},
		{
			name:                  "target-dir",
			gitRepoPluginDisabled: true,
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitURL,
						Revision:   revision,
						Directory:  "target_dir",
					},
				},
			},
			isExpectedFailure: true,
		},
		{
			name: "target-dir-no-revision",
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitURL,
						Directory:  "target_dir",
					},
				},
			},
			expecteds: []expectedCommand{
				{
					cmd: []string{"git", "clone", "--", gitURL, "target_dir"},
					dir: "",
				},
			},
		},
		{
			name:                  "target-dir-no-revision",
			gitRepoPluginDisabled: true,
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitURL,
						Directory:  "target_dir",
					},
				},
			},
			isExpectedFailure: true,
		},
		{
			name: "only-git-clone",
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitURL,
					},
				},
			},
			expecteds: []expectedCommand{
				{
					cmd: []string{"git", "clone", "--", gitURL},
					dir: "",
				},
			},
		},
		{
			name:                  "only-git-clone",
			gitRepoPluginDisabled: true,
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitURL,
					},
				},
			},
			isExpectedFailure: true,
		},
		{
			name: "no-target-dir",
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitURL,
						Revision:   revision,
						Directory:  "",
					},
				},
			},
			expecteds: []expectedCommand{
				{
					cmd: []string{"git", "clone", "--", gitURL},
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
		},
		{
			name:                  "no-target-dir",
			gitRepoPluginDisabled: true,
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitURL,
						Revision:   revision,
						Directory:  "",
					},
				},
			},
			isExpectedFailure: true,
		},
		{
			name: "current-dir",
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitURL,
						Revision:   revision,
						Directory:  ".",
					},
				},
			},
			expecteds: []expectedCommand{
				{
					cmd: []string{"git", "clone", "--", gitURL, "."},
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
		},
		{
			name:                  "current-dir",
			gitRepoPluginDisabled: true,
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitURL,
						Revision:   revision,
						Directory:  ".",
					},
				},
			},
			isExpectedFailure: true,
		},
		{
			name: "current-dir-mess",
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitURL,
						Revision:   revision,
						Directory:  "./.",
					},
				},
			},
			expecteds: []expectedCommand{
				{
					cmd: []string{"git", "clone", "--", gitURL, "./."},
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
		},
		{
			name:                  "current-dir-mess",
			gitRepoPluginDisabled: true,
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitURL,
						Revision:   revision,
						Directory:  "./.",
					},
				},
			},
			isExpectedFailure: true,
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
			name:                  "invalid-repository",
			gitRepoPluginDisabled: true,
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
						Repository: gitURL,
						Revision:   "--bar",
					},
				},
			},
			isExpectedFailure: true,
		},
		{
			name:                  "invalid-revision",
			gitRepoPluginDisabled: true,
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitURL,
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
						Repository: gitURL,
						Directory:  "-b",
					},
				},
			},
			isExpectedFailure: true,
		},
		{
			name:                  "invalid-directory",
			gitRepoPluginDisabled: true,
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitURL,
						Directory:  "-b",
					},
				},
			},
			isExpectedFailure: true,
		},
		{
			name: "invalid-revision-directory-combo",
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitURL,
						Revision:   "main",
						Directory:  "foo/bar",
					},
				},
			},
			isExpectedFailure: true,
		},
		{
			name:                  "invalid-revision-directory-combo",
			gitRepoPluginDisabled: true,
			vol: &v1.Volume{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GitRepo: &v1.GitRepoVolumeSource{
						Repository: gitURL,
						Revision:   "main",
						Directory:  "foo/bar",
					},
				},
			},
			isExpectedFailure: true,
		},
	}

	for _, sc := range scenarios {
		t.Run(fmt.Sprintf("%s/gitRepoPluginDisabled:%v", sc.name, sc.gitRepoPluginDisabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GitRepoVolumeDriver, !sc.gitRepoPluginDisabled)
			allErrs := doTestPlugin(t, sc)
			if len(allErrs) == 0 && sc.isExpectedFailure {
				t.Errorf("Unexpected success for scenario: %s", sc.name)
			}
			if len(allErrs) > 0 && !sc.isExpectedFailure {
				t.Errorf("Unexpected failure for scenario: %s - %+v", sc.name, allErrs)
			}
		})

	}

}

func doTestPlugin(t *testing.T, sc scenario) []error {
	allErrs := []error{}

	plugMgr := volume.VolumePluginMgr{}
	rootDir, host := newTestHost(t)
	defer os.RemoveAll(rootDir)
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, host)

	plug, err := plugMgr.FindPluginByName("kubernetes.io/git-repo")
	if err != nil {
		allErrs = append(allErrs,
			fmt.Errorf("can't find the plugin by name"))
		return allErrs
	}
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, err := plug.NewMounter(volume.NewSpecFromVolume(sc.vol), pod)

	if err != nil {
		allErrs = append(allErrs,
			fmt.Errorf("failed to make a new Mounter: %w", err))
		return allErrs
	}
	if mounter == nil {
		allErrs = append(allErrs,
			fmt.Errorf("got a nil Mounter"))
		return allErrs
	}

	path := mounter.GetPath()
	suffix := filepath.Join("pods/poduid/volumes/kubernetes.io~git-repo", sc.vol.Name)
	if !strings.HasSuffix(path, suffix) {
		allErrs = append(allErrs,
			fmt.Errorf("got unexpected path: %s", path))
		return allErrs
	}

	// Test setUp()
	setUpErrs := doTestSetUp(sc, mounter)
	allErrs = append(allErrs, setUpErrs...)

	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			allErrs = append(allErrs,
				fmt.Errorf("SetUp() failed, volume path not created: %s", path))
			return allErrs
		}
		allErrs = append(allErrs,
			fmt.Errorf("SetUp() failed: %v", err))
		return allErrs

	}

	// gitRepo volume should create its own empty wrapper path
	podWrapperMetadataDir := fmt.Sprintf("%v/pods/poduid/plugins/kubernetes.io~empty-dir/wrapped_%v", rootDir, sc.vol.Name)

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
			fmt.Errorf("failed to make a new Unmounter: %w", err))
		return allErrs
	}
	if unmounter == nil {
		allErrs = append(allErrs,
			fmt.Errorf("got a nil Unmounter"))
		return allErrs
	}

	if err := unmounter.TearDown(); err != nil {
		allErrs = append(allErrs,
			fmt.Errorf("expected success, got: %w", err))
		return allErrs
	}
	if _, err := os.Stat(path); err == nil {
		allErrs = append(allErrs,
			fmt.Errorf("TearDown() failed, volume path still exists: %s", path))
	} else if !os.IsNotExist(err) {
		allErrs = append(allErrs,
			fmt.Errorf("TearDown() failed: %w", err))
	}
	return allErrs
}

func doTestSetUp(sc scenario, mounter volume.Mounter) []error {
	expecteds := sc.expecteds
	allErrs := []error{}

	// Construct combined outputs from expected commands
	var fakeOutputs []fakeexec.FakeAction
	var fcmd fakeexec.FakeCmd
	for _, expected := range expecteds {
		expected := expected
		if expected.cmd[1] == "clone" {
			// Calculate the subdirectory clone would create (if any)
			// git clone -- https://github.com/kubernetes/kubernetes.git target_dir --> target_dir
			// git clone -- https://github.com/kubernetes/kubernetes.git            --> kubernetes
			// git clone -- https://github.com/kubernetes/kubernetes.git .          --> .
			// git clone -- https://github.com/kubernetes/kubernetes.git ./.        --> .
			cloneSubdir := path.Base(expected.cmd[len(expected.cmd)-1])
			if cloneSubdir == "kubernetes.git" {
				cloneSubdir = "kubernetes"
			}
			fakeOutputs = append(fakeOutputs, func() ([]byte, []byte, error) {
				// git clone, it creates new dir/files
				os.MkdirAll(filepath.Join(fcmd.Dirs[0], expected.dir, cloneSubdir), 0750)
				return []byte{}, nil, nil
			})
		} else {
			// git checkout || git reset, they create nothing
			fakeOutputs = append(fakeOutputs, func() ([]byte, []byte, error) {
				return []byte{}, nil, nil
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
	fake := &fakeexec.FakeExec{
		CommandScript: fakeAction,
	}

	g := mounter.(*gitRepoVolumeMounter)
	g.exec = fake

	err := g.SetUp(volume.MounterArgs{})
	if err != nil {
		allErrs = append(allErrs, err)
	}

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
		expectedPaths = append(expectedPaths, filepath.Join(g.GetPath(), expected.dir))
	}
	if len(fcmd.Dirs) != len(expectedPaths) || !reflect.DeepEqual(expectedPaths, fcmd.Dirs) {
		allErrs = append(allErrs,
			fmt.Errorf("unexpected directories: %v, expected: %v", fcmd.Dirs, expectedPaths))
	}

	return allErrs
}
