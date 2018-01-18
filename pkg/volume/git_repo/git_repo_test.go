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
	"reflect"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/empty_dir"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

const (
	gitUrl            = "https://github.com/kubernetes/kubernetes.git"
	revision          = "2a30ce65c5ab586b98916d83385c5983edd353a1"
	gitRepositoryName = "kubernetes"
)

func newTestHost(t *testing.T) (string, volume.VolumeHost) {
	tempDir, err := ioutil.TempDir("/tmp", "git_repo_test.")
	if err != nil {
		t.Fatalf("can't make a temp rootdir: %v", err)
	}
	return tempDir, volumetest.NewFakeVolumeHost(tempDir, nil, empty_dir.ProbeVolumePlugins())
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
type expectedCommand []string

type testScenario struct {
	name              string
	vol               *v1.Volume
	repositoryDir     string
	expecteds         []expectedCommand
	isExpectedFailure bool
}

func TestPlugin(t *testing.T) {
	scenarios := []testScenario{
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
			repositoryDir: "target_dir",
			expecteds: []expectedCommand{
				[]string{"git", "-C", "volume-dir", "clone", gitUrl, "target_dir"},
				[]string{"git", "-C", "volume-dir/target_dir", "checkout", revision},
				[]string{"git", "-C", "volume-dir/target_dir", "reset", "--hard"},
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
			repositoryDir: "target_dir",
			expecteds: []expectedCommand{
				[]string{"git", "-C", "volume-dir", "clone", gitUrl, "target_dir"},
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
			repositoryDir: "kubernetes",
			expecteds: []expectedCommand{
				[]string{"git", "-C", "volume-dir", "clone", gitUrl},
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
			repositoryDir: "kubernetes",
			expecteds: []expectedCommand{
				[]string{"git", "-C", "volume-dir", "clone", gitUrl},
				[]string{"git", "-C", "volume-dir/kubernetes", "checkout", revision},
				[]string{"git", "-C", "volume-dir/kubernetes", "reset", "--hard"},
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
			repositoryDir: "",
			expecteds: []expectedCommand{
				[]string{"git", "-C", "volume-dir", "clone", gitUrl, "."},
				[]string{"git", "-C", "volume-dir", "checkout", revision},
				[]string{"git", "-C", "volume-dir", "reset", "--hard"},
			},
			isExpectedFailure: false,
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

func doTestPlugin(scenario testScenario, t *testing.T) []error {
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

func doTestSetUp(scenario testScenario, mounter volume.Mounter) []error {
	expecteds := scenario.expecteds
	allErrs := []error{}

	var commandLog []expectedCommand
	execCallback := func(cmd string, args ...string) ([]byte, error) {
		if len(args) < 2 {
			return nil, fmt.Errorf("expected at least 2 arguments, got %q", args)
		}
		if args[0] != "-C" {
			return nil, fmt.Errorf("expected the first argument to be \"-C\", got %q", args[0])
		}
		// command is 'git -C <dir> <command> <args>
		gitDir := args[1]
		gitCommand := args[2]
		if gitCommand == "clone" {
			// Clone creates a directory
			if scenario.repositoryDir != "" {
				os.MkdirAll(path.Join(gitDir, scenario.repositoryDir), 0750)
			}
		}
		// add the command to log with de-randomized gitDir
		args[1] = strings.Replace(gitDir, mounter.GetPath(), "volume-dir", 1)
		cmdline := append([]string{cmd}, args...)
		commandLog = append(commandLog, cmdline)
		return []byte{}, nil
	}
	g := mounter.(*gitRepoVolumeMounter)
	g.mounter = &mount.FakeMounter{}
	g.exec = mount.NewFakeExec(execCallback)

	g.SetUp(nil)

	if !reflect.DeepEqual(expecteds, commandLog) {
		allErrs = append(allErrs,
			fmt.Errorf("unexpected commands: %v, expected: %v", commandLog, expecteds))
	}

	return allErrs
}
