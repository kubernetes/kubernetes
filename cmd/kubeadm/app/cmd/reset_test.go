/*
Copyright 2016 The Kubernetes Authors.

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

package cmd

import (
	"errors"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"

	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
)

func assertExists(t *testing.T, path string) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Errorf("file/directory does not exist; error: %s", err)
		t.Errorf("file/directory does not exist: %s", path)
	}
}

func assertNotExists(t *testing.T, path string) {
	if _, err := os.Stat(path); err == nil {
		t.Errorf("file/dir exists: %s", path)
	}
}

// assertDirEmpty verifies a directory either does not exist, or is empty.
func assertDirEmpty(t *testing.T, path string) {
	dac := preflight.DirAvailableCheck{Path: path}
	_, errors := dac.Check()
	if len(errors) != 0 {
		t.Errorf("directory not empty: [%v]", errors)
	}
}

func TestConfigDirCleaner(t *testing.T) {
	tests := map[string]struct {
		setupDirs       []string
		setupFiles      []string
		verifyExists    []string
		verifyNotExists []string
	}{
		"simple reset": {
			setupDirs: []string{
				"manifests",
				"pki",
			},
			setupFiles: []string{
				"manifests/etcd.yaml",
				"manifests/kube-apiserver.yaml",
				"pki/ca.pem",
				kubeadmconstants.AdminKubeConfigFileName,
				kubeadmconstants.KubeletKubeConfigFileName,
			},
			verifyExists: []string{
				"manifests",
				"pki",
			},
		},
		"partial reset": {
			setupDirs: []string{
				"pki",
			},
			setupFiles: []string{
				"pki/ca.pem",
				kubeadmconstants.KubeletKubeConfigFileName,
			},
			verifyExists: []string{
				"pki",
			},
			verifyNotExists: []string{
				"manifests",
			},
		},
		"preserve cloud-config": {
			setupDirs: []string{
				"manifests",
				"pki",
			},
			setupFiles: []string{
				"manifests/etcd.yaml",
				"manifests/kube-apiserver.yaml",
				"pki/ca.pem",
				kubeadmconstants.AdminKubeConfigFileName,
				kubeadmconstants.KubeletKubeConfigFileName,
				"cloud-config",
			},
			verifyExists: []string{
				"manifests",
				"pki",
				"cloud-config",
			},
		},
		"preserve hidden files and directories": {
			setupDirs: []string{
				"manifests",
				"pki",
				".mydir",
			},
			setupFiles: []string{
				"manifests/etcd.yaml",
				"manifests/kube-apiserver.yaml",
				"pki/ca.pem",
				kubeadmconstants.AdminKubeConfigFileName,
				kubeadmconstants.KubeletKubeConfigFileName,
				".cloud-config",
				".mydir/.myfile",
			},
			verifyExists: []string{
				"manifests",
				"pki",
				".cloud-config",
				".mydir",
				".mydir/.myfile",
			},
		},
		"no-op reset": {
			verifyNotExists: []string{
				"pki",
				"manifests",
			},
		},
	}

	for name, test := range tests {
		t.Logf("Running test: %s", name)

		// Create a temporary directory for our fake config dir:
		tmpDir, err := ioutil.TempDir("", "kubeadm-reset-test")
		if err != nil {
			t.Errorf("Unable to create temporary directory: %s", err)
		}
		defer os.RemoveAll(tmpDir)

		for _, createDir := range test.setupDirs {
			err := os.Mkdir(filepath.Join(tmpDir, createDir), 0700)
			if err != nil {
				t.Errorf("Unable to setup test config directory: %s", err)
			}
		}

		for _, createFile := range test.setupFiles {
			fullPath := filepath.Join(tmpDir, createFile)
			f, err := os.Create(fullPath)
			if err != nil {
				t.Errorf("Unable to create test file: %s", err)
			}
			defer f.Close()
		}

		resetConfigDir(tmpDir, filepath.Join(tmpDir, "pki"))

		// Verify the files we cleanup implicitly in every test:
		assertExists(t, tmpDir)
		assertNotExists(t, filepath.Join(tmpDir, kubeadmconstants.AdminKubeConfigFileName))
		assertNotExists(t, filepath.Join(tmpDir, kubeadmconstants.KubeletKubeConfigFileName))
		assertDirEmpty(t, filepath.Join(tmpDir, "manifests"))
		assertDirEmpty(t, filepath.Join(tmpDir, "pki"))

		// Verify the files as requested by the test:
		for _, path := range test.verifyExists {
			assertExists(t, filepath.Join(tmpDir, path))
		}
		for _, path := range test.verifyNotExists {
			assertNotExists(t, filepath.Join(tmpDir, path))
		}
	}
}

type fakeDockerChecker struct {
	warnings []error
	errors   []error
}

func (c *fakeDockerChecker) Check() (warnings, errors []error) {
	return c.warnings, c.errors
}

func (c *fakeDockerChecker) Name() string {
	return "FakeDocker"
}

func newFakeDockerChecker(warnings, errors []error) preflight.Checker {
	return &fakeDockerChecker{warnings: warnings, errors: errors}
}

func TestResetWithDocker(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		RunScript: []fakeexec.FakeRunAction{
			func() ([]byte, []byte, error) { return nil, nil, nil },
			func() ([]byte, []byte, error) { return nil, nil, errors.New("docker error") },
			func() ([]byte, []byte, error) { return nil, nil, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	resetWithDocker(&fexec, newFakeDockerChecker(nil, nil))
	if fcmd.RunCalls != 1 {
		t.Errorf("expected 1 call to Run, got %d", fcmd.RunCalls)
	}
	resetWithDocker(&fexec, newFakeDockerChecker(nil, nil))
	if fcmd.RunCalls != 2 {
		t.Errorf("expected 2 calls to Run, got %d", fcmd.RunCalls)
	}
	resetWithDocker(&fexec, newFakeDockerChecker(nil, []error{errors.New("test error")}))
	if fcmd.RunCalls != 2 {
		t.Errorf("expected 2 calls to Run, got %d", fcmd.RunCalls)
	}
}

func TestResetWithCrictl(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// 2: socket path provided, not runnning with crictl (1x CombinedOutput, 2x Run)
			func() ([]byte, error) { return []byte("1"), nil },
			// 3: socket path provided, crictl fails, reset with docker (1x CombinedOuput fail, 1x Run)
			func() ([]byte, error) { return nil, errors.New("crictl list err") },
		},
		RunScript: []fakeexec.FakeRunAction{
			// 1: socket path not provided, running with docker
			func() ([]byte, []byte, error) { return nil, nil, nil },
			// 2: socket path provided, now runnning with crictl (1x CombinedOutput, 2x Run)
			func() ([]byte, []byte, error) { return nil, nil, nil },
			func() ([]byte, []byte, error) { return nil, nil, nil },
			// 3: socket path provided, crictl fails, reset with docker (1x CombinedOuput, 1x Run)
			func() ([]byte, []byte, error) { return nil, nil, nil },
			// 4: running with no socket and docker fails (1x Run)
			func() ([]byte, []byte, error) { return nil, nil, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}

	// 1: socket path not provided, running with docker
	resetWithCrictl(&fexec, newFakeDockerChecker(nil, nil), "", "crictl")
	if fcmd.RunCalls != 1 {
		t.Errorf("expected 1 call to Run, got %d", fcmd.RunCalls)
	}
	if !strings.Contains(fcmd.RunLog[0][2], "docker") {
		t.Errorf("expected a call to docker, got %v", fcmd.RunLog[0])
	}

	// 2: socket path provided, now runnning with crictl (1x CombinedOutput, 2x Run)
	resetWithCrictl(&fexec, newFakeDockerChecker(nil, nil), "/test.sock", "crictl")
	if fcmd.RunCalls != 3 {
		t.Errorf("expected 3 calls to Run, got %d", fcmd.RunCalls)
	}
	if !strings.Contains(fcmd.RunLog[1][2], "crictl") {
		t.Errorf("expected a call to crictl, got %v", fcmd.RunLog[0])
	}
	if !strings.Contains(fcmd.RunLog[2][2], "crictl") {
		t.Errorf("expected a call to crictl, got %v", fcmd.RunLog[0])
	}

	// 3: socket path provided, crictl fails, reset with docker
	resetWithCrictl(&fexec, newFakeDockerChecker(nil, nil), "/test.sock", "crictl")
	if fcmd.RunCalls != 4 {
		t.Errorf("expected 4 calls to Run, got %d", fcmd.RunCalls)
	}
	if !strings.Contains(fcmd.RunLog[3][2], "docker") {
		t.Errorf("expected a call to docker, got %v", fcmd.RunLog[0])
	}

	// 4: running with no socket and docker fails (1x Run)
	resetWithCrictl(&fexec, newFakeDockerChecker(nil, []error{errors.New("test error")}), "", "crictl")
	if fcmd.RunCalls != 4 {
		t.Errorf("expected 4 calls to Run, got %d", fcmd.RunCalls)
	}
}

func TestReset(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			func() ([]byte, error) { return []byte("1"), nil },
			func() ([]byte, error) { return []byte("1"), nil },
			func() ([]byte, error) { return []byte("1"), nil },
		},
		RunScript: []fakeexec.FakeRunAction{
			func() ([]byte, []byte, error) { return nil, nil, nil },
			func() ([]byte, []byte, error) { return nil, nil, nil },
			func() ([]byte, []byte, error) { return nil, nil, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
		LookPathFunc: func(cmd string) (string, error) { return cmd, nil },
	}

	reset(&fexec, newFakeDockerChecker(nil, nil), "/test.sock")
	if fcmd.RunCalls != 2 {
		t.Errorf("expected 2 call to Run, got %d", fcmd.RunCalls)
	}
	if !strings.Contains(fcmd.RunLog[0][2], "crictl") {
		t.Errorf("expected a call to crictl, got %v", fcmd.RunLog[0])
	}

	fexec.LookPathFunc = func(cmd string) (string, error) { return "", errors.New("no crictl") }
	reset(&fexec, newFakeDockerChecker(nil, nil), "/test.sock")
	if fcmd.RunCalls != 3 {
		t.Errorf("expected 3 calls to Run, got %d", fcmd.RunCalls)
	}
	if !strings.Contains(fcmd.RunLog[2][2], "docker") {
		t.Errorf("expected a call to docker, got %v", fcmd.RunLog[0])
	}
}
