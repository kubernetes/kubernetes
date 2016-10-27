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

package cmd

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
)

func assertExists(t *testing.T, path string) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Errorf("file/dir does not exist error: %s", err)
		t.Errorf("file/dir does not exist: %s", path)
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
				"manifests/etcd.json",
				"manifests/kube-apiserver.json",
				"pki/ca.pem",
				"admin.conf",
				"kubelet.conf",
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
				"kubelet.conf",
			},
			verifyExists: []string{
				"pki",
			},
			verifyNotExists: []string{
				"manifests",
			},
		},
		"preserve cloud-config.json": {
			setupDirs: []string{
				"manifests",
				"pki",
			},
			setupFiles: []string{
				"manifests/etcd.json",
				"manifests/kube-apiserver.json",
				"pki/ca.pem",
				"admin.conf",
				"kubelet.conf",
				"cloud-config.json",
			},
			verifyExists: []string{
				"manifests",
				"pki",
				"cloud-config.json",
			},
		},
		"preserve hidden files and directories": {
			setupDirs: []string{
				"manifests",
				"pki",
				".mydir",
			},
			setupFiles: []string{
				"manifests/etcd.json",
				"manifests/kube-apiserver.json",
				"pki/ca.pem",
				"admin.conf",
				"kubelet.conf",
				".cloud-config.json",
				".mydir/.myfile",
			},
			verifyExists: []string{
				"manifests",
				"pki",
				".cloud-config.json",
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
			t.Errorf("Unable to create temp directory: %s", err)
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
			defer f.Close()
			if err != nil {
				t.Errorf("Unable to create test file: %s", err)
			}
		}

		resetConfigDir(tmpDir)

		// Verify the files we cleanup implicitly in every test:
		assertExists(t, tmpDir)
		assertNotExists(t, filepath.Join(tmpDir, "admin.conf"))
		assertNotExists(t, filepath.Join(tmpDir, "kubelet.conf"))
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
