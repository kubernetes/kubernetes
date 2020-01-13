/*
Copyright 2019 The Kubernetes Authors.

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

package config

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"

	kubectrlmgrconfig "k8s.io/kubernetes/pkg/controller/apis/config"
	utilfiles "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/files"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
)

const (
	configDir   = "/test-cmc-dir"
	ctrlmgrFile = "controller-manager-file"
)

func addFile(fs utilfs.Filesystem, path string, file string) error {
	if err := utilfiles.EnsureDir(fs, filepath.Dir(path)); err != nil {
		return err
	}
	return utilfiles.ReplaceFile(fs, path, []byte(file))
}

func TestLoadConfig(t *testing.T) {
	testCases := []struct {
		desc   string
		file   *string
		expect *kubectrlmgrconfig.KubeControllerManagerConfiguration
		errFn  func(err error) bool
	}{
		{
			desc: "missing file",
			errFn: func(err error) bool {
				if err == nil {
					return false
				}
				return true
			},
		},
	}

	for _, test := range testCases {
		t.Run(test.desc, func(t *testing.T) {
			fs := utilfs.NewFakeFs()
			path := filepath.Join(configDir, ctrlmgrFile)
			if test.file != nil {
				if err := utilfiles.EnsureDir(fs, filepath.Dir(path)); err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if err := utilfiles.ReplaceFile(fs, path, []byte(*test.file)); err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			}
			loader, err := NewFsLoader(fs, path)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			cmc, err := loader.Load()
			if test.errFn != nil {
				assert.True(t, test.errFn(err), test.desc)
			}
			assert.Equal(t, test.expect, cmc, test.desc)
		})
	}
}
