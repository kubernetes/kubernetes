/*
Copyright 2017 The Kubernetes Authors.

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

package configfiles

import (
	"fmt"
	"path/filepath"

	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	utilcodec "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/codec"
	utilfs "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/filesystem"
)

// Loader loads configuration from a storage layer
type Loader interface {
	// Load loads and returns the KubeletConfiguration from the storage layer, or an error if a configuration could not be loaded
	Load() (*kubeletconfig.KubeletConfiguration, error)
}

// fsLoader loads configuration from `configDir`
type fsLoader struct {
	// fs is the filesystem where the config files exist; can be mocked for testing
	fs utilfs.Filesystem
	// configDir is the absolute path to the directory containing the configuration files
	configDir string
}

// NewFSLoader returns a Loader that loads a KubeletConfiguration from the files in `configDir`
func NewFSLoader(fs utilfs.Filesystem, configDir string) Loader {
	return &fsLoader{
		fs:        fs,
		configDir: configDir,
	}
}

func (loader *fsLoader) Load() (*kubeletconfig.KubeletConfiguration, error) {
	errfmt := fmt.Sprintf("failed to load Kubelet config files from %q, error: ", loader.configDir) + "%v"

	// require the config be in a file called "kubelet"
	path := filepath.Join(loader.configDir, "kubelet")
	data, err := loader.fs.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read init config file %q, error: %v", path, err)
	}

	// no configuration is an error, some parameters are required
	if len(data) == 0 {
		return nil, fmt.Errorf(errfmt, fmt.Errorf("config file was empty, but some parameters are required"))
	}

	return utilcodec.DecodeKubeletConfiguration(data)
}
