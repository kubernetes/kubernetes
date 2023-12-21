//go:build linux
// +build linux

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

package cadvisor

import (
	"fmt"
	"strings"

	cadvisorfs "github.com/google/cadvisor/fs"
)

// LabelCrioContainers is a label to allow for cadvisor to track writeable layers
// separately from read-only layers.
// Once CAdvisor upstream changes are merged, we should remove this constant
const LabelCrioContainers string = "crio-containers"

// imageFsInfoProvider knows how to translate the configured runtime
// to its file system label for images.
type imageFsInfoProvider struct {
	runtimeEndpoint string
}

// ImageFsInfoLabel returns the image fs label for the configured runtime.
// For remote runtimes, it handles additional runtimes natively understood by cAdvisor.
func (i *imageFsInfoProvider) ImageFsInfoLabel() (string, error) {
	if detectCrioWorkaround(i) {
		return cadvisorfs.LabelCrioImages, nil
	}
	return "", fmt.Errorf("no imagefs label for configured runtime")
}

// ContainerFsInfoLabel returns the container fs label for the configured runtime.
// For remote runtimes, it handles addition runtimes natively understood by cAdvisor.
func (i *imageFsInfoProvider) ContainerFsInfoLabel() (string, error) {
	if detectCrioWorkaround(i) {
		return LabelCrioContainers, nil
	}
	return "", fmt.Errorf("no containerfs label for configured runtime")
}

// This is a temporary workaround to get stats for cri-o from cadvisor
// and should be removed.
// Related to https://github.com/kubernetes/kubernetes/issues/51798
func detectCrioWorkaround(i *imageFsInfoProvider) bool {
	return strings.HasSuffix(i.runtimeEndpoint, CrioSocketSuffix)
}

// NewImageFsInfoProvider returns a provider for the specified runtime configuration.
func NewImageFsInfoProvider(runtimeEndpoint string) ImageFsInfoProvider {
	return &imageFsInfoProvider{runtimeEndpoint: runtimeEndpoint}
}
