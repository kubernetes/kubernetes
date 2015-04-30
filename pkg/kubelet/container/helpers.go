/*
Copyright 2015 Google Inc. All rights reserved.

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

package container

import (
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

// HandlerRunner runs a lifecycle handler for a container.
type HandlerRunner interface {
	Run(containerID string, pod *api.Pod, container *api.Container, handler *api.Handler) error
}

// RunContainerOptionsGenerator generates the options that necessary for
// container runtime to run a container.
// TODO(yifan): Remove netMode, ipcMode.
type RunContainerOptionsGenerator interface {
	GenerateRunContainerOptions(pod *api.Pod, container *api.Container, netMode, ipcMode string) (*RunContainerOptions, error)
}

// Trims runtime prefix from ID or image name (e.g.: docker://busybox -> busybox).
func TrimRuntimePrefix(fullString string) string {
	const prefixSeparator = "://"

	idx := strings.Index(fullString, prefixSeparator)
	if idx < 0 {
		return fullString
	}
	return fullString[idx+len(prefixSeparator):]
}
