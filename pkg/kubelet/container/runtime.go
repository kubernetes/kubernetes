/*
Copyright 2015 CoreOS Inc. All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume"
)

// Runtime interface defines the interfaces that should be implemented
// by a container runtime.
type Runtime interface {
	Version() (map[string]string, error)
	ListPods() ([]*api.Pod, error)
	RunPod(*api.BoundPod, map[string]volume.Interface) error
	KillPod(*api.Pod) error
	// TODO(yifan): How about adding a field in api.Container.VolumeMounts so we
	// can have a cleaner interface?
	RunContainerInPod(api.Container, *api.Pod, map[string]volume.Interface) error
	KillContainerInPod(api.Container, *api.Pod) error
	// TODO(yifan): Pull/Remove images
}
