/*
Copyright 2014 Google Inc. All rights reserved.

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

package registrytest

import "github.com/GoogleCloudPlatform/kubernetes/pkg/api"

type PodRegistryStorage struct {
	PodRegistry
	machine string
}

func (rs *PodRegistryStorage) CreatePod(machine string, pod api.Pod) error {
	rs.PodRegistry.Pod = &pod
	rs.machine = machine
	return rs.PodRegistry.Err
}
