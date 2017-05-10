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

package gpu

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
)

type GPUManagerInitialzerFunc func(ActivePodsLister, dockertools.DockerInterface, string) (GPUManager, error)

type ActivePodsLister interface {
	// Returns a list of active pods on the node.
	GetActivePods() []*v1.Pod
}

var (
	mapperForManager = make(map[string]GPUManagerInitialzerFunc)
)

func RegisterGPUManagerInitializer(name string, fn GPUManagerInitialzerFunc) {
	if _, ok := mapperForManager[name]; ok {
		return
	}

	mapperForManager[name] = fn
}

func GetGPUManagerInitializer(name string) (GPUManagerInitialzerFunc, error) {
	initializer, ok := mapperForManager[name]

	if !ok {
		return nil, fmt.Errorf("Can't find GPUManagerRuntime %s", name)
	}

	return initializer, nil
}
