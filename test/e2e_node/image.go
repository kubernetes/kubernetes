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

package e2e_node

import (
	"fmt"

	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
)

type ConformanceImage struct {
	Image   kubecontainer.ImageSpec
	Runtime kubecontainer.Runtime
}

func NewConformanceImage(containerRuntime string, image string) (ci ConformanceImage, err error) {
	ci.Image = kubecontainer.ImageSpec{Image: image}
	if containerRuntime == "docker" {
		ci.Runtime = dockerRuntime()
		return ci, nil
	}
	return ci, fmt.Errorf("Unsupported runtime : %s.", containerRuntime)
}

//TODO: do not expose kubelet implementation details after we refactor the runtime API.
func dockerRuntime() kubecontainer.Runtime {
	dockerClient := dockertools.ConnectToDockerOrDie("", 0)
	pm := kubepod.NewBasicPodManager(nil)
	dm := dockertools.NewDockerManager(
		dockerClient,
		nil, nil, nil, pm, nil,
		"", 0, 0, "",
		nil, nil, nil, nil, nil, nil, nil,
		false, nil, true, false, false, "",
	)

	return dm
}

func (ci *ConformanceImage) Pull() error {
	return ci.Runtime.PullImage(ci.Image, nil)
}

func (ci *ConformanceImage) Present() (bool, error) {
	return ci.Runtime.IsImagePresent(ci.Image)
}

func (ci *ConformanceImage) List() ([]string, error) {
	if images, err := ci.Runtime.ListImages(); err != nil {
		return nil, err
	} else {
		var tags []string
		for _, image := range images {
			tags = append(tags, image.RepoTags...)
		}
		return tags, nil
	}
}

func (ci *ConformanceImage) Remove() error {
	return ci.Runtime.RemoveImage(ci.Image)
}

func (ci *ConformanceImage) GetTag() string {
	return ci.Image.Image
}
