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

package dockershim

import (
	"testing"

	dockertypes "github.com/docker/engine-api/types"

	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
)

func TestRemoveImage(t *testing.T) {
	ds, fakeDocker, _ := newTestDockerService()
	id := "1111"
	fakeDocker.Image = &dockertypes.ImageInspect{ID: id, RepoTags: []string{"foo"}}
	ds.RemoveImage(&runtimeApi.ImageSpec{Image: &id})
	fakeDocker.AssertCallDetails(dockertools.NewCalledDetail("inspect_image", nil),
		dockertools.NewCalledDetail("remove_image", []interface{}{id, dockertypes.ImageRemoveOptions{PruneChildren: true}}))
}

func TestRemoveImageWithMultipleTags(t *testing.T) {
	ds, fakeDocker, _ := newTestDockerService()
	id := "1111"
	fakeDocker.Image = &dockertypes.ImageInspect{ID: id, RepoTags: []string{"foo", "bar"}}
	ds.RemoveImage(&runtimeApi.ImageSpec{Image: &id})
	fakeDocker.AssertCallDetails(dockertools.NewCalledDetail("inspect_image", nil),
		dockertools.NewCalledDetail("remove_image", []interface{}{"foo", dockertypes.ImageRemoveOptions{PruneChildren: true}}),
		dockertools.NewCalledDetail("remove_image", []interface{}{"bar", dockertypes.ImageRemoveOptions{PruneChildren: true}}))
}
