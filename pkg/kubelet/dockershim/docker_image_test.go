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
	"fmt"
	"testing"

	"github.com/docker/docker/pkg/jsonmessage"
	dockertypes "github.com/docker/engine-api/types"
	"github.com/stretchr/testify/assert"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
)

func TestRemoveImage(t *testing.T) {
	ds, fakeDocker, _ := newTestDockerService()
	id := "1111"
	fakeDocker.InjectImageInspects([]dockertypes.ImageInspect{{ID: id, RepoTags: []string{"foo"}}})
	ds.RemoveImage(&runtimeapi.ImageSpec{Image: id})
	fakeDocker.AssertCallDetails(libdocker.NewCalledDetail("inspect_image", nil),
		libdocker.NewCalledDetail("remove_image", []interface{}{id, dockertypes.ImageRemoveOptions{PruneChildren: true}}))
}

func TestRemoveImageWithMultipleTags(t *testing.T) {
	ds, fakeDocker, _ := newTestDockerService()
	id := "1111"
	fakeDocker.InjectImageInspects([]dockertypes.ImageInspect{{ID: id, RepoTags: []string{"foo", "bar"}}})
	ds.RemoveImage(&runtimeapi.ImageSpec{Image: id})
	fakeDocker.AssertCallDetails(libdocker.NewCalledDetail("inspect_image", nil),
		libdocker.NewCalledDetail("remove_image", []interface{}{"foo", dockertypes.ImageRemoveOptions{PruneChildren: true}}),
		libdocker.NewCalledDetail("remove_image", []interface{}{"bar", dockertypes.ImageRemoveOptions{PruneChildren: true}}))
}

func TestPullWithJSONError(t *testing.T) {
	ds, fakeDocker, _ := newTestDockerService()
	tests := map[string]struct {
		image         *runtimeapi.ImageSpec
		err           error
		expectedError string
	}{
		"Json error": {
			&runtimeapi.ImageSpec{Image: "ubuntu"},
			&jsonmessage.JSONError{Code: 50, Message: "Json error"},
			"Json error",
		},
		"Bad gateway": {
			&runtimeapi.ImageSpec{Image: "ubuntu"},
			&jsonmessage.JSONError{Code: 502, Message: "<!doctype html>\n<html class=\"no-js\" lang=\"\">\n    <head>\n  </head>\n    <body>\n   <h1>Oops, there was an error!</h1>\n        <p>We have been contacted of this error, feel free to check out <a href=\"http://status.docker.com/\">status.docker.com</a>\n           to see if there is a bigger issue.</p>\n\n    </body>\n</html>"},
			"RegistryUnavailable",
		},
	}
	for key, test := range tests {
		fakeDocker.InjectError("pull", test.err)
		_, err := ds.PullImage(test.image, &runtimeapi.AuthConfig{})
		assert.Error(t, err, fmt.Sprintf("TestCase [%s]", key))
		assert.Contains(t, err.Error(), test.expectedError)
	}
}
