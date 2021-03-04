// +build !dockerless

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

	dockertypes "github.com/docker/docker/api/types"
	"github.com/docker/docker/pkg/jsonmessage"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
)

func TestRemoveImage(t *testing.T) {
	tests := map[string]struct {
		image         dockertypes.ImageInspect
		calledDetails []libdocker.CalledDetail
	}{
		"single tag": {
			dockertypes.ImageInspect{ID: "1111", RepoTags: []string{"foo"}},
			[]libdocker.CalledDetail{
				libdocker.NewCalledDetail("inspect_image", nil),
				libdocker.NewCalledDetail("remove_image", []interface{}{"foo", dockertypes.ImageRemoveOptions{PruneChildren: true}}),
				libdocker.NewCalledDetail("remove_image", []interface{}{"1111", dockertypes.ImageRemoveOptions{PruneChildren: true}}),
			},
		},
		"multiple tags": {
			dockertypes.ImageInspect{ID: "2222", RepoTags: []string{"foo", "bar"}},
			[]libdocker.CalledDetail{
				libdocker.NewCalledDetail("inspect_image", nil),
				libdocker.NewCalledDetail("remove_image", []interface{}{"foo", dockertypes.ImageRemoveOptions{PruneChildren: true}}),
				libdocker.NewCalledDetail("remove_image", []interface{}{"bar", dockertypes.ImageRemoveOptions{PruneChildren: true}}),
				libdocker.NewCalledDetail("remove_image", []interface{}{"2222", dockertypes.ImageRemoveOptions{PruneChildren: true}}),
			},
		},
		"single tag multiple repo digests": {
			dockertypes.ImageInspect{ID: "3333", RepoTags: []string{"foo"}, RepoDigests: []string{"foo@3333", "example.com/foo@3333"}},
			[]libdocker.CalledDetail{
				libdocker.NewCalledDetail("inspect_image", nil),
				libdocker.NewCalledDetail("remove_image", []interface{}{"foo", dockertypes.ImageRemoveOptions{PruneChildren: true}}),
				libdocker.NewCalledDetail("remove_image", []interface{}{"foo@3333", dockertypes.ImageRemoveOptions{PruneChildren: true}}),
				libdocker.NewCalledDetail("remove_image", []interface{}{"example.com/foo@3333", dockertypes.ImageRemoveOptions{PruneChildren: true}}),
				libdocker.NewCalledDetail("remove_image", []interface{}{"3333", dockertypes.ImageRemoveOptions{PruneChildren: true}}),
			},
		},
		"no tags multiple repo digests": {
			dockertypes.ImageInspect{ID: "4444", RepoTags: []string{}, RepoDigests: []string{"foo@4444", "example.com/foo@4444"}},
			[]libdocker.CalledDetail{
				libdocker.NewCalledDetail("inspect_image", nil),
				libdocker.NewCalledDetail("remove_image", []interface{}{"foo@4444", dockertypes.ImageRemoveOptions{PruneChildren: true}}),
				libdocker.NewCalledDetail("remove_image", []interface{}{"example.com/foo@4444", dockertypes.ImageRemoveOptions{PruneChildren: true}}),
				libdocker.NewCalledDetail("remove_image", []interface{}{"4444", dockertypes.ImageRemoveOptions{PruneChildren: true}}),
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			ds, fakeDocker, _ := newTestDockerService()
			fakeDocker.InjectImageInspects([]dockertypes.ImageInspect{test.image})
			ds.RemoveImage(getTestCTX(), &runtimeapi.RemoveImageRequest{Image: &runtimeapi.ImageSpec{Image: test.image.ID}})
			err := fakeDocker.AssertCallDetails(test.calledDetails...)
			assert.NoError(t, err)
		})
	}
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
		_, err := ds.PullImage(getTestCTX(), &runtimeapi.PullImageRequest{Image: test.image, Auth: &runtimeapi.AuthConfig{}})
		require.Error(t, err, fmt.Sprintf("TestCase [%s]", key))
		assert.Contains(t, err.Error(), test.expectedError)
	}
}
