/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package images

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	runtime "k8s.io/kubernetes/pkg/kubelet/container"
	ctest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

func TestPuller(t *testing.T) {
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:            "test_pod",
			Namespace:       "test-ns",
			UID:             "bar",
			ResourceVersion: "42",
			SelfLink:        "/api/v1/pods/foo",
		}}

	cases := []struct {
		containerImage     string
		policy             api.PullPolicy
		calledFunctions    [][]string
		pullerErr          error
		expectedErr        []error
		deleteUnusedImages bool
	}{
		{ // pull missing image
			containerImage:  "missing_image",
			policy:          api.PullIfNotPresent,
			calledFunctions: [][]string{{"ListImages", "GetPods", "PullImage"}},
			pullerErr:       nil,
			expectedErr:     []error{nil}},

		{ // image present, dont pull
			containerImage: "present_image",
			policy:         api.PullIfNotPresent,
			calledFunctions: [][]string{
				{"ListImages", "GetPods"},
				{"ListImages", "GetPods"},
				{"ListImages", "GetPods"}},
			pullerErr:   nil,
			expectedErr: []error{nil, nil, nil}},
		// image present, pull it
		{containerImage: "present_image",
			policy:          api.PullAlways,
			calledFunctions: [][]string{{"ListImages", "GetPods", "PullImage"}},
			pullerErr:       nil,
			expectedErr:     []error{nil}},
		// missing image, error PullNever
		{containerImage: "missing_image",
			policy: api.PullNever,
			calledFunctions: [][]string{
				{"ListImages", "GetPods"},
				{"ListImages", "GetPods"},
				{"ListImages", "GetPods"}},
			pullerErr:   nil,
			expectedErr: []error{ErrImageNeverPull, ErrImageNeverPull, ErrImageNeverPull}},
		// missing image, unable to fetch
		{containerImage: "typo_image",
			policy: api.PullIfNotPresent,
			calledFunctions: [][]string{{"ListImages", "GetPods", "PullImage"},
				{"ListImages", "GetPods", "PullImage", "PullImage"},
				{"ListImages", "GetPods", "PullImage", "PullImage"},
				{"ListImages", "GetPods", "PullImage", "PullImage", "PullImage"},
				{"ListImages", "GetPods", "PullImage", "PullImage", "PullImage"},
				{"ListImages", "GetPods", "PullImage", "PullImage", "PullImage"}},
			pullerErr:   errors.New("404"),
			expectedErr: []error{ErrImagePull, ErrImagePull, ErrImagePullBackOff, ErrImagePull, ErrImagePullBackOff, ErrImagePullBackOff}},
		// image present, then GCed, so pull it.
		{containerImage: "present_image",
			policy: api.PullIfNotPresent,
			calledFunctions: [][]string{{"ListImages", "GetPods"},
				{"ListImages", "GetPods", "PullImage"},
				{"ListImages", "GetPods", "PullImage"}},
			pullerErr:          nil,
			expectedErr:        []error{nil},
			deleteUnusedImages: true},
	}

	for i, c := range cases {
		container := &api.Container{
			Name:            "container_name",
			Image:           c.containerImage,
			ImagePullPolicy: c.policy,
		}

		backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
		fakeClock := util.NewFakeClock(time.Now())
		backOff.Clock = fakeClock

		fakeRuntime := &ctest.FakeRuntime{
			ImageList: []runtime.Image{{RepoTags: []string{"present_image"}, ID: "", Size: 0}},
			PullErr:   c.pullerErr,
		}

		fakeRecorder := &record.FakeRecorder{}
		manager, err := NewImageManager(fakeRecorder, fakeRuntime, backOff, false /*parallel*/)
		assert.Nil(t, err, "image manager creation failed")

		for tick, expected := range c.expectedErr {
			fakeClock.Step(time.Second)
			err := manager.EnsureImageExists(pod, container, nil)
			assert.Nil(t, fakeRuntime.AssertCalls(c.calledFunctions[tick]), "in test %d tick=%d", i, tick)
			assert.Equal(t, expected, err, "in test %d tick=%d", i, tick)
			if c.deleteUnusedImages {
				manager.DeleteUnusedImages()
			}
		}

	}
}

func TestGarbageCollection(t *testing.T) {
}
