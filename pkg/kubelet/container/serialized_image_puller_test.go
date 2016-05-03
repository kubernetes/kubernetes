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

package container_test

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	. "k8s.io/kubernetes/pkg/kubelet/container"
	ctest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

func TestSerializedPuller(t *testing.T) {
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:            "test_pod",
			Namespace:       "test-ns",
			UID:             "bar",
			ResourceVersion: "42",
			SelfLink:        "/api/v1/pods/foo",
		}}

	cases := []struct {
		containerImage  string
		policy          api.PullPolicy
		calledFunctions []string
		inspectErr      error
		pullerErr       error
		expectedErr     []error
	}{
		{ // pull missing image
			containerImage:  "missing_image",
			policy:          api.PullIfNotPresent,
			calledFunctions: []string{"IsImagePresent", "PullImage"},
			inspectErr:      nil,
			pullerErr:       nil,
			expectedErr:     []error{nil}},

		{ // image present, dont pull
			containerImage:  "present_image",
			policy:          api.PullIfNotPresent,
			calledFunctions: []string{"IsImagePresent"},
			inspectErr:      nil,
			pullerErr:       nil,
			expectedErr:     []error{nil, nil, nil}},
		// image present, pull it
		{containerImage: "present_image",
			policy:          api.PullAlways,
			calledFunctions: []string{"IsImagePresent", "PullImage"},
			inspectErr:      nil,
			pullerErr:       nil,
			expectedErr:     []error{nil, nil, nil}},
		// missing image, error PullNever
		{containerImage: "missing_image",
			policy:          api.PullNever,
			calledFunctions: []string{"IsImagePresent"},
			inspectErr:      nil,
			pullerErr:       nil,
			expectedErr:     []error{ErrImageNeverPull, ErrImageNeverPull, ErrImageNeverPull}},
		// missing image, unable to inspect
		{containerImage: "missing_image",
			policy:          api.PullIfNotPresent,
			calledFunctions: []string{"IsImagePresent"},
			inspectErr:      errors.New("unknown inspectError"),
			pullerErr:       nil,
			expectedErr:     []error{ErrImageInspect, ErrImageInspect, ErrImageInspect}},
		// missing image, unable to fetch
		{containerImage: "typo_image",
			policy:          api.PullIfNotPresent,
			calledFunctions: []string{"IsImagePresent", "PullImage"},
			inspectErr:      nil,
			pullerErr:       errors.New("404"),
			expectedErr:     []error{ErrImagePull, ErrImagePull, ErrImagePullBackOff, ErrImagePull, ErrImagePullBackOff, ErrImagePullBackOff}},
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

		fakeRuntime := &ctest.FakeRuntime{}
		fakeRecorder := &record.FakeRecorder{}
		puller := NewSerializedImagePuller(fakeRecorder, fakeRuntime, backOff)

		fakeRuntime.ImageList = []Image{{"present_image", nil, nil, 0}}
		fakeRuntime.Err = c.pullerErr
		fakeRuntime.InspectErr = c.inspectErr

		for tick, expected := range c.expectedErr {
			fakeClock.Step(time.Second)
			err, _ := puller.PullImage(pod, container, nil)
			fakeRuntime.AssertCalls(c.calledFunctions)
			assert.Equal(t, expected, err, "in test %d tick=%d", i, tick)
		}

	}
}
