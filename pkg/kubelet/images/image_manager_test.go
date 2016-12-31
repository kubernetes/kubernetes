/*
Copyright 2015 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/record"
	. "k8s.io/kubernetes/pkg/kubelet/container"
	ctest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/util/clock"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

type pullerTestCase struct {
	containerImage  string
	policy          v1.PullPolicy
	calledFunctions []string
	inspectErr      error
	pullerErr       error
	expectedErr     []error
}

func pullerTestCases() []pullerTestCase {
	return []pullerTestCase{
		{ // pull missing image
			containerImage:  "missing_image",
			policy:          v1.PullIfNotPresent,
			calledFunctions: []string{"GetImageRef", "PullImage"},
			inspectErr:      nil,
			pullerErr:       nil,
			expectedErr:     []error{nil}},

		{ // image present, don't pull
			containerImage:  "present_image",
			policy:          v1.PullIfNotPresent,
			calledFunctions: []string{"GetImageRef"},
			inspectErr:      nil,
			pullerErr:       nil,
			expectedErr:     []error{nil, nil, nil}},
		// image present, pull it
		{containerImage: "present_image",
			policy:          v1.PullAlways,
			calledFunctions: []string{"GetImageRef", "PullImage"},
			inspectErr:      nil,
			pullerErr:       nil,
			expectedErr:     []error{nil, nil, nil}},
		// missing image, error PullNever
		{containerImage: "missing_image",
			policy:          v1.PullNever,
			calledFunctions: []string{"GetImageRef"},
			inspectErr:      nil,
			pullerErr:       nil,
			expectedErr:     []error{ErrImageNeverPull, ErrImageNeverPull, ErrImageNeverPull}},
		// missing image, unable to inspect
		{containerImage: "missing_image",
			policy:          v1.PullIfNotPresent,
			calledFunctions: []string{"GetImageRef"},
			inspectErr:      errors.New("unknown inspectError"),
			pullerErr:       nil,
			expectedErr:     []error{ErrImageInspect, ErrImageInspect, ErrImageInspect}},
		// missing image, unable to fetch
		{containerImage: "typo_image",
			policy:          v1.PullIfNotPresent,
			calledFunctions: []string{"GetImageRef", "PullImage"},
			inspectErr:      nil,
			pullerErr:       errors.New("404"),
			expectedErr:     []error{ErrImagePull, ErrImagePull, ErrImagePullBackOff, ErrImagePull, ErrImagePullBackOff, ErrImagePullBackOff}},
	}
}

func pullerTestEnv(c pullerTestCase, serialized bool) (puller ImageManager, fakeClock *clock.FakeClock, fakeRuntime *ctest.FakeRuntime, container *v1.Container) {
	container = &v1.Container{
		Name:            "container_name",
		Image:           c.containerImage,
		ImagePullPolicy: c.policy,
	}

	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	fakeClock = clock.NewFakeClock(time.Now())
	backOff.Clock = fakeClock

	fakeRuntime = &ctest.FakeRuntime{}
	fakeRecorder := &record.FakeRecorder{}

	fakeRuntime.ImageList = []Image{{ID: "present_image"}}
	fakeRuntime.Err = c.pullerErr
	fakeRuntime.InspectErr = c.inspectErr

	puller = NewImageManager(fakeRecorder, fakeRuntime, backOff, serialized, 0, 0)
	return
}

func TestParallelPuller(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name:            "test_pod",
			Namespace:       "test-ns",
			UID:             "bar",
			ResourceVersion: "42",
			SelfLink:        "/api/v1/pods/foo",
		}}

	cases := pullerTestCases()

	for i, c := range cases {
		puller, fakeClock, fakeRuntime, container := pullerTestEnv(c, false)

		for tick, expected := range c.expectedErr {
			fakeClock.Step(time.Second)
			_, _, err := puller.EnsureImageExists(pod, container, nil)
			fakeRuntime.AssertCalls(c.calledFunctions)
			assert.Equal(t, expected, err, "in test %d tick=%d", i, tick)
		}
	}
}

func TestSerializedPuller(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name:            "test_pod",
			Namespace:       "test-ns",
			UID:             "bar",
			ResourceVersion: "42",
			SelfLink:        "/api/v1/pods/foo",
		}}

	cases := pullerTestCases()

	for i, c := range cases {
		puller, fakeClock, fakeRuntime, container := pullerTestEnv(c, true)

		for tick, expected := range c.expectedErr {
			fakeClock.Step(time.Second)
			_, _, err := puller.EnsureImageExists(pod, container, nil)
			fakeRuntime.AssertCalls(c.calledFunctions)
			assert.Equal(t, expected, err, "in test %d tick=%d", i, tick)
		}
	}
}

func TestApplyDefaultImageTag(t *testing.T) {
	for _, testCase := range []struct {
		Input  string
		Output string
	}{
		{Input: "root", Output: "root:latest"},
		{Input: "root:tag", Output: "root:tag"},
		{Input: "root@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", Output: "root@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
	} {
		image, err := applyDefaultImageTag(testCase.Input)
		if err != nil {
			t.Errorf("applyDefaultImageTag(%s) failed: %v", testCase.Input, err)
		} else if image != testCase.Output {
			t.Errorf("Expected image reference: %q, got %q", testCase.Output, image)
		}
	}
}
