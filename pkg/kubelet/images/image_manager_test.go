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
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	. "k8s.io/kubernetes/pkg/kubelet/container"
	ctest "k8s.io/kubernetes/pkg/kubelet/container/testing"
)

type pullerExpects struct {
	calls []string
	err   error
}

type pullerTestCase struct {
	containerImage string
	policy         v1.PullPolicy
	inspectErr     error
	pullerErr      error
	expected       []pullerExpects
}

func pullerTestCases() []pullerTestCase {
	return []pullerTestCase{
		{ // pull missing image
			containerImage: "missing_image",
			policy:         v1.PullIfNotPresent,
			inspectErr:     nil,
			pullerErr:      nil,
			expected: []pullerExpects{
				{[]string{"GetImageRef", "PullImage"}, nil},
			}},

		{ // image present, don't pull
			containerImage: "present_image",
			policy:         v1.PullIfNotPresent,
			inspectErr:     nil,
			pullerErr:      nil,
			expected: []pullerExpects{
				{[]string{"GetImageRef"}, nil},
				{[]string{"GetImageRef"}, nil},
				{[]string{"GetImageRef"}, nil},
			}},
		// image present, pull it
		{containerImage: "present_image",
			policy:     v1.PullAlways,
			inspectErr: nil,
			pullerErr:  nil,
			expected: []pullerExpects{
				{[]string{"GetImageRef", "PullImage"}, nil},
				{[]string{"GetImageRef", "PullImage"}, nil},
				{[]string{"GetImageRef", "PullImage"}, nil},
			}},
		// missing image, error PullNever
		{containerImage: "missing_image",
			policy:     v1.PullNever,
			inspectErr: nil,
			pullerErr:  nil,
			expected: []pullerExpects{
				{[]string{"GetImageRef"}, ErrImageNeverPull},
				{[]string{"GetImageRef"}, ErrImageNeverPull},
				{[]string{"GetImageRef"}, ErrImageNeverPull},
			}},
		// missing image, unable to inspect
		{containerImage: "missing_image",
			policy:     v1.PullIfNotPresent,
			inspectErr: errors.New("unknown inspectError"),
			pullerErr:  nil,
			expected: []pullerExpects{
				{[]string{"GetImageRef"}, ErrImageInspect},
				{[]string{"GetImageRef"}, ErrImageInspect},
				{[]string{"GetImageRef"}, ErrImageInspect},
			}},
		// missing image, unable to fetch
		{containerImage: "typo_image",
			policy:     v1.PullIfNotPresent,
			inspectErr: nil,
			pullerErr:  errors.New("404"),
			expected: []pullerExpects{
				{[]string{"GetImageRef", "PullImage"}, ErrImagePull},
				{[]string{"GetImageRef", "PullImage"}, ErrImagePull},
				{[]string{"GetImageRef"}, ErrImagePullBackOff},
				{[]string{"GetImageRef", "PullImage"}, ErrImagePull},
				{[]string{"GetImageRef"}, ErrImagePullBackOff},
				{[]string{"GetImageRef"}, ErrImagePullBackOff},
			}},
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

	fakeRuntime.ImageList = []Image{{ID: "present_image:latest"}}
	fakeRuntime.Err = c.pullerErr
	fakeRuntime.InspectErr = c.inspectErr

	puller = NewImageManager(fakeRecorder, fakeRuntime, backOff, serialized, 0, 0)
	return
}

func TestParallelPuller(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test_pod",
			Namespace:       "test-ns",
			UID:             "bar",
			ResourceVersion: "42",
			SelfLink:        "/api/v1/pods/foo",
		}}

	cases := pullerTestCases()

	useSerializedEnv := false
	for i, c := range cases {
		puller, fakeClock, fakeRuntime, container := pullerTestEnv(c, useSerializedEnv)

		for tick, expected := range c.expected {
			fakeRuntime.CalledFunctions = nil
			fakeClock.Step(time.Second)
			_, _, err := puller.EnsureImageExists(pod, container, nil, nil)
			assert.NoError(t, fakeRuntime.AssertCalls(expected.calls), "in test %d tick=%d", i, tick)
			assert.Equal(t, expected.err, err, "in test %d tick=%d", i, tick)
		}
	}
}

func TestSerializedPuller(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test_pod",
			Namespace:       "test-ns",
			UID:             "bar",
			ResourceVersion: "42",
			SelfLink:        "/api/v1/pods/foo",
		}}

	cases := pullerTestCases()

	useSerializedEnv := true
	for i, c := range cases {
		puller, fakeClock, fakeRuntime, container := pullerTestEnv(c, useSerializedEnv)

		for tick, expected := range c.expected {
			fakeRuntime.CalledFunctions = nil
			fakeClock.Step(time.Second)
			_, _, err := puller.EnsureImageExists(pod, container, nil, nil)
			assert.NoError(t, fakeRuntime.AssertCalls(expected.calls), "in test %d tick=%d", i, tick)
			assert.Equal(t, expected.err, err, "in test %d tick=%d", i, tick)
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

func TestPullAndListImageWithPodAnnotations(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test_pod",
			Namespace:       "test-ns",
			UID:             "bar",
			ResourceVersion: "42",
			SelfLink:        "/api/v1/pods/foo",
			Annotations: map[string]string{
				"kubernetes.io/runtimehandler": "handler_name",
			},
		}}
	c := pullerTestCase{ // pull missing image
		containerImage: "missing_image",
		policy:         v1.PullIfNotPresent,
		inspectErr:     nil,
		pullerErr:      nil,
		expected: []pullerExpects{
			{[]string{"GetImageRef", "PullImage"}, nil},
		}}

	useSerializedEnv := true
	puller, fakeClock, fakeRuntime, container := pullerTestEnv(c, useSerializedEnv)
	fakeRuntime.CalledFunctions = nil
	fakeRuntime.ImageList = []Image{}
	fakeClock.Step(time.Second)

	_, _, err := puller.EnsureImageExists(pod, container, nil, nil)
	assert.NoError(t, fakeRuntime.AssertCalls(c.expected[0].calls), "tick=%d", 0)
	assert.Equal(t, c.expected[0].err, err, "tick=%d", 0)

	images, _ := fakeRuntime.ListImages()
	assert.Equal(t, 1, len(images), "ListImages() count")

	image := images[0]
	assert.Equal(t, "missing_image:latest", image.ID, "Image ID")

	expectedAnnotations := []Annotation{
		{
			Name:  "kubernetes.io/runtimehandler",
			Value: "handler_name",
		}}
	assert.Equal(t, expectedAnnotations, image.Spec.Annotations, "image spec annotations")
}
