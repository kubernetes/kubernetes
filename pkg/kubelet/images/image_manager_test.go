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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	ctest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	testingclock "k8s.io/utils/clock/testing"
)

type pullerExpects struct {
	calls []string
	err   error
}

type pullerTestCase struct {
	testName       string
	containerImage string
	policy         v1.PullPolicy
	inspectErr     error
	pullerErr      error
	qps            float32
	burst          int
	expected       []pullerExpects
}

func pullerTestCases() []pullerTestCase {
	return []pullerTestCase{
		{ // pull missing image
			testName:       "image missing, pull",
			containerImage: "missing_image",
			policy:         v1.PullIfNotPresent,
			inspectErr:     nil,
			pullerErr:      nil,
			qps:            0.0,
			burst:          0,
			expected: []pullerExpects{
				{[]string{"GetImageRef", "PullImage"}, nil},
			}},

		{ // image present, don't pull
			testName:       "image present, don't pull ",
			containerImage: "present_image",
			policy:         v1.PullIfNotPresent,
			inspectErr:     nil,
			pullerErr:      nil,
			qps:            0.0,
			burst:          0,
			expected: []pullerExpects{
				{[]string{"GetImageRef"}, nil},
				{[]string{"GetImageRef"}, nil},
				{[]string{"GetImageRef"}, nil},
			}},
		// image present, pull it
		{containerImage: "present_image",
			testName:   "image present, pull ",
			policy:     v1.PullAlways,
			inspectErr: nil,
			pullerErr:  nil,
			qps:        0.0,
			burst:      0,
			expected: []pullerExpects{
				{[]string{"GetImageRef", "PullImage"}, nil},
				{[]string{"GetImageRef", "PullImage"}, nil},
				{[]string{"GetImageRef", "PullImage"}, nil},
			}},
		// missing image, error PullNever
		{containerImage: "missing_image",
			testName:   "image missing, never pull",
			policy:     v1.PullNever,
			inspectErr: nil,
			pullerErr:  nil,
			qps:        0.0,
			burst:      0,
			expected: []pullerExpects{
				{[]string{"GetImageRef"}, ErrImageNeverPull},
				{[]string{"GetImageRef"}, ErrImageNeverPull},
				{[]string{"GetImageRef"}, ErrImageNeverPull},
			}},
		// missing image, unable to inspect
		{containerImage: "missing_image",
			testName:   "image missing, pull if not present",
			policy:     v1.PullIfNotPresent,
			inspectErr: errors.New("unknown inspectError"),
			pullerErr:  nil,
			qps:        0.0,
			burst:      0,
			expected: []pullerExpects{
				{[]string{"GetImageRef"}, ErrImageInspect},
				{[]string{"GetImageRef"}, ErrImageInspect},
				{[]string{"GetImageRef"}, ErrImageInspect},
			}},
		// missing image, unable to fetch
		{containerImage: "typo_image",
			testName:   "image missing, unable to fetch",
			policy:     v1.PullIfNotPresent,
			inspectErr: nil,
			pullerErr:  errors.New("404"),
			qps:        0.0,
			burst:      0,
			expected: []pullerExpects{
				{[]string{"GetImageRef", "PullImage"}, ErrImagePull},
				{[]string{"GetImageRef", "PullImage"}, ErrImagePull},
				{[]string{"GetImageRef"}, ErrImagePullBackOff},
				{[]string{"GetImageRef", "PullImage"}, ErrImagePull},
				{[]string{"GetImageRef"}, ErrImagePullBackOff},
				{[]string{"GetImageRef"}, ErrImagePullBackOff},
			}},
		// image present, non-zero qps, try to pull
		{containerImage: "present_image",
			testName:   "image present and qps>0, pull",
			policy:     v1.PullAlways,
			inspectErr: nil,
			pullerErr:  nil,
			qps:        400.0,
			burst:      600,
			expected: []pullerExpects{
				{[]string{"GetImageRef", "PullImage"}, nil},
				{[]string{"GetImageRef", "PullImage"}, nil},
				{[]string{"GetImageRef", "PullImage"}, nil},
			}},
		// image present, non-zero qps, try to pull when qps exceeded
		{containerImage: "present_image",
			testName:   "image present and excessive qps rate, pull",
			policy:     v1.PullAlways,
			inspectErr: nil,
			pullerErr:  nil,
			qps:        2000.0,
			burst:      0,
			expected: []pullerExpects{
				{[]string{"GetImageRef"}, ErrImagePull},
				{[]string{"GetImageRef"}, ErrImagePull},
				{[]string{"GetImageRef"}, ErrImagePullBackOff},
			}},
	}
}

func pullerTestEnv(c pullerTestCase, serialized bool) (puller ImageManager, fakeClock *testingclock.FakeClock, fakeRuntime *ctest.FakeRuntime, container *v1.Container) {
	container = &v1.Container{
		Name:            "container_name",
		Image:           c.containerImage,
		ImagePullPolicy: c.policy,
	}

	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	fakeClock = testingclock.NewFakeClock(time.Now())
	backOff.Clock = fakeClock

	fakeRuntime = &ctest.FakeRuntime{}
	fakeRecorder := &record.FakeRecorder{}

	fakeRuntime.ImageList = []kubecontainer.Image{{ID: "present_image:latest"}}
	fakeRuntime.Err = c.pullerErr
	fakeRuntime.InspectErr = c.inspectErr

	puller = NewImageManager(fakeRecorder, fakeRuntime, backOff, serialized, c.qps, c.burst, nil)
	return
}

func TestParallelPuller(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test_pod",
			Namespace:       "test-ns",
			UID:             "bar",
			ResourceVersion: "42",
		}}

	cases := pullerTestCases()

	useSerializedEnv := false
	for _, c := range cases {
		puller, fakeClock, fakeRuntime, container := pullerTestEnv(c, useSerializedEnv)

		t.Run(c.testName, func(t *testing.T) {
			for _, expected := range c.expected {
				fakeRuntime.CalledFunctions = nil
				fakeClock.Step(time.Second)
				_, _, err := puller.EnsureImageExists(pod, container, nil, nil)
				fakeRuntime.AssertCalls(expected.calls)
				assert.Equal(t, expected.err, err)
			}
		})
	}
}

func TestSerializedPuller(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test_pod",
			Namespace:       "test-ns",
			UID:             "bar",
			ResourceVersion: "42",
		}}

	cases := pullerTestCases()

	useSerializedEnv := true
	for _, c := range cases {
		puller, fakeClock, fakeRuntime, container := pullerTestEnv(c, useSerializedEnv)

		t.Run(c.testName, func(t *testing.T) {
			for _, expected := range c.expected {
				fakeRuntime.CalledFunctions = nil
				fakeClock.Step(time.Second)
				_, _, err := puller.EnsureImageExists(pod, container, nil, nil)
				fakeRuntime.AssertCalls(expected.calls)
				assert.Equal(t, expected.err, err)
			}
		})
	}
}

func TestApplyDefaultImageTag(t *testing.T) {
	for _, testCase := range []struct {
		testName string
		Input    string
		Output   string
	}{
		{testName: "root", Input: "root", Output: "root:latest"},
		{testName: "root:tag", Input: "root:tag", Output: "root:tag"},
		{testName: "root@sha", Input: "root@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", Output: "root@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
	} {
		t.Run(testCase.testName, func(t *testing.T) {
			image, err := applyDefaultImageTag(testCase.Input)
			if err != nil {
				t.Errorf("applyDefaultImageTag(%s) failed: %v", testCase.Input, err)
			} else if image != testCase.Output {
				t.Errorf("Expected image reference: %q, got %q", testCase.Output, image)
			}
		})
	}
}

func TestPullAndListImageWithPodAnnotations(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test_pod",
			Namespace:       "test-ns",
			UID:             "bar",
			ResourceVersion: "42",
			Annotations: map[string]string{
				"kubernetes.io/runtimehandler": "handler_name",
			},
		}}
	c := pullerTestCase{ // pull missing image
		testName:       "test pull and list image with pod annotations",
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
	fakeRuntime.ImageList = []kubecontainer.Image{}
	fakeClock.Step(time.Second)

	t.Run(c.testName, func(t *testing.T) {
		_, _, err := puller.EnsureImageExists(pod, container, nil, nil)
		fakeRuntime.AssertCalls(c.expected[0].calls)
		assert.Equal(t, c.expected[0].err, err, "tick=%d", 0)

		images, _ := fakeRuntime.ListImages()
		assert.Equal(t, 1, len(images), "ListImages() count")

		image := images[0]
		assert.Equal(t, "missing_image:latest", image.ID, "Image ID")

		expectedAnnotations := []kubecontainer.Annotation{
			{
				Name:  "kubernetes.io/runtimehandler",
				Value: "handler_name",
			}}
		assert.Equal(t, expectedAnnotations, image.Spec.Annotations, "image spec annotations")
	})
}

func TestShouldPullImage(t *testing.T) {
	pullIfNotPresent := &v1.Container{
		Name:            "container_name",
		Image:           "container_image",
		ImagePullPolicy: v1.PullIfNotPresent,
	}
	pullNever := &v1.Container{
		Name:            "container_name",
		Image:           "container_image",
		ImagePullPolicy: v1.PullNever,
	}
	pullAlways := &v1.Container{
		Name:            "container_name",
		Image:           "container_image",
		ImagePullPolicy: v1.PullAlways,
	}
	tests := []struct {
		description       string
		container         *v1.Container
		imagePresent      bool
		pulledBySecret    bool
		ensuredBySecret   bool
		expectedWithFGOff bool
		expectedWithFGOn  bool
	}{
		{
			description:       "PullAlways should always pull esp. if not present",
			container:         pullAlways,
			imagePresent:      false,
			pulledBySecret:    false,
			ensuredBySecret:   false,
			expectedWithFGOff: true,
			expectedWithFGOn:  true,
		},
		{
			description:       "PullAlways should always pull even if present and not pulled by secret",
			container:         pullAlways,
			imagePresent:      true,
			pulledBySecret:    false,
			ensuredBySecret:   false,
			expectedWithFGOff: true,
			expectedWithFGOn:  true,
		},
		{
			description:       "PullAlways should always pull even if present and ensuredBySecret",
			container:         pullAlways,
			imagePresent:      true,
			pulledBySecret:    true,
			ensuredBySecret:   true,
			expectedWithFGOff: true,
			expectedWithFGOn:  true,
		},
		{
			description:       "PullIfNotPresent should pull if not present",
			container:         pullIfNotPresent,
			imagePresent:      false,
			pulledBySecret:    false,
			ensuredBySecret:   false,
			expectedWithFGOff: true,
			expectedWithFGOn:  true,
		},
		{
			description:       "PullIfNotPresent should pull if not present even if pulledBySecret and ensuredBySecret",
			container:         pullIfNotPresent,
			imagePresent:      false,
			pulledBySecret:    true,
			ensuredBySecret:   true,
			expectedWithFGOff: true,
			expectedWithFGOn:  true,
		},
		{
			description:       "PullIfNotPresent should not pull if present (and secrets are not involved)",
			container:         pullIfNotPresent,
			imagePresent:      true,
			pulledBySecret:    false,
			ensuredBySecret:   false,
			expectedWithFGOff: false,
			expectedWithFGOn:  false,
		},
		{
			description:       "PullIfNotPresent should not pull if present (and secrets are involved and match)",
			container:         pullIfNotPresent,
			imagePresent:      true,
			pulledBySecret:    true,
			ensuredBySecret:   true,
			expectedWithFGOff: false,
			expectedWithFGOn:  false,
		},
		{
			description:       "PullIfNotPresent should pull if present and secrets are involved and no match, unless ensure fg is off",
			container:         pullIfNotPresent,
			imagePresent:      true,
			pulledBySecret:    true,
			ensuredBySecret:   false,
			expectedWithFGOff: false,
			expectedWithFGOn:  true,
		},
		{
			description:       "PullNever should never report pull, but we'll throw error in EnsureImageExists()",
			container:         pullNever,
			imagePresent:      false,
			pulledBySecret:    false,
			ensuredBySecret:   false,
			expectedWithFGOff: false,
			expectedWithFGOn:  false,
		},
		{
			description:       "PullNever should never pull even if pulled by secret and not ensured by secret",
			container:         pullNever,
			imagePresent:      true,
			pulledBySecret:    true,
			ensuredBySecret:   false,
			expectedWithFGOff: false,
			expectedWithFGOn:  false,
		},
	}

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletEnsureSecretPulledImages, true)()

	for i, test := range tests {
		sp := shouldPullImage(test.container, test.imagePresent, test.pulledBySecret, test.ensuredBySecret)
		if utilfeature.DefaultFeatureGate.Enabled(features.KubeletEnsureSecretPulledImages) {
			assert.Equal(t, test.expectedWithFGOn, sp, "TestCase[%d]: %s ensured image fg enabled", i, test.description)
		} else {
			assert.Equal(t, test.expectedWithFGOff, sp, "TestCase[%d]: %s ensured image fg disabled", i, test.description)
		}
	}
}
