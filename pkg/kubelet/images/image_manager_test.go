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
	"context"
	"errors"
	"fmt"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	crierrors "k8s.io/cri-api/pkg/errors"
	"k8s.io/kubernetes/pkg/features"
	. "k8s.io/kubernetes/pkg/kubelet/container"
	ctest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	testingclock "k8s.io/utils/clock/testing"
	utilpointer "k8s.io/utils/pointer"
)

type pullerExpects struct {
	calls                           []string
	err                             error
	shouldRecordStartedPullingTime  bool
	shouldRecordFinishedPullingTime bool
}

type singleContainerPullerTestCase struct {
	testName       string
	containerImage string
	policy         v1.PullPolicy
	inspectErr     error
	pullerErr      error
	qps            float32
	burst          int
	expected       []pullerExpects
}

func pullerTestCases() []singleContainerPullerTestCase {
	return []singleContainerPullerTestCase{
		{ // pull missing image
			testName:       "image missing, pull",
			containerImage: "missing_image",
			policy:         v1.PullIfNotPresent,
			inspectErr:     nil,
			pullerErr:      nil,
			qps:            0.0,
			burst:          0,
			expected: []pullerExpects{
				{[]string{"GetImageRef", "PullImage", "GetImageSize"}, nil, true, true},
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
				{[]string{"GetImageRef"}, nil, false, false},
				{[]string{"GetImageRef"}, nil, false, false},
				{[]string{"GetImageRef"}, nil, false, false},
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
				{[]string{"GetImageRef", "PullImage", "GetImageSize"}, nil, true, true},
				{[]string{"GetImageRef", "PullImage", "GetImageSize"}, nil, true, true},
				{[]string{"GetImageRef", "PullImage", "GetImageSize"}, nil, true, true},
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
				{[]string{"GetImageRef"}, ErrImageNeverPull, false, false},
				{[]string{"GetImageRef"}, ErrImageNeverPull, false, false},
				{[]string{"GetImageRef"}, ErrImageNeverPull, false, false},
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
				{[]string{"GetImageRef"}, ErrImageInspect, false, false},
				{[]string{"GetImageRef"}, ErrImageInspect, false, false},
				{[]string{"GetImageRef"}, ErrImageInspect, false, false},
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
				{[]string{"GetImageRef", "PullImage"}, ErrImagePull, true, false},
				{[]string{"GetImageRef", "PullImage"}, ErrImagePull, true, false},
				{[]string{"GetImageRef"}, ErrImagePullBackOff, false, false},
				{[]string{"GetImageRef", "PullImage"}, ErrImagePull, true, false},
				{[]string{"GetImageRef"}, ErrImagePullBackOff, false, false},
				{[]string{"GetImageRef"}, ErrImagePullBackOff, false, false},
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
				{[]string{"GetImageRef", "PullImage", "GetImageSize"}, nil, true, true},
				{[]string{"GetImageRef", "PullImage", "GetImageSize"}, nil, true, true},
				{[]string{"GetImageRef", "PullImage", "GetImageSize"}, nil, true, true},
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
				{[]string{"GetImageRef"}, ErrImagePull, true, false},
				{[]string{"GetImageRef"}, ErrImagePull, true, false},
				{[]string{"GetImageRef"}, ErrImagePullBackOff, false, false},
			}},
		// error case if image name fails validation due to invalid reference format
		{containerImage: "FAILED_IMAGE",
			testName:   "invalid image name, no pull",
			policy:     v1.PullAlways,
			inspectErr: nil,
			pullerErr:  nil,
			qps:        0.0,
			burst:      0,
			expected: []pullerExpects{
				{[]string(nil), ErrInvalidImageName, false, false},
			}},
		// error case if image name contains http
		{containerImage: "http://url",
			testName:   "invalid image name with http, no pull",
			policy:     v1.PullAlways,
			inspectErr: nil,
			pullerErr:  nil,
			qps:        0.0,
			burst:      0,
			expected: []pullerExpects{
				{[]string(nil), ErrInvalidImageName, false, false},
			}},
		// error case if image name contains sha256
		{containerImage: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
			testName:   "invalid image name with sha256, no pull",
			policy:     v1.PullAlways,
			inspectErr: nil,
			pullerErr:  nil,
			qps:        0.0,
			burst:      0,
			expected: []pullerExpects{
				{[]string(nil), ErrInvalidImageName, false, false},
			}},
	}
}

type mockPodPullingTimeRecorder struct {
	sync.Mutex
	startedPullingRecorded  bool
	finishedPullingRecorded bool
}

func (m *mockPodPullingTimeRecorder) RecordImageStartedPulling(podUID types.UID) {
	m.Lock()
	defer m.Unlock()
	m.startedPullingRecorded = true
}

func (m *mockPodPullingTimeRecorder) RecordImageFinishedPulling(podUID types.UID) {
	m.Lock()
	defer m.Unlock()
	m.finishedPullingRecorded = true
}

func (m *mockPodPullingTimeRecorder) reset() {
	m.Lock()
	defer m.Unlock()
	m.startedPullingRecorded = false
	m.finishedPullingRecorded = false
}

func pullerTestEnvWithSingleContainer(t *testing.T, c singleContainerPullerTestCase, serialized bool, maxParallelImagePulls *int32) (puller ImageManager, fakeClock *testingclock.FakeClock, fakeRuntime *ctest.FakeRuntime, container *v1.Container, fakePodPullingTimeRecorder *mockPodPullingTimeRecorder) {
	container = &v1.Container{
		Name:            "container_name",
		Image:           c.containerImage,
		ImagePullPolicy: c.policy,
	}
	fakeClock, fakeRuntime, fakePodPullingTimeRecorder, puller = pullerTestEnv(t, pullerTestConfig{
		qps:                   c.qps,
		burst:                 c.burst,
		serialized:            serialized,
		maxParallelImagePulls: maxParallelImagePulls,
		inspectErr:            c.inspectErr,
		pullerErr:             c.pullerErr,
	})
	return
}

type pullerTestConfig struct {
	inspectErr            error
	pullerErr             error
	qps                   float32
	burst                 int
	serialized            bool
	maxParallelImagePulls *int32
}

func pullerTestEnv(t *testing.T, c pullerTestConfig) (*testingclock.FakeClock, *ctest.FakeRuntime, *mockPodPullingTimeRecorder, ImageManager) {
	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	fakeClock := testingclock.NewFakeClock(time.Now())
	backOff.Clock = fakeClock

	fakeRuntime := &ctest.FakeRuntime{T: t}
	fakeRecorder := &record.FakeRecorder{}

	fakeRuntime.ImageList = []Image{{ID: "present_image:latest"}}
	fakeRuntime.Err = c.pullerErr
	fakeRuntime.InspectErr = c.inspectErr

	fakePodPullingTimeRecorder := &mockPodPullingTimeRecorder{}

	puller := NewImageManager(fakeRecorder, fakeRuntime, backOff, c.serialized, c.maxParallelImagePulls, c.qps, c.burst, fakePodPullingTimeRecorder)
	return fakeClock, fakeRuntime, fakePodPullingTimeRecorder, puller
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
		t.Run(c.testName, func(t *testing.T) {
			ctx := context.Background()
			puller, fakeClock, fakeRuntime, container, fakePodPullingTimeRecorder := pullerTestEnvWithSingleContainer(t, c, useSerializedEnv, nil)

			for _, expected := range c.expected {
				fakeRuntime.CalledFunctions = nil
				fakeClock.Step(time.Second)

				_, _, err := puller.EnsureImageExists(ctx, nil, pod, container.Image, nil, nil, "", container.ImagePullPolicy)
				fakeRuntime.AssertCalls(expected.calls)
				assert.Equal(t, expected.err, err)
				assert.Equal(t, expected.shouldRecordStartedPullingTime, fakePodPullingTimeRecorder.startedPullingRecorded)
				assert.Equal(t, expected.shouldRecordFinishedPullingTime, fakePodPullingTimeRecorder.finishedPullingRecorded)
				fakePodPullingTimeRecorder.reset()
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
		t.Run(c.testName, func(t *testing.T) {
			ctx := context.Background()
			puller, fakeClock, fakeRuntime, container, fakePodPullingTimeRecorder := pullerTestEnvWithSingleContainer(t, c, useSerializedEnv, nil)

			for _, expected := range c.expected {
				fakeRuntime.CalledFunctions = nil
				fakeClock.Step(time.Second)

				_, _, err := puller.EnsureImageExists(ctx, nil, pod, container.Image, nil, nil, "", container.ImagePullPolicy)
				fakeRuntime.AssertCalls(expected.calls)
				assert.Equal(t, expected.err, err)
				assert.Equal(t, expected.shouldRecordStartedPullingTime, fakePodPullingTimeRecorder.startedPullingRecorded)
				assert.Equal(t, expected.shouldRecordFinishedPullingTime, fakePodPullingTimeRecorder.finishedPullingRecorded)
				fakePodPullingTimeRecorder.reset()
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
		{testName: "root:latest@sha", Input: "root:latest@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", Output: "root:latest@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
		{testName: "root:latest", Input: "root:latest", Output: "root:latest"},
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
	c := singleContainerPullerTestCase{ // pull missing image
		testName:       "test pull and list image with pod annotations",
		containerImage: "missing_image",
		policy:         v1.PullIfNotPresent,
		inspectErr:     nil,
		pullerErr:      nil,
		expected: []pullerExpects{
			{[]string{"GetImageRef", "PullImage", "GetImageSize"}, nil, true, true},
		}}

	useSerializedEnv := true
	t.Run(c.testName, func(t *testing.T) {
		ctx := context.Background()
		puller, fakeClock, fakeRuntime, container, fakePodPullingTimeRecorder := pullerTestEnvWithSingleContainer(t, c, useSerializedEnv, nil)
		fakeRuntime.CalledFunctions = nil
		fakeRuntime.ImageList = []Image{}
		fakeClock.Step(time.Second)

		_, _, err := puller.EnsureImageExists(ctx, nil, pod, container.Image, nil, nil, "", container.ImagePullPolicy)
		fakeRuntime.AssertCalls(c.expected[0].calls)
		assert.Equal(t, c.expected[0].err, err, "tick=%d", 0)
		assert.Equal(t, c.expected[0].shouldRecordStartedPullingTime, fakePodPullingTimeRecorder.startedPullingRecorded)
		assert.Equal(t, c.expected[0].shouldRecordFinishedPullingTime, fakePodPullingTimeRecorder.finishedPullingRecorded)

		images, _ := fakeRuntime.ListImages(ctx)
		assert.Len(t, images, 1, "ListImages() count")

		image := images[0]
		assert.Equal(t, "missing_image:latest", image.ID, "Image ID")
		assert.Equal(t, "", image.Spec.RuntimeHandler, "image.Spec.RuntimeHandler not empty", "ImageID", image.ID)

		expectedAnnotations := []Annotation{
			{
				Name:  "kubernetes.io/runtimehandler",
				Value: "handler_name",
			}}
		assert.Equal(t, expectedAnnotations, image.Spec.Annotations, "image spec annotations")
	})
}

func TestPullAndListImageWithRuntimeHandlerInImageCriAPIFeatureGate(t *testing.T) {
	runtimeHandler := "handler_name"
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test_pod",
			Namespace:       "test-ns",
			UID:             "bar",
			ResourceVersion: "42",
			Annotations: map[string]string{
				"kubernetes.io/runtimehandler": runtimeHandler,
			},
		},
		Spec: v1.PodSpec{
			RuntimeClassName: &runtimeHandler,
		},
	}
	c := singleContainerPullerTestCase{ // pull missing image
		testName:       "test pull and list image with pod annotations",
		containerImage: "missing_image",
		policy:         v1.PullIfNotPresent,
		inspectErr:     nil,
		pullerErr:      nil,
		expected: []pullerExpects{
			{[]string{"GetImageRef", "PullImage", "GetImageSize"}, nil, true, true},
		}}

	useSerializedEnv := true
	t.Run(c.testName, func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RuntimeClassInImageCriAPI, true)
		ctx := context.Background()
		puller, fakeClock, fakeRuntime, container, fakePodPullingTimeRecorder := pullerTestEnvWithSingleContainer(t, c, useSerializedEnv, nil)
		fakeRuntime.CalledFunctions = nil
		fakeRuntime.ImageList = []Image{}
		fakeClock.Step(time.Second)

		_, _, err := puller.EnsureImageExists(ctx, nil, pod, container.Image, nil, nil, runtimeHandler, container.ImagePullPolicy)
		fakeRuntime.AssertCalls(c.expected[0].calls)
		assert.Equal(t, c.expected[0].err, err, "tick=%d", 0)
		assert.Equal(t, c.expected[0].shouldRecordStartedPullingTime, fakePodPullingTimeRecorder.startedPullingRecorded)
		assert.Equal(t, c.expected[0].shouldRecordFinishedPullingTime, fakePodPullingTimeRecorder.finishedPullingRecorded)

		images, _ := fakeRuntime.ListImages(ctx)
		assert.Len(t, images, 1, "ListImages() count")

		image := images[0]
		assert.Equal(t, "missing_image:latest", image.ID, "Image ID")

		// when RuntimeClassInImageCriAPI feature gate is enabled, check runtime
		// handler information for every image in the ListImages() response
		assert.Equal(t, runtimeHandler, image.Spec.RuntimeHandler, "runtime handler returned not as expected", "Image ID", image)

		expectedAnnotations := []Annotation{
			{
				Name:  "kubernetes.io/runtimehandler",
				Value: "handler_name",
			}}
		assert.Equal(t, expectedAnnotations, image.Spec.Annotations, "image spec annotations")
	})
}

func TestMaxParallelImagePullsLimitWithDifferentImages(t *testing.T) {
	ctx := context.Background()

	var (
		maxParallelImagePulls = 5
		podCountSum           = maxParallelImagePulls + 2 // 7 pods total
	)

	// 7 pods with different image spec
	pods := []*v1.Pod{}
	for i := 0; i < podCountSum; i++ {
		pods = append(pods, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            fmt.Sprintf("test_pod_%d", i),
				Namespace:       "test-ns",
				UID:             types.UID(fmt.Sprintf("bar_%d", i)),
				ResourceVersion: strconv.Itoa(i),
				Annotations:     map[string]string{"key1": "val1"},
			},
			Spec: v1.PodSpec{
				RuntimeClassName: utilpointer.String("default"),
				Containers: []v1.Container{
					{
						Name:            "container1",
						Image:           fmt.Sprintf("missing_image_%d", i),
						ImagePullPolicy: v1.PullAlways,
					},
				},
			},
		})
		if i > 4 {
			// The last pods[5,6] with same image id
			pods[i].Spec.Containers[0].Image = "missing_image_same_id"
			// The last pods[5,6] with other annotation
			pods[i].Annotations["key2"] = "val2"
		}
		// The last pods[6] with different runtimeHandler
		if i > 5 {
			pods[i].Spec.RuntimeClassName = utilpointer.String("other")
		}
	}

	// ParallelImagePuller with max concurrent count 5
	testConf := pullerTestConfig{
		inspectErr:            nil,
		pullerErr:             nil,
		qps:                   0.0,
		burst:                 0,
		serialized:            false,
		maxParallelImagePulls: utilpointer.Int32(int32(maxParallelImagePulls)),
	}
	fakeClock, fakeRuntime, _, puller := pullerTestEnv(t, testConf)
	fakeRuntime.BlockImagePulls = true
	fakeRuntime.CalledFunctions = nil
	fakeRuntime.T = t
	fakeClock.Step(time.Second)

	var wg sync.WaitGroup

	// First 5 EnsureImageExists should result in runtime calls
	for i := 0; i < maxParallelImagePulls; i++ {
		wg.Add(1)
		go func(pod *v1.Pod) {
			container := pod.Spec.Containers[0]
			_, _, err := puller.EnsureImageExists(ctx, nil, pod, container.Image, nil, nil, *pod.Spec.RuntimeClassName, container.ImagePullPolicy)
			assert.Nil(t, err)
			wg.Done()
		}(pods[i])
	}
	time.Sleep(1 * time.Second)
	fakeRuntime.AssertCallCounts("PullImage", 5)

	// Next two EnsureImageExists should be blocked because maxParallelImagePulls is hit
	for i := maxParallelImagePulls; i < podCountSum; i++ {
		wg.Add(1)
		go func(pod *v1.Pod) {
			container := pod.Spec.Containers[0]
			_, _, err := puller.EnsureImageExists(ctx, nil, pod, container.Image, nil, nil, *pod.Spec.RuntimeClassName, container.ImagePullPolicy)
			assert.Nil(t, err)
			wg.Done()
		}(pods[i])
	}
	time.Sleep(1 * time.Second)
	fakeRuntime.AssertCallCounts("PullImage", 5)

	// Unblock two image pulls from runtime, and two EnsureImageExists can go through
	fakeRuntime.UnblockImagePulls(2)
	time.Sleep(1 * time.Second)
	fakeRuntime.AssertCallCounts("PullImage", 7)

	// Unblock the remaining 5 image pulls from runtime, and all EnsureImageExists can go through
	fakeRuntime.UnblockImagePulls(5)

	wg.Wait()
	fakeRuntime.AssertCallCounts("PullImage", 7)
}

func TestParallelImagePullsSeriallyWithSameImage(t *testing.T) {
	ctx := context.Background()

	maxParallelImagePulls := 5
	distinctImageIDCount := maxParallelImagePulls - 1

	pods := []*v1.Pod{}

	// 7 Pods with 4 distinct image
	for i := 0; i < maxParallelImagePulls+2; i++ {
		imageIndex := i % distinctImageIDCount
		pods = append(pods, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            fmt.Sprintf("test_pod_%d", i),
				Namespace:       "test-ns",
				UID:             types.UID(fmt.Sprintf("bar_%d", i)),
				ResourceVersion: strconv.Itoa(i),
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:            "container1",
						Image:           fmt.Sprintf("missing_image_%d", imageIndex),
						ImagePullPolicy: v1.PullAlways,
					},
				},
			},
		})
	}

	// ParallelImagePuller with max concurrent count 5
	testConf := pullerTestConfig{
		inspectErr:            nil,
		pullerErr:             nil,
		qps:                   0.0,
		burst:                 0,
		serialized:            false,
		maxParallelImagePulls: utilpointer.Int32(int32(maxParallelImagePulls)),
	}
	fakeClock, fakeRuntime, _, puller := pullerTestEnv(t, testConf)
	fakeRuntime.BlockImagePulls = true
	fakeRuntime.CalledFunctions = nil
	fakeRuntime.T = t
	fakeClock.Step(time.Second)

	for i := 0; i < maxParallelImagePulls+2; i++ {
		go func(pod *v1.Pod) {
			container := pod.Spec.Containers[0]
			_, _, err := puller.EnsureImageExists(ctx, nil, pod, container.Image, nil, nil, "", container.ImagePullPolicy)
			assert.NoError(t, err)
		}(pods[i])
	}

	time.Sleep(1 * time.Second)
	// 7 EnsureImageExists should result in runtime calls of 4.
	// Same image pull requests should be serially.
	fakeRuntime.AssertCallCounts("PullImage", 4)

}

func TestEvalCRIPullErr(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name   string
		input  error
		assert func(string, error)
	}{
		{
			name:  "fallback error",
			input: errors.New("test"),
			assert: func(msg string, err error) {
				assert.ErrorIs(t, err, ErrImagePull)
				assert.Contains(t, msg, "test")
			},
		},
		{
			name:  "registry is unavailable",
			input: crierrors.ErrRegistryUnavailable,
			assert: func(msg string, err error) {
				assert.ErrorIs(t, err, crierrors.ErrRegistryUnavailable)
				assert.Equal(t, msg, "image pull failed for test because the registry is unavailable")
			},
		},
		{
			name:  "registry is unavailable with additional error message",
			input: fmt.Errorf("%v: foo", crierrors.ErrRegistryUnavailable),
			assert: func(msg string, err error) {
				assert.ErrorIs(t, err, crierrors.ErrRegistryUnavailable)
				assert.Equal(t, msg, "image pull failed for test because the registry is unavailable: foo")
			},
		},
		{
			name:  "signature is invalid",
			input: crierrors.ErrSignatureValidationFailed,
			assert: func(msg string, err error) {
				assert.ErrorIs(t, err, crierrors.ErrSignatureValidationFailed)
				assert.Equal(t, msg, "image pull failed for test because the signature validation failed")
			},
		},
		{
			name:  "signature is invalid with additional error message (wrapped)",
			input: fmt.Errorf("%w: bar", crierrors.ErrSignatureValidationFailed),
			assert: func(msg string, err error) {
				assert.ErrorIs(t, err, crierrors.ErrSignatureValidationFailed)
				assert.Equal(t, msg, "image pull failed for test because the signature validation failed: bar")
			},
		},
	} {
		testInput := tc.input
		testAssert := tc.assert

		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			msg, err := evalCRIPullErr("test", testInput)
			testAssert(msg, err)
		})
	}
}

func TestFilterAnnotationsWithRuntimeNoEffect(t *testing.T) {
	tests := []struct {
		name     string
		input    map[string]string
		expected []Annotation
	}{
		{
			name: "No runtime no effect keys",
			input: map[string]string{
				"key1": "value1",
				"key2": "value2",
			},
			expected: []Annotation{
				{Name: "key1", Value: "value1"},
				{Name: "key2", Value: "value2"},
			},
		},
		{
			name: "With runtime no effect keys",
			input: map[string]string{
				RuntimeNoEffectAnnotationKey: `["key2","key3"]`,
				"key1":                       "value1",
				"key2":                       "value2",
				"key3":                       "value3",
			},
			expected: []Annotation{
				{Name: "key1", Value: "value1"},
			},
		},
		{
			name: "Invalid runtime no effect keys",
			input: map[string]string{
				RuntimeNoEffectAnnotationKey: "invalid json",
				"key1":                       "value1",
			},
			expected: []Annotation{
				{Name: "key1", Value: "value1"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actual := filterAnnotationsWithRuntimeNoEffect(tt.input)
			assert.Equal(t, tt.expected, actual)
		})
	}
}
