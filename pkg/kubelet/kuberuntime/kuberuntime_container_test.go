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

package kuberuntime

import (
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

// TestRemoveContainer tests removing the container and its corresponding container logs.
func TestRemoveContainer(t *testing.T) {
	fakeRuntime, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "bar",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "foo",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
				},
			},
		},
	}

	// Create fake sandbox and container
	_, fakeContainers := makeAndSetFakePod(t, m, fakeRuntime, pod)
	assert.Equal(t, len(fakeContainers), 1)

	containerID := fakeContainers[0].Id
	fakeOS := m.osInterface.(*containertest.FakeOS)
	err = m.removeContainer(containerID)
	assert.NoError(t, err)
	// Verify container log is removed
	expectedContainerLogPath := filepath.Join(podLogsRootDirectory, "new_bar_12345678", "foo", "0.log")
	expectedContainerLogSymlink := legacyLogSymlink(containerID, "foo", "bar", "new")
	assert.Equal(t, fakeOS.Removes, []string{expectedContainerLogPath, expectedContainerLogSymlink})
	// Verify container is removed
	assert.Contains(t, fakeRuntime.Called, "RemoveContainer")
	containers, err := fakeRuntime.ListContainers(&runtimeapi.ContainerFilter{Id: containerID})
	assert.NoError(t, err)
	assert.Empty(t, containers)
}

// TestKillContainer tests killing the container in a Pod.
func TestKillContainer(t *testing.T) {
	_, _, m, _ := createTestRuntimeManager()

	tests := []struct {
		caseName            string
		pod                 *v1.Pod
		containerID         kubecontainer.ContainerID
		containerName       string
		reason              string
		gracePeriodOverride int64
		succeed             bool
	}{
		{
			caseName: "Failed to find container in pods, expect to return error",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{UID: "pod1_id", Name: "pod1", Namespace: "default"},
				Spec:       v1.PodSpec{Containers: []v1.Container{{Name: "empty_container"}}},
			},
			containerID:         kubecontainer.ContainerID{Type: "docker", ID: "not_exist_container_id"},
			containerName:       "not_exist_container",
			reason:              "unknown reason",
			gracePeriodOverride: 0,
			succeed:             false,
		},
	}

	for _, test := range tests {
		err := m.killContainer(test.pod, test.containerID, test.containerName, test.reason, &test.gracePeriodOverride)
		if test.succeed != (err == nil) {
			t.Errorf("%s: expected %v, got %v (%v)", test.caseName, test.succeed, (err == nil), err)
		}
	}
}

// TestToKubeContainerStatus tests the converting the CRI container status to
// the internal type (i.e., toKubeContainerStatus()) for containers in
// different states.
func TestToKubeContainerStatus(t *testing.T) {
	cid := &kubecontainer.ContainerID{Type: "testRuntime", ID: "dummyid"}
	meta := &runtimeapi.ContainerMetadata{Name: "cname", Attempt: 3}
	imageSpec := &runtimeapi.ImageSpec{Image: "fimage"}
	var (
		createdAt  int64 = 327
		startedAt  int64 = 999
		finishedAt int64 = 1278
	)

	for desc, test := range map[string]struct {
		input    *runtimeapi.ContainerStatus
		expected *kubecontainer.ContainerStatus
	}{
		"created container": {
			input: &runtimeapi.ContainerStatus{
				Id:        cid.ID,
				Metadata:  meta,
				Image:     imageSpec,
				State:     runtimeapi.ContainerState_CONTAINER_CREATED,
				CreatedAt: createdAt,
			},
			expected: &kubecontainer.ContainerStatus{
				ID:        *cid,
				Image:     imageSpec.Image,
				State:     kubecontainer.ContainerStateCreated,
				CreatedAt: time.Unix(0, createdAt),
			},
		},
		"running container": {
			input: &runtimeapi.ContainerStatus{
				Id:        cid.ID,
				Metadata:  meta,
				Image:     imageSpec,
				State:     runtimeapi.ContainerState_CONTAINER_RUNNING,
				CreatedAt: createdAt,
				StartedAt: startedAt,
			},
			expected: &kubecontainer.ContainerStatus{
				ID:        *cid,
				Image:     imageSpec.Image,
				State:     kubecontainer.ContainerStateRunning,
				CreatedAt: time.Unix(0, createdAt),
				StartedAt: time.Unix(0, startedAt),
			},
		},
		"exited container": {
			input: &runtimeapi.ContainerStatus{
				Id:         cid.ID,
				Metadata:   meta,
				Image:      imageSpec,
				State:      runtimeapi.ContainerState_CONTAINER_EXITED,
				CreatedAt:  createdAt,
				StartedAt:  startedAt,
				FinishedAt: finishedAt,
				ExitCode:   int32(121),
				Reason:     "GotKilled",
				Message:    "The container was killed",
			},
			expected: &kubecontainer.ContainerStatus{
				ID:         *cid,
				Image:      imageSpec.Image,
				State:      kubecontainer.ContainerStateExited,
				CreatedAt:  time.Unix(0, createdAt),
				StartedAt:  time.Unix(0, startedAt),
				FinishedAt: time.Unix(0, finishedAt),
				ExitCode:   121,
				Reason:     "GotKilled",
				Message:    "The container was killed",
			},
		},
		"unknown container": {
			input: &runtimeapi.ContainerStatus{
				Id:        cid.ID,
				Metadata:  meta,
				Image:     imageSpec,
				State:     runtimeapi.ContainerState_CONTAINER_UNKNOWN,
				CreatedAt: createdAt,
				StartedAt: startedAt,
			},
			expected: &kubecontainer.ContainerStatus{
				ID:        *cid,
				Image:     imageSpec.Image,
				State:     kubecontainer.ContainerStateUnknown,
				CreatedAt: time.Unix(0, createdAt),
				StartedAt: time.Unix(0, startedAt),
			},
		},
	} {
		actual := toKubeContainerStatus(test.input, cid.Type)
		assert.Equal(t, test.expected, actual, desc)
	}
}

func TestLifeCycleHook(t *testing.T) {

	// Setup
	fakeRuntime, _, m, _ := createTestRuntimeManager()

	gracePeriod := int64(30)
	cID := kubecontainer.ContainerID{
		Type: "docker",
		ID:   "foo",
	}

	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "bar",
			Namespace: "default",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "foo",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					Command:         []string{"testCommand"},
					WorkingDir:      "testWorkingDir",
				},
			},
		},
	}
	cmdPostStart := &v1.Lifecycle{
		PostStart: &v1.Handler{
			Exec: &v1.ExecAction{
				Command: []string{"PostStartCMD"},
			},
		},
	}

	httpLifeCycle := &v1.Lifecycle{
		PreStop: &v1.Handler{
			HTTPGet: &v1.HTTPGetAction{
				Host: "testHost.com",
				Path: "/GracefulExit",
			},
		},
	}

	cmdLifeCycle := &v1.Lifecycle{
		PreStop: &v1.Handler{
			Exec: &v1.ExecAction{
				Command: []string{"PreStopCMD"},
			},
		},
	}

	fakeRunner := &containertest.FakeContainerCommandRunner{}
	fakeHTTP := &fakeHTTP{}

	lcHanlder := lifecycle.NewHandlerRunner(
		fakeHTTP,
		fakeRunner,
		nil)

	m.runner = lcHanlder

	// Configured and works as expected
	t.Run("PreStop-CMDExec", func(t *testing.T) {
		testPod.Spec.Containers[0].Lifecycle = cmdLifeCycle
		m.killContainer(testPod, cID, "foo", "testKill", &gracePeriod)
		if fakeRunner.Cmd[0] != cmdLifeCycle.PreStop.Exec.Command[0] {
			t.Errorf("CMD Prestop hook was not invoked")
		}
	})

	// Configured and working HTTP hook
	t.Run("PreStop-HTTPGet", func(t *testing.T) {
		defer func() { fakeHTTP.url = "" }()
		testPod.Spec.Containers[0].Lifecycle = httpLifeCycle
		m.killContainer(testPod, cID, "foo", "testKill", &gracePeriod)

		if !strings.Contains(fakeHTTP.url, httpLifeCycle.PreStop.HTTPGet.Host) {
			t.Errorf("HTTP Prestop hook was not invoked")
		}
	})

	// When there is no time to run PreStopHook
	t.Run("PreStop-NoTimeToRun", func(t *testing.T) {
		gracePeriodLocal := int64(0)

		testPod.DeletionGracePeriodSeconds = &gracePeriodLocal
		testPod.Spec.TerminationGracePeriodSeconds = &gracePeriodLocal

		m.killContainer(testPod, cID, "foo", "testKill", &gracePeriodLocal)

		if strings.Contains(fakeHTTP.url, httpLifeCycle.PreStop.HTTPGet.Host) {
			t.Errorf("HTTP Should not execute when gracePeriod is 0")
		}
	})

	// Post Start script
	t.Run("PostStart-CmdExe", func(t *testing.T) {

		// Fake all the things you need before trying to create a container
		fakeSandBox, _ := makeAndSetFakePod(t, m, fakeRuntime, testPod)
		fakeSandBoxConfig, _ := m.generatePodSandboxConfig(testPod, 0)
		testPod.Spec.Containers[0].Lifecycle = cmdPostStart
		testContainer := &testPod.Spec.Containers[0]
		fakePodStatus := &kubecontainer.PodStatus{
			ContainerStatuses: []*kubecontainer.ContainerStatus{
				{
					ID: kubecontainer.ContainerID{
						Type: "docker",
						ID:   testContainer.Name,
					},
					Name:      testContainer.Name,
					State:     kubecontainer.ContainerStateCreated,
					CreatedAt: time.Unix(0, time.Now().Unix()),
				},
			},
		}

		// Now try to create a container, which should in turn invoke PostStart Hook
		_, err := m.startContainer(fakeSandBox.Id, fakeSandBoxConfig, containerStartSpec(testContainer), testPod, fakePodStatus, nil, "", []string{})
		if err != nil {
			t.Errorf("startContainer error =%v", err)
		}
		if fakeRunner.Cmd[0] != cmdPostStart.PostStart.Exec.Command[0] {
			t.Errorf("CMD PostStart hook was not invoked")
		}
	})
}

func TestStartSpec(t *testing.T) {
	podStatus := &kubecontainer.PodStatus{
		ContainerStatuses: []*kubecontainer.ContainerStatus{
			{
				ID: kubecontainer.ContainerID{
					Type: "docker",
					ID:   "docker-something-something",
				},
				Name: "target",
			},
		},
	}

	for _, tc := range []struct {
		name string
		spec *startSpec
		want *kubecontainer.ContainerID
	}{
		{
			"Regular Container",
			containerStartSpec(&v1.Container{
				Name: "test",
			}),
			nil,
		},
		{
			"Ephemeral Container w/o Target",
			ephemeralContainerStartSpec(&v1.EphemeralContainer{
				EphemeralContainerCommon: v1.EphemeralContainerCommon{
					Name: "test",
				},
			}),
			nil,
		},
		{
			"Ephemeral Container w/ Target",
			ephemeralContainerStartSpec(&v1.EphemeralContainer{
				EphemeralContainerCommon: v1.EphemeralContainerCommon{
					Name: "test",
				},
				TargetContainerName: "target",
			}),
			&kubecontainer.ContainerID{
				Type: "docker",
				ID:   "docker-something-something",
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EphemeralContainers, true)()
			if got, err := tc.spec.getTargetID(podStatus); err != nil {
				t.Fatalf("%v: getTargetID got unexpected error: %v", t.Name(), err)
			} else if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("%v: getTargetID got unexpected result. diff:\n%v", t.Name(), diff)
			}
		})

		// Test with feature disabled in self-contained section which can be removed when feature flag is removed.
		t.Run(fmt.Sprintf("%s (disabled)", tc.name), func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EphemeralContainers, false)()
			if got, err := tc.spec.getTargetID(podStatus); err != nil {
				t.Fatalf("%v: getTargetID got unexpected error: %v", t.Name(), err)
			} else if got != nil {
				t.Errorf("%v: getTargetID got: %v, wanted nil", t.Name(), got)
			}
		})
	}
}
