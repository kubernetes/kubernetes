// +build linux

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

package dockertools

import (
	"fmt"
	"net"
	"path"
	"strconv"
	"testing"

	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/network/mock_network"
	"k8s.io/kubernetes/pkg/security/apparmor"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
)

func TestGetSecurityOpts(t *testing.T) {
	const containerName = "bar"
	pod := func(annotations map[string]string) *v1.Pod {
		p := makePod("foo", &v1.PodSpec{
			Containers: []v1.Container{
				{Name: containerName},
			},
		})
		p.Annotations = annotations
		return p
	}

	tests := []struct {
		msg          string
		pod          *v1.Pod
		expectedOpts []string
	}{{
		msg:          "No security annotations",
		pod:          pod(nil),
		expectedOpts: []string{"seccomp=unconfined"},
	}, {
		msg: "Seccomp default",
		pod: pod(map[string]string{
			v1.SeccompContainerAnnotationKeyPrefix + containerName: "docker/default",
		}),
		expectedOpts: nil,
	}, {
		msg: "AppArmor runtime/default",
		pod: pod(map[string]string{
			apparmor.ContainerAnnotationKeyPrefix + containerName: apparmor.ProfileRuntimeDefault,
		}),
		expectedOpts: []string{"seccomp=unconfined"},
	}, {
		msg: "AppArmor local profile",
		pod: pod(map[string]string{
			apparmor.ContainerAnnotationKeyPrefix + containerName: apparmor.ProfileNamePrefix + "foo",
		}),
		expectedOpts: []string{"seccomp=unconfined", "apparmor=foo"},
	}, {
		msg: "AppArmor and seccomp profile",
		pod: pod(map[string]string{
			v1.SeccompContainerAnnotationKeyPrefix + containerName: "docker/default",
			apparmor.ContainerAnnotationKeyPrefix + containerName:  apparmor.ProfileNamePrefix + "foo",
		}),
		expectedOpts: []string{"apparmor=foo"},
	}}

	dm, _ := newTestDockerManagerWithVersion("1.11.1", "1.23")
	for i, test := range tests {
		securityOpts, err := dm.getSecurityOpts(test.pod, containerName)
		assert.NoError(t, err, "TestCase[%d]: %s", i, test.msg)
		opts, err := dm.fmtDockerOpts(securityOpts)
		assert.NoError(t, err, "TestCase[%d]: %s", i, test.msg)
		assert.Len(t, opts, len(test.expectedOpts), "TestCase[%d]: %s", i, test.msg)
		for _, opt := range test.expectedOpts {
			assert.Contains(t, opts, opt, "TestCase[%d]: %s", i, test.msg)
		}
	}
}

func TestSeccompIsUnconfinedByDefaultWithDockerV110(t *testing.T) {
	dm, fakeDocker := newTestDockerManagerWithVersion("1.10.1", "1.22")
	// We want to capture events.
	recorder := record.NewFakeRecorder(20)
	dm.recorder = recorder

	pod := makePod("foo", &v1.PodSpec{
		Containers: []v1.Container{
			{Name: "bar"},
		},
	})

	runSyncPod(t, dm, fakeDocker, pod, nil, false)

	verifyCalls(t, fakeDocker, []string{
		// Create pod infra container.
		"create", "start", "inspect_container", "inspect_container",
		// Create container.
		"create", "start", "inspect_container",
	})

	fakeDocker.Lock()
	if len(fakeDocker.Created) != 2 ||
		!matchString(t, "/k8s_POD\\.[a-f0-9]+_foo_new_", fakeDocker.Created[0]) ||
		!matchString(t, "/k8s_bar\\.[a-f0-9]+_foo_new_", fakeDocker.Created[1]) {
		t.Errorf("unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()

	newContainer, err := fakeDocker.InspectContainer(fakeDocker.Created[1])
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	assert.Contains(t, newContainer.HostConfig.SecurityOpt, "seccomp:unconfined", "Pods with Docker versions >= 1.10 must not have seccomp disabled by default")

	cid := utilstrings.ShortenString(fakeDocker.Created[1], 12)
	assert.NoError(t, expectEvent(recorder, v1.EventTypeNormal, events.CreatedContainer,
		fmt.Sprintf("Created container with docker id %s; Security:[seccomp=unconfined]", cid)))
}

func TestUnconfinedSeccompProfileWithDockerV110(t *testing.T) {
	dm, fakeDocker := newTestDockerManagerWithVersion("1.10.1", "1.22")
	pod := makePod("foo4", &v1.PodSpec{
		Containers: []v1.Container{
			{Name: "bar4"},
		},
	})
	pod.Annotations = map[string]string{
		v1.SeccompPodAnnotationKey: "unconfined",
	}

	runSyncPod(t, dm, fakeDocker, pod, nil, false)

	verifyCalls(t, fakeDocker, []string{
		// Create pod infra container.
		"create", "start", "inspect_container", "inspect_container",
		// Create container.
		"create", "start", "inspect_container",
	})

	fakeDocker.Lock()
	if len(fakeDocker.Created) != 2 ||
		!matchString(t, "/k8s_POD\\.[a-f0-9]+_foo4_new_", fakeDocker.Created[0]) ||
		!matchString(t, "/k8s_bar4\\.[a-f0-9]+_foo4_new_", fakeDocker.Created[1]) {
		t.Errorf("unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()

	newContainer, err := fakeDocker.InspectContainer(fakeDocker.Created[1])
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	assert.Contains(t, newContainer.HostConfig.SecurityOpt, "seccomp:unconfined", "Pods created with a secccomp annotation of unconfined should have seccomp:unconfined.")
}

func TestDefaultSeccompProfileWithDockerV110(t *testing.T) {
	dm, fakeDocker := newTestDockerManagerWithVersion("1.10.1", "1.22")
	pod := makePod("foo1", &v1.PodSpec{
		Containers: []v1.Container{
			{Name: "bar1"},
		},
	})
	pod.Annotations = map[string]string{
		v1.SeccompPodAnnotationKey: "docker/default",
	}

	runSyncPod(t, dm, fakeDocker, pod, nil, false)

	verifyCalls(t, fakeDocker, []string{
		// Create pod infra container.
		"create", "start", "inspect_container", "inspect_container",
		// Create container.
		"create", "start", "inspect_container",
	})

	fakeDocker.Lock()
	if len(fakeDocker.Created) != 2 ||
		!matchString(t, "/k8s_POD\\.[a-f0-9]+_foo1_new_", fakeDocker.Created[0]) ||
		!matchString(t, "/k8s_bar1\\.[a-f0-9]+_foo1_new_", fakeDocker.Created[1]) {
		t.Errorf("unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()

	newContainer, err := fakeDocker.InspectContainer(fakeDocker.Created[1])
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	assert.NotContains(t, newContainer.HostConfig.SecurityOpt, "seccomp:unconfined", "Pods created with a secccomp annotation of docker/default should have empty security opt.")
}

func TestSeccompContainerAnnotationTrumpsPod(t *testing.T) {
	dm, fakeDocker := newTestDockerManagerWithVersion("1.10.1", "1.22")
	pod := makePod("foo2", &v1.PodSpec{
		Containers: []v1.Container{
			{Name: "bar2"},
		},
	})
	pod.Annotations = map[string]string{
		v1.SeccompPodAnnotationKey:                      "unconfined",
		v1.SeccompContainerAnnotationKeyPrefix + "bar2": "docker/default",
	}

	runSyncPod(t, dm, fakeDocker, pod, nil, false)

	verifyCalls(t, fakeDocker, []string{
		// Create pod infra container.
		"create", "start", "inspect_container", "inspect_container",
		// Create container.
		"create", "start", "inspect_container",
	})

	fakeDocker.Lock()
	if len(fakeDocker.Created) != 2 ||
		!matchString(t, "/k8s_POD\\.[a-f0-9]+_foo2_new_", fakeDocker.Created[0]) ||
		!matchString(t, "/k8s_bar2\\.[a-f0-9]+_foo2_new_", fakeDocker.Created[1]) {
		t.Errorf("unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()

	newContainer, err := fakeDocker.InspectContainer(fakeDocker.Created[1])
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	assert.NotContains(t, newContainer.HostConfig.SecurityOpt, "seccomp:unconfined", "Container annotation should trump the pod annotation for seccomp.")
}

func TestSecurityOptsAreNilWithDockerV19(t *testing.T) {
	dm, fakeDocker := newTestDockerManagerWithVersion("1.9.1", "1.21")
	pod := makePod("foo", &v1.PodSpec{
		Containers: []v1.Container{
			{Name: "bar"},
		},
	})

	runSyncPod(t, dm, fakeDocker, pod, nil, false)

	verifyCalls(t, fakeDocker, []string{
		// Create pod infra container.
		"create", "start", "inspect_container", "inspect_container",
		// Create container.
		"create", "start", "inspect_container",
	})

	fakeDocker.Lock()
	if len(fakeDocker.Created) != 2 ||
		!matchString(t, "/k8s_POD\\.[a-f0-9]+_foo_new_", fakeDocker.Created[0]) ||
		!matchString(t, "/k8s_bar\\.[a-f0-9]+_foo_new_", fakeDocker.Created[1]) {
		t.Errorf("unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()

	newContainer, err := fakeDocker.InspectContainer(fakeDocker.Created[1])
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	assert.NotContains(t, newContainer.HostConfig.SecurityOpt, "seccomp:unconfined", "Pods with Docker versions < 1.10 must not have seccomp disabled by default")
}

func TestCreateAppArmorContanier(t *testing.T) {
	dm, fakeDocker := newTestDockerManagerWithVersion("1.11.1", "1.23")
	// We want to capture events.
	recorder := record.NewFakeRecorder(20)
	dm.recorder = recorder

	pod := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
			Annotations: map[string]string{
				apparmor.ContainerAnnotationKeyPrefix + "test": apparmor.ProfileNamePrefix + "test-profile",
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{Name: "test"},
			},
		},
	}

	runSyncPod(t, dm, fakeDocker, pod, nil, false)

	verifyCalls(t, fakeDocker, []string{
		// Create pod infra container.
		"create", "start", "inspect_container", "inspect_container",
		// Create container.
		"create", "start", "inspect_container",
	})

	fakeDocker.Lock()
	if len(fakeDocker.Created) != 2 ||
		!matchString(t, "/k8s_POD\\.[a-f0-9]+_foo_new_", fakeDocker.Created[0]) ||
		!matchString(t, "/k8s_test\\.[a-f0-9]+_foo_new_", fakeDocker.Created[1]) {
		t.Errorf("unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()

	// Verify security opts.
	newContainer, err := fakeDocker.InspectContainer(fakeDocker.Created[1])
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	securityOpts := newContainer.HostConfig.SecurityOpt
	assert.Contains(t, securityOpts, "apparmor=test-profile", "Container should have apparmor security opt")

	cid := utilstrings.ShortenString(fakeDocker.Created[1], 12)
	assert.NoError(t, expectEvent(recorder, v1.EventTypeNormal, events.CreatedContainer,
		fmt.Sprintf("Created container with docker id %s; Security:[seccomp=unconfined apparmor=test-profile]", cid)))
}

func TestSeccompLocalhostProfileIsLoaded(t *testing.T) {
	tests := []struct {
		annotations    map[string]string
		expectedSecOpt string
		expectedSecMsg string
		expectedError  string
	}{
		{
			annotations: map[string]string{
				v1.SeccompPodAnnotationKey: "localhost/test",
			},
			expectedSecOpt: `seccomp={"foo":"bar"}`,
			expectedSecMsg: "seccomp=test(md5:21aeae45053385adebd25311f9dd9cb1)",
		},
		{
			annotations: map[string]string{
				v1.SeccompPodAnnotationKey: "localhost/sub/subtest",
			},
			expectedSecOpt: `seccomp={"abc":"def"}`,
			expectedSecMsg: "seccomp=sub/subtest(md5:07c9bcb4db631f7ca191d6e0bca49f76)",
		},
		{
			annotations: map[string]string{
				v1.SeccompPodAnnotationKey: "localhost/not-existing",
			},
			expectedError: "cannot load seccomp profile",
		},
	}

	for i, test := range tests {
		dm, fakeDocker := newTestDockerManagerWithVersion("1.11.0", "1.23")
		// We want to capture events.
		recorder := record.NewFakeRecorder(20)
		dm.recorder = recorder

		dm.seccompProfileRoot = path.Join("fixtures", "seccomp")

		pod := makePod("foo2", &v1.PodSpec{
			Containers: []v1.Container{
				{Name: "bar2"},
			},
		})
		pod.Annotations = test.annotations

		result := runSyncPod(t, dm, fakeDocker, pod, nil, test.expectedError != "")
		if test.expectedError != "" {
			assert.Contains(t, result.Error().Error(), test.expectedError)
			continue
		}

		verifyCalls(t, fakeDocker, []string{
			// Create pod infra container.
			"create", "start", "inspect_container", "inspect_container",
			// Create container.
			"create", "start", "inspect_container",
		})

		fakeDocker.Lock()
		if len(fakeDocker.Created) != 2 ||
			!matchString(t, "/k8s_POD\\.[a-f0-9]+_foo2_new_", fakeDocker.Created[0]) ||
			!matchString(t, "/k8s_bar2\\.[a-f0-9]+_foo2_new_", fakeDocker.Created[1]) {
			t.Errorf("unexpected containers created %v", fakeDocker.Created)
		}
		fakeDocker.Unlock()

		newContainer, err := fakeDocker.InspectContainer(fakeDocker.Created[1])
		if err != nil {
			t.Fatalf("unexpected error %v", err)
		}
		assert.Contains(t, newContainer.HostConfig.SecurityOpt, test.expectedSecOpt, "The compacted seccomp json profile should be loaded.")

		cid := utilstrings.ShortenString(fakeDocker.Created[1], 12)
		assert.NoError(t, expectEvent(recorder, v1.EventTypeNormal, events.CreatedContainer,
			fmt.Sprintf("Created container with docker id %s; Security:[%s]", cid, test.expectedSecMsg)),
			"testcase %d", i)
	}
}

func TestGetPodStatusFromNetworkPlugin(t *testing.T) {
	cases := []struct {
		pod                *v1.Pod
		fakePodIP          string
		containerID        string
		infraContainerID   string
		networkStatusError error
		expectRunning      bool
		expectUnknown      bool
	}{
		{
			pod: &v1.Pod{
				ObjectMeta: v1.ObjectMeta{
					UID:       "12345678",
					Name:      "foo",
					Namespace: "new",
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "container"}},
				},
			},
			fakePodIP:          "10.10.10.10",
			containerID:        "123",
			infraContainerID:   "9876",
			networkStatusError: nil,
			expectRunning:      true,
			expectUnknown:      false,
		},
		{
			pod: &v1.Pod{
				ObjectMeta: v1.ObjectMeta{
					UID:       "12345678",
					Name:      "foo",
					Namespace: "new",
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "container"}},
				},
			},
			fakePodIP:          "",
			containerID:        "123",
			infraContainerID:   "9876",
			networkStatusError: fmt.Errorf("CNI plugin error"),
			expectRunning:      false,
			expectUnknown:      true,
		},
	}
	for _, test := range cases {
		dm, fakeDocker := newTestDockerManager()
		ctrl := gomock.NewController(t)
		fnp := mock_network.NewMockNetworkPlugin(ctrl)
		dm.networkPlugin = fnp

		fakeDocker.SetFakeRunningContainers([]*FakeContainer{
			{
				ID:      test.containerID,
				Name:    fmt.Sprintf("/k8s_container_%s_%s_%s_42", test.pod.Name, test.pod.Namespace, test.pod.UID),
				Running: true,
			},
			{
				ID:      test.infraContainerID,
				Name:    fmt.Sprintf("/k8s_POD.%s_%s_%s_%s_42", strconv.FormatUint(generatePodInfraContainerHash(test.pod), 16), test.pod.Name, test.pod.Namespace, test.pod.UID),
				Running: true,
			},
		})

		fnp.EXPECT().Name().Return("someNetworkPlugin").AnyTimes()
		var podNetworkStatus *network.PodNetworkStatus
		if test.fakePodIP != "" {
			podNetworkStatus = &network.PodNetworkStatus{IP: net.ParseIP(test.fakePodIP)}
		}
		fnp.EXPECT().GetPodNetworkStatus(test.pod.Namespace, test.pod.Name, kubecontainer.DockerID(test.infraContainerID).ContainerID()).Return(podNetworkStatus, test.networkStatusError)

		podStatus, err := dm.GetPodStatus(test.pod.UID, test.pod.Name, test.pod.Namespace)
		if err != nil {
			t.Fatal(err)
		}
		if podStatus.IP != test.fakePodIP {
			t.Errorf("Got wrong ip, expected %v, got %v", test.fakePodIP, podStatus.IP)
		}

		expectedStatesCount := 0
		var expectedState kubecontainer.ContainerState
		if test.expectRunning {
			expectedState = kubecontainer.ContainerStateRunning
		} else if test.expectUnknown {
			expectedState = kubecontainer.ContainerStateUnknown
		} else {
			t.Errorf("Some state has to be expected")
		}
		for _, containerStatus := range podStatus.ContainerStatuses {
			if containerStatus.State == expectedState {
				expectedStatesCount++
			}
		}
		if expectedStatesCount < 1 {
			t.Errorf("Invalid count of containers with expected state")
		}
	}
}
