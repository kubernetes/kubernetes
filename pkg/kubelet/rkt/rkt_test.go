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

package rkt

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"testing"
	"time"

	appcschema "github.com/appc/spec/schema"
	appctypes "github.com/appc/spec/schema/types"
	rktapi "github.com/coreos/rkt/api/v1alpha"
	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
)

func mustMarshalPodManifest(man *appcschema.PodManifest) []byte {
	manblob, err := json.Marshal(man)
	if err != nil {
		panic(err)
	}
	return manblob
}

func mustMarshalImageManifest(man *appcschema.ImageManifest) []byte {
	manblob, err := json.Marshal(man)
	if err != nil {
		panic(err)
	}
	return manblob
}

func mustRktHash(hash string) *appctypes.Hash {
	h, err := appctypes.NewHash(hash)
	if err != nil {
		panic(err)
	}
	return h
}

func makeRktPod(rktPodState rktapi.PodState,
	rktPodID, podUID, podName, podNamespace,
	podIP, podCreationTs, podRestartCount string,
	appNames, imgIDs, imgNames, containerHashes []string,
	appStates []rktapi.AppState, exitcodes []int32) *rktapi.Pod {

	podManifest := &appcschema.PodManifest{
		ACKind:    appcschema.PodManifestKind,
		ACVersion: appcschema.AppContainerVersion,
		Annotations: appctypes.Annotations{
			appctypes.Annotation{
				Name:  *appctypes.MustACIdentifier(k8sRktKubeletAnno),
				Value: k8sRktKubeletAnnoValue,
			},
			appctypes.Annotation{
				Name:  *appctypes.MustACIdentifier(k8sRktUIDAnno),
				Value: podUID,
			},
			appctypes.Annotation{
				Name:  *appctypes.MustACIdentifier(k8sRktNameAnno),
				Value: podName,
			},
			appctypes.Annotation{
				Name:  *appctypes.MustACIdentifier(k8sRktNamespaceAnno),
				Value: podNamespace,
			},
			appctypes.Annotation{
				Name:  *appctypes.MustACIdentifier(k8sRktCreationTimeAnno),
				Value: podCreationTs,
			},
			appctypes.Annotation{
				Name:  *appctypes.MustACIdentifier(k8sRktRestartCountAnno),
				Value: podRestartCount,
			},
		},
	}

	appNum := len(appNames)
	if appNum != len(imgNames) ||
		appNum != len(imgIDs) ||
		appNum != len(containerHashes) ||
		appNum != len(appStates) {
		panic("inconsistent app number")
	}

	apps := make([]*rktapi.App, appNum)
	for i := range appNames {
		apps[i] = &rktapi.App{
			Name:  appNames[i],
			State: appStates[i],
			Image: &rktapi.Image{
				Id:   imgIDs[i],
				Name: imgNames[i],
				Manifest: mustMarshalImageManifest(
					&appcschema.ImageManifest{
						ACKind:    appcschema.ImageManifestKind,
						ACVersion: appcschema.AppContainerVersion,
						Name:      *appctypes.MustACIdentifier(imgNames[i]),
						Annotations: appctypes.Annotations{
							appctypes.Annotation{
								Name:  *appctypes.MustACIdentifier(k8sRktContainerHashAnno),
								Value: containerHashes[i],
							},
						},
					},
				),
			},
			ExitCode: exitcodes[i],
		}
		podManifest.Apps = append(podManifest.Apps, appcschema.RuntimeApp{
			Name:  *appctypes.MustACName(appNames[i]),
			Image: appcschema.RuntimeImage{ID: *mustRktHash("sha512-foo")},
			Annotations: appctypes.Annotations{
				appctypes.Annotation{
					Name:  *appctypes.MustACIdentifier(k8sRktContainerHashAnno),
					Value: containerHashes[i],
				},
			},
		})
	}

	return &rktapi.Pod{
		Id:       rktPodID,
		State:    rktPodState,
		Networks: []*rktapi.Network{{Name: defaultNetworkName, Ipv4: podIP}},
		Apps:     apps,
		Manifest: mustMarshalPodManifest(podManifest),
	}
}

func TestCheckVersion(t *testing.T) {
	fr := newFakeRktInterface()
	fs := newFakeSystemd()
	r := &Runtime{apisvc: fr, systemd: fs}

	fr.info = rktapi.Info{
		RktVersion:  "1.2.3+git",
		AppcVersion: "1.2.4+git",
		ApiVersion:  "1.2.6-alpha",
	}
	fs.version = "100"
	tests := []struct {
		minimumRktBinVersion     string
		recommendedRktBinVersion string
		minimumAppcVersion       string
		minimumRktApiVersion     string
		minimumSystemdVersion    string
		err                      error
		calledGetInfo            bool
		calledSystemVersion      bool
	}{
		// Good versions.
		{
			"1.2.3",
			"1.2.3",
			"1.2.4",
			"1.2.5",
			"99",
			nil,
			true,
			true,
		},
		// Good versions.
		{
			"1.2.3+git",
			"1.2.3+git",
			"1.2.4+git",
			"1.2.6-alpha",
			"100",
			nil,
			true,
			true,
		},
		// Requires greater binary version.
		{
			"1.2.4",
			"1.2.4",
			"1.2.4",
			"1.2.6-alpha",
			"100",
			fmt.Errorf("rkt: binary version is too old(%v), requires at least %v", fr.info.RktVersion, "1.2.4"),
			true,
			true,
		},
		// Requires greater Appc version.
		{
			"1.2.3",
			"1.2.3",
			"1.2.5",
			"1.2.6-alpha",
			"100",
			fmt.Errorf("rkt: appc version is too old(%v), requires at least %v", fr.info.AppcVersion, "1.2.5"),
			true,
			true,
		},
		// Requires greater API version.
		{
			"1.2.3",
			"1.2.3",
			"1.2.4",
			"1.2.6",
			"100",
			fmt.Errorf("rkt: API version is too old(%v), requires at least %v", fr.info.ApiVersion, "1.2.6"),
			true,
			true,
		},
		// Requires greater API version.
		{
			"1.2.3",
			"1.2.3",
			"1.2.4",
			"1.2.7",
			"100",
			fmt.Errorf("rkt: API version is too old(%v), requires at least %v", fr.info.ApiVersion, "1.2.7"),
			true,
			true,
		},
		// Requires greater systemd version.
		{
			"1.2.3",
			"1.2.3",
			"1.2.4",
			"1.2.7",
			"101",
			fmt.Errorf("rkt: systemd version(%v) is too old, requires at least %v", fs.version, "101"),
			false,
			true,
		},
	}

	for i, tt := range tests {
		testCaseHint := fmt.Sprintf("test case #%d", i)
		err := r.checkVersion(tt.minimumRktBinVersion, tt.recommendedRktBinVersion, tt.minimumAppcVersion, tt.minimumRktApiVersion, tt.minimumSystemdVersion)
		assert.Equal(t, tt.err, err, testCaseHint)

		if tt.calledGetInfo {
			assert.Equal(t, fr.called, []string{"GetInfo"}, testCaseHint)
		}
		if tt.calledSystemVersion {
			assert.Equal(t, fs.called, []string{"Version"}, testCaseHint)
		}
		if err == nil {
			assert.Equal(t, fr.info.RktVersion, r.binVersion.String(), testCaseHint)
			assert.Equal(t, fr.info.AppcVersion, r.appcVersion.String(), testCaseHint)
			assert.Equal(t, fr.info.ApiVersion, r.apiVersion.String(), testCaseHint)
		}
		fr.CleanCalls()
		fs.CleanCalls()
	}
}

func TestListImages(t *testing.T) {
	fr := newFakeRktInterface()
	fs := newFakeSystemd()
	r := &Runtime{apisvc: fr, systemd: fs}

	tests := []struct {
		images   []*rktapi.Image
		expected []kubecontainer.Image
	}{
		{nil, []kubecontainer.Image{}},
		{
			[]*rktapi.Image{
				{
					Id:      "sha512-a2fb8f390702",
					Name:    "quay.io/coreos/alpine-sh",
					Version: "latest",
				},
			},
			[]kubecontainer.Image{
				{
					ID:       "sha512-a2fb8f390702",
					RepoTags: []string{"quay.io/coreos/alpine-sh:latest"},
				},
			},
		},
		{
			[]*rktapi.Image{
				{
					Id:      "sha512-a2fb8f390702",
					Name:    "quay.io/coreos/alpine-sh",
					Version: "latest",
					Size:    400,
				},
				{
					Id:      "sha512-c6b597f42816",
					Name:    "coreos.com/rkt/stage1-coreos",
					Version: "0.10.0",
					Size:    400,
				},
			},
			[]kubecontainer.Image{
				{
					ID:       "sha512-a2fb8f390702",
					RepoTags: []string{"quay.io/coreos/alpine-sh:latest"},
					Size:     400,
				},
				{
					ID:       "sha512-c6b597f42816",
					RepoTags: []string{"coreos.com/rkt/stage1-coreos:0.10.0"},
					Size:     400,
				},
			},
		},
	}

	for i, tt := range tests {
		fr.images = tt.images

		images, err := r.ListImages()
		if err != nil {
			t.Errorf("%v", err)
		}
		assert.Equal(t, tt.expected, images)
		assert.Equal(t, fr.called, []string{"ListImages"}, fmt.Sprintf("test case %d: unexpected called list", i))

		fr.CleanCalls()
	}
}

func TestGetPods(t *testing.T) {
	fr := newFakeRktInterface()
	fs := newFakeSystemd()
	r := &Runtime{apisvc: fr, systemd: fs}

	tests := []struct {
		pods   []*rktapi.Pod
		result []*kubecontainer.Pod
	}{
		// No pods.
		{},
		// One pod.
		{
			[]*rktapi.Pod{
				makeRktPod(rktapi.PodState_POD_STATE_RUNNING,
					"uuid-4002", "42", "guestbook", "default",
					"10.10.10.42", "100000", "7",
					[]string{"app-1", "app-2"},
					[]string{"img-id-1", "img-id-2"},
					[]string{"img-name-1", "img-name-2"},
					[]string{"1001", "1002"},
					[]rktapi.AppState{rktapi.AppState_APP_STATE_RUNNING, rktapi.AppState_APP_STATE_EXITED},
					[]int32{0, 0},
				),
			},
			[]*kubecontainer.Pod{
				{
					ID:        "42",
					Name:      "guestbook",
					Namespace: "default",
					Containers: []*kubecontainer.Container{
						{
							ID:      kubecontainer.BuildContainerID("rkt", "uuid-4002:app-1"),
							Name:    "app-1",
							Image:   "img-name-1",
							Hash:    1001,
							Created: 100000,
							State:   "running",
						},
						{
							ID:      kubecontainer.BuildContainerID("rkt", "uuid-4002:app-2"),
							Name:    "app-2",
							Image:   "img-name-2",
							Hash:    1002,
							Created: 100000,
							State:   "exited",
						},
					},
				},
			},
		},
		// Multiple pods.
		{
			[]*rktapi.Pod{
				makeRktPod(rktapi.PodState_POD_STATE_RUNNING,
					"uuid-4002", "42", "guestbook", "default",
					"10.10.10.42", "100000", "7",
					[]string{"app-1", "app-2"},
					[]string{"img-id-1", "img-id-2"},
					[]string{"img-name-1", "img-name-2"},
					[]string{"1001", "1002"},
					[]rktapi.AppState{rktapi.AppState_APP_STATE_RUNNING, rktapi.AppState_APP_STATE_EXITED},
					[]int32{0, 0},
				),
				makeRktPod(rktapi.PodState_POD_STATE_EXITED,
					"uuid-4003", "43", "guestbook", "default",
					"10.10.10.43", "90000", "7",
					[]string{"app-11", "app-22"},
					[]string{"img-id-11", "img-id-22"},
					[]string{"img-name-11", "img-name-22"},
					[]string{"10011", "10022"},
					[]rktapi.AppState{rktapi.AppState_APP_STATE_RUNNING, rktapi.AppState_APP_STATE_EXITED},
					[]int32{0, 0},
				),
			},
			[]*kubecontainer.Pod{
				{
					ID:        "42",
					Name:      "guestbook",
					Namespace: "default",
					Containers: []*kubecontainer.Container{
						{
							ID:      kubecontainer.BuildContainerID("rkt", "uuid-4002:app-1"),
							Name:    "app-1",
							Image:   "img-name-1",
							Hash:    1001,
							Created: 100000,
							State:   "running",
						},
						{
							ID:      kubecontainer.BuildContainerID("rkt", "uuid-4002:app-2"),
							Name:    "app-2",
							Image:   "img-name-2",
							Hash:    1002,
							Created: 100000,
							State:   "exited",
						},
					},
				},
				{
					ID:        "43",
					Name:      "guestbook",
					Namespace: "default",
					Containers: []*kubecontainer.Container{
						{
							ID:      kubecontainer.BuildContainerID("rkt", "uuid-4003:app-11"),
							Name:    "app-11",
							Image:   "img-name-11",
							Hash:    10011,
							Created: 90000,
							State:   "running",
						},
						{
							ID:      kubecontainer.BuildContainerID("rkt", "uuid-4003:app-22"),
							Name:    "app-22",
							Image:   "img-name-22",
							Hash:    10022,
							Created: 90000,
							State:   "exited",
						},
					},
				},
			},
		},
	}

	for i, tt := range tests {
		testCaseHint := fmt.Sprintf("test case #%d", i)
		fr.pods = tt.pods

		pods, err := r.GetPods(true)
		if err != nil {
			t.Errorf("test case #%d: unexpected error: %v", i, err)
		}

		assert.Equal(t, tt.result, pods, testCaseHint)
		assert.Equal(t, []string{"ListPods"}, fr.called, fmt.Sprintf("test case %d: unexpected called list", i))

		fr.CleanCalls()
	}
}

func TestGetPodsFilters(t *testing.T) {
	fr := newFakeRktInterface()
	fs := newFakeSystemd()
	r := &Runtime{apisvc: fr, systemd: fs}

	for _, test := range []struct {
		All             bool
		ExpectedFilters []*rktapi.PodFilter
	}{
		{
			true,
			[]*rktapi.PodFilter{
				{
					Annotations: []*rktapi.KeyValue{
						{
							Key:   k8sRktKubeletAnno,
							Value: k8sRktKubeletAnnoValue,
						},
					},
				},
			},
		},
		{
			false,
			[]*rktapi.PodFilter{
				{
					States: []rktapi.PodState{rktapi.PodState_POD_STATE_RUNNING},
					Annotations: []*rktapi.KeyValue{
						{
							Key:   k8sRktKubeletAnno,
							Value: k8sRktKubeletAnnoValue,
						},
					},
				},
			},
		},
	} {
		_, err := r.GetPods(test.All)
		if err != nil {
			t.Errorf("%v", err)
		}
		assert.Equal(t, test.ExpectedFilters, fr.podFilters, "filters didn't match when all=%b", test.All)
	}
}

func TestGetPodStatus(t *testing.T) {
	fr := newFakeRktInterface()
	fs := newFakeSystemd()
	r := &Runtime{apisvc: fr, systemd: fs}

	tests := []struct {
		pods   []*rktapi.Pod
		result *kubecontainer.PodStatus
	}{
		// No pods.
		{
			nil,
			&kubecontainer.PodStatus{ID: "42", Name: "guestbook", Namespace: "default"},
		},
		// One pod.
		{
			[]*rktapi.Pod{
				makeRktPod(rktapi.PodState_POD_STATE_RUNNING,
					"uuid-4002", "42", "guestbook", "default",
					"10.10.10.42", "100000", "7",
					[]string{"app-1", "app-2"},
					[]string{"img-id-1", "img-id-2"},
					[]string{"img-name-1", "img-name-2"},
					[]string{"1001", "1002"},
					[]rktapi.AppState{rktapi.AppState_APP_STATE_RUNNING, rktapi.AppState_APP_STATE_EXITED},
					[]int32{0, 0},
				),
			},
			&kubecontainer.PodStatus{
				ID:        "42",
				Name:      "guestbook",
				Namespace: "default",
				IP:        "10.10.10.42",
				ContainerStatuses: []*kubecontainer.ContainerStatus{
					{
						ID:           kubecontainer.BuildContainerID("rkt", "uuid-4002:app-1"),
						Name:         "app-1",
						State:        kubecontainer.ContainerStateRunning,
						CreatedAt:    time.Unix(100000, 0),
						StartedAt:    time.Unix(100000, 0),
						Image:        "img-name-1",
						ImageID:      "rkt://img-id-1",
						Hash:         1001,
						RestartCount: 7,
					},
					{
						ID:           kubecontainer.BuildContainerID("rkt", "uuid-4002:app-2"),
						Name:         "app-2",
						State:        kubecontainer.ContainerStateExited,
						CreatedAt:    time.Unix(100000, 0),
						StartedAt:    time.Unix(100000, 0),
						Image:        "img-name-2",
						ImageID:      "rkt://img-id-2",
						Hash:         1002,
						RestartCount: 7,
						Reason:       "Completed",
					},
				},
			},
		},
		// Multiple pods.
		{
			[]*rktapi.Pod{
				makeRktPod(rktapi.PodState_POD_STATE_EXITED,
					"uuid-4002", "42", "guestbook", "default",
					"10.10.10.42", "90000", "7",
					[]string{"app-1", "app-2"},
					[]string{"img-id-1", "img-id-2"},
					[]string{"img-name-1", "img-name-2"},
					[]string{"1001", "1002"},
					[]rktapi.AppState{rktapi.AppState_APP_STATE_RUNNING, rktapi.AppState_APP_STATE_EXITED},
					[]int32{0, 0},
				),
				makeRktPod(rktapi.PodState_POD_STATE_RUNNING, // The latest pod is running.
					"uuid-4003", "42", "guestbook", "default",
					"10.10.10.42", "100000", "10",
					[]string{"app-1", "app-2"},
					[]string{"img-id-1", "img-id-2"},
					[]string{"img-name-1", "img-name-2"},
					[]string{"1001", "1002"},
					[]rktapi.AppState{rktapi.AppState_APP_STATE_RUNNING, rktapi.AppState_APP_STATE_EXITED},
					[]int32{0, 1},
				),
			},
			&kubecontainer.PodStatus{
				ID:        "42",
				Name:      "guestbook",
				Namespace: "default",
				IP:        "10.10.10.42",
				// Result should contain all containers.
				ContainerStatuses: []*kubecontainer.ContainerStatus{
					{
						ID:           kubecontainer.BuildContainerID("rkt", "uuid-4002:app-1"),
						Name:         "app-1",
						State:        kubecontainer.ContainerStateRunning,
						CreatedAt:    time.Unix(90000, 0),
						StartedAt:    time.Unix(90000, 0),
						Image:        "img-name-1",
						ImageID:      "rkt://img-id-1",
						Hash:         1001,
						RestartCount: 7,
					},
					{
						ID:           kubecontainer.BuildContainerID("rkt", "uuid-4002:app-2"),
						Name:         "app-2",
						State:        kubecontainer.ContainerStateExited,
						CreatedAt:    time.Unix(90000, 0),
						StartedAt:    time.Unix(90000, 0),
						Image:        "img-name-2",
						ImageID:      "rkt://img-id-2",
						Hash:         1002,
						RestartCount: 7,
						Reason:       "Completed",
					},
					{
						ID:           kubecontainer.BuildContainerID("rkt", "uuid-4003:app-1"),
						Name:         "app-1",
						State:        kubecontainer.ContainerStateRunning,
						CreatedAt:    time.Unix(100000, 0),
						StartedAt:    time.Unix(100000, 0),
						Image:        "img-name-1",
						ImageID:      "rkt://img-id-1",
						Hash:         1001,
						RestartCount: 10,
					},
					{
						ID:           kubecontainer.BuildContainerID("rkt", "uuid-4003:app-2"),
						Name:         "app-2",
						State:        kubecontainer.ContainerStateExited,
						CreatedAt:    time.Unix(100000, 0),
						StartedAt:    time.Unix(100000, 0),
						Image:        "img-name-2",
						ImageID:      "rkt://img-id-2",
						Hash:         1002,
						RestartCount: 10,
						ExitCode:     1,
						Reason:       "Error",
					},
				},
			},
		},
	}

	for i, tt := range tests {
		testCaseHint := fmt.Sprintf("test case #%d", i)
		fr.pods = tt.pods

		status, err := r.GetPodStatus("42", "guestbook", "default")
		if err != nil {
			t.Errorf("test case #%d: unexpected error: %v", i, err)
		}

		assert.Equal(t, tt.result, status, testCaseHint)
		assert.Equal(t, []string{"ListPods"}, fr.called, testCaseHint)
		fr.CleanCalls()
	}
}

func generateCapRetainIsolator(t *testing.T, caps ...string) appctypes.Isolator {
	retain, err := appctypes.NewLinuxCapabilitiesRetainSet(caps...)
	if err != nil {
		t.Fatalf("Error generating cap retain isolator", err)
	}
	return retain.AsIsolator()
}

func generateCapRevokeIsolator(t *testing.T, caps ...string) appctypes.Isolator {
	revoke, err := appctypes.NewLinuxCapabilitiesRevokeSet(caps...)
	if err != nil {
		t.Fatalf("Error generating cap revoke isolator", err)
	}
	return revoke.AsIsolator()
}

func generateCPUIsolator(t *testing.T, request, limit string) appctypes.Isolator {
	cpu, err := appctypes.NewResourceCPUIsolator(request, limit)
	if err != nil {
		t.Fatalf("Error generating cpu resource isolator", err)
	}
	return cpu.AsIsolator()
}

func generateMemoryIsolator(t *testing.T, request, limit string) appctypes.Isolator {
	memory, err := appctypes.NewResourceMemoryIsolator(request, limit)
	if err != nil {
		t.Fatalf("Error generating memory resource isolator", err)
	}
	return memory.AsIsolator()
}

func baseApp(t *testing.T) *appctypes.App {
	return &appctypes.App{
		Exec:              appctypes.Exec{"/bin/foo"},
		SupplementaryGIDs: []int{4, 5, 6},
		WorkingDirectory:  "/foo",
		Environment: []appctypes.EnvironmentVariable{
			{"env-foo", "bar"},
		},
		MountPoints: []appctypes.MountPoint{
			{Name: *appctypes.MustACName("mnt-foo"), Path: "/mnt-foo", ReadOnly: false},
		},
		Ports: []appctypes.Port{
			{Name: *appctypes.MustACName("port-foo"), Protocol: "TCP", Port: 4242},
		},
		Isolators: []appctypes.Isolator{
			generateCapRetainIsolator(t, "CAP_SYS_ADMIN"),
			generateCapRevokeIsolator(t, "CAP_NET_ADMIN"),
			generateCPUIsolator(t, "100m", "200m"),
			generateMemoryIsolator(t, "10M", "20M"),
		},
	}
}

func baseAppWithRootUserGroup(t *testing.T) *appctypes.App {
	app := baseApp(t)
	app.User, app.Group = "0", "0"
	return app
}

type envByName []appctypes.EnvironmentVariable

func (s envByName) Len() int           { return len(s) }
func (s envByName) Less(i, j int) bool { return s[i].Name < s[j].Name }
func (s envByName) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

type mountsByName []appctypes.MountPoint

func (s mountsByName) Len() int           { return len(s) }
func (s mountsByName) Less(i, j int) bool { return s[i].Name < s[j].Name }
func (s mountsByName) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

type portsByName []appctypes.Port

func (s portsByName) Len() int           { return len(s) }
func (s portsByName) Less(i, j int) bool { return s[i].Name < s[j].Name }
func (s portsByName) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

type isolatorsByName []appctypes.Isolator

func (s isolatorsByName) Len() int           { return len(s) }
func (s isolatorsByName) Less(i, j int) bool { return s[i].Name < s[j].Name }
func (s isolatorsByName) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

func sortAppFields(app *appctypes.App) {
	sort.Sort(envByName(app.Environment))
	sort.Sort(mountsByName(app.MountPoints))
	sort.Sort(portsByName(app.Ports))
	sort.Sort(isolatorsByName(app.Isolators))
}

func TestSetApp(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("rkt_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	rootUser := int64(0)
	nonRootUser := int64(42)
	runAsNonRootTrue := true
	fsgid := int64(3)

	tests := []struct {
		container *api.Container
		opts      *kubecontainer.RunContainerOptions
		ctx       *api.SecurityContext
		podCtx    *api.PodSecurityContext
		expect    *appctypes.App
		err       error
	}{
		// Nothing should change, but the "User" and "Group" should be filled.
		{
			container: &api.Container{},
			opts:      &kubecontainer.RunContainerOptions{},
			ctx:       nil,
			podCtx:    nil,
			expect:    baseAppWithRootUserGroup(t),
			err:       nil,
		},

		// error verifying non-root.
		{
			container: &api.Container{},
			opts:      &kubecontainer.RunContainerOptions{},
			ctx: &api.SecurityContext{
				RunAsNonRoot: &runAsNonRootTrue,
				RunAsUser:    &rootUser,
			},
			podCtx: nil,
			expect: nil,
			err:    fmt.Errorf("container has no runAsUser and image will run as root"),
		},

		// app should be changed.
		{
			container: &api.Container{
				Command:    []string{"/bin/bar"},
				Args:       []string{"hello", "world"},
				WorkingDir: tmpDir,
				Resources: api.ResourceRequirements{
					Limits:   api.ResourceList{"cpu": resource.MustParse("50m"), "memory": resource.MustParse("50M")},
					Requests: api.ResourceList{"cpu": resource.MustParse("5m"), "memory": resource.MustParse("5M")},
				},
			},
			opts: &kubecontainer.RunContainerOptions{
				Envs: []kubecontainer.EnvVar{
					{Name: "env-bar", Value: "foo"},
				},
				Mounts: []kubecontainer.Mount{
					{Name: "mnt-bar", ContainerPath: "/mnt-bar", ReadOnly: true},
				},
				PortMappings: []kubecontainer.PortMapping{
					{Name: "port-bar", Protocol: api.ProtocolTCP, ContainerPort: 1234},
				},
			},
			ctx: &api.SecurityContext{
				Capabilities: &api.Capabilities{
					Add:  []api.Capability{"CAP_SYS_CHROOT", "CAP_SYS_BOOT"},
					Drop: []api.Capability{"CAP_SETUID", "CAP_SETGID"},
				},
				RunAsUser:    &nonRootUser,
				RunAsNonRoot: &runAsNonRootTrue,
			},
			podCtx: &api.PodSecurityContext{
				SupplementalGroups: []int64{1, 2},
				FSGroup:            &fsgid,
			},
			expect: &appctypes.App{
				Exec:              appctypes.Exec{"/bin/bar", "hello", "world"},
				User:              "42",
				Group:             "0",
				SupplementaryGIDs: []int{1, 2, 3},
				WorkingDirectory:  tmpDir,
				Environment: []appctypes.EnvironmentVariable{
					{"env-foo", "bar"},
					{"env-bar", "foo"},
				},
				MountPoints: []appctypes.MountPoint{
					{Name: *appctypes.MustACName("mnt-foo"), Path: "/mnt-foo", ReadOnly: false},
					{Name: *appctypes.MustACName("mnt-bar"), Path: "/mnt-bar", ReadOnly: true},
				},
				Ports: []appctypes.Port{
					{Name: *appctypes.MustACName("port-foo"), Protocol: "TCP", Port: 4242},
					{Name: *appctypes.MustACName("port-bar"), Protocol: "TCP", Port: 1234},
				},
				Isolators: []appctypes.Isolator{
					generateCapRetainIsolator(t, "CAP_SYS_CHROOT", "CAP_SYS_BOOT"),
					generateCapRevokeIsolator(t, "CAP_SETUID", "CAP_SETGID"),
					generateCPUIsolator(t, "5m", "50m"),
					generateMemoryIsolator(t, "5M", "50M"),
				},
			},
		},

		// app should be changed. (env, mounts, ports, are overrided).
		{
			container: &api.Container{
				Name:       "hello-world",
				Command:    []string{"/bin/bar", "$(env-foo)"},
				Args:       []string{"hello", "world", "$(env-bar)"},
				WorkingDir: tmpDir,
				Resources: api.ResourceRequirements{
					Limits:   api.ResourceList{"cpu": resource.MustParse("50m")},
					Requests: api.ResourceList{"memory": resource.MustParse("5M")},
				},
			},
			opts: &kubecontainer.RunContainerOptions{
				Envs: []kubecontainer.EnvVar{
					{Name: "env-foo", Value: "foo"},
					{Name: "env-bar", Value: "bar"},
				},
				Mounts: []kubecontainer.Mount{
					{Name: "mnt-foo", ContainerPath: "/mnt-bar", ReadOnly: true},
				},
				PortMappings: []kubecontainer.PortMapping{
					{Name: "port-foo", Protocol: api.ProtocolTCP, ContainerPort: 1234},
				},
			},
			ctx: &api.SecurityContext{
				Capabilities: &api.Capabilities{
					Add:  []api.Capability{"CAP_SYS_CHROOT", "CAP_SYS_BOOT"},
					Drop: []api.Capability{"CAP_SETUID", "CAP_SETGID"},
				},
				RunAsUser:    &nonRootUser,
				RunAsNonRoot: &runAsNonRootTrue,
			},
			podCtx: &api.PodSecurityContext{
				SupplementalGroups: []int64{1, 2},
				FSGroup:            &fsgid,
			},
			expect: &appctypes.App{
				Exec:              appctypes.Exec{"/bin/bar", "foo", "hello", "world", "bar"},
				User:              "42",
				Group:             "0",
				SupplementaryGIDs: []int{1, 2, 3},
				WorkingDirectory:  tmpDir,
				Environment: []appctypes.EnvironmentVariable{
					{"env-foo", "foo"},
					{"env-bar", "bar"},
				},
				MountPoints: []appctypes.MountPoint{
					{Name: *appctypes.MustACName("mnt-foo"), Path: "/mnt-bar", ReadOnly: true},
				},
				Ports: []appctypes.Port{
					{Name: *appctypes.MustACName("port-foo"), Protocol: "TCP", Port: 1234},
				},
				Isolators: []appctypes.Isolator{
					generateCapRetainIsolator(t, "CAP_SYS_CHROOT", "CAP_SYS_BOOT"),
					generateCapRevokeIsolator(t, "CAP_SETUID", "CAP_SETGID"),
					generateCPUIsolator(t, "50m", "50m"),
					generateMemoryIsolator(t, "5M", "5M"),
				},
			},
		},
	}

	for i, tt := range tests {
		testCaseHint := fmt.Sprintf("test case #%d", i)
		app := baseApp(t)
		err := setApp(app, tt.container, tt.opts, tt.ctx, tt.podCtx)
		if err == nil && tt.err != nil || err != nil && tt.err == nil {
			t.Errorf("%s: expect %v, saw %v", testCaseHint, tt.err, err)
		}
		if err == nil {
			sortAppFields(tt.expect)
			sortAppFields(app)
			assert.Equal(t, tt.expect, app, testCaseHint)
		}
	}
}

func TestGenerateRunCommand(t *testing.T) {
	tests := []struct {
		pod  *api.Pod
		uuid string

		dnsServers  []string
		dnsSearches []string
		err         error

		expect string
	}{
		// Case #0, returns error.
		{
			&api.Pod{
				Spec: api.PodSpec{},
			},
			"rkt-uuid-foo",
			[]string{},
			[]string{},
			fmt.Errorf("failed to get cluster dns"),
			"",
		},
		// Case #1, returns no dns, with private-net.
		{
			&api.Pod{},
			"rkt-uuid-foo",
			[]string{},
			[]string{},
			nil,
			"/bin/rkt/rkt --debug=false --insecure-options=image,ondisk --local-config=/var/rkt/local/data --dir=/var/data run-prepared --net=rkt.kubernetes.io rkt-uuid-foo",
		},
		// Case #2, returns no dns, with host-net.
		{
			&api.Pod{
				Spec: api.PodSpec{
					SecurityContext: &api.PodSecurityContext{
						HostNetwork: true,
					},
				},
			},
			"rkt-uuid-foo",
			[]string{},
			[]string{},
			nil,
			"/bin/rkt/rkt --debug=false --insecure-options=image,ondisk --local-config=/var/rkt/local/data --dir=/var/data run-prepared --net=host rkt-uuid-foo",
		},
		// Case #3, returns dns, dns searches, with private-net.
		{
			&api.Pod{
				Spec: api.PodSpec{
					SecurityContext: &api.PodSecurityContext{
						HostNetwork: false,
					},
				},
			},
			"rkt-uuid-foo",
			[]string{"127.0.0.1"},
			[]string{"."},
			nil,
			"/bin/rkt/rkt --debug=false --insecure-options=image,ondisk --local-config=/var/rkt/local/data --dir=/var/data run-prepared --net=rkt.kubernetes.io --dns=127.0.0.1 --dns-search=. --dns-opt=ndots:5 rkt-uuid-foo",
		},
		// Case #4, returns dns, dns searches, with host-network.
		{
			&api.Pod{
				Spec: api.PodSpec{
					SecurityContext: &api.PodSecurityContext{
						HostNetwork: true,
					},
				},
			},
			"rkt-uuid-foo",
			[]string{"127.0.0.1"},
			[]string{"."},
			nil,
			"/bin/rkt/rkt --debug=false --insecure-options=image,ondisk --local-config=/var/rkt/local/data --dir=/var/data run-prepared --net=host --dns=127.0.0.1 --dns-search=. --dns-opt=ndots:5 rkt-uuid-foo",
		},
	}

	rkt := &Runtime{
		rktBinAbsPath: "/bin/rkt/rkt",
		config: &Config{
			Path:            "/bin/rkt/rkt",
			Stage1Image:     "/bin/rkt/stage1-coreos.aci",
			Dir:             "/var/data",
			InsecureOptions: "image,ondisk",
			LocalConfigDir:  "/var/rkt/local/data",
		},
	}

	for i, tt := range tests {
		testCaseHint := fmt.Sprintf("test case #%d", i)
		rkt.runtimeHelper = &fakeRuntimeHelper{tt.dnsServers, tt.dnsSearches, tt.err}

		result, err := rkt.generateRunCommand(tt.pod, tt.uuid)
		assert.Equal(t, tt.err, err, testCaseHint)
		assert.Equal(t, tt.expect, result, testCaseHint)
	}
}
