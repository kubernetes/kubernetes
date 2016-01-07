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
	"testing"
	"time"

	appcschema "github.com/appc/spec/schema"
	appctypes "github.com/appc/spec/schema/types"
	rktapi "github.com/coreos/rkt/api/v1alpha"
	"github.com/stretchr/testify/assert"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
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
	appStates []rktapi.AppState) *rktapi.Pod {

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
		assert.Equal(t, err, tt.err, testCaseHint)

		if tt.calledGetInfo {
			assert.Equal(t, fr.called, []string{"GetInfo"}, testCaseHint)
		}
		if tt.calledSystemVersion {
			assert.Equal(t, fs.called, []string{"Version"}, testCaseHint)
		}
		if err == nil {
			assert.Equal(t, r.binVersion.String(), fr.info.RktVersion, testCaseHint)
			assert.Equal(t, r.appcVersion.String(), fr.info.AppcVersion, testCaseHint)
			assert.Equal(t, r.apiVersion.String(), fr.info.ApiVersion, testCaseHint)
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
					ID:   "sha512-a2fb8f390702",
					Tags: []string{"quay.io/coreos/alpine-sh:latest"},
				},
			},
		},
		{
			[]*rktapi.Image{
				{
					Id:      "sha512-a2fb8f390702",
					Name:    "quay.io/coreos/alpine-sh",
					Version: "latest",
				},
				{
					Id:      "sha512-c6b597f42816",
					Name:    "coreos.com/rkt/stage1-coreos",
					Version: "0.10.0",
				},
			},
			[]kubecontainer.Image{
				{
					ID:   "sha512-a2fb8f390702",
					Tags: []string{"quay.io/coreos/alpine-sh:latest"},
				},
				{
					ID:   "sha512-c6b597f42816",
					Tags: []string{"coreos.com/rkt/stage1-coreos:0.10.0"},
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
				),
				makeRktPod(rktapi.PodState_POD_STATE_EXITED,
					"uuid-4003", "43", "guestbook", "default",
					"10.10.10.43", "90000", "7",
					[]string{"app-11", "app-22"},
					[]string{"img-id-11", "img-id-22"},
					[]string{"img-name-11", "img-name-22"},
					[]string{"10011", "10022"},
					[]rktapi.AppState{rktapi.AppState_APP_STATE_RUNNING, rktapi.AppState_APP_STATE_EXITED},
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
				),
				makeRktPod(rktapi.PodState_POD_STATE_RUNNING, // The latest pod is running.
					"uuid-4003", "42", "guestbook", "default",
					"10.10.10.42", "100000", "10",
					[]string{"app-1", "app-2"},
					[]string{"img-id-1", "img-id-2"},
					[]string{"img-name-1", "img-name-2"},
					[]string{"1001", "1002"},
					[]rktapi.AppState{rktapi.AppState_APP_STATE_RUNNING, rktapi.AppState_APP_STATE_EXITED},
				),
			},
			&kubecontainer.PodStatus{
				ID:        "42",
				Name:      "guestbook",
				Namespace: "default",
				IP:        "10.10.10.42",
				// Result should contain all contianers.
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
