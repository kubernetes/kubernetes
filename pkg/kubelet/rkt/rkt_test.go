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

package rkt

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	appcschema "github.com/appc/spec/schema"
	appctypes "github.com/appc/spec/schema/types"
	rktapi "github.com/coreos/rkt/api/v1alpha"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertesting "k8s.io/kubernetes/pkg/kubelet/container/testing"
	kubetesting "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/network/kubenet"
	"k8s.io/kubernetes/pkg/kubelet/network/mock_network"
	"k8s.io/kubernetes/pkg/kubelet/types"
	kubetypes "k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/errors"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
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
	rktPodID, podUID, podName, podNamespace string, podCreatedAt, podStartedAt int64,
	podRestartCount string, appNames, imgIDs, imgNames,
	containerHashes []string, appStates []rktapi.AppState,
	exitcodes []int32, ips map[string]string) *rktapi.Pod {

	podManifest := &appcschema.PodManifest{
		ACKind:    appcschema.PodManifestKind,
		ACVersion: appcschema.AppContainerVersion,
		Annotations: appctypes.Annotations{
			appctypes.Annotation{
				Name:  *appctypes.MustACIdentifier(k8sRktKubeletAnno),
				Value: k8sRktKubeletAnnoValue,
			},
			appctypes.Annotation{
				Name:  *appctypes.MustACIdentifier(types.KubernetesPodUIDLabel),
				Value: podUID,
			},
			appctypes.Annotation{
				Name:  *appctypes.MustACIdentifier(types.KubernetesPodNameLabel),
				Value: podName,
			},
			appctypes.Annotation{
				Name:  *appctypes.MustACIdentifier(types.KubernetesPodNamespaceLabel),
				Value: podNamespace,
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
				Id:      imgIDs[i],
				Name:    imgNames[i],
				Version: "latest",
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

	var networks []*rktapi.Network
	for name, ip := range ips {
		networks = append(networks, &rktapi.Network{Name: name, Ipv4: ip})
	}

	return &rktapi.Pod{
		Id:        rktPodID,
		State:     rktPodState,
		Apps:      apps,
		Manifest:  mustMarshalPodManifest(podManifest),
		StartedAt: podStartedAt,
		CreatedAt: podCreatedAt,
		Networks:  networks,
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
		minimumRktBinVersion  string
		minimumRktApiVersion  string
		minimumSystemdVersion string
		err                   error
		calledGetInfo         bool
		calledSystemVersion   bool
	}{
		// Good versions.
		{
			"1.2.3",
			"1.2.5",
			"99",
			nil,
			true,
			true,
		},
		// Good versions.
		{
			"1.2.3+git",
			"1.2.6-alpha",
			"100",
			nil,
			true,
			true,
		},
		// Requires greater binary version.
		{
			"1.2.4",
			"1.2.6-alpha",
			"100",
			fmt.Errorf("rkt: binary version is too old(%v), requires at least %v", fr.info.RktVersion, "1.2.4"),
			true,
			true,
		},
		// Requires greater API version.
		{
			"1.2.3",
			"1.2.6",
			"100",
			fmt.Errorf("rkt: API version is too old(%v), requires at least %v", fr.info.ApiVersion, "1.2.6"),
			true,
			true,
		},
		// Requires greater API version.
		{
			"1.2.3",
			"1.2.7",
			"100",
			fmt.Errorf("rkt: API version is too old(%v), requires at least %v", fr.info.ApiVersion, "1.2.7"),
			true,
			true,
		},
		// Requires greater systemd version.
		{
			"1.2.3",
			"1.2.7",
			"101",
			fmt.Errorf("rkt: systemd version(%v) is too old, requires at least %v", fs.version, "101"),
			false,
			true,
		},
	}

	for i, tt := range tests {
		testCaseHint := fmt.Sprintf("test case #%d", i)
		err := r.checkVersion(tt.minimumRktBinVersion, tt.minimumRktApiVersion, tt.minimumSystemdVersion)
		assert.Equal(t, tt.err, err, testCaseHint)

		if tt.calledGetInfo {
			assert.Equal(t, fr.called, []string{"GetInfo"}, testCaseHint)
		}
		if tt.calledSystemVersion {
			assert.Equal(t, fs.called, []string{"Version"}, testCaseHint)
		}
		if err == nil {
			assert.Equal(t, fr.info.RktVersion, r.versions.binVersion.String(), testCaseHint)
			assert.Equal(t, fr.info.ApiVersion, r.versions.apiVersion.String(), testCaseHint)
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
		{
			[]*rktapi.Image{
				{
					Id:      "sha512-a2fb8f390702",
					Name:    "quay.io_443/coreos/alpine-sh",
					Version: "latest",
					Annotations: []*rktapi.KeyValue{
						{
							Key:   appcDockerRegistryURL,
							Value: "quay.io:443",
						},
						{
							Key:   appcDockerRepository,
							Value: "coreos/alpine-sh",
						},
					},
					Size: 400,
				},
			},
			[]kubecontainer.Image{
				{
					ID:       "sha512-a2fb8f390702",
					RepoTags: []string{"quay.io:443/coreos/alpine-sh:latest"},
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

	ns := func(seconds int64) int64 {
		return seconds * 1e9
	}

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
					ns(10), ns(10), "7",
					[]string{"app-1", "app-2"},
					[]string{"img-id-1", "img-id-2"},
					[]string{"img-name-1", "img-name-2"},
					[]string{"1001", "1002"},
					[]rktapi.AppState{rktapi.AppState_APP_STATE_RUNNING, rktapi.AppState_APP_STATE_EXITED},
					[]int32{0, 0},
					nil,
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
							Image:   "img-name-1:latest",
							ImageID: "img-id-1",
							Hash:    1001,
							State:   "running",
						},
						{
							ID:      kubecontainer.BuildContainerID("rkt", "uuid-4002:app-2"),
							Name:    "app-2",
							Image:   "img-name-2:latest",
							ImageID: "img-id-2",
							Hash:    1002,
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
					ns(10), ns(20), "7",
					[]string{"app-1", "app-2"},
					[]string{"img-id-1", "img-id-2"},
					[]string{"img-name-1", "img-name-2"},
					[]string{"1001", "1002"},
					[]rktapi.AppState{rktapi.AppState_APP_STATE_RUNNING, rktapi.AppState_APP_STATE_EXITED},
					[]int32{0, 0},
					nil,
				),
				makeRktPod(rktapi.PodState_POD_STATE_EXITED,
					"uuid-4003", "43", "guestbook", "default",
					ns(30), ns(40), "7",
					[]string{"app-11", "app-22"},
					[]string{"img-id-11", "img-id-22"},
					[]string{"img-name-11", "img-name-22"},
					[]string{"10011", "10022"},
					[]rktapi.AppState{rktapi.AppState_APP_STATE_EXITED, rktapi.AppState_APP_STATE_EXITED},
					[]int32{0, 0},
					nil,
				),
				makeRktPod(rktapi.PodState_POD_STATE_EXITED,
					"uuid-4004", "43", "guestbook", "default",
					ns(50), ns(60), "8",
					[]string{"app-11", "app-22"},
					[]string{"img-id-11", "img-id-22"},
					[]string{"img-name-11", "img-name-22"},
					[]string{"10011", "10022"},
					[]rktapi.AppState{rktapi.AppState_APP_STATE_RUNNING, rktapi.AppState_APP_STATE_RUNNING},
					[]int32{0, 0},
					nil,
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
							Image:   "img-name-1:latest",
							ImageID: "img-id-1",
							Hash:    1001,
							State:   "running",
						},
						{
							ID:      kubecontainer.BuildContainerID("rkt", "uuid-4002:app-2"),
							Name:    "app-2",
							Image:   "img-name-2:latest",
							ImageID: "img-id-2",
							Hash:    1002,
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
							Image:   "img-name-11:latest",
							ImageID: "img-id-11",
							Hash:    10011,
							State:   "exited",
						},
						{
							ID:      kubecontainer.BuildContainerID("rkt", "uuid-4003:app-22"),
							Name:    "app-22",
							Image:   "img-name-22:latest",
							ImageID: "img-id-22",
							Hash:    10022,
							State:   "exited",
						},
						{
							ID:      kubecontainer.BuildContainerID("rkt", "uuid-4004:app-11"),
							Name:    "app-11",
							Image:   "img-name-11:latest",
							ImageID: "img-id-11",
							Hash:    10011,
							State:   "running",
						},
						{
							ID:      kubecontainer.BuildContainerID("rkt", "uuid-4004:app-22"),
							Name:    "app-22",
							Image:   "img-name-22:latest",
							ImageID: "img-id-22",
							Hash:    10022,
							State:   "running",
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
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	fr := newFakeRktInterface()
	fs := newFakeSystemd()
	fnp := mock_network.NewMockNetworkPlugin(ctrl)
	fos := &containertesting.FakeOS{}
	frh := &fakeRuntimeHelper{}
	r := &Runtime{
		apisvc:        fr,
		systemd:       fs,
		runtimeHelper: frh,
		os:            fos,
		networkPlugin: fnp,
	}

	ns := func(seconds int64) int64 {
		return seconds * 1e9
	}

	tests := []struct {
		networkPluginName string
		pods              []*rktapi.Pod
		result            *kubecontainer.PodStatus
	}{
		// # case 0, No pods.
		{
			kubenet.KubenetPluginName,
			nil,
			&kubecontainer.PodStatus{ID: "42", Name: "guestbook", Namespace: "default"},
		},
		// # case 1, One pod.
		{
			kubenet.KubenetPluginName,
			[]*rktapi.Pod{
				makeRktPod(rktapi.PodState_POD_STATE_RUNNING,
					"uuid-4002", "42", "guestbook", "default",
					ns(10), ns(20), "7",
					[]string{"app-1", "app-2"},
					[]string{"img-id-1", "img-id-2"},
					[]string{"img-name-1", "img-name-2"},
					[]string{"1001", "1002"},
					[]rktapi.AppState{rktapi.AppState_APP_STATE_RUNNING, rktapi.AppState_APP_STATE_EXITED},
					[]int32{0, 0},
					nil,
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
						CreatedAt:    time.Unix(10, 0),
						StartedAt:    time.Unix(20, 0),
						FinishedAt:   time.Unix(0, 30),
						Image:        "img-name-1:latest",
						ImageID:      "rkt://img-id-1",
						Hash:         1001,
						RestartCount: 7,
					},
					{
						ID:           kubecontainer.BuildContainerID("rkt", "uuid-4002:app-2"),
						Name:         "app-2",
						State:        kubecontainer.ContainerStateExited,
						CreatedAt:    time.Unix(10, 0),
						StartedAt:    time.Unix(20, 0),
						FinishedAt:   time.Unix(0, 30),
						Image:        "img-name-2:latest",
						ImageID:      "rkt://img-id-2",
						Hash:         1002,
						RestartCount: 7,
						Reason:       "Completed",
					},
				},
			},
		},
		// # case 2, One pod with no-op network plugin name.
		{
			network.DefaultPluginName,
			[]*rktapi.Pod{
				makeRktPod(rktapi.PodState_POD_STATE_RUNNING,
					"uuid-4002", "42", "guestbook", "default",
					ns(10), ns(20), "7",
					[]string{"app-1", "app-2"},
					[]string{"img-id-1", "img-id-2"},
					[]string{"img-name-1", "img-name-2"},
					[]string{"1001", "1002"},
					[]rktapi.AppState{rktapi.AppState_APP_STATE_RUNNING, rktapi.AppState_APP_STATE_EXITED},
					[]int32{0, 0},
					map[string]string{defaultNetworkName: "10.10.10.22"},
				),
			},
			&kubecontainer.PodStatus{
				ID:        "42",
				Name:      "guestbook",
				Namespace: "default",
				IP:        "10.10.10.22",
				ContainerStatuses: []*kubecontainer.ContainerStatus{
					{
						ID:           kubecontainer.BuildContainerID("rkt", "uuid-4002:app-1"),
						Name:         "app-1",
						State:        kubecontainer.ContainerStateRunning,
						CreatedAt:    time.Unix(10, 0),
						StartedAt:    time.Unix(20, 0),
						FinishedAt:   time.Unix(0, 30),
						Image:        "img-name-1:latest",
						ImageID:      "rkt://img-id-1",
						Hash:         1001,
						RestartCount: 7,
					},
					{
						ID:           kubecontainer.BuildContainerID("rkt", "uuid-4002:app-2"),
						Name:         "app-2",
						State:        kubecontainer.ContainerStateExited,
						CreatedAt:    time.Unix(10, 0),
						StartedAt:    time.Unix(20, 0),
						FinishedAt:   time.Unix(0, 30),
						Image:        "img-name-2:latest",
						ImageID:      "rkt://img-id-2",
						Hash:         1002,
						RestartCount: 7,
						Reason:       "Completed",
					},
				},
			},
		},
		// # case 3, Multiple pods.
		{
			kubenet.KubenetPluginName,
			[]*rktapi.Pod{
				makeRktPod(rktapi.PodState_POD_STATE_EXITED,
					"uuid-4002", "42", "guestbook", "default",
					ns(10), ns(20), "7",
					[]string{"app-1", "app-2"},
					[]string{"img-id-1", "img-id-2"},
					[]string{"img-name-1", "img-name-2"},
					[]string{"1001", "1002"},
					[]rktapi.AppState{rktapi.AppState_APP_STATE_RUNNING, rktapi.AppState_APP_STATE_EXITED},
					[]int32{0, 0},
					nil,
				),
				makeRktPod(rktapi.PodState_POD_STATE_RUNNING, // The latest pod is running.
					"uuid-4003", "42", "guestbook", "default",
					ns(10), ns(20), "10",
					[]string{"app-1", "app-2"},
					[]string{"img-id-1", "img-id-2"},
					[]string{"img-name-1", "img-name-2"},
					[]string{"1001", "1002"},
					[]rktapi.AppState{rktapi.AppState_APP_STATE_RUNNING, rktapi.AppState_APP_STATE_EXITED},
					[]int32{0, 1},
					nil,
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
						CreatedAt:    time.Unix(10, 0),
						StartedAt:    time.Unix(20, 0),
						FinishedAt:   time.Unix(0, 30),
						Image:        "img-name-1:latest",
						ImageID:      "rkt://img-id-1",
						Hash:         1001,
						RestartCount: 7,
					},
					{
						ID:           kubecontainer.BuildContainerID("rkt", "uuid-4002:app-2"),
						Name:         "app-2",
						State:        kubecontainer.ContainerStateExited,
						CreatedAt:    time.Unix(10, 0),
						StartedAt:    time.Unix(20, 0),
						FinishedAt:   time.Unix(0, 30),
						Image:        "img-name-2:latest",
						ImageID:      "rkt://img-id-2",
						Hash:         1002,
						RestartCount: 7,
						Reason:       "Completed",
					},
					{
						ID:           kubecontainer.BuildContainerID("rkt", "uuid-4003:app-1"),
						Name:         "app-1",
						State:        kubecontainer.ContainerStateRunning,
						CreatedAt:    time.Unix(10, 0),
						StartedAt:    time.Unix(20, 0),
						FinishedAt:   time.Unix(0, 30),
						Image:        "img-name-1:latest",
						ImageID:      "rkt://img-id-1",
						Hash:         1001,
						RestartCount: 10,
					},
					{
						ID:           kubecontainer.BuildContainerID("rkt", "uuid-4003:app-2"),
						Name:         "app-2",
						State:        kubecontainer.ContainerStateExited,
						CreatedAt:    time.Unix(10, 0),
						StartedAt:    time.Unix(20, 0),
						FinishedAt:   time.Unix(0, 30),
						Image:        "img-name-2:latest",
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

		podTimes := map[string]time.Time{}
		for _, pod := range tt.pods {
			podTimes[podFinishedMarkerPath(r.runtimeHelper.GetPodDir(tt.result.ID), pod.Id)] = tt.result.ContainerStatuses[0].FinishedAt
		}

		r.os.(*containertesting.FakeOS).StatFn = func(name string) (os.FileInfo, error) {
			podTime, ok := podTimes[name]
			if !ok {
				t.Errorf("osStat called with %v, but only knew about %#v", name, podTimes)
			}
			mockFI := containertesting.NewMockFileInfo(ctrl)
			mockFI.EXPECT().ModTime().Return(podTime)
			return mockFI, nil
		}
		fnp.EXPECT().Name().Return(tt.networkPluginName)

		if tt.networkPluginName == kubenet.KubenetPluginName {
			if tt.result.IP != "" {
				fnp.EXPECT().GetPodNetworkStatus("default", "guestbook", kubecontainer.ContainerID{ID: "42"}).
					Return(&network.PodNetworkStatus{IP: net.ParseIP(tt.result.IP)}, nil)
			} else {
				fnp.EXPECT().GetPodNetworkStatus("default", "guestbook", kubecontainer.ContainerID{ID: "42"}).
					Return(nil, fmt.Errorf("no such network"))
			}
		}

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
		t.Fatalf("Error generating cap retain isolator: %v", err)
	}
	return retain.AsIsolator()
}

func generateCapRevokeIsolator(t *testing.T, caps ...string) appctypes.Isolator {
	revoke, err := appctypes.NewLinuxCapabilitiesRevokeSet(caps...)
	if err != nil {
		t.Fatalf("Error generating cap revoke isolator: %v", err)
	}
	return revoke.AsIsolator()
}

func generateCPUIsolator(t *testing.T, request, limit string) appctypes.Isolator {
	cpu, err := appctypes.NewResourceCPUIsolator(request, limit)
	if err != nil {
		t.Fatalf("Error generating cpu resource isolator: %v", err)
	}
	return cpu.AsIsolator()
}

func generateMemoryIsolator(t *testing.T, request, limit string) appctypes.Isolator {
	memory, err := appctypes.NewResourceMemoryIsolator(request, limit)
	if err != nil {
		t.Fatalf("Error generating memory resource isolator: %v", err)
	}
	return memory.AsIsolator()
}

func baseApp(t *testing.T) *appctypes.App {
	return &appctypes.App{
		User:              "0",
		Group:             "0",
		Exec:              appctypes.Exec{"/bin/foo", "bar"},
		SupplementaryGIDs: []int{4, 5, 6},
		WorkingDirectory:  "/foo",
		Environment: []appctypes.EnvironmentVariable{
			{Name: "env-foo", Value: "bar"},
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

func baseImageManifest(t *testing.T) *appcschema.ImageManifest {
	img := &appcschema.ImageManifest{App: baseApp(t)}
	entrypoint, err := json.Marshal([]string{"/bin/foo"})
	if err != nil {
		t.Fatal(err)
	}
	cmd, err := json.Marshal([]string{"bar"})
	if err != nil {
		t.Fatal(err)
	}
	img.Annotations.Set(*appctypes.MustACIdentifier(appcDockerEntrypoint), string(entrypoint))
	img.Annotations.Set(*appctypes.MustACIdentifier(appcDockerCmd), string(cmd))
	return img
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

type sortedStringList []string

func (s sortedStringList) Len() int           { return len(s) }
func (s sortedStringList) Less(i, j int) bool { return s[i] < s[j] }
func (s sortedStringList) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

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
		container        *api.Container
		mountPoints      []appctypes.MountPoint
		containerPorts   []appctypes.Port
		envs             []kubecontainer.EnvVar
		ctx              *api.SecurityContext
		podCtx           *api.PodSecurityContext
		supplementalGids []int64
		expect           *appctypes.App
		err              error
	}{
		// Nothing should change, but the "User" and "Group" should be filled.
		{
			container:        &api.Container{},
			mountPoints:      []appctypes.MountPoint{},
			containerPorts:   []appctypes.Port{},
			envs:             []kubecontainer.EnvVar{},
			ctx:              nil,
			podCtx:           nil,
			supplementalGids: nil,
			expect:           baseAppWithRootUserGroup(t),
			err:              nil,
		},

		// error verifying non-root.
		{
			container:      &api.Container{},
			mountPoints:    []appctypes.MountPoint{},
			containerPorts: []appctypes.Port{},
			envs:           []kubecontainer.EnvVar{},
			ctx: &api.SecurityContext{
				RunAsNonRoot: &runAsNonRootTrue,
				RunAsUser:    &rootUser,
			},
			podCtx:           nil,
			supplementalGids: nil,
			expect:           nil,
			err:              fmt.Errorf("container has no runAsUser and image will run as root"),
		},

		// app's args should be changed.
		{
			container: &api.Container{
				Args: []string{"foo"},
			},
			mountPoints:      []appctypes.MountPoint{},
			containerPorts:   []appctypes.Port{},
			envs:             []kubecontainer.EnvVar{},
			ctx:              nil,
			podCtx:           nil,
			supplementalGids: nil,
			expect: &appctypes.App{
				Exec:              appctypes.Exec{"/bin/foo", "foo"},
				User:              "0",
				Group:             "0",
				SupplementaryGIDs: []int{4, 5, 6},
				WorkingDirectory:  "/foo",
				Environment: []appctypes.EnvironmentVariable{
					{Name: "env-foo", Value: "bar"},
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
			},
			err: nil,
		},

		// app should be changed.
		{
			container: &api.Container{
				Command:    []string{"/bin/bar", "$(env-bar)"},
				WorkingDir: tmpDir,
				Resources: api.ResourceRequirements{
					Limits:   api.ResourceList{"cpu": resource.MustParse("50m"), "memory": resource.MustParse("50M")},
					Requests: api.ResourceList{"cpu": resource.MustParse("5m"), "memory": resource.MustParse("5M")},
				},
			},
			mountPoints: []appctypes.MountPoint{
				{Name: *appctypes.MustACName("mnt-bar"), Path: "/mnt-bar", ReadOnly: true},
			},
			containerPorts: []appctypes.Port{
				{Name: *appctypes.MustACName("port-bar"), Protocol: "TCP", Port: 1234},
			},
			envs: []kubecontainer.EnvVar{
				{Name: "env-bar", Value: "foo"},
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
			supplementalGids: []int64{4},
			expect: &appctypes.App{
				Exec:              appctypes.Exec{"/bin/bar", "foo"},
				User:              "42",
				Group:             "0",
				SupplementaryGIDs: []int{1, 2, 3, 4},
				WorkingDirectory:  tmpDir,
				Environment: []appctypes.EnvironmentVariable{
					{Name: "env-foo", Value: "bar"},
					{Name: "env-bar", Value: "foo"},
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
				Command:    []string{"/bin/hello", "$(env-foo)"},
				Args:       []string{"hello", "world", "$(env-bar)"},
				WorkingDir: tmpDir,
				Resources: api.ResourceRequirements{
					Limits:   api.ResourceList{"cpu": resource.MustParse("50m")},
					Requests: api.ResourceList{"memory": resource.MustParse("5M")},
				},
			},
			mountPoints: []appctypes.MountPoint{
				{Name: *appctypes.MustACName("mnt-foo"), Path: "/mnt-foo", ReadOnly: true},
			},
			containerPorts: []appctypes.Port{
				{Name: *appctypes.MustACName("port-foo"), Protocol: "TCP", Port: 1234},
			},
			envs: []kubecontainer.EnvVar{
				{Name: "env-foo", Value: "foo"},
				{Name: "env-bar", Value: "bar"},
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
			supplementalGids: []int64{4},
			expect: &appctypes.App{
				Exec:              appctypes.Exec{"/bin/hello", "foo", "hello", "world", "bar"},
				User:              "42",
				Group:             "0",
				SupplementaryGIDs: []int{1, 2, 3, 4},
				WorkingDirectory:  tmpDir,
				Environment: []appctypes.EnvironmentVariable{
					{Name: "env-foo", Value: "foo"},
					{Name: "env-bar", Value: "bar"},
				},
				MountPoints: []appctypes.MountPoint{
					{Name: *appctypes.MustACName("mnt-foo"), Path: "/mnt-foo", ReadOnly: true},
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
		img := baseImageManifest(t)

		err := setApp(img, tt.container,
			tt.mountPoints, tt.containerPorts, tt.envs,
			tt.ctx, tt.podCtx, tt.supplementalGids)

		if err == nil && tt.err != nil || err != nil && tt.err == nil {
			t.Errorf("%s: expect %v, saw %v", testCaseHint, tt.err, err)
		}
		if err == nil {
			sortAppFields(tt.expect)
			sortAppFields(img.App)
			assert.Equal(t, tt.expect, img.App, testCaseHint)
		}
	}
}

func TestGenerateRunCommand(t *testing.T) {
	hostName := "test-hostname"
	boolTrue := true
	boolFalse := false

	tests := []struct {
		networkPlugin network.NetworkPlugin
		pod           *api.Pod
		uuid          string
		netnsName     string

		dnsServers  []string
		dnsSearches []string
		hostName    string
		err         error

		expect string
	}{
		// Case #0, returns error.
		{
			kubenet.NewPlugin("/tmp"),
			&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "pod-name-foo",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{{Name: "container-foo"}},
				},
			},
			"rkt-uuid-foo",
			"default",
			[]string{},
			[]string{},
			"",
			fmt.Errorf("failed to get cluster dns"),
			"",
		},
		// Case #1, returns no dns, with private-net.
		{
			kubenet.NewPlugin("/tmp"),
			&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "pod-name-foo",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{{Name: "container-foo"}},
				},
			},
			"rkt-uuid-foo",
			"default",
			[]string{},
			[]string{},
			"pod-hostname-foo",
			nil,
			"/usr/bin/nsenter --net=/var/run/netns/default -- /bin/rkt/rkt --insecure-options=image,ondisk --local-config=/var/rkt/local/data --dir=/var/data run-prepared --net=host --hostname=pod-hostname-foo rkt-uuid-foo",
		},
		// Case #2, returns no dns, with host-net.
		{
			kubenet.NewPlugin("/tmp"),
			&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "pod-name-foo",
				},
				Spec: api.PodSpec{
					SecurityContext: &api.PodSecurityContext{
						HostNetwork: true,
					},
					Containers: []api.Container{{Name: "container-foo"}},
				},
			},
			"rkt-uuid-foo",
			"",
			[]string{},
			[]string{},
			"",
			nil,
			fmt.Sprintf("/bin/rkt/rkt --insecure-options=image,ondisk --local-config=/var/rkt/local/data --dir=/var/data run-prepared --net=host --hostname=%s rkt-uuid-foo", hostName),
		},
		// Case #3, returns dns, dns searches, with private-net.
		{
			kubenet.NewPlugin("/tmp"),
			&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "pod-name-foo",
				},
				Spec: api.PodSpec{
					SecurityContext: &api.PodSecurityContext{
						HostNetwork: false,
					},
					Containers: []api.Container{{Name: "container-foo"}},
				},
			},
			"rkt-uuid-foo",
			"default",
			[]string{"127.0.0.1"},
			[]string{"."},
			"pod-hostname-foo",
			nil,
			"/usr/bin/nsenter --net=/var/run/netns/default -- /bin/rkt/rkt --insecure-options=image,ondisk --local-config=/var/rkt/local/data --dir=/var/data run-prepared --net=host --dns=127.0.0.1 --dns-search=. --dns-opt=ndots:5 --hostname=pod-hostname-foo rkt-uuid-foo",
		},
		// Case #4, returns no dns, dns searches, with host-network.
		{
			kubenet.NewPlugin("/tmp"),
			&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "pod-name-foo",
				},
				Spec: api.PodSpec{
					SecurityContext: &api.PodSecurityContext{
						HostNetwork: true,
					},
					Containers: []api.Container{{Name: "container-foo"}},
				},
			},
			"rkt-uuid-foo",
			"",
			[]string{"127.0.0.1"},
			[]string{"."},
			"pod-hostname-foo",
			nil,
			fmt.Sprintf("/bin/rkt/rkt --insecure-options=image,ondisk --local-config=/var/rkt/local/data --dir=/var/data run-prepared --net=host --hostname=%s rkt-uuid-foo", hostName),
		},
		// Case #5, with no-op plugin, returns --net=rkt.kubernetes.io, with dns and dns search.
		{
			&network.NoopNetworkPlugin{},
			&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "pod-name-foo",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{{Name: "container-foo"}},
				},
			},
			"rkt-uuid-foo",
			"default",
			[]string{"127.0.0.1"},
			[]string{"."},
			"pod-hostname-foo",
			nil,
			"/bin/rkt/rkt --insecure-options=image,ondisk --local-config=/var/rkt/local/data --dir=/var/data run-prepared --net=rkt.kubernetes.io --dns=127.0.0.1 --dns-search=. --dns-opt=ndots:5 --hostname=pod-hostname-foo rkt-uuid-foo",
		},
		// Case #6, if all containers are privileged, the result should have 'insecure-options=all-run'
		{
			kubenet.NewPlugin("/tmp"),
			&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "pod-name-foo",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "container-foo", SecurityContext: &api.SecurityContext{Privileged: &boolTrue}},
						{Name: "container-bar", SecurityContext: &api.SecurityContext{Privileged: &boolTrue}},
					},
				},
			},
			"rkt-uuid-foo",
			"default",
			[]string{},
			[]string{},
			"pod-hostname-foo",
			nil,
			"/usr/bin/nsenter --net=/var/run/netns/default -- /bin/rkt/rkt --insecure-options=image,ondisk,all-run --local-config=/var/rkt/local/data --dir=/var/data run-prepared --net=host --hostname=pod-hostname-foo rkt-uuid-foo",
		},
		// Case #7, if not all containers are privileged, the result should not have 'insecure-options=all-run'
		{
			kubenet.NewPlugin("/tmp"),
			&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "pod-name-foo",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "container-foo", SecurityContext: &api.SecurityContext{Privileged: &boolTrue}},
						{Name: "container-bar", SecurityContext: &api.SecurityContext{Privileged: &boolFalse}},
					},
				},
			},
			"rkt-uuid-foo",
			"default",
			[]string{},
			[]string{},
			"pod-hostname-foo",
			nil,
			"/usr/bin/nsenter --net=/var/run/netns/default -- /bin/rkt/rkt --insecure-options=image,ondisk --local-config=/var/rkt/local/data --dir=/var/data run-prepared --net=host --hostname=pod-hostname-foo rkt-uuid-foo",
		},
	}

	rkt := &Runtime{
		nsenterPath: "/usr/bin/nsenter",
		os:          &kubetesting.FakeOS{HostName: hostName},
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
		rkt.networkPlugin = tt.networkPlugin
		rkt.runtimeHelper = &fakeRuntimeHelper{tt.dnsServers, tt.dnsSearches, tt.hostName, "", tt.err}
		rkt.execer = &utilexec.FakeExec{CommandScript: []utilexec.FakeCommandAction{func(cmd string, args ...string) utilexec.Cmd {
			return utilexec.InitFakeCmd(&utilexec.FakeCmd{}, cmd, args...)
		}}}

		// a command should be created of this form, but the returned command shouldn't be called (asserted by having no expectations on it)

		result, err := rkt.generateRunCommand(tt.pod, tt.uuid, tt.netnsName)
		assert.Equal(t, tt.err, err, testCaseHint)
		assert.Equal(t, tt.expect, result, testCaseHint)
	}
}

func TestLifeCycleHooks(t *testing.T) {
	runner := lifecycle.NewFakeHandlerRunner()
	fr := newFakeRktInterface()
	fs := newFakeSystemd()

	rkt := &Runtime{
		runner:              runner,
		apisvc:              fr,
		systemd:             fs,
		containerRefManager: kubecontainer.NewRefManager(),
	}

	tests := []struct {
		pod           *api.Pod
		runtimePod    *kubecontainer.Pod
		postStartRuns []string
		preStopRuns   []string
		err           error
	}{
		{
			// Case 0, container without any hooks.
			&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name:      "pod-1",
					Namespace: "ns-1",
					UID:       "uid-1",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "container-name-1"},
					},
				},
			},
			&kubecontainer.Pod{
				Containers: []*kubecontainer.Container{
					{ID: kubecontainer.BuildContainerID("rkt", "id-1")},
				},
			},
			[]string{},
			[]string{},
			nil,
		},
		{
			// Case 1, containers with post-start and pre-stop hooks.
			&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name:      "pod-1",
					Namespace: "ns-1",
					UID:       "uid-1",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "container-name-1",
							Lifecycle: &api.Lifecycle{
								PostStart: &api.Handler{
									Exec: &api.ExecAction{},
								},
							},
						},
						{
							Name: "container-name-2",
							Lifecycle: &api.Lifecycle{
								PostStart: &api.Handler{
									HTTPGet: &api.HTTPGetAction{},
								},
							},
						},
						{
							Name: "container-name-3",
							Lifecycle: &api.Lifecycle{
								PreStop: &api.Handler{
									Exec: &api.ExecAction{},
								},
							},
						},
						{
							Name: "container-name-4",
							Lifecycle: &api.Lifecycle{
								PreStop: &api.Handler{
									HTTPGet: &api.HTTPGetAction{},
								},
							},
						},
					},
				},
			},
			&kubecontainer.Pod{
				Containers: []*kubecontainer.Container{
					{
						ID:   kubecontainer.ParseContainerID("rkt://uuid:container-name-4"),
						Name: "container-name-4",
					},
					{
						ID:   kubecontainer.ParseContainerID("rkt://uuid:container-name-3"),
						Name: "container-name-3",
					},
					{
						ID:   kubecontainer.ParseContainerID("rkt://uuid:container-name-2"),
						Name: "container-name-2",
					},
					{
						ID:   kubecontainer.ParseContainerID("rkt://uuid:container-name-1"),
						Name: "container-name-1",
					},
				},
			},
			[]string{
				"exec on pod: pod-1_ns-1(uid-1), container: container-name-1: rkt://uuid:container-name-1",
				"http-get on pod: pod-1_ns-1(uid-1), container: container-name-2: rkt://uuid:container-name-2",
			},
			[]string{
				"exec on pod: pod-1_ns-1(uid-1), container: container-name-3: rkt://uuid:container-name-3",
				"http-get on pod: pod-1_ns-1(uid-1), container: container-name-4: rkt://uuid:container-name-4",
			},
			nil,
		},
		{
			// Case 2, one container with invalid hooks.
			&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name:      "pod-1",
					Namespace: "ns-1",
					UID:       "uid-1",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "container-name-1",
							Lifecycle: &api.Lifecycle{
								PostStart: &api.Handler{},
								PreStop:   &api.Handler{},
							},
						},
					},
				},
			},
			&kubecontainer.Pod{
				Containers: []*kubecontainer.Container{
					{
						ID:   kubecontainer.ParseContainerID("rkt://uuid:container-name-1"),
						Name: "container-name-1",
					},
				},
			},
			[]string{},
			[]string{},
			errors.NewAggregate([]error{fmt.Errorf("Invalid handler: %v", &api.Handler{})}),
		},
	}

	for i, tt := range tests {
		testCaseHint := fmt.Sprintf("test case #%d", i)

		pod := &rktapi.Pod{Id: "uuid"}
		for _, c := range tt.runtimePod.Containers {
			pod.Apps = append(pod.Apps, &rktapi.App{
				Name:  c.Name,
				State: rktapi.AppState_APP_STATE_RUNNING,
			})
		}
		fr.pods = []*rktapi.Pod{pod}

		// Run post-start hooks
		err := rkt.runLifecycleHooks(tt.pod, tt.runtimePod, lifecyclePostStartHook)
		assert.Equal(t, tt.err, err, testCaseHint)

		sort.Sort(sortedStringList(tt.postStartRuns))
		sort.Sort(sortedStringList(runner.HandlerRuns))

		assert.Equal(t, tt.postStartRuns, runner.HandlerRuns, testCaseHint)

		runner.Reset()

		// Run pre-stop hooks.
		err = rkt.runLifecycleHooks(tt.pod, tt.runtimePod, lifecyclePreStopHook)
		assert.Equal(t, tt.err, err, testCaseHint)

		sort.Sort(sortedStringList(tt.preStopRuns))
		sort.Sort(sortedStringList(runner.HandlerRuns))

		assert.Equal(t, tt.preStopRuns, runner.HandlerRuns, testCaseHint)

		runner.Reset()
	}
}

func TestImageStats(t *testing.T) {
	fr := newFakeRktInterface()
	rkt := &Runtime{apisvc: fr}

	fr.images = []*rktapi.Image{
		{Size: 100},
		{Size: 200},
		{Size: 300},
	}

	result, err := rkt.ImageStats()
	assert.NoError(t, err)
	assert.Equal(t, result, &kubecontainer.ImageStats{TotalStorageBytes: 600})
}

func TestGarbageCollect(t *testing.T) {
	fr := newFakeRktInterface()
	fs := newFakeSystemd()
	cli := newFakeRktCli()
	fakeOS := kubetesting.NewFakeOS()
	getter := newFakePodGetter()

	rkt := &Runtime{
		os:                  fakeOS,
		cli:                 cli,
		apisvc:              fr,
		podGetter:           getter,
		systemd:             fs,
		containerRefManager: kubecontainer.NewRefManager(),
	}

	fakeApp := &rktapi.App{Name: "app-foo"}

	tests := []struct {
		gcPolicy             kubecontainer.ContainerGCPolicy
		apiPods              []*api.Pod
		pods                 []*rktapi.Pod
		serviceFilesOnDisk   []string
		expectedCommands     []string
		expectedServiceFiles []string
	}{
		// All running pods, should not be gc'd.
		// Dead, new pods should not be gc'd.
		// Dead, old pods should be gc'd.
		// Deleted pods should be gc'd.
		// Service files without corresponded pods should be removed.
		{
			kubecontainer.ContainerGCPolicy{
				MinAge:        0,
				MaxContainers: 0,
			},
			[]*api.Pod{
				{ObjectMeta: api.ObjectMeta{UID: "pod-uid-1"}},
				{ObjectMeta: api.ObjectMeta{UID: "pod-uid-2"}},
				{ObjectMeta: api.ObjectMeta{UID: "pod-uid-3"}},
				{ObjectMeta: api.ObjectMeta{UID: "pod-uid-4"}},
			},
			[]*rktapi.Pod{
				{
					Id:        "deleted-foo",
					State:     rktapi.PodState_POD_STATE_EXITED,
					CreatedAt: time.Now().Add(time.Hour).UnixNano(),
					StartedAt: time.Now().Add(time.Hour).UnixNano(),
					Apps:      []*rktapi.App{fakeApp},
					Annotations: []*rktapi.KeyValue{
						{
							Key:   types.KubernetesPodUIDLabel,
							Value: "pod-uid-0",
						},
					},
				},
				{
					Id:        "running-foo",
					State:     rktapi.PodState_POD_STATE_RUNNING,
					CreatedAt: 0,
					StartedAt: 0,
					Apps:      []*rktapi.App{fakeApp},
					Annotations: []*rktapi.KeyValue{
						{
							Key:   types.KubernetesPodUIDLabel,
							Value: "pod-uid-1",
						},
					},
				},
				{
					Id:        "running-bar",
					State:     rktapi.PodState_POD_STATE_RUNNING,
					CreatedAt: 0,
					StartedAt: 0,
					Apps:      []*rktapi.App{fakeApp},
					Annotations: []*rktapi.KeyValue{
						{
							Key:   types.KubernetesPodUIDLabel,
							Value: "pod-uid-2",
						},
					},
				},
				{
					Id:        "dead-old",
					State:     rktapi.PodState_POD_STATE_EXITED,
					CreatedAt: 0,
					StartedAt: 0,
					Apps:      []*rktapi.App{fakeApp},
					Annotations: []*rktapi.KeyValue{
						{
							Key:   types.KubernetesPodUIDLabel,
							Value: "pod-uid-3",
						},
					},
				},
				{
					Id:        "dead-new",
					State:     rktapi.PodState_POD_STATE_EXITED,
					CreatedAt: time.Now().Add(time.Hour).UnixNano(),
					StartedAt: time.Now().Add(time.Hour).UnixNano(),
					Apps:      []*rktapi.App{fakeApp},
					Annotations: []*rktapi.KeyValue{
						{
							Key:   types.KubernetesPodUIDLabel,
							Value: "pod-uid-4",
						},
					},
				},
			},
			[]string{"k8s_dead-old.service", "k8s_deleted-foo.service", "k8s_non-existing-bar.service"},
			[]string{"rkt rm dead-old", "rkt rm deleted-foo"},
			[]string{"/run/systemd/system/k8s_dead-old.service", "/run/systemd/system/k8s_deleted-foo.service", "/run/systemd/system/k8s_non-existing-bar.service"},
		},
		// gcPolicy.MaxContainers should be enforced.
		// Oldest ones are removed first.
		{
			kubecontainer.ContainerGCPolicy{
				MinAge:        0,
				MaxContainers: 1,
			},
			[]*api.Pod{
				{ObjectMeta: api.ObjectMeta{UID: "pod-uid-0"}},
				{ObjectMeta: api.ObjectMeta{UID: "pod-uid-1"}},
				{ObjectMeta: api.ObjectMeta{UID: "pod-uid-2"}},
			},
			[]*rktapi.Pod{
				{
					Id:        "dead-2",
					State:     rktapi.PodState_POD_STATE_EXITED,
					CreatedAt: 2,
					StartedAt: 2,
					Apps:      []*rktapi.App{fakeApp},
					Annotations: []*rktapi.KeyValue{
						{
							Key:   types.KubernetesPodUIDLabel,
							Value: "pod-uid-2",
						},
					},
				},
				{
					Id:        "dead-1",
					State:     rktapi.PodState_POD_STATE_EXITED,
					CreatedAt: 1,
					StartedAt: 1,
					Apps:      []*rktapi.App{fakeApp},
					Annotations: []*rktapi.KeyValue{
						{
							Key:   types.KubernetesPodUIDLabel,
							Value: "pod-uid-1",
						},
					},
				},
				{
					Id:        "dead-0",
					State:     rktapi.PodState_POD_STATE_EXITED,
					CreatedAt: 0,
					StartedAt: 0,
					Apps:      []*rktapi.App{fakeApp},
					Annotations: []*rktapi.KeyValue{
						{
							Key:   types.KubernetesPodUIDLabel,
							Value: "pod-uid-0",
						},
					},
				},
			},
			[]string{"k8s_dead-0.service", "k8s_dead-1.service", "k8s_dead-2.service"},
			[]string{"rkt rm dead-0", "rkt rm dead-1"},
			[]string{"/run/systemd/system/k8s_dead-0.service", "/run/systemd/system/k8s_dead-1.service"},
		},
	}

	for i, tt := range tests {
		testCaseHint := fmt.Sprintf("test case #%d", i)

		ctrl := gomock.NewController(t)

		fakeOS.ReadDirFn = func(dirname string) ([]os.FileInfo, error) {
			serviceFileNames := tt.serviceFilesOnDisk
			var fileInfos []os.FileInfo

			for _, name := range serviceFileNames {
				mockFI := containertesting.NewMockFileInfo(ctrl)
				mockFI.EXPECT().Name().Return(name)
				fileInfos = append(fileInfos, mockFI)
			}
			return fileInfos, nil
		}

		fr.pods = tt.pods
		for _, p := range tt.apiPods {
			getter.pods[p.UID] = p
		}

		allSourcesReady := true
		err := rkt.GarbageCollect(tt.gcPolicy, allSourcesReady)
		assert.NoError(t, err, testCaseHint)

		sort.Sort(sortedStringList(tt.expectedCommands))
		sort.Sort(sortedStringList(cli.cmds))

		assert.Equal(t, tt.expectedCommands, cli.cmds, testCaseHint)

		sort.Sort(sortedStringList(tt.expectedServiceFiles))
		sort.Sort(sortedStringList(fakeOS.Removes))
		sort.Sort(sortedStringList(fs.resetFailedUnits))

		assert.Equal(t, tt.expectedServiceFiles, fakeOS.Removes, testCaseHint)
		var expectedService []string
		for _, f := range tt.expectedServiceFiles {
			expectedService = append(expectedService, filepath.Base(f))
		}
		assert.Equal(t, expectedService, fs.resetFailedUnits, testCaseHint)

		// Cleanup after each test.
		cli.Reset()
		ctrl.Finish()
		fakeOS.Removes = []string{}
		fs.resetFailedUnits = []string{}
		getter.pods = make(map[kubetypes.UID]*api.Pod)
	}
}

type annotationsByName []appctypes.Annotation

func (a annotationsByName) Len() int           { return len(a) }
func (a annotationsByName) Less(x, y int) bool { return a[x].Name < a[y].Name }
func (a annotationsByName) Swap(x, y int)      { a[x], a[y] = a[y], a[x] }

func TestMakePodManifestAnnotations(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	fr := newFakeRktInterface()
	fs := newFakeSystemd()
	r := &Runtime{apisvc: fr, systemd: fs}

	testCases := []struct {
		in     *api.Pod
		out    *appcschema.PodManifest
		outerr error
	}{
		{
			in: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					UID:       "uid-1",
					Name:      "name-1",
					Namespace: "namespace-1",
					Annotations: map[string]string{
						k8sRktStage1NameAnno: "stage1-override-img",
					},
				},
			},
			out: &appcschema.PodManifest{
				Annotations: []appctypes.Annotation{
					{
						Name:  "io.kubernetes.container.name",
						Value: "POD",
					},
					{
						Name:  appctypes.ACIdentifier(k8sRktStage1NameAnno),
						Value: "stage1-override-img",
					},
					{
						Name:  appctypes.ACIdentifier(types.KubernetesPodUIDLabel),
						Value: "uid-1",
					},
					{
						Name:  appctypes.ACIdentifier(types.KubernetesPodNameLabel),
						Value: "name-1",
					},
					{
						Name:  appctypes.ACIdentifier(k8sRktKubeletAnno),
						Value: "true",
					},
					{
						Name:  appctypes.ACIdentifier(types.KubernetesPodNamespaceLabel),
						Value: "namespace-1",
					},
					{
						Name:  appctypes.ACIdentifier(k8sRktRestartCountAnno),
						Value: "0",
					},
				},
			},
		},
	}

	for i, testCase := range testCases {
		hint := fmt.Sprintf("case #%d", i)

		result, err := r.makePodManifest(testCase.in, "", []api.Secret{})
		assert.Equal(t, testCase.outerr, err, hint)
		if err == nil {
			sort.Sort(annotationsByName(result.Annotations))
			sort.Sort(annotationsByName(testCase.out.Annotations))
			assert.Equal(t, testCase.out.Annotations, result.Annotations, hint)
		}
	}
}

func TestPreparePodArgs(t *testing.T) {
	r := &Runtime{
		config: &Config{},
	}

	testCases := []struct {
		manifest     appcschema.PodManifest
		stage1Config string
		cmd          []string
	}{
		{
			appcschema.PodManifest{
				Annotations: appctypes.Annotations{
					{
						Name:  k8sRktStage1NameAnno,
						Value: "stage1-image",
					},
				},
			},
			"",
			[]string{"prepare", "--quiet", "--pod-manifest", "file", "--stage1-name=stage1-image"},
		},
		{
			appcschema.PodManifest{
				Annotations: appctypes.Annotations{
					{
						Name:  k8sRktStage1NameAnno,
						Value: "stage1-image",
					},
				},
			},
			"stage1-image0",
			[]string{"prepare", "--quiet", "--pod-manifest", "file", "--stage1-name=stage1-image"},
		},
		{
			appcschema.PodManifest{
				Annotations: appctypes.Annotations{},
			},
			"stage1-image0",
			[]string{"prepare", "--quiet", "--pod-manifest", "file", "--stage1-name=stage1-image0"},
		},
		{
			appcschema.PodManifest{
				Annotations: appctypes.Annotations{},
			},
			"",
			[]string{"prepare", "--quiet", "--pod-manifest", "file"},
		},
	}

	for i, testCase := range testCases {
		r.config.Stage1Image = testCase.stage1Config
		cmd := r.preparePodArgs(&testCase.manifest, "file")
		assert.Equal(t, testCase.cmd, cmd, fmt.Sprintf("Test case #%d", i))
	}
}
