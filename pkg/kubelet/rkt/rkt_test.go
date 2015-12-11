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

	appcschema "github.com/appc/spec/schema"
	appctypes "github.com/appc/spec/schema/types"
	rktapi "github.com/coreos/rkt/api/v1alpha"
	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/types"
)

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
		images []*rktapi.Image
	}{
		{},
		{
			[]*rktapi.Image{
				{
					Id:      "sha512-a2fb8f390702",
					Name:    "quay.io/coreos/alpine-sh",
					Version: "latest",
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
		},
	}

	for i, tt := range tests {
		fr.images = tt.images

		images, err := r.ListImages()
		if err != nil {
			t.Errorf("%v", err)
		}
		assert.Equal(t, len(images), len(tt.images), fmt.Sprintf("test case %d: mismatched number of images", i))
		for i, image := range images {
			assert.Equal(t, image.ID, tt.images[i].Id, fmt.Sprintf("test case %d: mismatched image IDs", i))
			assert.Equal(t, []string{tt.images[i].Name}, image.Tags, fmt.Sprintf("test case %d: mismatched image tags", i))
		}

		assert.Equal(t, fr.called, []string{"ListImages"}, fmt.Sprintf("test case %d: unexpected called list", i))

		fr.CleanCalls()
	}
}

func TestGetPods(t *testing.T) {
	fr := newFakeRktInterface()
	fs := newFakeSystemd()
	r := &Runtime{apisvc: fr, systemd: fs}

	tests := []struct {
		k8sUID        types.UID
		k8sName       string
		k8sNamespace  string
		k8sCreation   int64
		k8sRestart    int
		k8sContHashes []uint64
		rktPodState   rktapi.PodState
		pods          []*rktapi.Pod
	}{
		{},
		{
			k8sUID:        types.UID("0"),
			k8sName:       "guestbook",
			k8sNamespace:  "default",
			k8sCreation:   10000000000,
			k8sRestart:    1,
			k8sContHashes: []uint64{2353434678},
			rktPodState:   rktapi.PodState_POD_STATE_RUNNING,
			pods: []*rktapi.Pod{
				{
					State: rktapi.PodState_POD_STATE_RUNNING,
					Apps: []*rktapi.App{
						{
							Name: "test",
							Image: &rktapi.Image{
								Name: "test",
								Manifest: mustMarshalImageManifest(
									&appcschema.ImageManifest{
										ACKind:    appcschema.ImageManifestKind,
										ACVersion: appcschema.AppContainerVersion,
										Name:      *appctypes.MustACIdentifier("test"),
										Annotations: appctypes.Annotations{
											appctypes.Annotation{
												Name:  *appctypes.MustACIdentifier(k8sRktContainerHashAnno),
												Value: "2353434678",
											},
										},
									},
								),
							},
						},
					},
					Manifest: mustMarshalPodManifest(
						&appcschema.PodManifest{
							ACKind:    appcschema.PodManifestKind,
							ACVersion: appcschema.AppContainerVersion,
							Annotations: appctypes.Annotations{
								appctypes.Annotation{
									Name:  *appctypes.MustACIdentifier(k8sRktKubeletAnno),
									Value: k8sRktKubeletAnnoValue,
								},
								appctypes.Annotation{
									Name:  *appctypes.MustACIdentifier(k8sRktUIDAnno),
									Value: "0",
								},
								appctypes.Annotation{
									Name:  *appctypes.MustACIdentifier(k8sRktNameAnno),
									Value: "guestbook",
								},
								appctypes.Annotation{
									Name:  *appctypes.MustACIdentifier(k8sRktNamespaceAnno),
									Value: "default",
								},
								appctypes.Annotation{
									Name:  *appctypes.MustACIdentifier(k8sRktCreationTimeAnno),
									Value: "10000000000",
								},
								appctypes.Annotation{
									Name:  *appctypes.MustACIdentifier(k8sRktRestartCountAnno),
									Value: "1",
								},
							},
						},
					),
				},
			},
		},
		{
			k8sUID:        types.UID("1"),
			k8sName:       "test-pod",
			k8sNamespace:  "default",
			k8sCreation:   10000000001,
			k8sRestart:    3,
			k8sContHashes: []uint64{2353434682, 8732645},
			rktPodState:   rktapi.PodState_POD_STATE_EXITED,
			pods: []*rktapi.Pod{
				{
					State: rktapi.PodState_POD_STATE_EXITED,
					Apps: []*rktapi.App{
						{
							Name: "test",
							Image: &rktapi.Image{
								Name: "test",
								Manifest: mustMarshalImageManifest(
									&appcschema.ImageManifest{
										ACKind:    appcschema.ImageManifestKind,
										ACVersion: appcschema.AppContainerVersion,
										Name:      *appctypes.MustACIdentifier("test"),
										Annotations: appctypes.Annotations{
											appctypes.Annotation{
												Name:  *appctypes.MustACIdentifier(k8sRktContainerHashAnno),
												Value: "2353434682",
											},
										},
									},
								),
							},
						},
						{
							Name: "test2",
							Image: &rktapi.Image{
								Name: "test2",
								Manifest: mustMarshalImageManifest(
									&appcschema.ImageManifest{
										ACKind:    appcschema.ImageManifestKind,
										ACVersion: appcschema.AppContainerVersion,
										Name:      *appctypes.MustACIdentifier("test2"),
										Annotations: appctypes.Annotations{
											appctypes.Annotation{
												Name:  *appctypes.MustACIdentifier(k8sRktContainerHashAnno),
												Value: "8732645",
											},
										},
									},
								),
							},
						},
					},
					Manifest: mustMarshalPodManifest(
						&appcschema.PodManifest{
							ACKind:    appcschema.PodManifestKind,
							ACVersion: appcschema.AppContainerVersion,
							Annotations: appctypes.Annotations{
								appctypes.Annotation{
									Name:  *appctypes.MustACIdentifier(k8sRktKubeletAnno),
									Value: k8sRktKubeletAnnoValue,
								},
								appctypes.Annotation{
									Name:  *appctypes.MustACIdentifier(k8sRktUIDAnno),
									Value: "1",
								},
								appctypes.Annotation{
									Name:  *appctypes.MustACIdentifier(k8sRktNameAnno),
									Value: "test-pod",
								},
								appctypes.Annotation{
									Name:  *appctypes.MustACIdentifier(k8sRktNamespaceAnno),
									Value: "default",
								},
								appctypes.Annotation{
									Name:  *appctypes.MustACIdentifier(k8sRktCreationTimeAnno),
									Value: "10000000001",
								},
								appctypes.Annotation{
									Name:  *appctypes.MustACIdentifier(k8sRktRestartCountAnno),
									Value: "3",
								},
							},
						},
					),
				},
			},
		},
	}

	for i, tt := range tests {
		fr.pods = tt.pods

		pods, err := r.GetPods(true)
		if err != nil {
			t.Errorf("%v", err)
		}
		assert.Equal(t, len(pods), len(tt.pods), fmt.Sprintf("test case %d: mismatched number of pods", i))

		for j, pod := range pods {
			assert.Equal(t, pod.ID, tt.k8sUID, fmt.Sprintf("test case %d: mismatched UIDs", i))
			assert.Equal(t, pod.Name, tt.k8sName, fmt.Sprintf("test case %d: mismatched Names", i))
			assert.Equal(t, pod.Namespace, tt.k8sNamespace, fmt.Sprintf("test case %d: mismatched Namespaces", i))
			assert.Equal(t, len(pod.Containers), len(tt.pods[j].Apps), fmt.Sprintf("test case %d: mismatched number of containers", i))
			for k, cont := range pod.Containers {
				assert.Equal(t, cont.Created, tt.k8sCreation, fmt.Sprintf("test case %d: mismatched creation times", i))
				assert.Equal(t, cont.Hash, tt.k8sContHashes[k], fmt.Sprintf("test case %d: mismatched container hashes", i))
			}
		}

		var inspectPodCalls []string
		for range pods {
			inspectPodCalls = append(inspectPodCalls, "InspectPod")
		}
		assert.Equal(t, append([]string{"ListPods"}, inspectPodCalls...), fr.called, fmt.Sprintf("test case %d: unexpected called list", i))

		fr.CleanCalls()
	}
}

func TestGetPodsFilter(t *testing.T) {
	fr := newFakeRktInterface()
	fs := newFakeSystemd()
	r := &Runtime{apisvc: fr, systemd: fs}

	for _, test := range []struct {
		All            bool
		ExpectedFilter *rktapi.PodFilter
	}{
		{
			true,
			&rktapi.PodFilter{
				Annotations: []*rktapi.KeyValue{
					{
						Key:   k8sRktKubeletAnno,
						Value: k8sRktKubeletAnnoValue,
					},
				},
			},
		},
		{
			false,
			&rktapi.PodFilter{
				States: []rktapi.PodState{rktapi.PodState_POD_STATE_RUNNING},
				Annotations: []*rktapi.KeyValue{
					{
						Key:   k8sRktKubeletAnno,
						Value: k8sRktKubeletAnnoValue,
					},
				},
			},
		},
	} {
		_, err := r.GetPods(test.All)
		if err != nil {
			t.Errorf("%v", err)
		}
		assert.Equal(t, test.ExpectedFilter, fr.podFilter, "filters didn't match when all=%b", test.All)
	}
}

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
