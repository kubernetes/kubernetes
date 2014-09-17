/*
Copyright 2014 Google Inc. All rights reserved.

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

package config

import (
	"encoding/json"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestURLErrorNotExistNoUpdate(t *testing.T) {
	ch := make(chan interface{})
	NewSourceURL("http://localhost:49575/_not_found_", time.Millisecond, ch)
	select {
	case got := <-ch:
		t.Errorf("Expected no update, Got %#v", got)
	case <-time.After(2 * time.Millisecond):
	}
}

func TestExtractFromHttpBadness(t *testing.T) {
	ch := make(chan interface{}, 1)
	c := SourceURL{"http://localhost:49575/_not_found_", ch, nil}
	if err := c.extractFromURL(); err == nil {
		t.Errorf("Expected error")
	}
	expectEmptyChannel(t, ch)
}

func TestExtractInvalidManifest(t *testing.T) {
	var testCases = []struct {
		desc      string
		manifests interface{}
	}{
		{
			desc:      "No version",
			manifests: []api.ContainerManifest{{Version: ""}},
		},
		{
			desc:      "Invalid version",
			manifests: []api.ContainerManifest{{Version: "v1betta2"}},
		},
		{
			desc: "Invalid volume name",
			manifests: []api.ContainerManifest{
				{Version: "v1beta1", Volumes: []api.Volume{{Name: "_INVALID_"}}},
			},
		},
		{
			desc: "Duplicate volume names",
			manifests: []api.ContainerManifest{
				{
					Version: "v1beta1",
					Volumes: []api.Volume{{Name: "repeated"}, {Name: "repeated"}},
				},
			},
		},
		{
			desc: "Unspecified container name",
			manifests: []api.ContainerManifest{
				{
					Version: "v1beta1",
					Containers: []api.Container{{Name: ""}},
				},
			},
		},
		{
			desc: "Invalid container name",
			manifests: []api.ContainerManifest{
				{
					Version: "v1beta1",
					Containers: []api.Container{{Name: "_INVALID_"}},
				},
			},
		},
	}
	for _, testCase := range testCases {
		data, err := json.Marshal(testCase.manifests)
		if err != nil {
			t.Fatalf("%s: Some weird json problem: %v", testCase.desc, err)
		}
		fakeHandler := util.FakeHandler{
			StatusCode:   200,
			ResponseBody: string(data),
		}
		testServer := httptest.NewServer(&fakeHandler)
		ch := make(chan interface{}, 1)
		c := SourceURL{testServer.URL, ch, nil}
		if err := c.extractFromURL(); err == nil {
			t.Errorf("%s: Expected error", testCase.desc)
		}
	}
}

func TestExtractFromHTTP(t *testing.T) {
	var testCases = []struct {
		desc      string
		manifests interface{}
		expected  kubelet.PodUpdate
	}{
		{
			desc:      "Single manifest",
			manifests: api.ContainerManifest{Version: "v1beta1", ID: "foo"},
			expected: CreatePodUpdate(kubelet.SET,
				kubelet.Pod{
					Name: "foo",
					Manifest: api.ContainerManifest{
						Version:       "v1beta1",
						ID:            "foo",
						RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
					},
				}),
		},
		{
			desc: "Multiple manifests",
			manifests: []api.ContainerManifest{
				{Version: "v1beta1", ID: "", Containers: []api.Container{{Name: "1", Image: "foo"}}},
				{Version: "v1beta1", ID: "bar", Containers: []api.Container{{Name: "1", Image: "foo"}}},
			},
			expected: CreatePodUpdate(kubelet.SET,
				kubelet.Pod{
					Name:     "1",
					Manifest: api.ContainerManifest{Version: "v1beta1", ID: "", Containers: []api.Container{{Name: "1", Image: "foo"}}},
				},
				kubelet.Pod{
					Name:     "bar",
					Manifest: api.ContainerManifest{Version: "v1beta1", ID: "bar", Containers: []api.Container{{Name: "1", Image: "foo"}}},
				}),
		},
		{
			desc:      "Empty Array",
			manifests: []api.ContainerManifest{},
			expected:  CreatePodUpdate(kubelet.SET),
		},
	}
	for _, testCase := range testCases {
		data, err := json.Marshal(testCase.manifests)
		if err != nil {
			t.Fatalf("%s: Some weird json problem: %v", testCase.desc, err)
		}
		fakeHandler := util.FakeHandler{
			StatusCode:   200,
			ResponseBody: string(data),
		}
		testServer := httptest.NewServer(&fakeHandler)
		ch := make(chan interface{}, 1)
		c := SourceURL{testServer.URL, ch, nil}
		if err := c.extractFromURL(); err != nil {
			t.Errorf("%s: Unexpected error: %v", testCase.desc, err)
		}
		update := (<-ch).(kubelet.PodUpdate)
		if !reflect.DeepEqual(testCase.expected, update) {
			t.Errorf("%s: Expected: %#v, Got: %#v", testCase.desc, testCase.expected, update)
		}
		for i := range update.Pods {
			if errs := kubelet.ValidatePod(&update.Pods[i]); len(errs) != 0 {
				t.Errorf("%s: Expected no validation errors on %#v, Got %#v", testCase.desc, update.Pods[i], errs)
			}
		}
	}
}
