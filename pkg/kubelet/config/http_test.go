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
	c := SourceURL{"http://localhost:49575/_not_found_", ch}
	err := c.extractFromURL()
	if err == nil {
		t.Errorf("Expected error")
	}
	expectEmptyChannel(t, ch)
}

func TestExtractFromHttpSingle(t *testing.T) {
	manifests := []api.ContainerManifest{
		{Version: "v1beta1", ID: "foo"},
	}
	// Taking a single-manifest from a URL allows kubelet to be used
	// in the implementation of google's container VM image.
	data, err := json.Marshal(manifests[0])

	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(data),
	}
	testServer := httptest.NewServer(&fakeHandler)
	ch := make(chan interface{}, 1)
	c := SourceURL{testServer.URL, ch}

	err = c.extractFromURL()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	update := (<-ch).(kubelet.PodUpdate)
	expected := CreatePodUpdate(kubelet.SET, kubelet.Pod{Name: "foo", Manifest: manifests[0]})
	if !reflect.DeepEqual(expected, update) {
		t.Errorf("Expected: %#v, Got: %#v", expected, update)
	}
}

func TestExtractFromHttpMultiple(t *testing.T) {
	manifests := []api.ContainerManifest{
		{Version: "v1beta1", ID: ""},
		{Version: "v1beta1", ID: "bar"},
	}
	data, err := json.Marshal(manifests)
	if err != nil {
		t.Fatalf("Some weird json problem: %v", err)
	}

	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(data),
	}
	testServer := httptest.NewServer(&fakeHandler)
	ch := make(chan interface{}, 1)
	c := SourceURL{testServer.URL, ch}

	err = c.extractFromURL()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	update := (<-ch).(kubelet.PodUpdate)
	expected := CreatePodUpdate(kubelet.SET, kubelet.Pod{Name: "1", Manifest: manifests[0]}, kubelet.Pod{Name: "bar", Manifest: manifests[1]})
	if !reflect.DeepEqual(expected, update) {
		t.Errorf("Expected: %#v, Got: %#v", expected, update)
	}
}

func TestExtractFromHttpEmptyArray(t *testing.T) {
	manifests := []api.ContainerManifest{}
	data, err := json.Marshal(manifests)
	if err != nil {
		t.Fatalf("Some weird json problem: %v", err)
	}

	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(data),
	}
	testServer := httptest.NewServer(&fakeHandler)
	ch := make(chan interface{}, 1)
	c := SourceURL{testServer.URL, ch}

	err = c.extractFromURL()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	update := (<-ch).(kubelet.PodUpdate)
	expected := CreatePodUpdate(kubelet.SET)
	if !reflect.DeepEqual(expected, update) {
		t.Errorf("Expected: %#v, Got: %#v", expected, update)
	}
}
