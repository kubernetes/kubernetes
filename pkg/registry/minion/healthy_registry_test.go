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

package minion

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
)

type alwaysYes struct{}

func fakeHTTPResponse(status int) *http.Response {
	return &http.Response{
		StatusCode: status,
		Body:       ioutil.NopCloser(&bytes.Buffer{}),
	}
}

func (alwaysYes) Get(url string) (*http.Response, error) {
	return fakeHTTPResponse(http.StatusOK), nil
}

func TestBasicDelegation(t *testing.T) {
	mockMinionRegistry := registrytest.NewMinionRegistry([]string{"m1", "m2", "m3"})
	healthy := HealthyRegistry{
		delegate: mockMinionRegistry,
		client:   alwaysYes{},
	}
	list, err := healthy.List()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(list, mockMinionRegistry.Minions) {
		t.Errorf("Expected %v, Got %v", mockMinionRegistry.Minions, list)
	}
	err = healthy.Insert("foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	ok, err := healthy.Contains("m1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !ok {
		t.Errorf("Unexpected absence of 'm1'")
	}
	ok, err = healthy.Contains("m5")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if ok {
		t.Errorf("Unexpected presence of 'm5'")
	}
}

type notMinion struct {
	minion string
}

func (n *notMinion) Get(url string) (*http.Response, error) {
	if url != "http://"+n.minion+":10250/healthz" {
		return fakeHTTPResponse(http.StatusOK), nil
	} else {
		return fakeHTTPResponse(http.StatusInternalServerError), nil
	}
}

func TestFiltering(t *testing.T) {
	mockMinionRegistry := registrytest.NewMinionRegistry([]string{"m1", "m2", "m3"})
	healthy := HealthyRegistry{
		delegate: mockMinionRegistry,
		client:   &notMinion{minion: "m1"},
		port:     10250,
	}
	expected := []string{"m2", "m3"}
	list, err := healthy.List()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(list, expected) {
		t.Errorf("Expected %v, Got %v", expected, list)
	}
	ok, err := healthy.Contains("m1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if ok {
		t.Errorf("Unexpected presence of 'm1'")
	}
}
