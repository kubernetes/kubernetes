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

package client

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestHTTPKubeletClient(t *testing.T) {
	expectObj := api.PodInfo{
		"myID": api.ContainerStatus{},
	}
	body, err := json.Marshal(expectObj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()

	hostURL, err := url.Parse(testServer.URL)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	parts := strings.Split(hostURL.Host, ":")

	port, err := strconv.Atoi(parts[1])
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	podInfoGetter := &HTTPKubeletClient{
		Client: http.DefaultClient,
		Port:   uint(port),
	}
	gotObj, err := podInfoGetter.GetPodInfo(parts[0], api.NamespaceDefault, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// reflect.DeepEqual(expectObj, gotObj) doesn't handle blank times well
	if len(gotObj) != len(expectObj) {
		t.Errorf("Unexpected response.  Expected: %#v, received %#v", expectObj, gotObj)
	}
}

func TestHTTPKubeletClientNotFound(t *testing.T) {
	expectObj := api.PodInfo{
		"myID": api.ContainerStatus{},
	}
	_, err := json.Marshal(expectObj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	fakeHandler := util.FakeHandler{
		StatusCode:   404,
		ResponseBody: "Pod not found",
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()

	hostURL, err := url.Parse(testServer.URL)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	parts := strings.Split(hostURL.Host, ":")

	port, err := strconv.Atoi(parts[1])
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	podInfoGetter := &HTTPKubeletClient{
		Client: http.DefaultClient,
		Port:   uint(port),
	}
	_, err = podInfoGetter.GetPodInfo(parts[0], api.NamespaceDefault, "foo")
	if err != ErrPodInfoNotAvailable {
		t.Errorf("Expected %#v, Got %#v", ErrPodInfoNotAvailable, err)
	}
}
