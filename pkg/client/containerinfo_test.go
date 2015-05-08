/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"path"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	cadvisorApi "github.com/google/cadvisor/info/v1"
	cadvisorApiTest "github.com/google/cadvisor/info/v1/test"
)

func testHTTPContainerInfoGetter(
	req *cadvisorApi.ContainerInfoRequest,
	cinfo *cadvisorApi.ContainerInfo,
	podID string,
	containerID string,
	status int,
	t *testing.T,
) {
	expectedPath := "/stats"
	if len(podID) > 0 && len(containerID) > 0 {
		expectedPath = path.Join(expectedPath, podID, containerID)
	}
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if status != 0 {
			w.WriteHeader(status)
		}
		if strings.TrimRight(r.URL.Path, "/") != strings.TrimRight(expectedPath, "/") {
			t.Fatalf("Received request to an invalid path. Should be %v. got %v",
				expectedPath, r.URL.Path)
		}

		var receivedReq cadvisorApi.ContainerInfoRequest
		err := json.NewDecoder(r.Body).Decode(&receivedReq)
		if err != nil {
			t.Fatal(err)
		}
		// Note: This will not make a deep copy of req.
		// So changing req after Get*Info would be a race.
		expectedReq := req
		// Fill any empty fields with default value
		if !expectedReq.Equals(receivedReq) {
			t.Errorf("received wrong request")
		}
		err = json.NewEncoder(w).Encode(cinfo)
		if err != nil {
			t.Fatal(err)
		}
	}))
	defer ts.Close()
	hostURL, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	parts := strings.Split(hostURL.Host, ":")

	port, err := strconv.Atoi(parts[1])
	if err != nil {
		t.Fatal(err)
	}

	containerInfoGetter := &HTTPContainerInfoGetter{
		Client: http.DefaultClient,
		Port:   port,
	}

	var receivedContainerInfo *cadvisorApi.ContainerInfo
	if len(podID) > 0 && len(containerID) > 0 {
		receivedContainerInfo, err = containerInfoGetter.GetContainerInfo(parts[0], podID, containerID, req)
	} else {
		receivedContainerInfo, err = containerInfoGetter.GetRootInfo(parts[0], req)
	}
	if status == 0 || status == http.StatusOK {
		if err != nil {
			t.Errorf("received unexpected error: %v", err)
		}

		if !receivedContainerInfo.Eq(cinfo) {
			t.Error("received unexpected container info")
		}
	} else {
		if err == nil {
			t.Error("did not receive expected error.")
		}
	}
}

func TestHTTPContainerInfoGetterGetContainerInfoSuccessfully(t *testing.T) {
	req := &cadvisorApi.ContainerInfoRequest{
		NumStats: 10,
	}
	cinfo := cadvisorApiTest.GenerateRandomContainerInfo(
		"dockerIDWhichWillNotBeChecked", // docker ID
		2, // Number of cores
		req,
		1*time.Second,
	)
	testHTTPContainerInfoGetter(req, cinfo, "somePodID", "containerNameInK8S", 0, t)
}

func TestHTTPContainerInfoGetterGetRootInfoSuccessfully(t *testing.T) {
	req := &cadvisorApi.ContainerInfoRequest{
		NumStats: 10,
	}
	cinfo := cadvisorApiTest.GenerateRandomContainerInfo(
		"dockerIDWhichWillNotBeChecked", // docker ID
		2, // Number of cores
		req,
		1*time.Second,
	)
	testHTTPContainerInfoGetter(req, cinfo, "", "", 0, t)
}

func TestHTTPContainerInfoGetterGetContainerInfoWithError(t *testing.T) {
	req := &cadvisorApi.ContainerInfoRequest{
		NumStats: 10,
	}
	cinfo := cadvisorApiTest.GenerateRandomContainerInfo(
		"dockerIDWhichWillNotBeChecked", // docker ID
		2, // Number of cores
		req,
		1*time.Second,
	)
	testHTTPContainerInfoGetter(req, cinfo, "somePodID", "containerNameInK8S", http.StatusNotFound, t)
}

func TestHTTPContainerInfoGetterGetRootInfoWithError(t *testing.T) {
	req := &cadvisorApi.ContainerInfoRequest{
		NumStats: 10,
	}
	cinfo := cadvisorApiTest.GenerateRandomContainerInfo(
		"dockerIDWhichWillNotBeChecked", // docker ID
		2, // Number of cores
		req,
		1*time.Second,
	)
	testHTTPContainerInfoGetter(req, cinfo, "", "", http.StatusNotFound, t)
}

func TestHTTPGetMachineInfo(t *testing.T) {
	mspec := &cadvisorApi.MachineInfo{
		NumCores:       4,
		MemoryCapacity: 2048,
	}
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		err := json.NewEncoder(w).Encode(mspec)
		if err != nil {
			t.Fatal(err)
		}
	}))
	defer ts.Close()
	hostURL, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	parts := strings.Split(hostURL.Host, ":")

	port, err := strconv.Atoi(parts[1])
	if err != nil {
		t.Fatal(err)
	}

	containerInfoGetter := &HTTPContainerInfoGetter{
		Client: http.DefaultClient,
		Port:   port,
	}

	received, err := containerInfoGetter.GetMachineInfo(parts[0])
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(received, mspec) {
		t.Errorf("received wrong machine spec")
	}
}
