// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package client

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path"
	"reflect"
	"strings"
	"testing"
	"time"

	info "github.com/google/cadvisor/info/v1"
	itest "github.com/google/cadvisor/info/v1/test"

	"github.com/kr/pretty"
)

func testGetJsonData(
	expected interface{},
	f func() (interface{}, error),
) error {
	reply, err := f()
	if err != nil {
		return fmt.Errorf("unable to retrieve data: %v", err)
	}
	if !reflect.DeepEqual(reply, expected) {
		return pretty.Errorf("retrieved wrong data: %# v != %# v", reply, expected)
	}
	return nil
}

func cadvisorTestClient(path string, expectedPostObj *info.ContainerInfoRequest, replyObj interface{}, t *testing.T) (*Client, *httptest.Server, error) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == path {
			if expectedPostObj != nil {
				expectedPostObjEmpty := new(info.ContainerInfoRequest)
				decoder := json.NewDecoder(r.Body)
				if err := decoder.Decode(expectedPostObjEmpty); err != nil {
					t.Errorf("Received invalid object: %v", err)
				}
				if expectedPostObj.NumStats != expectedPostObjEmpty.NumStats ||
					expectedPostObj.Start.Unix() != expectedPostObjEmpty.Start.Unix() ||
					expectedPostObj.End.Unix() != expectedPostObjEmpty.End.Unix() {
					t.Errorf("Received unexpected object: %+v, expected: %+v", expectedPostObjEmpty, expectedPostObj)
				}
			}
			encoder := json.NewEncoder(w)
			encoder.Encode(replyObj)
		} else {
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprintf(w, "Page not found.")
		}
	}))
	client, err := NewClient(ts.URL)
	if err != nil {
		ts.Close()
		return nil, nil, err
	}
	return client, ts, err
}

// TestGetMachineInfo performs one test to check if MachineInfo()
// in a cAdvisor client returns the correct result.
func TestGetMachineinfo(t *testing.T) {
	minfo := &info.MachineInfo{
		NumCores:       8,
		MemoryCapacity: 31625871360,
		DiskMap: map[string]info.DiskInfo{
			"8:0": {
				Name:  "sda",
				Major: 8,
				Minor: 0,
				Size:  10737418240,
			},
		},
	}
	client, server, err := cadvisorTestClient("/api/v1.3/machine", nil, minfo, t)
	if err != nil {
		t.Fatalf("unable to get a client %v", err)
	}
	defer server.Close()
	returned, err := client.MachineInfo()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(returned, minfo) {
		t.Fatalf("received unexpected machine info")
	}
}

// TestGetContainerInfo generates a random container information object
// and then checks that ContainerInfo returns the expected result.
func TestGetContainerInfo(t *testing.T) {
	query := &info.ContainerInfoRequest{
		NumStats: 3,
	}
	containerName := "/some/container"
	cinfo := itest.GenerateRandomContainerInfo(containerName, 4, query, 1*time.Second)
	client, server, err := cadvisorTestClient(fmt.Sprintf("/api/v1.3/containers%v", containerName), query, cinfo, t)
	if err != nil {
		t.Fatalf("unable to get a client %v", err)
	}
	defer server.Close()
	returned, err := client.ContainerInfo(containerName, query)
	if err != nil {
		t.Fatal(err)
	}

	if !returned.Eq(cinfo) {
		t.Error("received unexpected ContainerInfo")
	}
}

// Test a request failing
func TestRequestFails(t *testing.T) {
	errorText := "there was an error"
	// Setup a server that simply fails.
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, errorText, 500)
	}))
	client, err := NewClient(ts.URL)
	if err != nil {
		ts.Close()
		t.Fatal(err)
	}
	defer ts.Close()

	_, err = client.ContainerInfo("/", &info.ContainerInfoRequest{NumStats: 3})
	if err == nil {
		t.Fatalf("Expected non-nil error")
	}
	expectedError := fmt.Sprintf("request failed with error: %q", errorText)
	if strings.Contains(err.Error(), expectedError) {
		t.Fatalf("Expected error %q but received %q", expectedError, err)
	}
}

func TestGetSubcontainersInfo(t *testing.T) {
	query := &info.ContainerInfoRequest{
		NumStats: 3,
	}
	containerName := "/some/container"
	cinfo := itest.GenerateRandomContainerInfo(containerName, 4, query, 1*time.Second)
	cinfo1 := itest.GenerateRandomContainerInfo(path.Join(containerName, "sub1"), 4, query, 1*time.Second)
	cinfo2 := itest.GenerateRandomContainerInfo(path.Join(containerName, "sub2"), 4, query, 1*time.Second)
	response := []info.ContainerInfo{
		*cinfo,
		*cinfo1,
		*cinfo2,
	}
	client, server, err := cadvisorTestClient(fmt.Sprintf("/api/v1.3/subcontainers%v", containerName), query, response, t)
	if err != nil {
		t.Fatalf("unable to get a client %v", err)
	}
	defer server.Close()
	returned, err := client.SubcontainersInfo(containerName, query)
	if err != nil {
		t.Fatal(err)
	}

	if len(returned) != 3 {
		t.Errorf("unexpected number of results: got %d, expected 3", len(returned))
	}
	if !returned[0].Eq(cinfo) {
		t.Error("received unexpected ContainerInfo")
	}
	if !returned[1].Eq(cinfo1) {
		t.Error("received unexpected ContainerInfo")
	}
	if !returned[2].Eq(cinfo2) {
		t.Error("received unexpected ContainerInfo")
	}
}
