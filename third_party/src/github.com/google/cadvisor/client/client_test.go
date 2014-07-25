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

package cadvisor

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/google/cadvisor/info"
	itest "github.com/google/cadvisor/info/test"
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

func cadvisorTestClient(path string, expectedPostObj, expectedPostObjEmpty, replyObj interface{}, t *testing.T) (*Client, *httptest.Server, error) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == path {
			if expectedPostObj != nil {
				decoder := json.NewDecoder(r.Body)
				err := decoder.Decode(expectedPostObjEmpty)
				if err != nil {
					t.Errorf("Recieved invalid object: %v", err)
				}
				if !reflect.DeepEqual(expectedPostObj, expectedPostObjEmpty) {
					t.Errorf("Recieved unexpected object: %+v", expectedPostObjEmpty)
				}
			}
			encoder := json.NewEncoder(w)
			encoder.Encode(replyObj)
		} else if r.URL.Path == "/api/v1.0/machine" {
			fmt.Fprint(w, `{"num_cores":8,"memory_capacity":31625871360}`)
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

func TestGetMachineinfo(t *testing.T) {
	minfo := &info.MachineInfo{
		NumCores:       8,
		MemoryCapacity: 31625871360,
	}
	client, server, err := cadvisorTestClient("/api/v1.0/machine", nil, nil, minfo, t)
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

func TestGetContainerInfo(t *testing.T) {
	query := &info.ContainerInfoRequest{
		NumStats:               3,
		NumSamples:             2,
		CpuUsagePercentiles:    []int{10, 50, 90},
		MemoryUsagePercentiles: []int{10, 80, 90},
	}
	containerName := "/some/container"
	cinfo := itest.GenerateRandomContainerInfo(containerName, 4, query, 1*time.Second)
	client, server, err := cadvisorTestClient(fmt.Sprintf("/api/v1.0/containers%v", containerName), query, &info.ContainerInfoRequest{}, cinfo, t)
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
