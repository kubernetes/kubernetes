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

package kubelet

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

type fakeKubelet struct {
	infoFunc  func(name string) (string, error)
	statsFunc func(name string) (*api.ContainerStats, error)
}

func (fk *fakeKubelet) GetContainerInfo(name string) (string, error) {
	return fk.infoFunc(name)
}

func (fk *fakeKubelet) GetContainerStats(name string) (*api.ContainerStats, error) {
	return fk.statsFunc(name)
}

type serverTestFramework struct {
	updateChan      chan manifestUpdate
	updateReader    *channelReader
	serverUnderTest *KubeletServer
	fakeKubelet     *fakeKubelet
	testHttpServer  *httptest.Server
}

func makeServerTest() *serverTestFramework {
	fw := &serverTestFramework{
		updateChan: make(chan manifestUpdate),
	}
	fw.updateReader = startReading(fw.updateChan)
	fw.fakeKubelet = &fakeKubelet{}
	fw.serverUnderTest = &KubeletServer{
		Kubelet:       fw.fakeKubelet,
		UpdateChannel: fw.updateChan,
	}
	fw.testHttpServer = httptest.NewServer(fw.serverUnderTest)
	return fw
}

func readResp(resp *http.Response) (string, error) {
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	return string(body), err
}

func TestContainer(t *testing.T) {
	fw := makeServerTest()
	expected := []api.ContainerManifest{
		{ID: "test_manifest"},
	}
	body := bytes.NewBuffer([]byte(util.MakeJSONString(expected[0]))) // Only send a single ContainerManifest
	resp, err := http.Post(fw.testHttpServer.URL+"/container", "application/json", body)
	if err != nil {
		t.Errorf("Post returned: %v", err)
	}
	resp.Body.Close()
	close(fw.updateChan)
	received := fw.updateReader.GetList()
	if len(received) != 1 {
		t.Errorf("Expected 1 manifest, but got %v", len(received))
	}
	if !reflect.DeepEqual(expected, received[0]) {
		t.Errorf("Expected %#v, but got %#v", expected, received[0])
	}
}

func TestContainers(t *testing.T) {
	fw := makeServerTest()
	expected := []api.ContainerManifest{
		{ID: "test_manifest_1"},
		{ID: "test_manifest_2"},
	}
	body := bytes.NewBuffer([]byte(util.MakeJSONString(expected)))
	resp, err := http.Post(fw.testHttpServer.URL+"/containers", "application/json", body)
	if err != nil {
		t.Errorf("Post returned: %v", err)
	}
	resp.Body.Close()
	close(fw.updateChan)
	received := fw.updateReader.GetList()
	if len(received) != 1 {
		t.Errorf("Expected 1 update, but got %v", len(received))
	}
	if !reflect.DeepEqual(expected, received[0]) {
		t.Errorf("Expected %#v, but got %#v", expected, received[0])
	}
}

func TestContainerInfo(t *testing.T) {
	fw := makeServerTest()
	expected := "good container info string"
	fw.fakeKubelet.infoFunc = func(name string) (string, error) {
		if name == "goodcontainer" {
			return expected, nil
		}
		return "", fmt.Errorf("bad container")
	}
	resp, err := http.Get(fw.testHttpServer.URL + "/containerInfo?container=goodcontainer")
	if err != nil {
		t.Errorf("Got error GETing: %v", err)
	}
	got, err := readResp(resp)
	if err != nil {
		t.Errorf("Error reading body: %v", err)
	}
	if got != expected {
		t.Errorf("Expected: '%v', got: '%v'", expected, got)
	}
}

func TestContainerStats(t *testing.T) {
	fw := makeServerTest()
	expectedStats := &api.ContainerStats{
		MaxMemoryUsage: 1024001,
		CpuUsagePercentiles: []api.Percentile{
			{50, 150},
			{80, 180},
			{90, 190},
		},
		MemoryUsagePercentiles: []api.Percentile{
			{50, 150},
			{80, 180},
			{90, 190},
		},
	}
	expectedContainerName := "goodcontainer"
	fw.fakeKubelet.statsFunc = func(name string) (*api.ContainerStats, error) {
		if name != expectedContainerName {
			return nil, fmt.Errorf("bad container name: %v", name)
		}
		return expectedStats, nil
	}

	resp, err := http.Get(fw.testHttpServer.URL + fmt.Sprintf("/containerStats?container=%v", expectedContainerName))
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()
	var receivedStats api.ContainerStats
	decoder := json.NewDecoder(resp.Body)
	err = decoder.Decode(&receivedStats)
	if err != nil {
		t.Fatalf("received invalid json data: %v", err)
	}
	if !reflect.DeepEqual(&receivedStats, expectedStats) {
		t.Errorf("received wrong data: %#v", receivedStats)
	}
}
