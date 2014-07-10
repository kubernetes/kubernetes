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
	"github.com/fsouza/go-dockerclient"
)

type fakeKubelet struct {
	infoFunc           func(name string) (api.PodInfo, error)
	containerStatsFunc func(podID, containerName string) (*api.ContainerStats, error)
	machineStatsFunc   func() (*api.ContainerStats, error)
}

func (fk *fakeKubelet) GetPodInfo(name string) (api.PodInfo, error) {
	return fk.infoFunc(name)
}

func (fk *fakeKubelet) GetContainerStats(podID, containerName string) (*api.ContainerStats, error) {
	return fk.containerStatsFunc(podID, containerName)
}

func (fk *fakeKubelet) GetMachineStats() (*api.ContainerStats, error) {
	return fk.machineStatsFunc()
}

type serverTestFramework struct {
	updateChan      chan manifestUpdate
	updateReader    *channelReader
	serverUnderTest *KubeletServer
	fakeKubelet     *fakeKubelet
	testHTTPServer  *httptest.Server
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
	fw.testHTTPServer = httptest.NewServer(fw.serverUnderTest)
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
	resp, err := http.Post(fw.testHTTPServer.URL+"/container", "application/json", body)
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
	resp, err := http.Post(fw.testHTTPServer.URL+"/containers", "application/json", body)
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

func TestPodInfo(t *testing.T) {
	fw := makeServerTest()
	expected := api.PodInfo{"goodpod": docker.Container{ID: "myContainerID"}}
	fw.fakeKubelet.infoFunc = func(name string) (api.PodInfo, error) {
		if name == "goodpod" {
			return expected, nil
		}
		return nil, fmt.Errorf("bad pod")
	}
	resp, err := http.Get(fw.testHTTPServer.URL + "/podInfo?podID=goodpod")
	if err != nil {
		t.Errorf("Got error GETing: %v", err)
	}
	got, err := readResp(resp)
	if err != nil {
		t.Errorf("Error reading body: %v", err)
	}
	expectedBytes, err := json.Marshal(expected)
	if err != nil {
		t.Fatalf("Unexpected marshal error %v", err)
	}
	if got != string(expectedBytes) {
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
	expectedPodID := "somepod"
	expectedContainerName := "goodcontainer"
	fw.fakeKubelet.containerStatsFunc = func(podID, containerName string) (*api.ContainerStats, error) {
		if podID != expectedPodID || containerName != expectedContainerName {
			return nil, fmt.Errorf("bad podID or containerName: podID=%v; containerName=%v", podID, containerName)
		}
		return expectedStats, nil
	}

	resp, err := http.Get(fw.testHTTPServer.URL + fmt.Sprintf("/stats/%v/%v", expectedPodID, expectedContainerName))
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

func TestMachineStats(t *testing.T) {
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
	fw.fakeKubelet.machineStatsFunc = func() (*api.ContainerStats, error) {
		return expectedStats, nil
	}

	resp, err := http.Get(fw.testHTTPServer.URL + "/stats")
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
