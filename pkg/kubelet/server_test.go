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
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"reflect"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/fsouza/go-dockerclient"
	"github.com/google/cadvisor/info"
)

type fakeKubelet struct {
	infoFunc          func(name string) (api.PodInfo, error)
	containerInfoFunc func(podFullName, containerName string, req *info.ContainerInfoRequest) (*info.ContainerInfo, error)
	rootInfoFunc      func(query *info.ContainerInfoRequest) (*info.ContainerInfo, error)
	machineInfoFunc   func() (*info.MachineInfo, error)
	logFunc           func(w http.ResponseWriter, req *http.Request)
	runFunc           func(podFullName, uuid, containerName string, cmd []string) ([]byte, error)
	containerLogsFunc func(podFullName, containerName, tail string, follow bool, stdout, stderr io.Writer) error
}

func (fk *fakeKubelet) GetPodInfo(name, uuid string) (api.PodInfo, error) {
	return fk.infoFunc(name)
}

func (fk *fakeKubelet) GetContainerInfo(podFullName, uuid, containerName string, req *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
	return fk.containerInfoFunc(podFullName, containerName, req)
}

func (fk *fakeKubelet) GetRootInfo(req *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
	return fk.rootInfoFunc(req)
}

func (fk *fakeKubelet) GetMachineInfo() (*info.MachineInfo, error) {
	return fk.machineInfoFunc()
}

func (fk *fakeKubelet) ServeLogs(w http.ResponseWriter, req *http.Request) {
	fk.logFunc(w, req)
}

func (fk *fakeKubelet) GetKubeletContainerLogs(podFullName, containerName, tail string, follow bool, stdout, stderr io.Writer) error {
	return fk.containerLogsFunc(podFullName, containerName, tail, follow, stdout, stderr)
}

func (fk *fakeKubelet) RunInContainer(podFullName, uuid, containerName string, cmd []string) ([]byte, error) {
	return fk.runFunc(podFullName, uuid, containerName, cmd)
}

type serverTestFramework struct {
	updateChan      chan interface{}
	updateReader    *channelReader
	serverUnderTest *Server
	fakeKubelet     *fakeKubelet
	testHTTPServer  *httptest.Server
}

func newServerTest() *serverTestFramework {
	fw := &serverTestFramework{
		updateChan: make(chan interface{}),
	}
	fw.updateReader = startReading(fw.updateChan)
	fw.fakeKubelet = &fakeKubelet{}
	server := NewServer(fw.fakeKubelet, fw.updateChan)
	fw.serverUnderTest = &server
	fw.testHTTPServer = httptest.NewServer(fw.serverUnderTest)
	return fw
}

func readResp(resp *http.Response) (string, error) {
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	return string(body), err
}

func TestContainer(t *testing.T) {
	fw := newServerTest()
	expected := []api.ContainerManifest{
		{ID: "test_manifest"},
	}
	body := bytes.NewBuffer([]byte(util.EncodeJSON(expected[0]))) // Only send a single ContainerManifest
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
	expectedPods := []Pod{{Name: "1", Manifest: expected[0]}}
	if !reflect.DeepEqual(expectedPods, received[0]) {
		t.Errorf("Expected %#v, but got %#v", expectedPods, received[0])
	}
}

func TestContainers(t *testing.T) {
	fw := newServerTest()
	expected := []api.ContainerManifest{
		{ID: "test_manifest_1"},
		{ID: "test_manifest_2"},
	}
	body := bytes.NewBuffer([]byte(util.EncodeJSON(expected)))
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
	expectedPods := []Pod{{Name: "1", Manifest: expected[0]}, {Name: "2", Manifest: expected[1]}}
	if !reflect.DeepEqual(expectedPods, received[0]) {
		t.Errorf("Expected %#v, but got %#v", expectedPods, received[0])
	}
}

func TestPodInfo(t *testing.T) {
	fw := newServerTest()
	expected := api.PodInfo{"goodpod": docker.Container{ID: "myContainerID"}}
	fw.fakeKubelet.infoFunc = func(name string) (api.PodInfo, error) {
		if name == "goodpod.etcd" {
			return expected, nil
		}
		return nil, fmt.Errorf("bad pod %s", name)
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

func TestContainerInfo(t *testing.T) {
	fw := newServerTest()
	expectedInfo := &info.ContainerInfo{}
	expectedPodID := "somepod"
	expectedContainerName := "goodcontainer"
	fw.fakeKubelet.containerInfoFunc = func(podID, containerName string, req *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
		if podID != expectedPodID || containerName != expectedContainerName {
			return nil, fmt.Errorf("bad podID or containerName: podID=%v; containerName=%v", podID, containerName)
		}
		return expectedInfo, nil
	}

	resp, err := http.Get(fw.testHTTPServer.URL + fmt.Sprintf("/stats/%v/%v", expectedPodID, expectedContainerName))
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()
	var receivedInfo info.ContainerInfo
	err = json.NewDecoder(resp.Body).Decode(&receivedInfo)
	if err != nil {
		t.Fatalf("received invalid json data: %v", err)
	}
	if !reflect.DeepEqual(&receivedInfo, expectedInfo) {
		t.Errorf("received wrong data: %#v", receivedInfo)
	}
}

func TestRootInfo(t *testing.T) {
	fw := newServerTest()
	expectedInfo := &info.ContainerInfo{}
	fw.fakeKubelet.rootInfoFunc = func(req *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
		return expectedInfo, nil
	}

	resp, err := http.Get(fw.testHTTPServer.URL + "/stats")
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()
	var receivedInfo info.ContainerInfo
	err = json.NewDecoder(resp.Body).Decode(&receivedInfo)
	if err != nil {
		t.Fatalf("received invalid json data: %v", err)
	}
	if !reflect.DeepEqual(&receivedInfo, expectedInfo) {
		t.Errorf("received wrong data: %#v", receivedInfo)
	}
}

func TestMachineInfo(t *testing.T) {
	fw := newServerTest()
	expectedInfo := &info.MachineInfo{
		NumCores:       4,
		MemoryCapacity: 1024,
	}
	fw.fakeKubelet.machineInfoFunc = func() (*info.MachineInfo, error) {
		return expectedInfo, nil
	}

	resp, err := http.Get(fw.testHTTPServer.URL + "/spec")
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()
	var receivedInfo info.MachineInfo
	err = json.NewDecoder(resp.Body).Decode(&receivedInfo)
	if err != nil {
		t.Fatalf("received invalid json data: %v", err)
	}
	if !reflect.DeepEqual(&receivedInfo, expectedInfo) {
		t.Errorf("received wrong data: %#v", receivedInfo)
	}
}

func TestServeLogs(t *testing.T) {
	fw := newServerTest()

	content := string(`<pre><a href="kubelet.log">kubelet.log</a><a href="google.log">google.log</a></pre>`)

	fw.fakeKubelet.logFunc = func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Header().Add("Content-Type", "text/html")
		w.Write([]byte(content))
	}

	resp, err := http.Get(fw.testHTTPServer.URL + "/logs/")
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()

	body, err := httputil.DumpResponse(resp, true)
	if err != nil {
		// copying the response body did not work
		t.Errorf("Cannot copy resp: %#v", err)
	}
	result := string(body)
	if !strings.Contains(result, "kubelet.log") || !strings.Contains(result, "google.log") {
		t.Errorf("Received wrong data: %s", result)
	}
}

func TestServeRunInContainer(t *testing.T) {
	fw := newServerTest()
	output := "foo bar"
	podName := "foo"
	expectedPodName := podName + ".etcd"
	expectedContainerName := "baz"
	expectedCommand := "ls -a"
	fw.fakeKubelet.runFunc = func(podFullName, uuid, containerName string, cmd []string) ([]byte, error) {
		if podFullName != expectedPodName {
			t.Errorf("expected %s, got %s", expectedPodName, podFullName)
		}
		if containerName != expectedContainerName {
			t.Errorf("expected %s, got %s", expectedContainerName, containerName)
		}
		if strings.Join(cmd, " ") != expectedCommand {
			t.Errorf("expected: %s, got %v", expectedCommand, cmd)
		}

		return []byte(output), nil
	}

	resp, err := http.Get(fw.testHTTPServer.URL + "/run/" + podName + "/" + expectedContainerName + "?cmd=ls%20-a")

	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		// copying the response body did not work
		t.Errorf("Cannot copy resp: %#v", err)
	}
	result := string(body)
	if result != output {
		t.Errorf("expected %s, got %s", output, result)
	}
}

func TestServeRunInContainerWithUUID(t *testing.T) {
	fw := newServerTest()
	output := "foo bar"
	podName := "foo"
	expectedPodName := podName + ".etcd"
	expectedUuid := "7e00838d_-_3523_-_11e4_-_8421_-_42010af0a720"
	expectedContainerName := "baz"
	expectedCommand := "ls -a"
	fw.fakeKubelet.runFunc = func(podFullName, uuid, containerName string, cmd []string) ([]byte, error) {
		if podFullName != expectedPodName {
			t.Errorf("expected %s, got %s", expectedPodName, podFullName)
		}
		if uuid != expectedUuid {
			t.Errorf("expected %s, got %s", expectedUuid, uuid)
		}
		if containerName != expectedContainerName {
			t.Errorf("expected %s, got %s", expectedContainerName, containerName)
		}
		if strings.Join(cmd, " ") != expectedCommand {
			t.Errorf("expected: %s, got %v", expectedCommand, cmd)
		}

		return []byte(output), nil
	}

	resp, err := http.Get(fw.testHTTPServer.URL + "/run/" + podName + "/" + expectedUuid + "/" + expectedContainerName + "?cmd=ls%20-a")

	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		// copying the response body did not work
		t.Errorf("Cannot copy resp: %#v", err)
	}
	result := string(body)
	if result != output {
		t.Errorf("expected %s, got %s", output, result)
	}
}

func TestContainerLogs(t *testing.T) {
	fw := newServerTest()
	output := "foo bar"
	podName := "foo"
	expectedPodName := podName + ".etcd"
	expectedContainerName := "baz"
	expectedTail := ""
	expectedFollow := false
	// expected := api.Container{"goodpod": docker.Container{ID: "myContainerID"}}
	fw.fakeKubelet.containerLogsFunc = func(podFullName, containerName, tail string, follow bool, stdout, stderr io.Writer) error {
		if podFullName != expectedPodName {
			t.Errorf("expected %s, got %s", expectedPodName, podFullName)
		}
		if containerName != expectedContainerName {
			t.Errorf("expected %s, got %s", expectedContainerName, containerName)
		}
		if tail != expectedTail {
			t.Errorf("expected %s, got %s", expectedTail, tail)
		}
		if follow != expectedFollow {
			t.Errorf("expected %t, got %t", expectedFollow, follow)
		}
		return nil
	}
	resp, err := http.Get(fw.testHTTPServer.URL + "/containerLogs/" + podName + "/" + expectedContainerName)
	if err != nil {
		t.Errorf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Errorf("Error reading container logs: %v", err)
	}
	result := string(body)
	if result != string(body) {
		t.Errorf("Expected: '%v', got: '%v'", output, result)
	}
}

func TestContainerLogsWithTail(t *testing.T) {
	fw := newServerTest()
	output := "foo bar"
	podName := "foo"
	expectedPodName := podName + ".etcd"
	expectedContainerName := "baz"
	expectedTail := "5"
	expectedFollow := false
	fw.fakeKubelet.containerLogsFunc = func(podFullName, containerName, tail string, follow bool, stdout, stderr io.Writer) error {
		if podFullName != expectedPodName {
			t.Errorf("expected %s, got %s", expectedPodName, podFullName)
		}
		if containerName != expectedContainerName {
			t.Errorf("expected %s, got %s", expectedContainerName, containerName)
		}
		if tail != expectedTail {
			t.Errorf("expected %s, got %s", expectedTail, tail)
		}
		if follow != expectedFollow {
			t.Errorf("expected %t, got %t", expectedFollow, follow)
		}
		return nil
	}
	resp, err := http.Get(fw.testHTTPServer.URL + "/containerLogs/" + podName + "/" + expectedContainerName + "?tail=5")
	if err != nil {
		t.Errorf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Errorf("Error reading container logs: %v", err)
	}
	result := string(body)
	if result != string(body) {
		t.Errorf("Expected: '%v', got: '%v'", output, result)
	}
}

func TestContainerLogsWithFollow(t *testing.T) {
	fw := newServerTest()
	output := "foo bar"
	podName := "foo"
	expectedPodName := podName + ".etcd"
	expectedContainerName := "baz"
	expectedTail := ""
	expectedFollow := true
	fw.fakeKubelet.containerLogsFunc = func(podFullName, containerName, tail string, follow bool, stdout, stderr io.Writer) error {
		if podFullName != expectedPodName {
			t.Errorf("expected %s, got %s", expectedPodName, podFullName)
		}
		if containerName != expectedContainerName {
			t.Errorf("expected %s, got %s", expectedContainerName, containerName)
		}
		if tail != expectedTail {
			t.Errorf("expected %s, got %s", expectedTail, tail)
		}
		if follow != expectedFollow {
			t.Errorf("expected %t, got %t", expectedFollow, follow)
		}
		return nil
	}
	resp, err := http.Get(fw.testHTTPServer.URL + "/containerLogs/" + podName + "/" + expectedContainerName + "?follow=1")
	if err != nil {
		t.Errorf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Errorf("Error reading container logs: %v", err)
	}
	result := string(body)
	if result != string(body) {
		t.Errorf("Expected: '%v', got: '%v'", output, result)
	}
}
