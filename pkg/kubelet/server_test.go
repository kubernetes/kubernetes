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
	"github.com/google/cadvisor/info"
)

type fakeKubelet struct {
	fullNameFunc      func(namespace, name string) (string, bool)
	infoFunc          func(name string) (api.PodInfo, error)
	containerInfoFunc func(podFullName, uid, containerName string, req *info.ContainerInfoRequest) (*info.ContainerInfo, error)
	rootInfoFunc      func(query *info.ContainerInfoRequest) (*info.ContainerInfo, error)
	machineInfoFunc   func() (*info.MachineInfo, error)
	boundPodsFunc     func() ([]api.BoundPod, error)
	logFunc           func(w http.ResponseWriter, req *http.Request)
	runFunc           func(podFullName, uuid, containerName string, cmd []string) ([]byte, error)
	containerLogsFunc func(podFullName, containerName, tail string, follow bool, stdout, stderr io.Writer) error
}

func (fk *fakeKubelet) GetPodFullName(namespace, name string) (string, bool) {
	return fk.fullNameFunc(namespace, name)
}

func (fk *fakeKubelet) GetPodInfo(name, uuid string) (api.PodInfo, error) {
	return fk.infoFunc(name)
}

func (fk *fakeKubelet) GetContainerInfo(podFullName, uuid, containerName string, req *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
	return fk.containerInfoFunc(podFullName, uuid, containerName, req)
}

func (fk *fakeKubelet) GetRootInfo(req *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
	return fk.rootInfoFunc(req)
}

func (fk *fakeKubelet) GetMachineInfo() (*info.MachineInfo, error) {
	return fk.machineInfoFunc()
}

func (fk *fakeKubelet) GetBoundPods() ([]api.BoundPod, error) {
	return fk.boundPodsFunc()
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
	fw.fakeKubelet = &fakeKubelet{
		fullNameFunc: func(namespace, name string) (string, bool) {
			return fmt.Sprintf("%s.%s.api", name, namespace), true
		},
	}
	server := NewServer(fw.fakeKubelet, true)
	fw.serverUnderTest = &server
	fw.testHTTPServer = httptest.NewServer(fw.serverUnderTest)
	return fw
}

// encodeJSON returns obj marshalled as a JSON string, panicing on any errors
func encodeJSON(obj interface{}) string {
	data, err := json.Marshal(obj)
	if err != nil {
		panic(err)
	}
	return string(data)
}

func readResp(resp *http.Response) (string, error) {
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	return string(body), err
}

func TestPodInfo(t *testing.T) {
	fw := newServerTest()
	expected := api.PodInfo{
		"goodpod": api.ContainerStatus{},
	}
	fw.fakeKubelet.infoFunc = func(name string) (api.PodInfo, error) {
		if name == "goodpod.default.api" {
			return expected, nil
		}
		return nil, fmt.Errorf("bad pod %s", name)
	}
	resp, err := http.Get(fw.testHTTPServer.URL + "/podInfo?podID=goodpod&podNamespace=default")
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
	podID := "somepod"
	expectedPodID := "somepod" + ".default.api"
	expectedContainerName := "goodcontainer"
	fw.fakeKubelet.containerInfoFunc = func(podID, uid, containerName string, req *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
		if podID != expectedPodID || containerName != expectedContainerName {
			return nil, fmt.Errorf("bad podID or containerName: podID=%v; containerName=%v", podID, containerName)
		}
		return expectedInfo, nil
	}

	resp, err := http.Get(fw.testHTTPServer.URL + fmt.Sprintf("/stats/%v/%v", podID, expectedContainerName))
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

func TestContainerInfoWithUidNamespace(t *testing.T) {
	fw := newServerTest()
	expectedInfo := &info.ContainerInfo{}
	podID := "somepod"
	expectedNamespace := "custom"
	expectedPodID := "somepod" + "." + expectedNamespace + ".api"
	expectedContainerName := "goodcontainer"
	expectedUid := "9b01b80f-8fb4-11e4-95ab-4200af06647"
	fw.fakeKubelet.containerInfoFunc = func(podID, uid, containerName string, req *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
		if podID != expectedPodID || uid != expectedUid || containerName != expectedContainerName {
			return nil, fmt.Errorf("bad podID or uid or containerName: podID=%v; uid=%v; containerName=%v", podID, uid, containerName)
		}
		return expectedInfo, nil
	}

	resp, err := http.Get(fw.testHTTPServer.URL + fmt.Sprintf("/stats/%v/%v/%v/%v", expectedNamespace, podID, expectedUid, expectedContainerName))
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
	podNamespace := "other"
	podName := "foo"
	expectedPodName := podName + "." + podNamespace + ".api"
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

	resp, err := http.Get(fw.testHTTPServer.URL + "/run/" + podNamespace + "/" + podName + "/" + expectedContainerName + "?cmd=ls%20-a")

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
	podNamespace := "other"
	podName := "foo"
	expectedPodName := podName + "." + podNamespace + ".api"
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

	resp, err := http.Get(fw.testHTTPServer.URL + "/run/" + podNamespace + "/" + podName + "/" + expectedUuid + "/" + expectedContainerName + "?cmd=ls%20-a")

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
	podNamespace := "other"
	podName := "foo"
	expectedPodName := podName + ".other.api"
	expectedContainerName := "baz"
	expectedTail := ""
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
	resp, err := http.Get(fw.testHTTPServer.URL + "/containerLogs/" + podNamespace + "/" + podName + "/" + expectedContainerName)
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
	podNamespace := "other"
	podName := "foo"
	expectedPodName := podName + ".other.api"
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
	resp, err := http.Get(fw.testHTTPServer.URL + "/containerLogs/" + podNamespace + "/" + podName + "/" + expectedContainerName + "?tail=5")
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
	podNamespace := "other"
	podName := "foo"
	expectedPodName := podName + ".other.api"
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
	resp, err := http.Get(fw.testHTTPServer.URL + "/containerLogs/" + podNamespace + "/" + podName + "/" + expectedContainerName + "?follow=1")
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
