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

package kubelet

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/httpstream"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/httpstream/spdy"
	cadvisorApi "github.com/google/cadvisor/info/v1"
)

type fakeKubelet struct {
	podByNameFunc                      func(namespace, name string) (*api.Pod, bool)
	containerInfoFunc                  func(podFullName string, uid types.UID, containerName string, req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error)
	rawInfoFunc                        func(query *cadvisorApi.ContainerInfoRequest) (map[string]*cadvisorApi.ContainerInfo, error)
	machineInfoFunc                    func() (*cadvisorApi.MachineInfo, error)
	podsFunc                           func() []*api.Pod
	logFunc                            func(w http.ResponseWriter, req *http.Request)
	runFunc                            func(podFullName string, uid types.UID, containerName string, cmd []string) ([]byte, error)
	containerVersionFunc               func() (kubecontainer.Version, error)
	execFunc                           func(pod string, uid types.UID, container string, cmd []string, in io.Reader, out, err io.WriteCloser, tty bool) error
	portForwardFunc                    func(name string, uid types.UID, port uint16, stream io.ReadWriteCloser) error
	containerLogsFunc                  func(podFullName, containerName, tail string, follow, pervious bool, stdout, stderr io.Writer) error
	streamingConnectionIdleTimeoutFunc func() time.Duration
	hostnameFunc                       func() string
	resyncInterval                     time.Duration
	loopEntryTime                      time.Time
}

func (fk *fakeKubelet) ResyncInterval() time.Duration {
	return fk.resyncInterval
}

func (fk *fakeKubelet) LatestLoopEntryTime() time.Time {
	return fk.loopEntryTime
}

func (fk *fakeKubelet) GetPodByName(namespace, name string) (*api.Pod, bool) {
	return fk.podByNameFunc(namespace, name)
}

func (fk *fakeKubelet) GetContainerInfo(podFullName string, uid types.UID, containerName string, req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error) {
	return fk.containerInfoFunc(podFullName, uid, containerName, req)
}

func (fk *fakeKubelet) GetRawContainerInfo(containerName string, req *cadvisorApi.ContainerInfoRequest, subcontainers bool) (map[string]*cadvisorApi.ContainerInfo, error) {
	return fk.rawInfoFunc(req)
}

func (fk *fakeKubelet) GetContainerRuntimeVersion() (kubecontainer.Version, error) {
	return fk.containerVersionFunc()
}

func (fk *fakeKubelet) GetCachedMachineInfo() (*cadvisorApi.MachineInfo, error) {
	return fk.machineInfoFunc()
}

func (fk *fakeKubelet) GetPods() []*api.Pod {
	return fk.podsFunc()
}

func (fk *fakeKubelet) ServeLogs(w http.ResponseWriter, req *http.Request) {
	fk.logFunc(w, req)
}

func (fk *fakeKubelet) GetKubeletContainerLogs(podFullName, containerName, tail string, follow, previous bool, stdout, stderr io.Writer) error {
	return fk.containerLogsFunc(podFullName, containerName, tail, follow, previous, stdout, stderr)
}

func (fk *fakeKubelet) GetHostname() string {
	return fk.hostnameFunc()
}

func (fk *fakeKubelet) RunInContainer(podFullName string, uid types.UID, containerName string, cmd []string) ([]byte, error) {
	return fk.runFunc(podFullName, uid, containerName, cmd)
}

func (fk *fakeKubelet) ExecInContainer(name string, uid types.UID, container string, cmd []string, in io.Reader, out, err io.WriteCloser, tty bool) error {
	return fk.execFunc(name, uid, container, cmd, in, out, err, tty)
}

func (fk *fakeKubelet) PortForward(name string, uid types.UID, port uint16, stream io.ReadWriteCloser) error {
	return fk.portForwardFunc(name, uid, port, stream)
}

func (fk *fakeKubelet) StreamingConnectionIdleTimeout() time.Duration {
	return fk.streamingConnectionIdleTimeoutFunc()
}

type serverTestFramework struct {
	serverUnderTest *Server
	fakeKubelet     *fakeKubelet
	testHTTPServer  *httptest.Server
}

func newServerTest() *serverTestFramework {
	fw := &serverTestFramework{}
	fw.fakeKubelet = &fakeKubelet{
		podByNameFunc: func(namespace, name string) (*api.Pod, bool) {
			return &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Namespace: namespace,
					Name:      name,
				},
			}, true
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

// A helper function to return the correct pod name.
func getPodName(name, namespace string) string {
	if namespace == "" {
		namespace = NamespaceDefault
	}
	return name + "_" + namespace
}

func TestContainerInfo(t *testing.T) {
	fw := newServerTest()
	expectedInfo := &cadvisorApi.ContainerInfo{}
	podID := "somepod"
	expectedPodID := getPodName(podID, "")
	expectedContainerName := "goodcontainer"
	fw.fakeKubelet.containerInfoFunc = func(podID string, uid types.UID, containerName string, req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error) {
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
	var receivedInfo cadvisorApi.ContainerInfo
	err = json.NewDecoder(resp.Body).Decode(&receivedInfo)
	if err != nil {
		t.Fatalf("received invalid json data: %v", err)
	}
	if !receivedInfo.Eq(expectedInfo) {
		t.Errorf("received wrong data: %#v", receivedInfo)
	}
}

func TestContainerInfoWithUidNamespace(t *testing.T) {
	fw := newServerTest()
	expectedInfo := &cadvisorApi.ContainerInfo{}
	podID := "somepod"
	expectedNamespace := "custom"
	expectedPodID := getPodName(podID, expectedNamespace)
	expectedContainerName := "goodcontainer"
	expectedUid := "9b01b80f-8fb4-11e4-95ab-4200af06647"
	fw.fakeKubelet.containerInfoFunc = func(podID string, uid types.UID, containerName string, req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error) {
		if podID != expectedPodID || string(uid) != expectedUid || containerName != expectedContainerName {
			return nil, fmt.Errorf("bad podID or uid or containerName: podID=%v; uid=%v; containerName=%v", podID, uid, containerName)
		}
		return expectedInfo, nil
	}

	resp, err := http.Get(fw.testHTTPServer.URL + fmt.Sprintf("/stats/%v/%v/%v/%v", expectedNamespace, podID, expectedUid, expectedContainerName))
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()
	var receivedInfo cadvisorApi.ContainerInfo
	err = json.NewDecoder(resp.Body).Decode(&receivedInfo)
	if err != nil {
		t.Fatalf("received invalid json data: %v", err)
	}
	if !receivedInfo.Eq(expectedInfo) {
		t.Errorf("received wrong data: %#v", receivedInfo)
	}
}

func TestContainerNotFound(t *testing.T) {
	fw := newServerTest()
	podID := "somepod"
	expectedNamespace := "custom"
	expectedContainerName := "slowstartcontainer"
	expectedUid := "9b01b80f-8fb4-11e4-95ab-4200af06647"
	fw.fakeKubelet.containerInfoFunc = func(podID string, uid types.UID, containerName string, req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error) {
		return nil, ErrContainerNotFound
	}
	resp, err := http.Get(fw.testHTTPServer.URL + fmt.Sprintf("/stats/%v/%v/%v/%v", expectedNamespace, podID, expectedUid, expectedContainerName))
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("Received status %d expecting %d", resp.StatusCode, http.StatusNotFound)
	}
	defer resp.Body.Close()
}

func TestRootInfo(t *testing.T) {
	fw := newServerTest()
	expectedInfo := &cadvisorApi.ContainerInfo{
		ContainerReference: cadvisorApi.ContainerReference{
			Name: "/",
		},
	}
	fw.fakeKubelet.rawInfoFunc = func(req *cadvisorApi.ContainerInfoRequest) (map[string]*cadvisorApi.ContainerInfo, error) {
		return map[string]*cadvisorApi.ContainerInfo{
			expectedInfo.Name: expectedInfo,
		}, nil
	}

	resp, err := http.Get(fw.testHTTPServer.URL + "/stats")
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()
	var receivedInfo cadvisorApi.ContainerInfo
	err = json.NewDecoder(resp.Body).Decode(&receivedInfo)
	if err != nil {
		t.Fatalf("received invalid json data: %v", err)
	}
	if !receivedInfo.Eq(expectedInfo) {
		t.Errorf("received wrong data: %#v, expected %#v", receivedInfo, expectedInfo)
	}
}

func TestSubcontainerContainerInfo(t *testing.T) {
	fw := newServerTest()
	const kubeletContainer = "/kubelet"
	const kubeletSubContainer = "/kubelet/sub"
	expectedInfo := map[string]*cadvisorApi.ContainerInfo{
		kubeletContainer: {
			ContainerReference: cadvisorApi.ContainerReference{
				Name: kubeletContainer,
			},
		},
		kubeletSubContainer: {
			ContainerReference: cadvisorApi.ContainerReference{
				Name: kubeletSubContainer,
			},
		},
	}
	fw.fakeKubelet.rawInfoFunc = func(req *cadvisorApi.ContainerInfoRequest) (map[string]*cadvisorApi.ContainerInfo, error) {
		return expectedInfo, nil
	}

	request := fmt.Sprintf("{\"containerName\":%q, \"subcontainers\": true}", kubeletContainer)
	resp, err := http.Post(fw.testHTTPServer.URL+"/stats/container", "application/json", bytes.NewBuffer([]byte(request)))
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()
	var receivedInfo map[string]*cadvisorApi.ContainerInfo
	err = json.NewDecoder(resp.Body).Decode(&receivedInfo)
	if err != nil {
		t.Fatalf("Received invalid json data: %v", err)
	}
	if len(receivedInfo) != len(expectedInfo) {
		t.Errorf("Received wrong data: %#v, expected %#v", receivedInfo, expectedInfo)
	}

	for _, containerName := range []string{kubeletContainer, kubeletSubContainer} {
		if _, ok := receivedInfo[containerName]; !ok {
			t.Errorf("Expected container %q to be present in result: %#v", containerName, receivedInfo)
		}
		if !receivedInfo[containerName].Eq(expectedInfo[containerName]) {
			t.Errorf("Invalid result for %q: Expected %#v, received %#v", containerName, expectedInfo[containerName], receivedInfo[containerName])
		}
	}
}

func TestMachineInfo(t *testing.T) {
	fw := newServerTest()
	expectedInfo := &cadvisorApi.MachineInfo{
		NumCores:       4,
		MemoryCapacity: 1024,
	}
	fw.fakeKubelet.machineInfoFunc = func() (*cadvisorApi.MachineInfo, error) {
		return expectedInfo, nil
	}

	resp, err := http.Get(fw.testHTTPServer.URL + "/spec")
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()
	var receivedInfo cadvisorApi.MachineInfo
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
	expectedPodName := getPodName(podName, podNamespace)
	expectedContainerName := "baz"
	expectedCommand := "ls -a"
	fw.fakeKubelet.runFunc = func(podFullName string, uid types.UID, containerName string, cmd []string) ([]byte, error) {
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

func TestServeRunInContainerWithUID(t *testing.T) {
	fw := newServerTest()
	output := "foo bar"
	podNamespace := "other"
	podName := "foo"
	expectedPodName := getPodName(podName, podNamespace)
	expectedUID := "7e00838d_-_3523_-_11e4_-_8421_-_42010af0a720"
	expectedContainerName := "baz"
	expectedCommand := "ls -a"
	fw.fakeKubelet.runFunc = func(podFullName string, uid types.UID, containerName string, cmd []string) ([]byte, error) {
		if podFullName != expectedPodName {
			t.Errorf("expected %s, got %s", expectedPodName, podFullName)
		}
		if string(uid) != expectedUID {
			t.Errorf("expected %s, got %s", expectedUID, uid)
		}
		if containerName != expectedContainerName {
			t.Errorf("expected %s, got %s", expectedContainerName, containerName)
		}
		if strings.Join(cmd, " ") != expectedCommand {
			t.Errorf("expected: %s, got %v", expectedCommand, cmd)
		}

		return []byte(output), nil
	}

	resp, err := http.Get(fw.testHTTPServer.URL + "/run/" + podNamespace + "/" + podName + "/" + expectedUID + "/" + expectedContainerName + "?cmd=ls%20-a")

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

func TestHealthCheck(t *testing.T) {
	fw := newServerTest()
	fw.fakeKubelet.containerVersionFunc = func() (kubecontainer.Version, error) {
		return dockertools.NewVersion("1.15")
	}
	fw.fakeKubelet.hostnameFunc = func() string {
		return "127.0.0.1"
	}

	// Test with correct hostname, Docker version
	assertHealthIsOk(t, fw.testHTTPServer.URL+"/healthz")

	//Test with incorrect hostname
	fw.fakeKubelet.hostnameFunc = func() string {
		return "fake"
	}
	assertHealthIsOk(t, fw.testHTTPServer.URL+"/healthz")

	//Test with old container runtime version
	fw.fakeKubelet.containerVersionFunc = func() (kubecontainer.Version, error) {
		return dockertools.NewVersion("1.1")
	}

	assertHealthFails(t, fw.testHTTPServer.URL+"/healthz", http.StatusInternalServerError)
}

func assertHealthFails(t *testing.T, httpURL string, expectedErrorCode int) {
	resp, err := http.Get(httpURL)
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != expectedErrorCode {
		t.Errorf("expected status code %d, got %d", expectedErrorCode, resp.StatusCode)
	}
}

func TestSyncLoopCheck(t *testing.T) {
	fw := newServerTest()
	fw.fakeKubelet.containerVersionFunc = func() (kubecontainer.Version, error) {
		return dockertools.NewVersion("1.15")
	}
	fw.fakeKubelet.hostnameFunc = func() string {
		return "127.0.0.1"
	}

	fw.fakeKubelet.resyncInterval = time.Minute
	fw.fakeKubelet.loopEntryTime = time.Now()

	// Test with correct hostname, Docker version
	assertHealthIsOk(t, fw.testHTTPServer.URL+"/healthz")

	fw.fakeKubelet.loopEntryTime = time.Now().Add(time.Minute * -10)
	assertHealthFails(t, fw.testHTTPServer.URL+"/healthz", http.StatusInternalServerError)
}

// returns http response status code from the HTTP GET
func assertHealthIsOk(t *testing.T, httpURL string) {
	resp, err := http.Get(httpURL)
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected status code %d, got %d", http.StatusOK, resp.StatusCode)
	}
	body, readErr := ioutil.ReadAll(resp.Body)
	if readErr != nil {
		// copying the response body did not work
		t.Fatalf("Cannot copy resp: %#v", readErr)
	}
	result := string(body)
	if !strings.Contains(result, "ok") {
		t.Errorf("expected body contains ok, got %s", result)
	}
}

func setPodByNameFunc(fw *serverTestFramework, namespace, pod, container string) {
	fw.fakeKubelet.podByNameFunc = func(namespace, name string) (*api.Pod, bool) {
		return &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Namespace: namespace,
				Name:      pod,
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name: container,
					},
				},
			},
		}, true
	}
}

func setGetContainerLogsFunc(fw *serverTestFramework, t *testing.T, expectedPodName, expectedContainerName, expectedTail string, expectedFollow, expectedPrevious bool, output string) {
	fw.fakeKubelet.containerLogsFunc = func(podFullName, containerName, tail string, follow, previous bool, stdout, stderr io.Writer) error {
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
		if previous != expectedPrevious {
			t.Errorf("expected %t, got %t", expectedPrevious, previous)
		}

		io.WriteString(stdout, output)
		return nil
	}
}

func TestContainerLogs(t *testing.T) {
	fw := newServerTest()
	output := "foo bar"
	podNamespace := "other"
	podName := "foo"
	expectedPodName := getPodName(podName, podNamespace)
	expectedContainerName := "baz"
	expectedTail := ""
	expectedFollow := false
	expectedPrevious := false
	setPodByNameFunc(fw, podNamespace, podName, expectedContainerName)
	setGetContainerLogsFunc(fw, t, expectedPodName, expectedContainerName, expectedTail, expectedFollow, expectedPrevious, output)
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
	if result != output {
		t.Errorf("Expected: '%v', got: '%v'", output, result)
	}
}

func TestContainerLogsWithTail(t *testing.T) {
	fw := newServerTest()
	output := "foo bar"
	podNamespace := "other"
	podName := "foo"
	expectedPodName := getPodName(podName, podNamespace)
	expectedContainerName := "baz"
	expectedTail := "5"
	expectedFollow := false
	expectedPrevious := false
	setPodByNameFunc(fw, podNamespace, podName, expectedContainerName)
	setGetContainerLogsFunc(fw, t, expectedPodName, expectedContainerName, expectedTail, expectedFollow, expectedPrevious, output)
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
	if result != output {
		t.Errorf("Expected: '%v', got: '%v'", output, result)
	}
}

func TestContainerLogsWithFollow(t *testing.T) {
	fw := newServerTest()
	output := "foo bar"
	podNamespace := "other"
	podName := "foo"
	expectedPodName := getPodName(podName, podNamespace)
	expectedContainerName := "baz"
	expectedTail := ""
	expectedFollow := true
	expectedPrevious := false
	setPodByNameFunc(fw, podNamespace, podName, expectedContainerName)
	setGetContainerLogsFunc(fw, t, expectedPodName, expectedContainerName, expectedTail, expectedFollow, expectedPrevious, output)
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
	if result != output {
		t.Errorf("Expected: '%v', got: '%v'", output, result)
	}
}

func TestServeExecInContainerIdleTimeout(t *testing.T) {
	fw := newServerTest()

	fw.fakeKubelet.streamingConnectionIdleTimeoutFunc = func() time.Duration {
		return 100 * time.Millisecond
	}

	podNamespace := "other"
	podName := "foo"
	expectedContainerName := "baz"

	url := fw.testHTTPServer.URL + "/exec/" + podNamespace + "/" + podName + "/" + expectedContainerName + "?c=ls&c=-a&" + api.ExecStdinParam + "=1"

	upgradeRoundTripper := spdy.NewSpdyRoundTripper(nil)
	c := &http.Client{Transport: upgradeRoundTripper}

	resp, err := c.Get(url)
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()

	upgradeRoundTripper.Dialer = &net.Dialer{
		Deadline: time.Now().Add(60 * time.Second),
		Timeout:  60 * time.Second,
	}
	conn, err := upgradeRoundTripper.NewConnection(resp)
	if err != nil {
		t.Fatalf("Unexpected error creating streaming connection: %s", err)
	}
	if conn == nil {
		t.Fatal("Unexpected nil connection")
	}

	<-conn.CloseChan()
}

func TestServeExecInContainer(t *testing.T) {
	tests := []struct {
		stdin              bool
		stdout             bool
		stderr             bool
		tty                bool
		responseStatusCode int
		uid                bool
	}{
		{responseStatusCode: http.StatusBadRequest},
		{stdin: true, responseStatusCode: http.StatusSwitchingProtocols},
		{stdout: true, responseStatusCode: http.StatusSwitchingProtocols},
		{stderr: true, responseStatusCode: http.StatusSwitchingProtocols},
		{stdout: true, stderr: true, responseStatusCode: http.StatusSwitchingProtocols},
		{stdout: true, stderr: true, tty: true, responseStatusCode: http.StatusSwitchingProtocols},
		{stdin: true, stdout: true, stderr: true, responseStatusCode: http.StatusSwitchingProtocols},
	}

	for i, test := range tests {
		fw := newServerTest()

		fw.fakeKubelet.streamingConnectionIdleTimeoutFunc = func() time.Duration {
			return 0
		}

		podNamespace := "other"
		podName := "foo"
		expectedPodName := getPodName(podName, podNamespace)
		expectedUid := "9b01b80f-8fb4-11e4-95ab-4200af06647"
		expectedContainerName := "baz"
		expectedCommand := "ls -a"
		expectedStdin := "stdin"
		expectedStdout := "stdout"
		expectedStderr := "stderr"
		execFuncDone := make(chan struct{})
		clientStdoutReadDone := make(chan struct{})
		clientStderrReadDone := make(chan struct{})

		fw.fakeKubelet.execFunc = func(podFullName string, uid types.UID, containerName string, cmd []string, in io.Reader, out, stderr io.WriteCloser, tty bool) error {
			defer close(execFuncDone)
			if podFullName != expectedPodName {
				t.Fatalf("%d: podFullName: expected %s, got %s", i, expectedPodName, podFullName)
			}
			if test.uid && string(uid) != expectedUid {
				t.Fatalf("%d: uid: expected %v, got %v", i, expectedUid, uid)
			}
			if containerName != expectedContainerName {
				t.Fatalf("%d: containerName: expected %s, got %s", i, expectedContainerName, containerName)
			}
			if strings.Join(cmd, " ") != expectedCommand {
				t.Fatalf("%d: cmd: expected: %s, got %v", i, expectedCommand, cmd)
			}

			if test.stdin {
				if in == nil {
					t.Fatalf("%d: stdin: expected non-nil", i)
				}
				b := make([]byte, 10)
				n, err := in.Read(b)
				if err != nil {
					t.Fatalf("%d: error reading from stdin: %v", i, err)
				}
				if e, a := expectedStdin, string(b[0:n]); e != a {
					t.Fatalf("%d: stdin: expected to read %v, got %v", i, e, a)
				}
			} else if in != nil {
				t.Fatalf("%d: stdin: expected nil: %#v", i, in)
			}

			if test.stdout {
				if out == nil {
					t.Fatalf("%d: stdout: expected non-nil", i)
				}
				_, err := out.Write([]byte(expectedStdout))
				if err != nil {
					t.Fatalf("%d:, error writing to stdout: %v", i, err)
				}
				out.Close()
				<-clientStdoutReadDone
			} else if out != nil {
				t.Fatalf("%d: stdout: expected nil: %#v", i, out)
			}

			if tty {
				if stderr != nil {
					t.Fatalf("%d: tty set but received non-nil stderr: %v", i, stderr)
				}
			} else if test.stderr {
				if stderr == nil {
					t.Fatalf("%d: stderr: expected non-nil", i)
				}
				_, err := stderr.Write([]byte(expectedStderr))
				if err != nil {
					t.Fatalf("%d:, error writing to stderr: %v", i, err)
				}
				stderr.Close()
				<-clientStderrReadDone
			} else if stderr != nil {
				t.Fatalf("%d: stderr: expected nil: %#v", i, stderr)
			}

			return nil
		}

		var url string
		if test.uid {
			url = fw.testHTTPServer.URL + "/exec/" + podNamespace + "/" + podName + "/" + expectedUid + "/" + expectedContainerName + "?command=ls&command=-a"
		} else {
			url = fw.testHTTPServer.URL + "/exec/" + podNamespace + "/" + podName + "/" + expectedContainerName + "?command=ls&command=-a"
		}
		if test.stdin {
			url += "&" + api.ExecStdinParam + "=1"
		}
		if test.stdout {
			url += "&" + api.ExecStdoutParam + "=1"
		}
		if test.stderr && !test.tty {
			url += "&" + api.ExecStderrParam + "=1"
		}
		if test.tty {
			url += "&" + api.ExecTTYParam + "=1"
		}

		var (
			resp                *http.Response
			err                 error
			upgradeRoundTripper httpstream.UpgradeRoundTripper
			c                   *http.Client
		)

		if test.responseStatusCode != http.StatusSwitchingProtocols {
			c = &http.Client{}
		} else {
			upgradeRoundTripper = spdy.NewRoundTripper(nil)
			c = &http.Client{Transport: upgradeRoundTripper}
		}

		resp, err = c.Get(url)
		if err != nil {
			t.Fatalf("%d: Got error GETing: %v", i, err)
		}
		defer resp.Body.Close()

		_, err = ioutil.ReadAll(resp.Body)
		if err != nil {
			t.Errorf("%d: Error reading response body: %v", i, err)
		}

		if e, a := test.responseStatusCode, resp.StatusCode; e != a {
			t.Fatalf("%d: response status: expected %v, got %v", i, e, a)
		}

		if test.responseStatusCode != http.StatusSwitchingProtocols {
			continue
		}

		conn, err := upgradeRoundTripper.NewConnection(resp)
		if err != nil {
			t.Fatalf("Unexpected error creating streaming connection: %s", err)
		}
		if conn == nil {
			t.Fatalf("%d: unexpected nil conn", i)
		}
		defer conn.Close()

		h := http.Header{}
		h.Set(api.StreamType, api.StreamTypeError)
		errorStream, err := conn.CreateStream(h)
		if err != nil {
			t.Fatalf("%d: error creating error stream: %v", i, err)
		}
		defer errorStream.Reset()

		if test.stdin {
			h.Set(api.StreamType, api.StreamTypeStdin)
			stream, err := conn.CreateStream(h)
			if err != nil {
				t.Fatalf("%d: error creating stdin stream: %v", i, err)
			}
			defer stream.Reset()
			_, err = stream.Write([]byte(expectedStdin))
			if err != nil {
				t.Fatalf("%d: error writing to stdin stream: %v", i, err)
			}
		}

		var stdoutStream httpstream.Stream
		if test.stdout {
			h.Set(api.StreamType, api.StreamTypeStdout)
			stdoutStream, err = conn.CreateStream(h)
			if err != nil {
				t.Fatalf("%d: error creating stdout stream: %v", i, err)
			}
			defer stdoutStream.Reset()
		}

		var stderrStream httpstream.Stream
		if test.stderr && !test.tty {
			h.Set(api.StreamType, api.StreamTypeStderr)
			stderrStream, err = conn.CreateStream(h)
			if err != nil {
				t.Fatalf("%d: error creating stderr stream: %v", i, err)
			}
			defer stderrStream.Reset()
		}

		if test.stdout {
			output := make([]byte, 10)
			n, err := stdoutStream.Read(output)
			close(clientStdoutReadDone)
			if err != nil {
				t.Fatalf("%d: error reading from stdout stream: %v", i, err)
			}
			if e, a := expectedStdout, string(output[0:n]); e != a {
				t.Fatalf("%d: stdout: expected '%v', got '%v'", i, e, a)
			}
		}

		if test.stderr && !test.tty {
			output := make([]byte, 10)
			n, err := stderrStream.Read(output)
			close(clientStderrReadDone)
			if err != nil {
				t.Fatalf("%d: error reading from stderr stream: %v", i, err)
			}
			if e, a := expectedStderr, string(output[0:n]); e != a {
				t.Fatalf("%d: stderr: expected '%v', got '%v'", i, e, a)
			}
		}

		<-execFuncDone
	}
}

func TestServePortForwardIdleTimeout(t *testing.T) {
	fw := newServerTest()

	fw.fakeKubelet.streamingConnectionIdleTimeoutFunc = func() time.Duration {
		return 100 * time.Millisecond
	}

	podNamespace := "other"
	podName := "foo"

	url := fw.testHTTPServer.URL + "/portForward/" + podNamespace + "/" + podName

	upgradeRoundTripper := spdy.NewRoundTripper(nil)
	c := &http.Client{Transport: upgradeRoundTripper}

	resp, err := c.Get(url)
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()

	conn, err := upgradeRoundTripper.NewConnection(resp)
	if err != nil {
		t.Fatalf("Unexpected error creating streaming connection: %s", err)
	}
	if conn == nil {
		t.Fatal("Unexpected nil connection")
	}
	defer conn.Close()

	<-conn.CloseChan()
}

func TestServePortForward(t *testing.T) {
	tests := []struct {
		port          string
		uid           bool
		clientData    string
		containerData string
		shouldError   bool
	}{
		{port: "", shouldError: true},
		{port: "abc", shouldError: true},
		{port: "-1", shouldError: true},
		{port: "65536", shouldError: true},
		{port: "0", shouldError: true},
		{port: "1", shouldError: false},
		{port: "8000", shouldError: false},
		{port: "8000", clientData: "client data", containerData: "container data", shouldError: false},
		{port: "65535", shouldError: false},
		{port: "65535", uid: true, shouldError: false},
	}

	podNamespace := "other"
	podName := "foo"
	expectedPodName := getPodName(podName, podNamespace)
	expectedUid := "9b01b80f-8fb4-11e4-95ab-4200af06647"

	for i, test := range tests {
		fw := newServerTest()

		fw.fakeKubelet.streamingConnectionIdleTimeoutFunc = func() time.Duration {
			return 0
		}

		portForwardFuncDone := make(chan struct{})

		fw.fakeKubelet.portForwardFunc = func(name string, uid types.UID, port uint16, stream io.ReadWriteCloser) error {
			defer close(portForwardFuncDone)

			if e, a := expectedPodName, name; e != a {
				t.Fatalf("%d: pod name: expected '%v', got '%v'", i, e, a)
			}

			if e, a := expectedUid, uid; test.uid && e != string(a) {
				t.Fatalf("%d: uid: expected '%v', got '%v'", i, e, a)
			}

			p, err := strconv.ParseUint(test.port, 10, 16)
			if err != nil {
				t.Fatalf("%d: error parsing port string '%s': %v", i, test.port, err)
			}
			if e, a := uint16(p), port; e != a {
				t.Fatalf("%d: port: expected '%v', got '%v'", i, e, a)
			}

			if test.clientData != "" {
				fromClient := make([]byte, 32)
				n, err := stream.Read(fromClient)
				if err != nil {
					t.Fatalf("%d: error reading client data: %v", i, err)
				}
				if e, a := test.clientData, string(fromClient[0:n]); e != a {
					t.Fatalf("%d: client data: expected to receive '%v', got '%v'", i, e, a)
				}
			}

			if test.containerData != "" {
				_, err := stream.Write([]byte(test.containerData))
				if err != nil {
					t.Fatalf("%d: error writing container data: %v", i, err)
				}
			}

			return nil
		}

		var url string
		if test.uid {
			url = fmt.Sprintf("%s/portForward/%s/%s/%s", fw.testHTTPServer.URL, podNamespace, podName, expectedUid)
		} else {
			url = fmt.Sprintf("%s/portForward/%s/%s", fw.testHTTPServer.URL, podNamespace, podName)
		}

		upgradeRoundTripper := spdy.NewRoundTripper(nil)
		c := &http.Client{Transport: upgradeRoundTripper}

		resp, err := c.Get(url)
		if err != nil {
			t.Fatalf("%d: Got error GETing: %v", i, err)
		}
		defer resp.Body.Close()

		conn, err := upgradeRoundTripper.NewConnection(resp)
		if err != nil {
			t.Fatalf("Unexpected error creating streaming connection: %s", err)
		}
		if conn == nil {
			t.Fatalf("%d: Unexpected nil connection", i)
		}
		defer conn.Close()

		headers := http.Header{}
		headers.Set("streamType", "error")
		headers.Set("port", test.port)
		errorStream, err := conn.CreateStream(headers)
		_ = errorStream
		haveErr := err != nil
		if e, a := test.shouldError, haveErr; e != a {
			t.Fatalf("%d: create stream: expected err=%t, got %t: %v", i, e, a, err)
		}

		if test.shouldError {
			continue
		}

		headers.Set("streamType", "data")
		headers.Set("port", test.port)
		dataStream, err := conn.CreateStream(headers)
		haveErr = err != nil
		if e, a := test.shouldError, haveErr; e != a {
			t.Fatalf("%d: create stream: expected err=%t, got %t: %v", i, e, a, err)
		}

		if test.clientData != "" {
			_, err := dataStream.Write([]byte(test.clientData))
			if err != nil {
				t.Fatalf("%d: unexpected error writing client data: %v", i, err)
			}
		}

		if test.containerData != "" {
			fromContainer := make([]byte, 32)
			n, err := dataStream.Read(fromContainer)
			if err != nil {
				t.Fatalf("%d: unexpected error reading container data: %v", i, err)
			}
			if e, a := test.containerData, string(fromContainer[0:n]); e != a {
				t.Fatalf("%d: expected to receive '%v' from container, got '%v'", i, e, a)
			}
		}

		<-portForwardFuncDone
	}
}
