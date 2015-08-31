// Copyright 2015 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/fsouza/go-dockerclient"
)

func TestNewServer(t *testing.T) {
	server, err := NewServer("127.0.0.1:0", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer server.listener.Close()
	conn, err := net.Dial("tcp", server.listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	conn.Close()
}

func TestServerStop(t *testing.T) {
	server, err := NewServer("127.0.0.1:0", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	server.Stop()
	_, err = net.Dial("tcp", server.listener.Addr().String())
	if err == nil {
		t.Error("Unexpected <nil> error when dialing to stopped server")
	}
}

func TestServerStopNoListener(t *testing.T) {
	server := DockerServer{}
	server.Stop()
}

func TestServerURL(t *testing.T) {
	server, err := NewServer("127.0.0.1:0", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer server.Stop()
	url := server.URL()
	if expected := "http://" + server.listener.Addr().String() + "/"; url != expected {
		t.Errorf("DockerServer.URL(): Want %q. Got %q.", expected, url)
	}
}

func TestServerURLNoListener(t *testing.T) {
	server := DockerServer{}
	url := server.URL()
	if url != "" {
		t.Errorf("DockerServer.URL(): Expected empty URL on handler mode, got %q.", url)
	}
}

func TestHandleWithHook(t *testing.T) {
	var called bool
	server, _ := NewServer("127.0.0.1:0", nil, func(*http.Request) { called = true })
	defer server.Stop()
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/containers/json?all=1", nil)
	server.ServeHTTP(recorder, request)
	if !called {
		t.Error("ServeHTTP did not call the hook function.")
	}
}

func TestSetHook(t *testing.T) {
	var called bool
	server, _ := NewServer("127.0.0.1:0", nil, nil)
	defer server.Stop()
	server.SetHook(func(*http.Request) { called = true })
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/containers/json?all=1", nil)
	server.ServeHTTP(recorder, request)
	if !called {
		t.Error("ServeHTTP did not call the hook function.")
	}
}

func TestCustomHandler(t *testing.T) {
	var called bool
	server, _ := NewServer("127.0.0.1:0", nil, nil)
	addContainers(server, 2)
	server.CustomHandler("/containers/json", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		fmt.Fprint(w, "Hello world")
	}))
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/containers/json?all=1", nil)
	server.ServeHTTP(recorder, request)
	if !called {
		t.Error("Did not call the custom handler")
	}
	if got := recorder.Body.String(); got != "Hello world" {
		t.Errorf("Wrong output for custom handler: want %q. Got %q.", "Hello world", got)
	}
}

func TestCustomHandlerRegexp(t *testing.T) {
	var called bool
	server, _ := NewServer("127.0.0.1:0", nil, nil)
	addContainers(server, 2)
	server.CustomHandler("/containers/.*/json", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		fmt.Fprint(w, "Hello world")
	}))
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/containers/.*/json?all=1", nil)
	server.ServeHTTP(recorder, request)
	if !called {
		t.Error("Did not call the custom handler")
	}
	if got := recorder.Body.String(); got != "Hello world" {
		t.Errorf("Wrong output for custom handler: want %q. Got %q.", "Hello world", got)
	}
}

func TestListContainers(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 2)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/containers/json?all=1", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Errorf("ListContainers: wrong status. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
	expected := make([]docker.APIContainers, 2)
	for i, container := range server.containers {
		expected[i] = docker.APIContainers{
			ID:      container.ID,
			Image:   container.Image,
			Command: strings.Join(container.Config.Cmd, " "),
			Created: container.Created.Unix(),
			Status:  container.State.String(),
			Ports:   container.NetworkSettings.PortMappingAPI(),
			Names:   []string{"/" + container.Name},
		}
	}
	var got []docker.APIContainers
	err := json.NewDecoder(recorder.Body).Decode(&got)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("ListContainers. Want %#v. Got %#v.", expected, got)
	}
}

func TestListRunningContainers(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 2)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/containers/json?all=0", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Errorf("ListRunningContainers: wrong status. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
	var got []docker.APIContainers
	err := json.NewDecoder(recorder.Body).Decode(&got)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Errorf("ListRunningContainers: Want 0. Got %d.", len(got))
	}
}

func TestCreateContainer(t *testing.T) {
	server := DockerServer{}
	server.imgIDs = map[string]string{"base": "a1234"}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	body := `{"Hostname":"", "User":"ubuntu", "Memory":0, "MemorySwap":0, "AttachStdin":false, "AttachStdout":true, "AttachStderr":true,
"PortSpecs":null, "Tty":false, "OpenStdin":false, "StdinOnce":false, "Env":null, "Cmd":["date"], "Image":"base", "Volumes":{}, "VolumesFrom":"","HostConfig":{"Binds":["/var/run/docker.sock:/var/run/docker.sock:rw"]}}`
	request, _ := http.NewRequest("POST", "/containers/create", strings.NewReader(body))
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusCreated {
		t.Errorf("CreateContainer: wrong status. Want %d. Got %d.", http.StatusCreated, recorder.Code)
	}
	var returned docker.Container
	err := json.NewDecoder(recorder.Body).Decode(&returned)
	if err != nil {
		t.Fatal(err)
	}
	stored := server.containers[0]
	if returned.ID != stored.ID {
		t.Errorf("CreateContainer: ID mismatch. Stored: %q. Returned: %q.", stored.ID, returned.ID)
	}
	if stored.State.Running {
		t.Errorf("CreateContainer should not set container to running state.")
	}
	if stored.Config.User != "ubuntu" {
		t.Errorf("CreateContainer: wrong config. Expected: %q. Returned: %q.", "ubuntu", stored.Config.User)
	}
	if stored.Config.Hostname != returned.ID[:12] {
		t.Errorf("CreateContainer: wrong hostname. Expected: %q. Returned: %q.", returned.ID[:12], stored.Config.Hostname)
	}
	expectedBind := []string{"/var/run/docker.sock:/var/run/docker.sock:rw"}
	if !reflect.DeepEqual(stored.HostConfig.Binds, expectedBind) {
		t.Errorf("CreateContainer: wrong host config. Expected: %v. Returned %v.", expectedBind, stored.HostConfig.Binds)
	}
}

func TestCreateContainerWithNotifyChannel(t *testing.T) {
	ch := make(chan *docker.Container, 1)
	server := DockerServer{}
	server.imgIDs = map[string]string{"base": "a1234"}
	server.cChan = ch
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	body := `{"Hostname":"", "User":"", "Memory":0, "MemorySwap":0, "AttachStdin":false, "AttachStdout":true, "AttachStderr":true,
"PortSpecs":null, "Tty":false, "OpenStdin":false, "StdinOnce":false, "Env":null, "Cmd":["date"], "Image":"base", "Volumes":{}, "VolumesFrom":""}`
	request, _ := http.NewRequest("POST", "/containers/create", strings.NewReader(body))
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusCreated {
		t.Errorf("CreateContainer: wrong status. Want %d. Got %d.", http.StatusCreated, recorder.Code)
	}
	if notified := <-ch; notified != server.containers[0] {
		t.Errorf("CreateContainer: did not notify the proper container. Want %q. Got %q.", server.containers[0].ID, notified.ID)
	}
}

func TestCreateContainerInvalidBody(t *testing.T) {
	server := DockerServer{}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("POST", "/containers/create", strings.NewReader("whaaaaaat---"))
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Errorf("CreateContainer: wrong status. Want %d. Got %d.", http.StatusBadRequest, recorder.Code)
	}
}

func TestCreateContainerDuplicateName(t *testing.T) {
	server := DockerServer{}
	server.buildMuxer()
	server.imgIDs = map[string]string{"base": "a1234"}
	addContainers(&server, 1)
	server.containers[0].Name = "mycontainer"
	recorder := httptest.NewRecorder()
	body := `{"Hostname":"", "User":"ubuntu", "Memory":0, "MemorySwap":0, "AttachStdin":false, "AttachStdout":true, "AttachStderr":true,
"PortSpecs":null, "Tty":false, "OpenStdin":false, "StdinOnce":false, "Env":null, "Cmd":["date"], "Image":"base", "Volumes":{}, "VolumesFrom":"","HostConfig":{"Binds":["/var/run/docker.sock:/var/run/docker.sock:rw"]}}`
	request, _ := http.NewRequest("POST", "/containers/create?name=mycontainer", strings.NewReader(body))
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusConflict {
		t.Errorf("CreateContainer: wrong status. Want %d. Got %d.", http.StatusConflict, recorder.Code)
	}
}

func TestCreateMultipleContainersEmptyName(t *testing.T) {
	server := DockerServer{}
	server.buildMuxer()
	server.imgIDs = map[string]string{"base": "a1234"}
	addContainers(&server, 1)
	server.containers[0].Name = ""
	recorder := httptest.NewRecorder()
	body := `{"Hostname":"", "User":"ubuntu", "Memory":0, "MemorySwap":0, "AttachStdin":false, "AttachStdout":true, "AttachStderr":true,
"PortSpecs":null, "Tty":false, "OpenStdin":false, "StdinOnce":false, "Env":null, "Cmd":["date"], "Image":"base", "Volumes":{}, "VolumesFrom":"","HostConfig":{"Binds":["/var/run/docker.sock:/var/run/docker.sock:rw"]}}`
	request, _ := http.NewRequest("POST", "/containers/create", strings.NewReader(body))
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusCreated {
		t.Errorf("CreateContainer: wrong status. Want %d. Got %d.", http.StatusCreated, recorder.Code)
	}
	var returned docker.Container
	err := json.NewDecoder(recorder.Body).Decode(&returned)
	if err != nil {
		t.Fatal(err)
	}
	stored := server.containers[1]
	if returned.ID != stored.ID {
		t.Errorf("CreateContainer: ID mismatch. Stored: %q. Returned: %q.", stored.ID, returned.ID)
	}
	if stored.State.Running {
		t.Errorf("CreateContainer should not set container to running state.")
	}
	if stored.Config.User != "ubuntu" {
		t.Errorf("CreateContainer: wrong config. Expected: %q. Returned: %q.", "ubuntu", stored.Config.User)
	}
	expectedBind := []string{"/var/run/docker.sock:/var/run/docker.sock:rw"}
	if !reflect.DeepEqual(stored.HostConfig.Binds, expectedBind) {
		t.Errorf("CreateContainer: wrong host config. Expected: %v. Returned %v.", expectedBind, stored.HostConfig.Binds)
	}
}

func TestCreateContainerInvalidName(t *testing.T) {
	server := DockerServer{}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	body := `{"Hostname":"", "User":"", "Memory":0, "MemorySwap":0, "AttachStdin":false, "AttachStdout":true, "AttachStderr":true,
"PortSpecs":null, "Tty":false, "OpenStdin":false, "StdinOnce":false, "Env":null, "Cmd":["date"],
"Image":"base", "Volumes":{}, "VolumesFrom":""}`
	request, _ := http.NewRequest("POST", "/containers/create?name=myapp/container1", strings.NewReader(body))
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusInternalServerError {
		t.Errorf("CreateContainer: wrong status. Want %d. Got %d.", http.StatusInternalServerError, recorder.Code)
	}
	expectedBody := "Invalid container name\n"
	if got := recorder.Body.String(); got != expectedBody {
		t.Errorf("CreateContainer: wrong body. Want %q. Got %q.", expectedBody, got)
	}
}

func TestCreateContainerImageNotFound(t *testing.T) {
	server := DockerServer{}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	body := `{"Hostname":"", "User":"", "Memory":0, "MemorySwap":0, "AttachStdin":false, "AttachStdout":true, "AttachStderr":true,
"PortSpecs":null, "Tty":false, "OpenStdin":false, "StdinOnce":false, "Env":null, "Cmd":["date"],
"Image":"base", "Volumes":{}, "VolumesFrom":""}`
	request, _ := http.NewRequest("POST", "/containers/create", strings.NewReader(body))
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNotFound {
		t.Errorf("CreateContainer: wrong status. Want %d. Got %d.", http.StatusNotFound, recorder.Code)
	}
}

func TestRenameContainer(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 2)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	newName := server.containers[0].Name + "abc"
	path := fmt.Sprintf("/containers/%s/rename?name=%s", server.containers[0].ID, newName)
	request, _ := http.NewRequest("POST", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNoContent {
		t.Errorf("RenameContainer: wrong status. Want %d. Got %d.", http.StatusNoContent, recorder.Code)
	}
	container := server.containers[0]
	if container.Name != newName {
		t.Errorf("RenameContainer: did not rename the container. Want %q. Got %q.", newName, container.Name)
	}
}

func TestRenameContainerNotFound(t *testing.T) {
	server := DockerServer{}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("POST", "/containers/blabla/rename?name=something", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNotFound {
		t.Errorf("RenameContainer: wrong status. Want %d. Got %d.", http.StatusNotFound, recorder.Code)
	}
}

func TestCommitContainer(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 2)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("POST", "/commit?container="+server.containers[0].ID, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Errorf("CommitContainer: wrong status. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
	expected := fmt.Sprintf(`{"ID":"%s"}`, server.images[0].ID)
	if got := recorder.Body.String(); got != expected {
		t.Errorf("CommitContainer: wrong response body. Want %q. Got %q.", expected, got)
	}
}

func TestCommitContainerComplete(t *testing.T) {
	server := DockerServer{}
	server.imgIDs = make(map[string]string)
	addContainers(&server, 2)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	queryString := "container=" + server.containers[0].ID + "&repo=tsuru/python&m=saving&author=developers"
	queryString += `&run={"Cmd": ["cat", "/world"],"PortSpecs":["22"]}`
	request, _ := http.NewRequest("POST", "/commit?"+queryString, nil)
	server.ServeHTTP(recorder, request)
	image := server.images[0]
	if image.Parent != server.containers[0].Image {
		t.Errorf("CommitContainer: wrong parent image. Want %q. Got %q.", server.containers[0].Image, image.Parent)
	}
	if image.Container != server.containers[0].ID {
		t.Errorf("CommitContainer: wrong container. Want %q. Got %q.", server.containers[0].ID, image.Container)
	}
	message := "saving"
	if image.Comment != message {
		t.Errorf("CommitContainer: wrong comment (commit message). Want %q. Got %q.", message, image.Comment)
	}
	author := "developers"
	if image.Author != author {
		t.Errorf("CommitContainer: wrong author. Want %q. Got %q.", author, image.Author)
	}
	if id := server.imgIDs["tsuru/python"]; id != image.ID {
		t.Errorf("CommitContainer: wrong ID saved for repository. Want %q. Got %q.", image.ID, id)
	}
	portSpecs := []string{"22"}
	if !reflect.DeepEqual(image.Config.PortSpecs, portSpecs) {
		t.Errorf("CommitContainer: wrong port spec in config. Want %#v. Got %#v.", portSpecs, image.Config.PortSpecs)
	}
	cmd := []string{"cat", "/world"}
	if !reflect.DeepEqual(image.Config.Cmd, cmd) {
		t.Errorf("CommitContainer: wrong cmd in config. Want %#v. Got %#v.", cmd, image.Config.Cmd)
	}
}

func TestCommitContainerWithTag(t *testing.T) {
	server := DockerServer{}
	server.imgIDs = make(map[string]string)
	addContainers(&server, 2)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	queryString := "container=" + server.containers[0].ID + "&repo=tsuru/python&tag=v1"
	request, _ := http.NewRequest("POST", "/commit?"+queryString, nil)
	server.ServeHTTP(recorder, request)
	image := server.images[0]
	if image.Parent != server.containers[0].Image {
		t.Errorf("CommitContainer: wrong parent image. Want %q. Got %q.", server.containers[0].Image, image.Parent)
	}
	if image.Container != server.containers[0].ID {
		t.Errorf("CommitContainer: wrong container. Want %q. Got %q.", server.containers[0].ID, image.Container)
	}
	if id := server.imgIDs["tsuru/python:v1"]; id != image.ID {
		t.Errorf("CommitContainer: wrong ID saved for repository. Want %q. Got %q.", image.ID, id)
	}
}

func TestCommitContainerInvalidRun(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 1)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("POST", "/commit?container="+server.containers[0].ID+"&run=abc---", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Errorf("CommitContainer. Wrong status. Want %d. Got %d.", http.StatusBadRequest, recorder.Code)
	}
}

func TestCommitContainerNotFound(t *testing.T) {
	server := DockerServer{}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("POST", "/commit?container=abc123", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNotFound {
		t.Errorf("CommitContainer. Wrong status. Want %d. Got %d.", http.StatusNotFound, recorder.Code)
	}
}

func TestInspectContainer(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 2)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s/json", server.containers[0].ID)
	request, _ := http.NewRequest("GET", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Errorf("InspectContainer: wrong status. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
	expected := server.containers[0]
	var got docker.Container
	err := json.NewDecoder(recorder.Body).Decode(&got)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Config, expected.Config) {
		t.Errorf("InspectContainer: wrong value. Want %#v. Got %#v.", *expected, got)
	}
	if !reflect.DeepEqual(got.NetworkSettings, expected.NetworkSettings) {
		t.Errorf("InspectContainer: wrong value. Want %#v. Got %#v.", *expected, got)
	}
	got.State.StartedAt = expected.State.StartedAt
	got.State.FinishedAt = expected.State.FinishedAt
	got.Config = expected.Config
	got.Created = expected.Created
	got.NetworkSettings = expected.NetworkSettings
	if !reflect.DeepEqual(got, *expected) {
		t.Errorf("InspectContainer: wrong value. Want %#v. Got %#v.", *expected, got)
	}
}

func TestInspectContainerNotFound(t *testing.T) {
	server := DockerServer{}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/containers/abc123/json", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNotFound {
		t.Errorf("InspectContainer: wrong status code. Want %d. Got %d.", http.StatusNotFound, recorder.Code)
	}
}

func TestTopContainer(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 1)
	server.containers[0].State.Running = true
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s/top", server.containers[0].ID)
	request, _ := http.NewRequest("GET", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Errorf("TopContainer: wrong status. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
	var got docker.TopResult
	err := json.NewDecoder(recorder.Body).Decode(&got)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Titles, []string{"UID", "PID", "PPID", "C", "STIME", "TTY", "TIME", "CMD"}) {
		t.Fatalf("TopContainer: Unexpected titles, got: %#v", got.Titles)
	}
	if len(got.Processes) != 1 {
		t.Fatalf("TopContainer: Unexpected process len, got: %d", len(got.Processes))
	}
	if got.Processes[0][len(got.Processes[0])-1] != "ls -la .." {
		t.Fatalf("TopContainer: Unexpected command name, got: %s", got.Processes[0][len(got.Processes[0])-1])
	}
}

func TestTopContainerNotFound(t *testing.T) {
	server := DockerServer{}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/containers/xyz/top", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNotFound {
		t.Errorf("TopContainer: wrong status. Want %d. Got %d.", http.StatusNotFound, recorder.Code)
	}
}

func TestTopContainerStopped(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 1)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s/top", server.containers[0].ID)
	request, _ := http.NewRequest("GET", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusInternalServerError {
		t.Errorf("TopContainer: wrong status. Want %d. Got %d.", http.StatusInternalServerError, recorder.Code)
	}
}

func TestStartContainer(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 1)
	server.buildMuxer()
	memory := int64(536870912)
	hostConfig := docker.HostConfig{Memory: memory}
	configBytes, err := json.Marshal(hostConfig)
	if err != nil {
		t.Fatal(err)
	}
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s/start", server.containers[0].ID)
	request, _ := http.NewRequest("POST", path, bytes.NewBuffer(configBytes))
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Errorf("StartContainer: wrong status code. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
	if !server.containers[0].State.Running {
		t.Error("StartContainer: did not set the container to running state")
	}
	if gotMemory := server.containers[0].HostConfig.Memory; gotMemory != memory {
		t.Errorf("StartContainer: wrong HostConfig. Wants %d of memory. Got %d", memory, gotMemory)
	}
}

func TestStartContainerWithNotifyChannel(t *testing.T) {
	ch := make(chan *docker.Container, 1)
	server := DockerServer{}
	server.cChan = ch
	addContainers(&server, 1)
	addContainers(&server, 1)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s/start", server.containers[1].ID)
	request, _ := http.NewRequest("POST", path, bytes.NewBuffer([]byte("{}")))
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Errorf("StartContainer: wrong status code. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
	if notified := <-ch; notified != server.containers[1] {
		t.Errorf("StartContainer: did not notify the proper container. Want %q. Got %q.", server.containers[1].ID, notified.ID)
	}
}

func TestStartContainerNotFound(t *testing.T) {
	server := DockerServer{}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := "/containers/abc123/start"
	request, _ := http.NewRequest("POST", path, bytes.NewBuffer([]byte("null")))
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNotFound {
		t.Errorf("StartContainer: wrong status code. Want %d. Got %d.", http.StatusNotFound, recorder.Code)
	}
}

func TestStartContainerAlreadyRunning(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 1)
	server.containers[0].State.Running = true
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s/start", server.containers[0].ID)
	request, _ := http.NewRequest("POST", path, bytes.NewBuffer([]byte("null")))
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Errorf("StartContainer: wrong status code. Want %d. Got %d.", http.StatusBadRequest, recorder.Code)
	}
}

func TestStopContainer(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 1)
	server.containers[0].State.Running = true
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s/stop", server.containers[0].ID)
	request, _ := http.NewRequest("POST", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNoContent {
		t.Errorf("StopContainer: wrong status code. Want %d. Got %d.", http.StatusNoContent, recorder.Code)
	}
	if server.containers[0].State.Running {
		t.Error("StopContainer: did not stop the container")
	}
}

func TestKillContainer(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 1)
	server.containers[0].State.Running = true
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s/kill", server.containers[0].ID)
	request, _ := http.NewRequest("POST", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNoContent {
		t.Errorf("KillContainer: wrong status code. Want %d. Got %d.", http.StatusNoContent, recorder.Code)
	}
	if server.containers[0].State.Running {
		t.Error("KillContainer: did not stop the container")
	}
}

func TestStopContainerWithNotifyChannel(t *testing.T) {
	ch := make(chan *docker.Container, 1)
	server := DockerServer{}
	server.cChan = ch
	addContainers(&server, 1)
	addContainers(&server, 1)
	server.containers[1].State.Running = true
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s/stop", server.containers[1].ID)
	request, _ := http.NewRequest("POST", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNoContent {
		t.Errorf("StopContainer: wrong status code. Want %d. Got %d.", http.StatusNoContent, recorder.Code)
	}
	if notified := <-ch; notified != server.containers[1] {
		t.Errorf("StopContainer: did not notify the proper container. Want %q. Got %q.", server.containers[1].ID, notified.ID)
	}
}

func TestStopContainerNotFound(t *testing.T) {
	server := DockerServer{}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := "/containers/abc123/stop"
	request, _ := http.NewRequest("POST", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNotFound {
		t.Errorf("StopContainer: wrong status code. Want %d. Got %d.", http.StatusNotFound, recorder.Code)
	}
}

func TestStopContainerNotRunning(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 1)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s/stop", server.containers[0].ID)
	request, _ := http.NewRequest("POST", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Errorf("StopContainer: wrong status code. Want %d. Got %d.", http.StatusBadRequest, recorder.Code)
	}
}

func TestPauseContainer(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 1)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s/pause", server.containers[0].ID)
	request, _ := http.NewRequest("POST", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNoContent {
		t.Errorf("PauseContainer: wrong status code. Want %d. Got %d.", http.StatusNoContent, recorder.Code)
	}
	if !server.containers[0].State.Paused {
		t.Error("PauseContainer: did not pause the container")
	}
}

func TestPauseContainerAlreadyPaused(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 1)
	server.containers[0].State.Paused = true
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s/pause", server.containers[0].ID)
	request, _ := http.NewRequest("POST", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Errorf("PauseContainer: wrong status code. Want %d. Got %d.", http.StatusBadRequest, recorder.Code)
	}
}

func TestPauseContainerNotFound(t *testing.T) {
	server := DockerServer{}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := "/containers/abc123/pause"
	request, _ := http.NewRequest("POST", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNotFound {
		t.Errorf("PauseContainer: wrong status code. Want %d. Got %d.", http.StatusNotFound, recorder.Code)
	}
}

func TestUnpauseContainer(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 1)
	server.containers[0].State.Paused = true
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s/unpause", server.containers[0].ID)
	request, _ := http.NewRequest("POST", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNoContent {
		t.Errorf("UnpauseContainer: wrong status code. Want %d. Got %d.", http.StatusNoContent, recorder.Code)
	}
	if server.containers[0].State.Paused {
		t.Error("UnpauseContainer: did not unpause the container")
	}
}

func TestUnpauseContainerNotPaused(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 1)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s/unpause", server.containers[0].ID)
	request, _ := http.NewRequest("POST", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Errorf("UnpauseContainer: wrong status code. Want %d. Got %d.", http.StatusBadRequest, recorder.Code)
	}
}

func TestUnpauseContainerNotFound(t *testing.T) {
	server := DockerServer{}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := "/containers/abc123/unpause"
	request, _ := http.NewRequest("POST", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNotFound {
		t.Errorf("UnpauseContainer: wrong status code. Want %d. Got %d.", http.StatusNotFound, recorder.Code)
	}
}

func TestWaitContainer(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 1)
	server.containers[0].State.Running = true
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s/wait", server.containers[0].ID)
	request, _ := http.NewRequest("POST", path, nil)
	go func() {
		server.cMut.Lock()
		server.containers[0].State.Running = false
		server.cMut.Unlock()
	}()
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Errorf("WaitContainer: wrong status. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
	expected := `{"StatusCode":0}` + "\n"
	if body := recorder.Body.String(); body != expected {
		t.Errorf("WaitContainer: wrong body. Want %q. Got %q.", expected, body)
	}
}

func TestWaitContainerStatus(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 1)
	server.buildMuxer()
	server.containers[0].State.ExitCode = 63
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s/wait", server.containers[0].ID)
	request, _ := http.NewRequest("POST", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Errorf("WaitContainer: wrong status. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
	expected := `{"StatusCode":63}` + "\n"
	if body := recorder.Body.String(); body != expected {
		t.Errorf("WaitContainer: wrong body. Want %q. Got %q.", expected, body)
	}
}

func TestWaitContainerNotFound(t *testing.T) {
	server := DockerServer{}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := "/containers/abc123/wait"
	request, _ := http.NewRequest("POST", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNotFound {
		t.Errorf("WaitContainer: wrong status code. Want %d. Got %d.", http.StatusNotFound, recorder.Code)
	}
}

func TestAttachContainer(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 1)
	server.containers[0].State.Running = true
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s/attach?logs=1", server.containers[0].ID)
	request, _ := http.NewRequest("POST", path, nil)
	server.ServeHTTP(recorder, request)
	lines := []string{
		fmt.Sprintf("\x01\x00\x00\x00\x03\x00\x00\x00Container %q is running", server.containers[0].ID),
		"What happened?",
		"Something happened",
	}
	expected := strings.Join(lines, "\n") + "\n"
	if body := recorder.Body.String(); body == expected {
		t.Errorf("AttachContainer: wrong body. Want %q. Got %q.", expected, body)
	}
}

func TestAttachContainerNotFound(t *testing.T) {
	server := DockerServer{}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := "/containers/abc123/attach?logs=1"
	request, _ := http.NewRequest("POST", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNotFound {
		t.Errorf("AttachContainer: wrong status. Want %d. Got %d.", http.StatusNotFound, recorder.Code)
	}
}

func TestRemoveContainer(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 1)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s", server.containers[0].ID)
	request, _ := http.NewRequest("DELETE", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNoContent {
		t.Errorf("RemoveContainer: wrong status. Want %d. Got %d.", http.StatusNoContent, recorder.Code)
	}
	if len(server.containers) > 0 {
		t.Error("RemoveContainer: did not remove the container.")
	}
}

func TestRemoveContainerByName(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 1)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s", server.containers[0].Name)
	request, _ := http.NewRequest("DELETE", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNoContent {
		t.Errorf("RemoveContainer: wrong status. Want %d. Got %d.", http.StatusNoContent, recorder.Code)
	}
	if len(server.containers) > 0 {
		t.Error("RemoveContainer: did not remove the container.")
	}
}

func TestRemoveContainerNotFound(t *testing.T) {
	server := DockerServer{}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/abc123")
	request, _ := http.NewRequest("DELETE", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNotFound {
		t.Errorf("RemoveContainer: wrong status. Want %d. Got %d.", http.StatusNotFound, recorder.Code)
	}
}

func TestRemoveContainerRunning(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 1)
	server.containers[0].State.Running = true
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s", server.containers[0].ID)
	request, _ := http.NewRequest("DELETE", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusInternalServerError {
		t.Errorf("RemoveContainer: wrong status. Want %d. Got %d.", http.StatusInternalServerError, recorder.Code)
	}
	if len(server.containers) < 1 {
		t.Error("RemoveContainer: should not remove the container.")
	}
}

func TestRemoveContainerRunningForce(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 1)
	server.containers[0].State.Running = true
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s?%s", server.containers[0].ID, "force=1")
	request, _ := http.NewRequest("DELETE", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNoContent {
		t.Errorf("RemoveContainer: wrong status. Want %d. Got %d.", http.StatusNoContent, recorder.Code)
	}
	if len(server.containers) > 0 {
		t.Error("RemoveContainer: did not remove the container.")
	}
}

func TestPullImage(t *testing.T) {
	server := DockerServer{imgIDs: make(map[string]string)}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("POST", "/images/create?fromImage=base", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Errorf("PullImage: wrong status. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
	if len(server.images) != 1 {
		t.Errorf("PullImage: Want 1 image. Got %d.", len(server.images))
	}
	if _, ok := server.imgIDs["base"]; !ok {
		t.Error("PullImage: Repository should not be empty.")
	}
}

func TestPullImageWithTag(t *testing.T) {
	server := DockerServer{imgIDs: make(map[string]string)}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("POST", "/images/create?fromImage=base&tag=tag", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Errorf("PullImage: wrong status. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
	if len(server.images) != 1 {
		t.Errorf("PullImage: Want 1 image. Got %d.", len(server.images))
	}
	if _, ok := server.imgIDs["base:tag"]; !ok {
		t.Error("PullImage: Repository should not be empty.")
	}
}

func TestPushImage(t *testing.T) {
	server := DockerServer{imgIDs: map[string]string{"tsuru/python": "a123"}}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("POST", "/images/tsuru/python/push", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Errorf("PushImage: wrong status. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
}

func TestPushImageWithTag(t *testing.T) {
	server := DockerServer{imgIDs: map[string]string{"tsuru/python:v1": "a123"}}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("POST", "/images/tsuru/python/push?tag=v1", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Errorf("PushImage: wrong status. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
}

func TestPushImageNotFound(t *testing.T) {
	server := DockerServer{}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("POST", "/images/tsuru/python/push", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNotFound {
		t.Errorf("PushImage: wrong status. Want %d. Got %d.", http.StatusNotFound, recorder.Code)
	}
}

func TestTagImage(t *testing.T) {
	server := DockerServer{imgIDs: map[string]string{"tsuru/python": "a123"}}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("POST", "/images/tsuru/python/tag?repo=tsuru/new-python", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusCreated {
		t.Errorf("TagImage: wrong status. Want %d. Got %d.", http.StatusCreated, recorder.Code)
	}
	if server.imgIDs["tsuru/python"] != server.imgIDs["tsuru/new-python"] {
		t.Errorf("TagImage: did not tag the image")
	}
}

func TestTagImageWithRepoAndTag(t *testing.T) {
	server := DockerServer{imgIDs: map[string]string{"tsuru/python": "a123"}}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("POST", "/images/tsuru/python/tag?repo=tsuru/new-python&tag=v1", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusCreated {
		t.Errorf("TagImage: wrong status. Want %d. Got %d.", http.StatusCreated, recorder.Code)
	}
	if server.imgIDs["tsuru/python"] != server.imgIDs["tsuru/new-python:v1"] {
		t.Errorf("TagImage: did not tag the image")
	}
}

func TestTagImageNotFound(t *testing.T) {
	server := DockerServer{}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("POST", "/images/tsuru/python/tag", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNotFound {
		t.Errorf("TagImage: wrong status. Want %d. Got %d.", http.StatusNotFound, recorder.Code)
	}
}

func addContainers(server *DockerServer, n int) {
	server.cMut.Lock()
	defer server.cMut.Unlock()
	for i := 0; i < n; i++ {
		date := time.Now().Add(time.Duration((rand.Int() % (i + 1))) * time.Hour)
		container := docker.Container{
			Name:    fmt.Sprintf("%x", rand.Int()%10000),
			ID:      fmt.Sprintf("%x", rand.Int()%10000),
			Created: date,
			Path:    "ls",
			Args:    []string{"-la", ".."},
			Config: &docker.Config{
				Hostname:     fmt.Sprintf("docker-%d", i),
				AttachStdout: true,
				AttachStderr: true,
				Env:          []string{"ME=you", fmt.Sprintf("NUMBER=%d", i)},
				Cmd:          []string{"ls", "-la", ".."},
				Image:        "base",
			},
			State: docker.State{
				Running:   false,
				Pid:       400 + i,
				ExitCode:  0,
				StartedAt: date,
			},
			Image: "b750fe79269d2ec9a3c593ef05b4332b1d1a02a62b4accb2c21d589ff2f5f2dc",
			NetworkSettings: &docker.NetworkSettings{
				IPAddress:   fmt.Sprintf("10.10.10.%d", i+2),
				IPPrefixLen: 24,
				Gateway:     "10.10.10.1",
				Bridge:      "docker0",
				PortMapping: map[string]docker.PortMapping{
					"Tcp": {"8888": fmt.Sprintf("%d", 49600+i)},
				},
			},
			ResolvConfPath: "/etc/resolv.conf",
		}
		server.containers = append(server.containers, &container)
	}
}

func addImages(server *DockerServer, n int, repo bool) {
	server.iMut.Lock()
	defer server.iMut.Unlock()
	if server.imgIDs == nil {
		server.imgIDs = make(map[string]string)
	}
	for i := 0; i < n; i++ {
		date := time.Now().Add(time.Duration((rand.Int() % (i + 1))) * time.Hour)
		image := docker.Image{
			ID:      fmt.Sprintf("%x", rand.Int()%10000),
			Created: date,
		}
		server.images = append(server.images, image)
		if repo {
			repo := "docker/python-" + image.ID
			server.imgIDs[repo] = image.ID
		}
	}
}

func TestListImages(t *testing.T) {
	server := DockerServer{}
	addImages(&server, 2, true)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/images/json?all=1", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Errorf("ListImages: wrong status. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
	expected := make([]docker.APIImages, 2)
	for i, image := range server.images {
		expected[i] = docker.APIImages{
			ID:       image.ID,
			Created:  image.Created.Unix(),
			RepoTags: []string{"docker/python-" + image.ID},
		}
	}
	var got []docker.APIImages
	err := json.NewDecoder(recorder.Body).Decode(&got)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("ListImages. Want %#v. Got %#v.", expected, got)
	}
}

func TestRemoveImage(t *testing.T) {
	server := DockerServer{}
	addImages(&server, 1, false)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/images/%s", server.images[0].ID)
	request, _ := http.NewRequest("DELETE", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNoContent {
		t.Errorf("RemoveImage: wrong status. Want %d. Got %d.", http.StatusNoContent, recorder.Code)
	}
	if len(server.images) > 0 {
		t.Error("RemoveImage: did not remove the image.")
	}
}

func TestRemoveImageByName(t *testing.T) {
	server := DockerServer{}
	addImages(&server, 1, true)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	imgName := "docker/python-" + server.images[0].ID
	path := "/images/" + imgName
	request, _ := http.NewRequest("DELETE", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNoContent {
		t.Errorf("RemoveImage: wrong status. Want %d. Got %d.", http.StatusNoContent, recorder.Code)
	}
	if len(server.images) > 0 {
		t.Error("RemoveImage: did not remove the image.")
	}
	_, ok := server.imgIDs[imgName]
	if ok {
		t.Error("RemoveImage: did not remove image tag name.")
	}
}

func TestRemoveImageWithMultipleTags(t *testing.T) {
	server := DockerServer{}
	addImages(&server, 1, true)
	server.buildMuxer()
	imgID := server.images[0].ID
	imgName := "docker/python-" + imgID
	server.imgIDs["docker/python-wat"] = imgID
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/images/%s", imgName)
	request, _ := http.NewRequest("DELETE", path, nil)
	server.ServeHTTP(recorder, request)
	_, ok := server.imgIDs[imgName]
	if ok {
		t.Error("RemoveImage: did not remove image tag name.")
	}
	id, ok := server.imgIDs["docker/python-wat"]
	if !ok {
		t.Error("RemoveImage: removed the wrong tag name.")
	}
	if id != imgID {
		t.Error("RemoveImage: disassociated the wrong ID from the tag")
	}
	if len(server.images) < 1 {
		t.Fatal("RemoveImage: removed the image, but should keep it")
	}
	if server.images[0].ID != imgID {
		t.Error("RemoveImage: changed the ID of the image!")
	}
}

func TestPrepareFailure(t *testing.T) {
	server := DockerServer{failures: make(map[string]string)}
	server.buildMuxer()
	errorID := "my_error"
	server.PrepareFailure(errorID, "containers/json")
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/containers/json?all=1", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Errorf("PrepareFailure: wrong status. Want %d. Got %d.", http.StatusBadRequest, recorder.Code)
	}
	if recorder.Body.String() != errorID+"\n" {
		t.Errorf("PrepareFailure: wrong message. Want %s. Got %s.", errorID, recorder.Body.String())
	}
}

func TestPrepareMultiFailures(t *testing.T) {
	server := DockerServer{multiFailures: []map[string]string{}}
	server.buildMuxer()
	errorID := "multi error"
	server.PrepareMultiFailures(errorID, "containers/json")
	server.PrepareMultiFailures(errorID, "containers/json")
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/containers/json?all=1", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Errorf("PrepareFailure: wrong status. Want %d. Got %d.", http.StatusBadRequest, recorder.Code)
	}
	if recorder.Body.String() != errorID+"\n" {
		t.Errorf("PrepareFailure: wrong message. Want %s. Got %s.", errorID, recorder.Body.String())
	}
	recorder = httptest.NewRecorder()
	request, _ = http.NewRequest("GET", "/containers/json?all=1", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Errorf("PrepareFailure: wrong status. Want %d. Got %d.", http.StatusBadRequest, recorder.Code)
	}
	if recorder.Body.String() != errorID+"\n" {
		t.Errorf("PrepareFailure: wrong message. Want %s. Got %s.", errorID, recorder.Body.String())
	}
	recorder = httptest.NewRecorder()
	request, _ = http.NewRequest("GET", "/containers/json?all=1", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Errorf("PrepareFailure: wrong status. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
	if recorder.Body.String() == errorID+"\n" {
		t.Errorf("PrepareFailure: wrong message. Want %s. Got %s.", errorID, recorder.Body.String())
	}
}

func TestRemoveFailure(t *testing.T) {
	server := DockerServer{failures: make(map[string]string)}
	server.buildMuxer()
	errorID := "my_error"
	server.PrepareFailure(errorID, "containers/json")
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/containers/json?all=1", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Errorf("PrepareFailure: wrong status. Want %d. Got %d.", http.StatusBadRequest, recorder.Code)
	}
	server.ResetFailure(errorID)
	recorder = httptest.NewRecorder()
	request, _ = http.NewRequest("GET", "/containers/json?all=1", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Errorf("RemoveFailure: wrong status. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
}

func TestResetMultiFailures(t *testing.T) {
	server := DockerServer{multiFailures: []map[string]string{}}
	server.buildMuxer()
	errorID := "multi error"
	server.PrepareMultiFailures(errorID, "containers/json")
	server.PrepareMultiFailures(errorID, "containers/json")
	if len(server.multiFailures) != 2 {
		t.Errorf("PrepareMultiFailures: error adding multi failures.")
	}
	server.ResetMultiFailures()
	if len(server.multiFailures) != 0 {
		t.Errorf("ResetMultiFailures: error reseting multi failures.")
	}
}

func TestMutateContainer(t *testing.T) {
	server := DockerServer{failures: make(map[string]string)}
	server.buildMuxer()
	server.containers = append(server.containers, &docker.Container{ID: "id123"})
	state := docker.State{Running: false, ExitCode: 1}
	err := server.MutateContainer("id123", state)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(server.containers[0].State, state) {
		t.Errorf("Wrong state after mutation.\nWant %#v.\nGot %#v.",
			state, server.containers[0].State)
	}
}

func TestMutateContainerNotFound(t *testing.T) {
	server := DockerServer{failures: make(map[string]string)}
	server.buildMuxer()
	state := docker.State{Running: false, ExitCode: 1}
	err := server.MutateContainer("id123", state)
	if err == nil {
		t.Error("Unexpected <nil> error")
	}
	if err.Error() != "container not found" {
		t.Errorf("wrong error message. Want %q. Got %q.", "container not found", err)
	}
}

func TestBuildImageWithContentTypeTar(t *testing.T) {
	server := DockerServer{imgIDs: make(map[string]string)}
	imageName := "teste"
	recorder := httptest.NewRecorder()
	tarFile, err := os.Open("data/dockerfile.tar")
	if err != nil {
		t.Fatal(err)
	}
	defer tarFile.Close()
	request, _ := http.NewRequest("POST", "/build?t=teste", tarFile)
	request.Header.Add("Content-Type", "application/tar")
	server.buildImage(recorder, request)
	if recorder.Body.String() == "miss Dockerfile" {
		t.Errorf("BuildImage: miss Dockerfile")
		return
	}
	if _, ok := server.imgIDs[imageName]; ok == false {
		t.Errorf("BuildImage: image %s not builded", imageName)
	}
}

func TestBuildImageWithRemoteDockerfile(t *testing.T) {
	server := DockerServer{imgIDs: make(map[string]string)}
	imageName := "teste"
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("POST", "/build?t=teste&remote=http://localhost/Dockerfile", nil)
	server.buildImage(recorder, request)
	if _, ok := server.imgIDs[imageName]; ok == false {
		t.Errorf("BuildImage: image %s not builded", imageName)
	}
}

func TestPing(t *testing.T) {
	server := DockerServer{}
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/_ping", nil)
	server.pingDocker(recorder, request)
	if recorder.Body.String() != "" {
		t.Errorf("Ping: Unexpected body: %s", recorder.Body.String())
	}
	if recorder.Code != http.StatusOK {
		t.Errorf("Ping: Expected code %d, got: %d", http.StatusOK, recorder.Code)
	}
}

func TestDefaultHandler(t *testing.T) {
	server, err := NewServer("127.0.0.1:0", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer server.listener.Close()
	if server.mux != server.DefaultHandler() {
		t.Fatalf("DefaultHandler: Expected to return server.mux, got: %#v", server.DefaultHandler())
	}
}

func TestCreateExecContainer(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 2)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	body := `{"Cmd": ["bash", "-c", "ls"]}`
	path := fmt.Sprintf("/containers/%s/exec", server.containers[0].ID)
	request, _ := http.NewRequest("POST", path, strings.NewReader(body))
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("CreateExec: wrong status. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
	serverExec := server.execs[0]
	var got docker.Exec
	err := json.NewDecoder(recorder.Body).Decode(&got)
	if err != nil {
		t.Fatal(err)
	}
	if got.ID != serverExec.ID {
		t.Errorf("CreateExec: wrong value. Want %#v. Got %#v.", serverExec.ID, got.ID)
	}
	expected := docker.ExecInspect{
		ID: got.ID,
		ProcessConfig: docker.ExecProcessConfig{
			EntryPoint: "bash",
			Arguments:  []string{"-c", "ls"},
		},
		Container: *server.containers[0],
	}
	if !reflect.DeepEqual(*serverExec, expected) {
		t.Errorf("InspectContainer: wrong value. Want:\n%#v\nGot:\n%#v\n", expected, *serverExec)
	}
}

func TestInspectExecContainer(t *testing.T) {
	server := DockerServer{}
	addContainers(&server, 1)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	body := `{"Cmd": ["bash", "-c", "ls"]}`
	path := fmt.Sprintf("/containers/%s/exec", server.containers[0].ID)
	request, _ := http.NewRequest("POST", path, strings.NewReader(body))
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("CreateExec: wrong status. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
	var got docker.Exec
	err := json.NewDecoder(recorder.Body).Decode(&got)
	if err != nil {
		t.Fatal(err)
	}
	path = fmt.Sprintf("/exec/%s/json", got.ID)
	request, _ = http.NewRequest("GET", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("CreateExec: wrong status. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
	var got2 docker.ExecInspect
	err = json.NewDecoder(recorder.Body).Decode(&got2)
	if err != nil {
		t.Fatal(err)
	}
	expected := docker.ExecInspect{
		ID: got.ID,
		ProcessConfig: docker.ExecProcessConfig{
			EntryPoint: "bash",
			Arguments:  []string{"-c", "ls"},
		},
		Container: *server.containers[0],
	}
	got2.Container.State.StartedAt = expected.Container.State.StartedAt
	got2.Container.State.FinishedAt = expected.Container.State.FinishedAt
	got2.Container.Config = expected.Container.Config
	got2.Container.Created = expected.Container.Created
	got2.Container.NetworkSettings = expected.Container.NetworkSettings
	if !reflect.DeepEqual(got2, expected) {
		t.Errorf("InspectContainer: wrong value. Want:\n%#v\nGot:\n%#v\n", expected, got2)
	}
}

func TestStartExecContainer(t *testing.T) {
	server, _ := NewServer("127.0.0.1:0", nil, nil)
	addContainers(server, 1)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	body := `{"Cmd": ["bash", "-c", "ls"]}`
	path := fmt.Sprintf("/containers/%s/exec", server.containers[0].ID)
	request, _ := http.NewRequest("POST", path, strings.NewReader(body))
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("CreateExec: wrong status. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
	var exec docker.Exec
	err := json.NewDecoder(recorder.Body).Decode(&exec)
	if err != nil {
		t.Fatal(err)
	}
	unleash := make(chan bool)
	server.PrepareExec(exec.ID, func() {
		<-unleash
	})
	codes := make(chan int, 1)
	sent := make(chan bool)
	go func() {
		recorder := httptest.NewRecorder()
		path := fmt.Sprintf("/exec/%s/start", exec.ID)
		body := `{"Tty":true}`
		request, _ := http.NewRequest("POST", path, strings.NewReader(body))
		close(sent)
		server.ServeHTTP(recorder, request)
		codes <- recorder.Code
	}()
	<-sent
	execInfo, err := waitExec(server.URL(), exec.ID, true, 5)
	if err != nil {
		t.Fatal(err)
	}
	if !execInfo.Running {
		t.Error("StartExec: expected exec to be running, but it's not running")
	}
	close(unleash)
	if code := <-codes; code != http.StatusOK {
		t.Errorf("StartExec: wrong status. Want %d. Got %d.", http.StatusOK, code)
	}
	execInfo, err = waitExec(server.URL(), exec.ID, false, 5)
	if err != nil {
		t.Fatal(err)
	}
	if execInfo.Running {
		t.Error("StartExec: expected exec to be not running after start returns, but it's running")
	}
}

func TestStartExecContainerWildcardCallback(t *testing.T) {
	server, _ := NewServer("127.0.0.1:0", nil, nil)
	addContainers(server, 1)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	body := `{"Cmd": ["bash", "-c", "ls"]}`
	path := fmt.Sprintf("/containers/%s/exec", server.containers[0].ID)
	request, _ := http.NewRequest("POST", path, strings.NewReader(body))
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("CreateExec: wrong status. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
	unleash := make(chan bool)
	server.PrepareExec("*", func() {
		<-unleash
	})
	var exec docker.Exec
	err := json.NewDecoder(recorder.Body).Decode(&exec)
	if err != nil {
		t.Fatal(err)
	}
	codes := make(chan int, 1)
	sent := make(chan bool)
	go func() {
		recorder := httptest.NewRecorder()
		path := fmt.Sprintf("/exec/%s/start", exec.ID)
		body := `{"Tty":true}`
		request, _ := http.NewRequest("POST", path, strings.NewReader(body))
		close(sent)
		server.ServeHTTP(recorder, request)
		codes <- recorder.Code
	}()
	<-sent
	execInfo, err := waitExec(server.URL(), exec.ID, true, 5)
	if err != nil {
		t.Fatal(err)
	}
	if !execInfo.Running {
		t.Error("StartExec: expected exec to be running, but it's not running")
	}
	close(unleash)
	if code := <-codes; code != http.StatusOK {
		t.Errorf("StartExec: wrong status. Want %d. Got %d.", http.StatusOK, code)
	}
	execInfo, err = waitExec(server.URL(), exec.ID, false, 5)
	if err != nil {
		t.Fatal(err)
	}
	if execInfo.Running {
		t.Error("StartExec: expected exec to be not running after start returns, but it's running")
	}
}

func TestStartExecContainerNotFound(t *testing.T) {
	server, _ := NewServer("127.0.0.1:0", nil, nil)
	addContainers(server, 1)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	body := `{"Tty":true}`
	request, _ := http.NewRequest("POST", "/exec/something-wat/start", strings.NewReader(body))
	server.ServeHTTP(recorder, request)
}

func waitExec(url, execID string, running bool, maxTry int) (*docker.ExecInspect, error) {
	client, err := docker.NewClient(url)
	if err != nil {
		return nil, err
	}
	exec, err := client.InspectExec(execID)
	for i := 0; i < maxTry && exec.Running != running && err == nil; i++ {
		time.Sleep(100e6)
		exec, err = client.InspectExec(exec.ID)
	}
	return exec, err
}

func TestStatsContainer(t *testing.T) {
	server, err := NewServer("127.0.0.1:0", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer server.Stop()
	addContainers(server, 2)
	server.buildMuxer()
	expected := docker.Stats{}
	expected.CPUStats.CPUUsage.TotalUsage = 20
	server.PrepareStats(server.containers[0].ID, func(id string) docker.Stats {
		return expected
	})
	recorder := httptest.NewRecorder()
	path := fmt.Sprintf("/containers/%s/stats?stream=false", server.containers[0].ID)
	request, _ := http.NewRequest("GET", path, nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Errorf("StatsContainer: wrong status. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
	body := recorder.Body.Bytes()
	var got docker.Stats
	err = json.Unmarshal(body, &got)
	if err != nil {
		t.Fatal(err)
	}
	got.Read = time.Time{}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("StatsContainer: wrong value. Want %#v. Got %#v.", expected, got)
	}
}

type safeWriter struct {
	sync.Mutex
	*httptest.ResponseRecorder
}

func (w *safeWriter) Write(buf []byte) (int, error) {
	w.Lock()
	defer w.Unlock()
	return w.ResponseRecorder.Write(buf)
}

func TestStatsContainerStream(t *testing.T) {
	server, err := NewServer("127.0.0.1:0", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer server.Stop()
	addContainers(server, 2)
	server.buildMuxer()
	expected := docker.Stats{}
	expected.CPUStats.CPUUsage.TotalUsage = 20
	server.PrepareStats(server.containers[0].ID, func(id string) docker.Stats {
		time.Sleep(50 * time.Millisecond)
		return expected
	})
	recorder := &safeWriter{
		ResponseRecorder: httptest.NewRecorder(),
	}
	path := fmt.Sprintf("/containers/%s/stats?stream=true", server.containers[0].ID)
	request, _ := http.NewRequest("GET", path, nil)
	go func() {
		server.ServeHTTP(recorder, request)
	}()
	time.Sleep(200 * time.Millisecond)
	recorder.Lock()
	defer recorder.Unlock()
	body := recorder.Body.Bytes()
	parts := bytes.Split(body, []byte("\n"))
	if len(parts) < 2 {
		t.Errorf("StatsContainer: wrong number of parts. Want at least 2. Got %#v.", len(parts))
	}
	var got docker.Stats
	err = json.Unmarshal(parts[0], &got)
	if err != nil {
		t.Fatal(err)
	}
	got.Read = time.Time{}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("StatsContainer: wrong value. Want %#v. Got %#v.", expected, got)
	}
}

func addNetworks(server *DockerServer, n int) {
	server.netMut.Lock()
	defer server.netMut.Unlock()
	for i := 0; i < n; i++ {
		netid := fmt.Sprintf("%x", rand.Int()%10000)
		network := docker.Network{
			Name: netid,
			ID:   fmt.Sprintf("%x", rand.Int()%10000),
			Type: "bridge",
			Endpoints: []*docker.Endpoint{
				&docker.Endpoint{
					Name:    "blah",
					ID:      fmt.Sprintf("%x", rand.Int()%10000),
					Network: netid,
				},
			},
		}
		server.networks = append(server.networks, &network)
	}
}

func TestListNetworks(t *testing.T) {
	server := DockerServer{}
	addNetworks(&server, 2)
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/networks", nil)
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Errorf("ListNetworks: wrong status. Want %d. Got %d.", http.StatusOK, recorder.Code)
	}
	expected := make([]docker.Network, 2)
	for i, network := range server.networks {
		expected[i] = docker.Network{
			ID:        network.ID,
			Name:      network.Name,
			Type:      network.Type,
			Endpoints: network.Endpoints,
		}
	}
	var got []docker.Network
	err := json.NewDecoder(recorder.Body).Decode(&got)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("ListNetworks. Want %#v. Got %#v.", expected, got)
	}
}

type createNetworkResponse struct {
	ID string `json:"ID"`
}

func TestCreateNetwork(t *testing.T) {
	server := DockerServer{}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	netid := fmt.Sprintf("%x", rand.Int()%10000)
	netname := fmt.Sprintf("%x", rand.Int()%10000)
	body := fmt.Sprintf(`{"ID": "%s", "Name": "%s", "Type": "bridge" }`, netid, netname)
	request, _ := http.NewRequest("POST", "/networks", strings.NewReader(body))
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusCreated {
		t.Errorf("CreateNetwork: wrong status. Want %d. Got %d.", http.StatusCreated, recorder.Code)
	}

	var returned createNetworkResponse
	err := json.NewDecoder(recorder.Body).Decode(&returned)
	if err != nil {
		t.Fatal(err)
	}
	stored := server.networks[0]
	if returned.ID != stored.ID {
		t.Errorf("CreateNetwork: ID mismatch. Stored: %q. Returned: %q.", stored.ID, returned)
	}
}

func TestCreateNetworkInvalidBody(t *testing.T) {
	server := DockerServer{}
	server.buildMuxer()
	recorder := httptest.NewRecorder()
	request, _ := http.NewRequest("POST", "/networks", strings.NewReader("whaaaaaat---"))
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Errorf("CreateNetwork: wrong status. Want %d. Got %d.", http.StatusBadRequest, recorder.Code)
	}
}

func TestCreateNetworkDuplicateName(t *testing.T) {
	server := DockerServer{}
	server.buildMuxer()
	addNetworks(&server, 1)
	server.networks[0].Name = "mynetwork"
	recorder := httptest.NewRecorder()
	body := fmt.Sprintf(`{"ID": "%s", "Name": "mynetwork", "Type": "bridge" }`, fmt.Sprintf("%x", rand.Int()%10000))
	request, _ := http.NewRequest("POST", "/networks", strings.NewReader(body))
	server.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusForbidden {
		t.Errorf("CreateNetwork: wrong status. Want %d. Got %d.", http.StatusForbidden, recorder.Code)
	}
}
