// Copyright 2015 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docker

import (
	"bytes"
	"encoding/json"
	"errors"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestStateString(t *testing.T) {
	started := time.Now().Add(-3 * time.Hour)
	var tests = []struct {
		input    State
		expected string
	}{
		{State{Running: true, Paused: true}, "^paused$"},
		{State{Running: true, StartedAt: started}, "^Up 3h.*$"},
		{State{Running: false, ExitCode: 7}, "^Exit 7$"},
	}
	for _, tt := range tests {
		re := regexp.MustCompile(tt.expected)
		if got := tt.input.String(); !re.MatchString(got) {
			t.Errorf("State.String(): wrong result. Want %q. Got %q.", tt.expected, got)
		}
	}
}

func TestListContainers(t *testing.T) {
	jsonContainers := `[
     {
             "Id": "8dfafdbc3a40",
             "Image": "base:latest",
             "Command": "echo 1",
             "Created": 1367854155,
             "Ports":[{"PrivatePort": 2222, "PublicPort": 3333, "Type": "tcp"}],
             "Status": "Exit 0"
     },
     {
             "Id": "9cd87474be90",
             "Image": "base:latest",
             "Command": "echo 222222",
             "Created": 1367854155,
             "Ports":[{"PrivatePort": 2222, "PublicPort": 3333, "Type": "tcp"}],
             "Status": "Exit 0"
     },
     {
             "Id": "3176a2479c92",
             "Image": "base:latest",
             "Command": "echo 3333333333333333",
             "Created": 1367854154,
             "Ports":[{"PrivatePort": 2221, "PublicPort": 3331, "Type": "tcp"}],
             "Status": "Exit 0"
     },
     {
             "Id": "4cb07b47f9fb",
             "Image": "base:latest",
             "Command": "echo 444444444444444444444444444444444",
             "Ports":[{"PrivatePort": 2223, "PublicPort": 3332, "Type": "tcp"}],
             "Created": 1367854152,
             "Status": "Exit 0"
     }
]`
	var expected []APIContainers
	err := json.Unmarshal([]byte(jsonContainers), &expected)
	if err != nil {
		t.Fatal(err)
	}
	client := newTestClient(&FakeRoundTripper{message: jsonContainers, status: http.StatusOK})
	containers, err := client.ListContainers(ListContainersOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(containers, expected) {
		t.Errorf("ListContainers: Expected %#v. Got %#v.", expected, containers)
	}
}

func TestListContainersParams(t *testing.T) {
	var tests = []struct {
		input  ListContainersOptions
		params map[string][]string
	}{
		{ListContainersOptions{}, map[string][]string{}},
		{ListContainersOptions{All: true}, map[string][]string{"all": {"1"}}},
		{ListContainersOptions{All: true, Limit: 10}, map[string][]string{"all": {"1"}, "limit": {"10"}}},
		{
			ListContainersOptions{All: true, Limit: 10, Since: "adf9983", Before: "abdeef"},
			map[string][]string{"all": {"1"}, "limit": {"10"}, "since": {"adf9983"}, "before": {"abdeef"}},
		},
		{
			ListContainersOptions{Filters: map[string][]string{"status": {"paused", "running"}}},
			map[string][]string{"filters": {"{\"status\":[\"paused\",\"running\"]}"}},
		},
		{
			ListContainersOptions{All: true, Filters: map[string][]string{"exited": {"0"}, "status": {"exited"}}},
			map[string][]string{"all": {"1"}, "filters": {"{\"exited\":[\"0\"],\"status\":[\"exited\"]}"}},
		},
	}
	fakeRT := &FakeRoundTripper{message: "[]", status: http.StatusOK}
	client := newTestClient(fakeRT)
	u, _ := url.Parse(client.getURL("/containers/json"))
	for _, tt := range tests {
		if _, err := client.ListContainers(tt.input); err != nil {
			t.Error(err)
		}
		got := map[string][]string(fakeRT.requests[0].URL.Query())
		if !reflect.DeepEqual(got, tt.params) {
			t.Errorf("Expected %#v, got %#v.", tt.params, got)
		}
		if path := fakeRT.requests[0].URL.Path; path != u.Path {
			t.Errorf("Wrong path on request. Want %q. Got %q.", u.Path, path)
		}
		if meth := fakeRT.requests[0].Method; meth != "GET" {
			t.Errorf("Wrong HTTP method. Want GET. Got %s.", meth)
		}
		fakeRT.Reset()
	}
}

func TestListContainersFailure(t *testing.T) {
	var tests = []struct {
		status  int
		message string
	}{
		{400, "bad parameter"},
		{500, "internal server error"},
	}
	for _, tt := range tests {
		client := newTestClient(&FakeRoundTripper{message: tt.message, status: tt.status})
		expected := Error{Status: tt.status, Message: tt.message}
		containers, err := client.ListContainers(ListContainersOptions{})
		if !reflect.DeepEqual(expected, *err.(*Error)) {
			t.Errorf("Wrong error in ListContainers. Want %#v. Got %#v.", expected, err)
		}
		if len(containers) > 0 {
			t.Errorf("ListContainers failure. Expected empty list. Got %#v.", containers)
		}
	}
}

func TestInspectContainer(t *testing.T) {
	jsonContainer := `{
             "Id": "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2",
             "AppArmorProfile": "Profile",
             "Created": "2013-05-07T14:51:42.087658+02:00",
             "Path": "date",
             "Args": [],
             "Config": {
                     "Hostname": "4fa6e0f0c678",
                     "User": "",
                     "Memory": 17179869184,
                     "MemorySwap": 34359738368,
                     "AttachStdin": false,
                     "AttachStdout": true,
                     "AttachStderr": true,
                     "PortSpecs": null,
                     "Tty": false,
                     "OpenStdin": false,
                     "StdinOnce": false,
                     "Env": null,
                     "Cmd": [
                             "date"
                     ],
                     "Image": "base",
                     "Volumes": {},
                     "VolumesFrom": "",
                     "SecurityOpt": [
                         "label:user:USER"
                      ]
             },
             "State": {
                     "Running": false,
                     "Pid": 0,
                     "ExitCode": 0,
                     "StartedAt": "2013-05-07T14:51:42.087658+02:00",
                     "Ghost": false
             },
             "Node": {
                  "ID": "4I4E:QR4I:Z733:QEZK:5X44:Q4T7:W2DD:JRDY:KB2O:PODO:Z5SR:XRB6",
                  "IP": "192.168.99.105",
                  "Addra": "192.168.99.105:2376",
                  "Name": "node-01",
                  "Cpus": 4,
                  "Memory": 1048436736,
                  "Labels": {
                      "executiondriver": "native-0.2",
                      "kernelversion": "3.18.5-tinycore64",
                      "operatingsystem": "Boot2Docker 1.5.0 (TCL 5.4); master : a66bce5 - Tue Feb 10 23:31:27 UTC 2015",
                      "provider": "virtualbox",
                      "storagedriver": "aufs"
                  }
              },
             "Image": "b750fe79269d2ec9a3c593ef05b4332b1d1a02a62b4accb2c21d589ff2f5f2dc",
             "NetworkSettings": {
                     "IpAddress": "",
                     "IpPrefixLen": 0,
                     "Gateway": "",
                     "Bridge": "",
                     "PortMapping": null
             },
             "SysInitPath": "/home/kitty/go/src/github.com/dotcloud/docker/bin/docker",
             "ResolvConfPath": "/etc/resolv.conf",
             "Volumes": {},
             "HostConfig": {
               "Binds": null,
               "ContainerIDFile": "",
               "LxcConf": [],
               "Privileged": false,
               "PortBindings": {
                 "80/tcp": [
                   {
                     "HostIp": "0.0.0.0",
                     "HostPort": "49153"
                   }
                 ]
               },
               "Links": null,
               "PublishAllPorts": false,
               "CgroupParent": "/mesos",
               "Memory": 17179869184,
               "MemorySwap": 34359738368
             }
}`
	var expected Container
	err := json.Unmarshal([]byte(jsonContainer), &expected)
	if err != nil {
		t.Fatal(err)
	}
	fakeRT := &FakeRoundTripper{message: jsonContainer, status: http.StatusOK}
	client := newTestClient(fakeRT)
	id := "4fa6e0f0c678"
	container, err := client.InspectContainer(id)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(*container, expected) {
		t.Errorf("InspectContainer(%q): Expected %#v. Got %#v.", id, expected, container)
	}
	expectedURL, _ := url.Parse(client.getURL("/containers/4fa6e0f0c678/json"))
	if gotPath := fakeRT.requests[0].URL.Path; gotPath != expectedURL.Path {
		t.Errorf("InspectContainer(%q): Wrong path in request. Want %q. Got %q.", id, expectedURL.Path, gotPath)
	}
}

func TestInspectContainerNegativeSwap(t *testing.T) {
	jsonContainer := `{
             "Id": "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2",
             "Created": "2013-05-07T14:51:42.087658+02:00",
             "Path": "date",
             "Args": [],
             "Config": {
                     "Hostname": "4fa6e0f0c678",
                     "User": "",
                     "Memory": 17179869184,
                     "MemorySwap": -1,
                     "AttachStdin": false,
                     "AttachStdout": true,
                     "AttachStderr": true,
                     "PortSpecs": null,
                     "Tty": false,
                     "OpenStdin": false,
                     "StdinOnce": false,
                     "Env": null,
                     "Cmd": [
                             "date"
                     ],
                     "Image": "base",
                     "Volumes": {},
                     "VolumesFrom": ""
             },
             "State": {
                     "Running": false,
                     "Pid": 0,
                     "ExitCode": 0,
                     "StartedAt": "2013-05-07T14:51:42.087658+02:00",
                     "Ghost": false
             },
             "Image": "b750fe79269d2ec9a3c593ef05b4332b1d1a02a62b4accb2c21d589ff2f5f2dc",
             "NetworkSettings": {
                     "IpAddress": "",
                     "IpPrefixLen": 0,
                     "Gateway": "",
                     "Bridge": "",
                     "PortMapping": null
             },
             "SysInitPath": "/home/kitty/go/src/github.com/dotcloud/docker/bin/docker",
             "ResolvConfPath": "/etc/resolv.conf",
             "Volumes": {},
             "HostConfig": {
               "Binds": null,
               "ContainerIDFile": "",
               "LxcConf": [],
               "Privileged": false,
               "PortBindings": {
                 "80/tcp": [
                   {
                     "HostIp": "0.0.0.0",
                     "HostPort": "49153"
                   }
                 ]
               },
               "Links": null,
               "PublishAllPorts": false
             }
}`
	var expected Container
	err := json.Unmarshal([]byte(jsonContainer), &expected)
	if err != nil {
		t.Fatal(err)
	}
	fakeRT := &FakeRoundTripper{message: jsonContainer, status: http.StatusOK}
	client := newTestClient(fakeRT)
	id := "4fa6e0f0c678"
	container, err := client.InspectContainer(id)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(*container, expected) {
		t.Errorf("InspectContainer(%q): Expected %#v. Got %#v.", id, expected, container)
	}
	expectedURL, _ := url.Parse(client.getURL("/containers/4fa6e0f0c678/json"))
	if gotPath := fakeRT.requests[0].URL.Path; gotPath != expectedURL.Path {
		t.Errorf("InspectContainer(%q): Wrong path in request. Want %q. Got %q.", id, expectedURL.Path, gotPath)
	}
}

func TestInspectContainerFailure(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "server error", status: 500})
	expected := Error{Status: 500, Message: "server error"}
	container, err := client.InspectContainer("abe033")
	if container != nil {
		t.Errorf("InspectContainer: Expected <nil> container, got %#v", container)
	}
	if !reflect.DeepEqual(expected, *err.(*Error)) {
		t.Errorf("InspectContainer: Wrong error information. Want %#v. Got %#v.", expected, err)
	}
}

func TestInspectContainerNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such container", status: 404})
	container, err := client.InspectContainer("abe033")
	if container != nil {
		t.Errorf("InspectContainer: Expected <nil> container, got %#v", container)
	}
	expected := &NoSuchContainer{ID: "abe033"}
	if !reflect.DeepEqual(err, expected) {
		t.Errorf("InspectContainer: Wrong error information. Want %#v. Got %#v.", expected, err)
	}
}

func TestContainerChanges(t *testing.T) {
	jsonChanges := `[
     {
             "Path":"/dev",
             "Kind":0
     },
     {
             "Path":"/dev/kmsg",
             "Kind":1
     },
     {
             "Path":"/test",
             "Kind":1
     }
]`
	var expected []Change
	err := json.Unmarshal([]byte(jsonChanges), &expected)
	if err != nil {
		t.Fatal(err)
	}
	fakeRT := &FakeRoundTripper{message: jsonChanges, status: http.StatusOK}
	client := newTestClient(fakeRT)
	id := "4fa6e0f0c678"
	changes, err := client.ContainerChanges(id)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(changes, expected) {
		t.Errorf("ContainerChanges(%q): Expected %#v. Got %#v.", id, expected, changes)
	}
	expectedURL, _ := url.Parse(client.getURL("/containers/4fa6e0f0c678/changes"))
	if gotPath := fakeRT.requests[0].URL.Path; gotPath != expectedURL.Path {
		t.Errorf("ContainerChanges(%q): Wrong path in request. Want %q. Got %q.", id, expectedURL.Path, gotPath)
	}
}

func TestContainerChangesFailure(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "server error", status: 500})
	expected := Error{Status: 500, Message: "server error"}
	changes, err := client.ContainerChanges("abe033")
	if changes != nil {
		t.Errorf("ContainerChanges: Expected <nil> changes, got %#v", changes)
	}
	if !reflect.DeepEqual(expected, *err.(*Error)) {
		t.Errorf("ContainerChanges: Wrong error information. Want %#v. Got %#v.", expected, err)
	}
}

func TestContainerChangesNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such container", status: 404})
	changes, err := client.ContainerChanges("abe033")
	if changes != nil {
		t.Errorf("ContainerChanges: Expected <nil> changes, got %#v", changes)
	}
	expected := &NoSuchContainer{ID: "abe033"}
	if !reflect.DeepEqual(err, expected) {
		t.Errorf("ContainerChanges: Wrong error information. Want %#v. Got %#v.", expected, err)
	}
}

func TestCreateContainer(t *testing.T) {
	jsonContainer := `{
             "Id": "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2",
	     "Warnings": []
}`
	var expected Container
	err := json.Unmarshal([]byte(jsonContainer), &expected)
	if err != nil {
		t.Fatal(err)
	}
	fakeRT := &FakeRoundTripper{message: jsonContainer, status: http.StatusOK}
	client := newTestClient(fakeRT)
	config := Config{AttachStdout: true, AttachStdin: true}
	opts := CreateContainerOptions{Name: "TestCreateContainer", Config: &config}
	container, err := client.CreateContainer(opts)
	if err != nil {
		t.Fatal(err)
	}
	id := "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"
	if container.ID != id {
		t.Errorf("CreateContainer: wrong ID. Want %q. Got %q.", id, container.ID)
	}
	req := fakeRT.requests[0]
	if req.Method != "POST" {
		t.Errorf("CreateContainer: wrong HTTP method. Want %q. Got %q.", "POST", req.Method)
	}
	expectedURL, _ := url.Parse(client.getURL("/containers/create"))
	if gotPath := req.URL.Path; gotPath != expectedURL.Path {
		t.Errorf("CreateContainer: Wrong path in request. Want %q. Got %q.", expectedURL.Path, gotPath)
	}
	var gotBody Config
	err = json.NewDecoder(req.Body).Decode(&gotBody)
	if err != nil {
		t.Fatal(err)
	}
}

func TestCreateContainerImageNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "No such image", status: http.StatusNotFound})
	config := Config{AttachStdout: true, AttachStdin: true}
	container, err := client.CreateContainer(CreateContainerOptions{Config: &config})
	if container != nil {
		t.Errorf("CreateContainer: expected <nil> container, got %#v.", container)
	}
	if !reflect.DeepEqual(err, ErrNoSuchImage) {
		t.Errorf("CreateContainer: Wrong error type. Want %#v. Got %#v.", ErrNoSuchImage, err)
	}
}

func TestCreateContainerWithHostConfig(t *testing.T) {
	fakeRT := &FakeRoundTripper{message: "{}", status: http.StatusOK}
	client := newTestClient(fakeRT)
	config := Config{}
	hostConfig := HostConfig{PublishAllPorts: true}
	opts := CreateContainerOptions{Name: "TestCreateContainerWithHostConfig", Config: &config, HostConfig: &hostConfig}
	_, err := client.CreateContainer(opts)
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	var gotBody map[string]interface{}
	err = json.NewDecoder(req.Body).Decode(&gotBody)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := gotBody["HostConfig"]; !ok {
		t.Errorf("CreateContainer: wrong body. HostConfig was not serialized")
	}
}

func TestStartContainer(t *testing.T) {
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusOK}
	client := newTestClient(fakeRT)
	id := "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"
	err := client.StartContainer(id, &HostConfig{})
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	if req.Method != "POST" {
		t.Errorf("StartContainer(%q): wrong HTTP method. Want %q. Got %q.", id, "POST", req.Method)
	}
	expectedURL, _ := url.Parse(client.getURL("/containers/" + id + "/start"))
	if gotPath := req.URL.Path; gotPath != expectedURL.Path {
		t.Errorf("StartContainer(%q): Wrong path in request. Want %q. Got %q.", id, expectedURL.Path, gotPath)
	}
	expectedContentType := "application/json"
	if contentType := req.Header.Get("Content-Type"); contentType != expectedContentType {
		t.Errorf("StartContainer(%q): Wrong content-type in request. Want %q. Got %q.", id, expectedContentType, contentType)
	}
}

func TestStartContainerNilHostConfig(t *testing.T) {
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusOK}
	client := newTestClient(fakeRT)
	id := "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"
	err := client.StartContainer(id, nil)
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	if req.Method != "POST" {
		t.Errorf("StartContainer(%q): wrong HTTP method. Want %q. Got %q.", id, "POST", req.Method)
	}
	expectedURL, _ := url.Parse(client.getURL("/containers/" + id + "/start"))
	if gotPath := req.URL.Path; gotPath != expectedURL.Path {
		t.Errorf("StartContainer(%q): Wrong path in request. Want %q. Got %q.", id, expectedURL.Path, gotPath)
	}
	expectedContentType := "application/json"
	if contentType := req.Header.Get("Content-Type"); contentType != expectedContentType {
		t.Errorf("StartContainer(%q): Wrong content-type in request. Want %q. Got %q.", id, expectedContentType, contentType)
	}
	var buf [4]byte
	req.Body.Read(buf[:])
	if string(buf[:]) != "null" {
		t.Errorf("Startcontainer(%q): Wrong body. Want null. Got %s", id, buf[:])
	}
}

func TestStartContainerNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such container", status: http.StatusNotFound})
	err := client.StartContainer("a2344", &HostConfig{})
	expected := &NoSuchContainer{ID: "a2344", Err: err.(*NoSuchContainer).Err}
	if !reflect.DeepEqual(err, expected) {
		t.Errorf("StartContainer: Wrong error returned. Want %#v. Got %#v.", expected, err)
	}
}

func TestStartContainerAlreadyRunning(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "container already running", status: http.StatusNotModified})
	err := client.StartContainer("a2334", &HostConfig{})
	expected := &ContainerAlreadyRunning{ID: "a2334"}
	if !reflect.DeepEqual(err, expected) {
		t.Errorf("StartContainer: Wrong error returned. Want %#v. Got %#v.", expected, err)
	}
}

func TestStopContainer(t *testing.T) {
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	id := "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"
	err := client.StopContainer(id, 10)
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	if req.Method != "POST" {
		t.Errorf("StopContainer(%q, 10): wrong HTTP method. Want %q. Got %q.", id, "POST", req.Method)
	}
	expectedURL, _ := url.Parse(client.getURL("/containers/" + id + "/stop"))
	if gotPath := req.URL.Path; gotPath != expectedURL.Path {
		t.Errorf("StopContainer(%q, 10): Wrong path in request. Want %q. Got %q.", id, expectedURL.Path, gotPath)
	}
}

func TestStopContainerNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such container", status: http.StatusNotFound})
	err := client.StopContainer("a2334", 10)
	expected := &NoSuchContainer{ID: "a2334"}
	if !reflect.DeepEqual(err, expected) {
		t.Errorf("StopContainer: Wrong error returned. Want %#v. Got %#v.", expected, err)
	}
}

func TestStopContainerNotRunning(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "container not running", status: http.StatusNotModified})
	err := client.StopContainer("a2334", 10)
	expected := &ContainerNotRunning{ID: "a2334"}
	if !reflect.DeepEqual(err, expected) {
		t.Errorf("StopContainer: Wrong error returned. Want %#v. Got %#v.", expected, err)
	}
}

func TestRestartContainer(t *testing.T) {
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	id := "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"
	err := client.RestartContainer(id, 10)
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	if req.Method != "POST" {
		t.Errorf("RestartContainer(%q, 10): wrong HTTP method. Want %q. Got %q.", id, "POST", req.Method)
	}
	expectedURL, _ := url.Parse(client.getURL("/containers/" + id + "/restart"))
	if gotPath := req.URL.Path; gotPath != expectedURL.Path {
		t.Errorf("RestartContainer(%q, 10): Wrong path in request. Want %q. Got %q.", id, expectedURL.Path, gotPath)
	}
}

func TestRestartContainerNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such container", status: http.StatusNotFound})
	err := client.RestartContainer("a2334", 10)
	expected := &NoSuchContainer{ID: "a2334"}
	if !reflect.DeepEqual(err, expected) {
		t.Errorf("RestartContainer: Wrong error returned. Want %#v. Got %#v.", expected, err)
	}
}

func TestPauseContainer(t *testing.T) {
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	id := "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"
	err := client.PauseContainer(id)
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	if req.Method != "POST" {
		t.Errorf("PauseContainer(%q): wrong HTTP method. Want %q. Got %q.", id, "POST", req.Method)
	}
	expectedURL, _ := url.Parse(client.getURL("/containers/" + id + "/pause"))
	if gotPath := req.URL.Path; gotPath != expectedURL.Path {
		t.Errorf("PauseContainer(%q): Wrong path in request. Want %q. Got %q.", id, expectedURL.Path, gotPath)
	}
}

func TestPauseContainerNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such container", status: http.StatusNotFound})
	err := client.PauseContainer("a2334")
	expected := &NoSuchContainer{ID: "a2334"}
	if !reflect.DeepEqual(err, expected) {
		t.Errorf("PauseContainer: Wrong error returned. Want %#v. Got %#v.", expected, err)
	}
}

func TestUnpauseContainer(t *testing.T) {
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	id := "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"
	err := client.UnpauseContainer(id)
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	if req.Method != "POST" {
		t.Errorf("PauseContainer(%q): wrong HTTP method. Want %q. Got %q.", id, "POST", req.Method)
	}
	expectedURL, _ := url.Parse(client.getURL("/containers/" + id + "/unpause"))
	if gotPath := req.URL.Path; gotPath != expectedURL.Path {
		t.Errorf("PauseContainer(%q): Wrong path in request. Want %q. Got %q.", id, expectedURL.Path, gotPath)
	}
}

func TestUnpauseContainerNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such container", status: http.StatusNotFound})
	err := client.UnpauseContainer("a2334")
	expected := &NoSuchContainer{ID: "a2334"}
	if !reflect.DeepEqual(err, expected) {
		t.Errorf("PauseContainer: Wrong error returned. Want %#v. Got %#v.", expected, err)
	}
}

func TestKillContainer(t *testing.T) {
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	id := "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"
	err := client.KillContainer(KillContainerOptions{ID: id})
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	if req.Method != "POST" {
		t.Errorf("KillContainer(%q): wrong HTTP method. Want %q. Got %q.", id, "POST", req.Method)
	}
	expectedURL, _ := url.Parse(client.getURL("/containers/" + id + "/kill"))
	if gotPath := req.URL.Path; gotPath != expectedURL.Path {
		t.Errorf("KillContainer(%q): Wrong path in request. Want %q. Got %q.", id, expectedURL.Path, gotPath)
	}
}

func TestKillContainerSignal(t *testing.T) {
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	id := "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"
	err := client.KillContainer(KillContainerOptions{ID: id, Signal: SIGTERM})
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	if req.Method != "POST" {
		t.Errorf("KillContainer(%q): wrong HTTP method. Want %q. Got %q.", id, "POST", req.Method)
	}
	if signal := req.URL.Query().Get("signal"); signal != "15" {
		t.Errorf("KillContainer(%q): Wrong query string in request. Want %q. Got %q.", id, "15", signal)
	}
}

func TestKillContainerNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such container", status: http.StatusNotFound})
	err := client.KillContainer(KillContainerOptions{ID: "a2334"})
	expected := &NoSuchContainer{ID: "a2334"}
	if !reflect.DeepEqual(err, expected) {
		t.Errorf("KillContainer: Wrong error returned. Want %#v. Got %#v.", expected, err)
	}
}

func TestRemoveContainer(t *testing.T) {
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusOK}
	client := newTestClient(fakeRT)
	id := "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"
	opts := RemoveContainerOptions{ID: id}
	err := client.RemoveContainer(opts)
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	if req.Method != "DELETE" {
		t.Errorf("RemoveContainer(%q): wrong HTTP method. Want %q. Got %q.", id, "DELETE", req.Method)
	}
	expectedURL, _ := url.Parse(client.getURL("/containers/" + id))
	if gotPath := req.URL.Path; gotPath != expectedURL.Path {
		t.Errorf("RemoveContainer(%q): Wrong path in request. Want %q. Got %q.", id, expectedURL.Path, gotPath)
	}
}

func TestRemoveContainerRemoveVolumes(t *testing.T) {
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusOK}
	client := newTestClient(fakeRT)
	id := "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"
	opts := RemoveContainerOptions{ID: id, RemoveVolumes: true}
	err := client.RemoveContainer(opts)
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	params := map[string][]string(req.URL.Query())
	expected := map[string][]string{"v": {"1"}}
	if !reflect.DeepEqual(params, expected) {
		t.Errorf("RemoveContainer(%q): wrong parameters. Want %#v. Got %#v.", id, expected, params)
	}
}

func TestRemoveContainerNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such container", status: http.StatusNotFound})
	err := client.RemoveContainer(RemoveContainerOptions{ID: "a2334"})
	expected := &NoSuchContainer{ID: "a2334"}
	if !reflect.DeepEqual(err, expected) {
		t.Errorf("RemoveContainer: Wrong error returned. Want %#v. Got %#v.", expected, err)
	}
}

func TestResizeContainerTTY(t *testing.T) {
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusOK}
	client := newTestClient(fakeRT)
	id := "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"
	err := client.ResizeContainerTTY(id, 40, 80)
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	if req.Method != "POST" {
		t.Errorf("ResizeContainerTTY(%q): wrong HTTP method. Want %q. Got %q.", id, "POST", req.Method)
	}
	expectedURL, _ := url.Parse(client.getURL("/containers/" + id + "/resize"))
	if gotPath := req.URL.Path; gotPath != expectedURL.Path {
		t.Errorf("ResizeContainerTTY(%q): Wrong path in request. Want %q. Got %q.", id, expectedURL.Path, gotPath)
	}
	got := map[string][]string(req.URL.Query())
	expectedParams := map[string][]string{
		"w": {"80"},
		"h": {"40"},
	}
	if !reflect.DeepEqual(got, expectedParams) {
		t.Errorf("Expected %#v, got %#v.", expectedParams, got)
	}
}

func TestWaitContainer(t *testing.T) {
	fakeRT := &FakeRoundTripper{message: `{"StatusCode": 56}`, status: http.StatusOK}
	client := newTestClient(fakeRT)
	id := "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"
	status, err := client.WaitContainer(id)
	if err != nil {
		t.Fatal(err)
	}
	if status != 56 {
		t.Errorf("WaitContainer(%q): wrong return. Want 56. Got %d.", id, status)
	}
	req := fakeRT.requests[0]
	if req.Method != "POST" {
		t.Errorf("WaitContainer(%q): wrong HTTP method. Want %q. Got %q.", id, "POST", req.Method)
	}
	expectedURL, _ := url.Parse(client.getURL("/containers/" + id + "/wait"))
	if gotPath := req.URL.Path; gotPath != expectedURL.Path {
		t.Errorf("WaitContainer(%q): Wrong path in request. Want %q. Got %q.", id, expectedURL.Path, gotPath)
	}
}

func TestWaitContainerNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such container", status: http.StatusNotFound})
	_, err := client.WaitContainer("a2334")
	expected := &NoSuchContainer{ID: "a2334"}
	if !reflect.DeepEqual(err, expected) {
		t.Errorf("WaitContainer: Wrong error returned. Want %#v. Got %#v.", expected, err)
	}
}

func TestCommitContainer(t *testing.T) {
	response := `{"Id":"596069db4bf5"}`
	client := newTestClient(&FakeRoundTripper{message: response, status: http.StatusOK})
	id := "596069db4bf5"
	image, err := client.CommitContainer(CommitContainerOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if image.ID != id {
		t.Errorf("CommitContainer: Wrong image id. Want %q. Got %q.", id, image.ID)
	}
}

func TestCommitContainerParams(t *testing.T) {
	cfg := Config{Memory: 67108864}
	json, _ := json.Marshal(&cfg)
	var tests = []struct {
		input  CommitContainerOptions
		params map[string][]string
		body   []byte
	}{
		{CommitContainerOptions{}, map[string][]string{}, nil},
		{CommitContainerOptions{Container: "44c004db4b17"}, map[string][]string{"container": {"44c004db4b17"}}, nil},
		{
			CommitContainerOptions{Container: "44c004db4b17", Repository: "tsuru/python", Message: "something"},
			map[string][]string{"container": {"44c004db4b17"}, "repo": {"tsuru/python"}, "m": {"something"}},
			nil,
		},
		{
			CommitContainerOptions{Container: "44c004db4b17", Run: &cfg},
			map[string][]string{"container": {"44c004db4b17"}},
			json,
		},
	}
	fakeRT := &FakeRoundTripper{message: "{}", status: http.StatusOK}
	client := newTestClient(fakeRT)
	u, _ := url.Parse(client.getURL("/commit"))
	for _, tt := range tests {
		if _, err := client.CommitContainer(tt.input); err != nil {
			t.Error(err)
		}
		got := map[string][]string(fakeRT.requests[0].URL.Query())
		if !reflect.DeepEqual(got, tt.params) {
			t.Errorf("Expected %#v, got %#v.", tt.params, got)
		}
		if path := fakeRT.requests[0].URL.Path; path != u.Path {
			t.Errorf("Wrong path on request. Want %q. Got %q.", u.Path, path)
		}
		if meth := fakeRT.requests[0].Method; meth != "POST" {
			t.Errorf("Wrong HTTP method. Want POST. Got %s.", meth)
		}
		if tt.body != nil {
			if requestBody, err := ioutil.ReadAll(fakeRT.requests[0].Body); err == nil {
				if bytes.Compare(requestBody, tt.body) != 0 {
					t.Errorf("Expected body %#v, got %#v", tt.body, requestBody)
				}
			} else {
				t.Errorf("Error reading request body: %#v", err)
			}
		}
		fakeRT.Reset()
	}
}

func TestCommitContainerFailure(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such container", status: http.StatusInternalServerError})
	_, err := client.CommitContainer(CommitContainerOptions{})
	if err == nil {
		t.Error("Expected non-nil error, got <nil>.")
	}
}

func TestCommitContainerNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such container", status: http.StatusNotFound})
	_, err := client.CommitContainer(CommitContainerOptions{})
	expected := &NoSuchContainer{ID: ""}
	if !reflect.DeepEqual(err, expected) {
		t.Errorf("CommitContainer: Wrong error returned. Want %#v. Got %#v.", expected, err)
	}
}

func TestAttachToContainerLogs(t *testing.T) {
	var req http.Request
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte{1, 0, 0, 0, 0, 0, 0, 19})
		w.Write([]byte("something happened!"))
		req = *r
	}))
	defer server.Close()
	client, _ := NewClient(server.URL)
	client.SkipServerVersionCheck = true
	var buf bytes.Buffer
	opts := AttachToContainerOptions{
		Container:    "a123456",
		OutputStream: &buf,
		Stdout:       true,
		Stderr:       true,
		Logs:         true,
	}
	err := client.AttachToContainer(opts)
	if err != nil {
		t.Fatal(err)
	}
	expected := "something happened!"
	if buf.String() != expected {
		t.Errorf("AttachToContainer for logs: wrong output. Want %q. Got %q.", expected, buf.String())
	}
	if req.Method != "POST" {
		t.Errorf("AttachToContainer: wrong HTTP method. Want POST. Got %s.", req.Method)
	}
	u, _ := url.Parse(client.getURL("/containers/a123456/attach"))
	if req.URL.Path != u.Path {
		t.Errorf("AttachToContainer for logs: wrong HTTP path. Want %q. Got %q.", u.Path, req.URL.Path)
	}
	expectedQs := map[string][]string{
		"logs":   {"1"},
		"stdout": {"1"},
		"stderr": {"1"},
	}
	got := map[string][]string(req.URL.Query())
	if !reflect.DeepEqual(got, expectedQs) {
		t.Errorf("AttachToContainer: wrong query string. Want %#v. Got %#v.", expectedQs, got)
	}
}

func TestAttachToContainer(t *testing.T) {
	var reader = strings.NewReader("send value")
	var req http.Request
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte{1, 0, 0, 0, 0, 0, 0, 5})
		w.Write([]byte("hello"))
		req = *r
	}))
	defer server.Close()
	client, _ := NewClient(server.URL)
	client.SkipServerVersionCheck = true
	var stdout, stderr bytes.Buffer
	opts := AttachToContainerOptions{
		Container:    "a123456",
		OutputStream: &stdout,
		ErrorStream:  &stderr,
		InputStream:  reader,
		Stdin:        true,
		Stdout:       true,
		Stderr:       true,
		Stream:       true,
		RawTerminal:  true,
	}
	err := client.AttachToContainer(opts)
	if err != nil {
		t.Fatal(err)
	}
	expected := map[string][]string{
		"stdin":  {"1"},
		"stdout": {"1"},
		"stderr": {"1"},
		"stream": {"1"},
	}
	got := map[string][]string(req.URL.Query())
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("AttachToContainer: wrong query string. Want %#v. Got %#v.", expected, got)
	}
}

func TestAttachToContainerSentinel(t *testing.T) {
	var reader = strings.NewReader("send value")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte{1, 0, 0, 0, 0, 0, 0, 5})
		w.Write([]byte("hello"))
	}))
	defer server.Close()
	client, _ := NewClient(server.URL)
	client.SkipServerVersionCheck = true
	var stdout, stderr bytes.Buffer
	success := make(chan struct{})
	opts := AttachToContainerOptions{
		Container:    "a123456",
		OutputStream: &stdout,
		ErrorStream:  &stderr,
		InputStream:  reader,
		Stdin:        true,
		Stdout:       true,
		Stderr:       true,
		Stream:       true,
		RawTerminal:  true,
		Success:      success,
	}
	go func() {
		if err := client.AttachToContainer(opts); err != nil {
			t.Error(err)
		}
	}()
	success <- <-success
}

func TestAttachToContainerNilStdout(t *testing.T) {
	var reader = strings.NewReader("send value")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte{1, 0, 0, 0, 0, 0, 0, 5})
		w.Write([]byte("hello"))
	}))
	defer server.Close()
	client, _ := NewClient(server.URL)
	client.SkipServerVersionCheck = true
	var stderr bytes.Buffer
	opts := AttachToContainerOptions{
		Container:    "a123456",
		OutputStream: nil,
		ErrorStream:  &stderr,
		InputStream:  reader,
		Stdin:        true,
		Stdout:       true,
		Stderr:       true,
		Stream:       true,
		RawTerminal:  true,
	}
	err := client.AttachToContainer(opts)
	if err != nil {
		t.Fatal(err)
	}
}

func TestAttachToContainerNilStderr(t *testing.T) {
	var reader = strings.NewReader("send value")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte{1, 0, 0, 0, 0, 0, 0, 5})
		w.Write([]byte("hello"))
	}))
	defer server.Close()
	client, _ := NewClient(server.URL)
	client.SkipServerVersionCheck = true
	var stdout bytes.Buffer
	opts := AttachToContainerOptions{
		Container:    "a123456",
		OutputStream: &stdout,
		InputStream:  reader,
		Stdin:        true,
		Stdout:       true,
		Stderr:       true,
		Stream:       true,
		RawTerminal:  true,
	}
	err := client.AttachToContainer(opts)
	if err != nil {
		t.Fatal(err)
	}
}

func TestAttachToContainerRawTerminalFalse(t *testing.T) {
	input := strings.NewReader("send value")
	var req http.Request
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		req = *r
		w.WriteHeader(http.StatusOK)
		hj, ok := w.(http.Hijacker)
		if !ok {
			t.Fatal("cannot hijack server connection")
		}
		conn, _, err := hj.Hijack()
		if err != nil {
			t.Fatal(err)
		}
		conn.Write([]byte{1, 0, 0, 0, 0, 0, 0, 5})
		conn.Write([]byte("hello"))
		conn.Write([]byte{2, 0, 0, 0, 0, 0, 0, 6})
		conn.Write([]byte("hello!"))
		conn.Close()
	}))
	defer server.Close()
	client, _ := NewClient(server.URL)
	client.SkipServerVersionCheck = true
	var stdout, stderr bytes.Buffer
	opts := AttachToContainerOptions{
		Container:    "a123456",
		OutputStream: &stdout,
		ErrorStream:  &stderr,
		InputStream:  input,
		Stdin:        true,
		Stdout:       true,
		Stderr:       true,
		Stream:       true,
		RawTerminal:  false,
	}
	err := client.AttachToContainer(opts)
	if err != nil {
		t.Fatal(err)
	}
	expected := map[string][]string{
		"stdin":  {"1"},
		"stdout": {"1"},
		"stderr": {"1"},
		"stream": {"1"},
	}
	got := map[string][]string(req.URL.Query())
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("AttachToContainer: wrong query string. Want %#v. Got %#v.", expected, got)
	}
	if stdout.String() != "hello" {
		t.Errorf("AttachToContainer: wrong content written to stdout. Want %q. Got %q.", "hello", stdout.String())
	}
	if stderr.String() != "hello!" {
		t.Errorf("AttachToContainer: wrong content written to stderr. Want %q. Got %q.", "hello!", stderr.String())
	}
}

func TestAttachToContainerWithoutContainer(t *testing.T) {
	var client Client
	err := client.AttachToContainer(AttachToContainerOptions{})
	expected := &NoSuchContainer{ID: ""}
	if !reflect.DeepEqual(err, expected) {
		t.Errorf("AttachToContainer: wrong error. Want %#v. Got %#v.", expected, err)
	}
}

func TestLogs(t *testing.T) {
	var req http.Request
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		prefix := []byte{1, 0, 0, 0, 0, 0, 0, 19}
		w.Write(prefix)
		w.Write([]byte("something happened!"))
		req = *r
	}))
	defer server.Close()
	client, _ := NewClient(server.URL)
	client.SkipServerVersionCheck = true
	var buf bytes.Buffer
	opts := LogsOptions{
		Container:    "a123456",
		OutputStream: &buf,
		Follow:       true,
		Stdout:       true,
		Stderr:       true,
		Timestamps:   true,
	}
	err := client.Logs(opts)
	if err != nil {
		t.Fatal(err)
	}
	expected := "something happened!"
	if buf.String() != expected {
		t.Errorf("Logs: wrong output. Want %q. Got %q.", expected, buf.String())
	}
	if req.Method != "GET" {
		t.Errorf("Logs: wrong HTTP method. Want GET. Got %s.", req.Method)
	}
	u, _ := url.Parse(client.getURL("/containers/a123456/logs"))
	if req.URL.Path != u.Path {
		t.Errorf("AttachToContainer for logs: wrong HTTP path. Want %q. Got %q.", u.Path, req.URL.Path)
	}
	expectedQs := map[string][]string{
		"follow":     {"1"},
		"stdout":     {"1"},
		"stderr":     {"1"},
		"timestamps": {"1"},
		"tail":       {"all"},
	}
	got := map[string][]string(req.URL.Query())
	if !reflect.DeepEqual(got, expectedQs) {
		t.Errorf("Logs: wrong query string. Want %#v. Got %#v.", expectedQs, got)
	}
}

func TestLogsNilStdoutDoesntFail(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		prefix := []byte{1, 0, 0, 0, 0, 0, 0, 19}
		w.Write(prefix)
		w.Write([]byte("something happened!"))
	}))
	defer server.Close()
	client, _ := NewClient(server.URL)
	client.SkipServerVersionCheck = true
	opts := LogsOptions{
		Container:  "a123456",
		Follow:     true,
		Stdout:     true,
		Stderr:     true,
		Timestamps: true,
	}
	err := client.Logs(opts)
	if err != nil {
		t.Fatal(err)
	}
}

func TestLogsNilStderrDoesntFail(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		prefix := []byte{2, 0, 0, 0, 0, 0, 0, 19}
		w.Write(prefix)
		w.Write([]byte("something happened!"))
	}))
	defer server.Close()
	client, _ := NewClient(server.URL)
	client.SkipServerVersionCheck = true
	opts := LogsOptions{
		Container:  "a123456",
		Follow:     true,
		Stdout:     true,
		Stderr:     true,
		Timestamps: true,
	}
	err := client.Logs(opts)
	if err != nil {
		t.Fatal(err)
	}
}

func TestLogsSpecifyingTail(t *testing.T) {
	var req http.Request
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		prefix := []byte{1, 0, 0, 0, 0, 0, 0, 19}
		w.Write(prefix)
		w.Write([]byte("something happened!"))
		req = *r
	}))
	defer server.Close()
	client, _ := NewClient(server.URL)
	client.SkipServerVersionCheck = true
	var buf bytes.Buffer
	opts := LogsOptions{
		Container:    "a123456",
		OutputStream: &buf,
		Follow:       true,
		Stdout:       true,
		Stderr:       true,
		Timestamps:   true,
		Tail:         "100",
	}
	err := client.Logs(opts)
	if err != nil {
		t.Fatal(err)
	}
	expected := "something happened!"
	if buf.String() != expected {
		t.Errorf("Logs: wrong output. Want %q. Got %q.", expected, buf.String())
	}
	if req.Method != "GET" {
		t.Errorf("Logs: wrong HTTP method. Want GET. Got %s.", req.Method)
	}
	u, _ := url.Parse(client.getURL("/containers/a123456/logs"))
	if req.URL.Path != u.Path {
		t.Errorf("AttachToContainer for logs: wrong HTTP path. Want %q. Got %q.", u.Path, req.URL.Path)
	}
	expectedQs := map[string][]string{
		"follow":     {"1"},
		"stdout":     {"1"},
		"stderr":     {"1"},
		"timestamps": {"1"},
		"tail":       {"100"},
	}
	got := map[string][]string(req.URL.Query())
	if !reflect.DeepEqual(got, expectedQs) {
		t.Errorf("Logs: wrong query string. Want %#v. Got %#v.", expectedQs, got)
	}
}

func TestLogsRawTerminal(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("something happened!"))
	}))
	defer server.Close()
	client, _ := NewClient(server.URL)
	client.SkipServerVersionCheck = true
	var buf bytes.Buffer
	opts := LogsOptions{
		Container:    "a123456",
		OutputStream: &buf,
		Follow:       true,
		RawTerminal:  true,
		Stdout:       true,
		Stderr:       true,
		Timestamps:   true,
		Tail:         "100",
	}
	err := client.Logs(opts)
	if err != nil {
		t.Fatal(err)
	}
	expected := "something happened!"
	if buf.String() != expected {
		t.Errorf("Logs: wrong output. Want %q. Got %q.", expected, buf.String())
	}
}

func TestLogsNoContainer(t *testing.T) {
	var client Client
	err := client.Logs(LogsOptions{})
	expected := &NoSuchContainer{ID: ""}
	if !reflect.DeepEqual(err, expected) {
		t.Errorf("AttachToContainer: wrong error. Want %#v. Got %#v.", expected, err)
	}
}

func TestNoSuchContainerError(t *testing.T) {
	var err = &NoSuchContainer{ID: "i345"}
	expected := "No such container: i345"
	if got := err.Error(); got != expected {
		t.Errorf("NoSuchContainer: wrong message. Want %q. Got %q.", expected, got)
	}
}

func TestNoSuchContainerErrorMessage(t *testing.T) {
	var err = &NoSuchContainer{ID: "i345", Err: errors.New("some advanced error info")}
	expected := "some advanced error info"
	if got := err.Error(); got != expected {
		t.Errorf("NoSuchContainer: wrong message. Want %q. Got %q.", expected, got)
	}
}

func TestExportContainer(t *testing.T) {
	content := "exported container tar content"
	out := stdoutMock{bytes.NewBufferString(content)}
	client := newTestClient(&FakeRoundTripper{status: http.StatusOK})
	opts := ExportContainerOptions{ID: "4fa6e0f0c678", OutputStream: out}
	err := client.ExportContainer(opts)
	if err != nil {
		t.Errorf("ExportContainer: caugh error %#v while exporting container, expected nil", err.Error())
	}
	if out.String() != content {
		t.Errorf("ExportContainer: wrong stdout. Want %#v. Got %#v.", content, out.String())
	}
}

func TestExportContainerViaUnixSocket(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("skipping test on %q", runtime.GOOS)
	}
	content := "exported container tar content"
	var buf []byte
	out := bytes.NewBuffer(buf)
	tempSocket := tempfile("export_socket")
	defer os.Remove(tempSocket)
	endpoint := "unix://" + tempSocket
	u, _ := parseEndpoint(endpoint, false)
	client := Client{
		HTTPClient:             http.DefaultClient,
		endpoint:               endpoint,
		endpointURL:            u,
		SkipServerVersionCheck: true,
	}
	listening := make(chan string)
	done := make(chan int)
	go runStreamConnServer(t, "unix", tempSocket, listening, done)
	<-listening // wait for server to start
	opts := ExportContainerOptions{ID: "4fa6e0f0c678", OutputStream: out}
	err := client.ExportContainer(opts)
	<-done // make sure server stopped
	if err != nil {
		t.Errorf("ExportContainer: caugh error %#v while exporting container, expected nil", err.Error())
	}
	if out.String() != content {
		t.Errorf("ExportContainer: wrong stdout. Want %#v. Got %#v.", content, out.String())
	}
}

func runStreamConnServer(t *testing.T, network, laddr string, listening chan<- string, done chan<- int) {
	defer close(done)
	l, err := net.Listen(network, laddr)
	if err != nil {
		t.Errorf("Listen(%q, %q) failed: %v", network, laddr, err)
		listening <- "<nil>"
		return
	}
	defer l.Close()
	listening <- l.Addr().String()
	c, err := l.Accept()
	if err != nil {
		t.Logf("Accept failed: %v", err)
		return
	}
	c.Write([]byte("HTTP/1.1 200 OK\n\nexported container tar content"))
	c.Close()
}

func tempfile(filename string) string {
	return os.TempDir() + "/" + filename + "." + strconv.Itoa(os.Getpid())
}

func TestExportContainerNoId(t *testing.T) {
	client := Client{}
	out := stdoutMock{bytes.NewBufferString("")}
	err := client.ExportContainer(ExportContainerOptions{OutputStream: out})
	e, ok := err.(*NoSuchContainer)
	if !ok {
		t.Errorf("ExportContainer: wrong error. Want NoSuchContainer. Got %#v.", e)
	}
	if e.ID != "" {
		t.Errorf("ExportContainer: wrong ID. Want %q. Got %q", "", e.ID)
	}
}

func TestCopyFromContainer(t *testing.T) {
	content := "File content"
	out := stdoutMock{bytes.NewBufferString(content)}
	client := newTestClient(&FakeRoundTripper{status: http.StatusOK})
	opts := CopyFromContainerOptions{
		Container:    "a123456",
		OutputStream: out,
	}
	err := client.CopyFromContainer(opts)
	if err != nil {
		t.Errorf("CopyFromContainer: caugh error %#v while copying from container, expected nil", err.Error())
	}
	if out.String() != content {
		t.Errorf("CopyFromContainer: wrong stdout. Want %#v. Got %#v.", content, out.String())
	}
}

func TestCopyFromContainerEmptyContainer(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{status: http.StatusOK})
	err := client.CopyFromContainer(CopyFromContainerOptions{})
	_, ok := err.(*NoSuchContainer)
	if !ok {
		t.Errorf("CopyFromContainer: invalid error returned. Want NoSuchContainer, got %#v.", err)
	}
}

func TestPassingNameOptToCreateContainerReturnsItInContainer(t *testing.T) {
	jsonContainer := `{
             "Id": "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2",
	     "Warnings": []
}`
	fakeRT := &FakeRoundTripper{message: jsonContainer, status: http.StatusOK}
	client := newTestClient(fakeRT)
	config := Config{AttachStdout: true, AttachStdin: true}
	opts := CreateContainerOptions{Name: "TestCreateContainer", Config: &config}
	container, err := client.CreateContainer(opts)
	if err != nil {
		t.Fatal(err)
	}
	if container.Name != "TestCreateContainer" {
		t.Errorf("Container name expected to be TestCreateContainer, was %s", container.Name)
	}
}

func TestAlwaysRestart(t *testing.T) {
	policy := AlwaysRestart()
	if policy.Name != "always" {
		t.Errorf("AlwaysRestart(): wrong policy name. Want %q. Got %q", "always", policy.Name)
	}
	if policy.MaximumRetryCount != 0 {
		t.Errorf("AlwaysRestart(): wrong MaximumRetryCount. Want 0. Got %d", policy.MaximumRetryCount)
	}
}

func TestRestartOnFailure(t *testing.T) {
	const retry = 5
	policy := RestartOnFailure(retry)
	if policy.Name != "on-failure" {
		t.Errorf("RestartOnFailure(%d): wrong policy name. Want %q. Got %q", retry, "on-failure", policy.Name)
	}
	if policy.MaximumRetryCount != retry {
		t.Errorf("RestartOnFailure(%d): wrong MaximumRetryCount. Want %d. Got %d", retry, retry, policy.MaximumRetryCount)
	}
}

func TestNeverRestart(t *testing.T) {
	policy := NeverRestart()
	if policy.Name != "no" {
		t.Errorf("NeverRestart(): wrong policy name. Want %q. Got %q", "always", policy.Name)
	}
	if policy.MaximumRetryCount != 0 {
		t.Errorf("NeverRestart(): wrong MaximumRetryCount. Want 0. Got %d", policy.MaximumRetryCount)
	}
}

func TestTopContainer(t *testing.T) {
	jsonTop := `{
  "Processes": [
    [
      "ubuntu",
      "3087",
      "815",
      "0",
      "01:44",
      "?",
      "00:00:00",
      "cmd1"
    ],
    [
      "root",
      "3158",
      "3087",
      "0",
      "01:44",
      "?",
      "00:00:01",
      "cmd2"
    ]
  ],
  "Titles": [
    "UID",
    "PID",
    "PPID",
    "C",
    "STIME",
    "TTY",
    "TIME",
    "CMD"
  ]
}`
	var expected TopResult
	err := json.Unmarshal([]byte(jsonTop), &expected)
	if err != nil {
		t.Fatal(err)
	}
	id := "4fa6e0f0"
	fakeRT := &FakeRoundTripper{message: jsonTop, status: http.StatusOK}
	client := newTestClient(fakeRT)
	processes, err := client.TopContainer(id, "")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(processes, expected) {
		t.Errorf("TopContainer: Expected %#v. Got %#v.", expected, processes)
	}
	if len(processes.Processes) != 2 || len(processes.Processes[0]) != 8 ||
		processes.Processes[0][7] != "cmd1" {
		t.Errorf("TopContainer: Process list to include cmd1. Got %#v.", processes)
	}
	expectedURI := "/containers/" + id + "/top"
	if !strings.HasSuffix(fakeRT.requests[0].URL.String(), expectedURI) {
		t.Errorf("TopContainer: Expected URI to have %q. Got %q.", expectedURI, fakeRT.requests[0].URL.String())
	}
}

func TestTopContainerNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such container", status: http.StatusNotFound})
	_, err := client.TopContainer("abef348", "")
	expected := &NoSuchContainer{ID: "abef348"}
	if !reflect.DeepEqual(err, expected) {
		t.Errorf("StopContainer: Wrong error returned. Want %#v. Got %#v.", expected, err)
	}
}

func TestTopContainerWithPsArgs(t *testing.T) {
	fakeRT := &FakeRoundTripper{message: "no such container", status: http.StatusNotFound}
	client := newTestClient(fakeRT)
	expectedErr := &NoSuchContainer{ID: "abef348"}
	if _, err := client.TopContainer("abef348", "aux"); !reflect.DeepEqual(expectedErr, err) {
		t.Errorf("TopContainer: Expected %v. Got %v.", expectedErr, err)
	}
	expectedURI := "/containers/abef348/top?ps_args=aux"
	if !strings.HasSuffix(fakeRT.requests[0].URL.String(), expectedURI) {
		t.Errorf("TopContainer: Expected URI to have %q. Got %q.", expectedURI, fakeRT.requests[0].URL.String())
	}
}

func TestStats(t *testing.T) {
	jsonStats1 := `{
       "read" : "2015-01-08T22:57:31.547920715Z",
       "network" : {
          "rx_dropped" : 0,
          "rx_bytes" : 648,
          "rx_errors" : 0,
          "tx_packets" : 8,
          "tx_dropped" : 0,
          "rx_packets" : 8,
          "tx_errors" : 0,
          "tx_bytes" : 648
       },
       "memory_stats" : {
          "stats" : {
             "total_pgmajfault" : 0,
             "cache" : 0,
             "mapped_file" : 0,
             "total_inactive_file" : 0,
             "pgpgout" : 414,
             "rss" : 6537216,
             "total_mapped_file" : 0,
             "writeback" : 0,
             "unevictable" : 0,
             "pgpgin" : 477,
             "total_unevictable" : 0,
             "pgmajfault" : 0,
             "total_rss" : 6537216,
             "total_rss_huge" : 6291456,
             "total_writeback" : 0,
             "total_inactive_anon" : 0,
             "rss_huge" : 6291456,
	     "hierarchical_memory_limit": 189204833,
             "total_pgfault" : 964,
             "total_active_file" : 0,
             "active_anon" : 6537216,
             "total_active_anon" : 6537216,
             "total_pgpgout" : 414,
             "total_cache" : 0,
             "inactive_anon" : 0,
             "active_file" : 0,
             "pgfault" : 964,
             "inactive_file" : 0,
             "total_pgpgin" : 477
          },
          "max_usage" : 6651904,
          "usage" : 6537216,
          "failcnt" : 0,
          "limit" : 67108864
       },
       "blkio_stats": {
          "io_service_bytes_recursive": [
             {
                "major": 8,
                "minor": 0,
                "op": "Read",
                "value": 428795731968
             },
             {
                "major": 8,
                "minor": 0,
                "op": "Write",
                "value": 388177920
             }
          ],
          "io_serviced_recursive": [
             {
                "major": 8,
                "minor": 0,
                "op": "Read",
                "value": 25994442
             },
             {
                "major": 8,
                "minor": 0,
                "op": "Write",
                "value": 1734
             }
          ],
          "io_queue_recursive": [],
          "io_service_time_recursive": [],
          "io_wait_time_recursive": [],
          "io_merged_recursive": [],
          "io_time_recursive": [],
          "sectors_recursive": []
       },
       "cpu_stats" : {
          "cpu_usage" : {
             "percpu_usage" : [
                16970827,
                1839451,
                7107380,
                10571290
             ],
             "usage_in_usermode" : 10000000,
             "total_usage" : 36488948,
             "usage_in_kernelmode" : 20000000
          },
          "system_cpu_usage" : 20091722000000000
       }
    }`
	// 1 second later, cache is 100
	jsonStats2 := `{
       "read" : "2015-01-08T22:57:32.547920715Z",
       "network" : {
          "rx_dropped" : 0,
          "rx_bytes" : 648,
          "rx_errors" : 0,
          "tx_packets" : 8,
          "tx_dropped" : 0,
          "rx_packets" : 8,
          "tx_errors" : 0,
          "tx_bytes" : 648
       },
       "memory_stats" : {
          "stats" : {
             "total_pgmajfault" : 0,
             "cache" : 100,
             "mapped_file" : 0,
             "total_inactive_file" : 0,
             "pgpgout" : 414,
             "rss" : 6537216,
             "total_mapped_file" : 0,
             "writeback" : 0,
             "unevictable" : 0,
             "pgpgin" : 477,
             "total_unevictable" : 0,
             "pgmajfault" : 0,
             "total_rss" : 6537216,
             "total_rss_huge" : 6291456,
             "total_writeback" : 0,
             "total_inactive_anon" : 0,
             "rss_huge" : 6291456,
             "total_pgfault" : 964,
             "total_active_file" : 0,
             "active_anon" : 6537216,
             "total_active_anon" : 6537216,
             "total_pgpgout" : 414,
             "total_cache" : 0,
             "inactive_anon" : 0,
             "active_file" : 0,
             "pgfault" : 964,
             "inactive_file" : 0,
             "total_pgpgin" : 477
          },
          "max_usage" : 6651904,
          "usage" : 6537216,
          "failcnt" : 0,
          "limit" : 67108864
       },
       "blkio_stats": {
          "io_service_bytes_recursive": [
             {
                "major": 8,
                "minor": 0,
                "op": "Read",
                "value": 428795731968
             },
             {
                "major": 8,
                "minor": 0,
                "op": "Write",
                "value": 388177920
             }
          ],
          "io_serviced_recursive": [
             {
                "major": 8,
                "minor": 0,
                "op": "Read",
                "value": 25994442
             },
             {
                "major": 8,
                "minor": 0,
                "op": "Write",
                "value": 1734
             }
          ],
          "io_queue_recursive": [],
          "io_service_time_recursive": [],
          "io_wait_time_recursive": [],
          "io_merged_recursive": [],
          "io_time_recursive": [],
          "sectors_recursive": []
       },
       "cpu_stats" : {
          "cpu_usage" : {
             "percpu_usage" : [
                16970827,
                1839451,
                7107380,
                10571290
             ],
             "usage_in_usermode" : 10000000,
             "total_usage" : 36488948,
             "usage_in_kernelmode" : 20000000
          },
          "system_cpu_usage" : 20091722000000000
       }
    }`
	var expected1 Stats
	var expected2 Stats
	err := json.Unmarshal([]byte(jsonStats1), &expected1)
	if err != nil {
		t.Fatal(err)
	}
	err = json.Unmarshal([]byte(jsonStats2), &expected2)
	if err != nil {
		t.Fatal(err)
	}
	id := "4fa6e0f0"

	var req http.Request
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(jsonStats1))
		w.Write([]byte(jsonStats2))
		req = *r
	}))
	defer server.Close()
	client, _ := NewClient(server.URL)
	client.SkipServerVersionCheck = true
	errC := make(chan error, 1)
	statsC := make(chan *Stats)
	go func() {
		errC <- client.Stats(StatsOptions{id, statsC})
		close(errC)
	}()
	var resultStats []*Stats
	for {
		stats, ok := <-statsC
		if !ok {
			break
		}
		resultStats = append(resultStats, stats)
	}
	err = <-errC
	if err != nil {
		t.Fatal(err)
	}
	if len(resultStats) != 2 {
		t.Fatalf("Stats: Expected 2 results. Got %d.", len(resultStats))
	}
	if !reflect.DeepEqual(resultStats[0], &expected1) {
		t.Errorf("Stats: Expected:\n%+v\nGot:\n%+v", expected1, resultStats[0])
	}
	if !reflect.DeepEqual(resultStats[1], &expected2) {
		t.Errorf("Stats: Expected:\n%+v\nGot:\n%+v", expected2, resultStats[1])
	}
	if req.Method != "GET" {
		t.Errorf("Stats: wrong HTTP method. Want GET. Got %s.", req.Method)
	}
	u, _ := url.Parse(client.getURL("/containers/" + id + "/stats"))
	if req.URL.Path != u.Path {
		t.Errorf("Stats: wrong HTTP path. Want %q. Got %q.", u.Path, req.URL.Path)
	}
}

func TestStatsContainerNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such container", status: http.StatusNotFound})
	statsC := make(chan *Stats)
	err := client.Stats(StatsOptions{"abef348", statsC})
	expected := &NoSuchContainer{ID: "abef348"}
	if !reflect.DeepEqual(err, expected) {
		t.Errorf("Stats: Wrong error returned. Want %#v. Got %#v.", expected, err)
	}
}

func TestRenameContainer(t *testing.T) {
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusOK}
	client := newTestClient(fakeRT)
	opts := RenameContainerOptions{ID: "something_old", Name: "something_new"}
	err := client.RenameContainer(opts)
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	if req.Method != "POST" {
		t.Errorf("RenameContainer: wrong HTTP method. Want %q. Got %q.", "POST", req.Method)
	}
	expectedURL, _ := url.Parse(client.getURL("/containers/something_old/rename?name=something_new"))
	if gotPath := req.URL.Path; gotPath != expectedURL.Path {
		t.Errorf("RenameContainer: Wrong path in request. Want %q. Got %q.", expectedURL.Path, gotPath)
	}
	expectedValues := expectedURL.Query()["name"]
	actualValues := req.URL.Query()["name"]
	if len(actualValues) != 1 || expectedValues[0] != actualValues[0] {
		t.Errorf("RenameContainer: Wrong params in request. Want %q. Got %q.", expectedValues, actualValues)
	}
}
