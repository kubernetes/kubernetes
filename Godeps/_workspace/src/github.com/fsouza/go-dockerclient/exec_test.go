// Copyright 2015 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docker

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"testing"
)

func TestExecCreate(t *testing.T) {
	jsonContainer := `{"Id": "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"}`
	var expected struct{ ID string }
	err := json.Unmarshal([]byte(jsonContainer), &expected)
	if err != nil {
		t.Fatal(err)
	}
	fakeRT := &FakeRoundTripper{message: jsonContainer, status: http.StatusOK}
	client := newTestClient(fakeRT)
	config := CreateExecOptions{
		Container:    "test",
		AttachStdin:  true,
		AttachStdout: true,
		AttachStderr: false,
		Tty:          false,
		Cmd:          []string{"touch", "/tmp/file"},
		User:         "a-user",
	}
	execObj, err := client.CreateExec(config)
	if err != nil {
		t.Fatal(err)
	}
	expectedID := "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"
	if execObj.ID != expectedID {
		t.Errorf("ExecCreate: wrong ID. Want %q. Got %q.", expectedID, execObj.ID)
	}
	req := fakeRT.requests[0]
	if req.Method != "POST" {
		t.Errorf("ExecCreate: wrong HTTP method. Want %q. Got %q.", "POST", req.Method)
	}
	expectedURL, _ := url.Parse(client.getURL("/containers/test/exec"))
	if gotPath := req.URL.Path; gotPath != expectedURL.Path {
		t.Errorf("ExecCreate: Wrong path in request. Want %q. Got %q.", expectedURL.Path, gotPath)
	}
	var gotBody struct{ ID string }
	err = json.NewDecoder(req.Body).Decode(&gotBody)
	if err != nil {
		t.Fatal(err)
	}
}

func TestExecStartDetached(t *testing.T) {
	execID := "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"
	fakeRT := &FakeRoundTripper{status: http.StatusOK}
	client := newTestClient(fakeRT)
	config := StartExecOptions{
		Detach: true,
	}
	err := client.StartExec(execID, config)
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	if req.Method != "POST" {
		t.Errorf("ExecStart: wrong HTTP method. Want %q. Got %q.", "POST", req.Method)
	}
	expectedURL, _ := url.Parse(client.getURL("/exec/" + execID + "/start"))
	if gotPath := req.URL.Path; gotPath != expectedURL.Path {
		t.Errorf("ExecCreate: Wrong path in request. Want %q. Got %q.", expectedURL.Path, gotPath)
	}
	t.Log(req.Body)
	var gotBody struct{ Detach bool }
	err = json.NewDecoder(req.Body).Decode(&gotBody)
	if err != nil {
		t.Fatal(err)
	}
	if !gotBody.Detach {
		t.Fatal("Expected Detach in StartExecOptions to be true")
	}
}

func TestExecStartAndAttach(t *testing.T) {
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
	execID := "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"
	opts := StartExecOptions{
		OutputStream: &stdout,
		ErrorStream:  &stderr,
		InputStream:  reader,
		RawTerminal:  true,
		Success:      success,
	}
	go func() {
		if err := client.StartExec(execID, opts); err != nil {
			t.Error(err)
		}
	}()
	<-success
}

func TestExecResize(t *testing.T) {
	execID := "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"
	fakeRT := &FakeRoundTripper{status: http.StatusOK}
	client := newTestClient(fakeRT)
	err := client.ResizeExecTTY(execID, 10, 20)
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	if req.Method != "POST" {
		t.Errorf("ExecStart: wrong HTTP method. Want %q. Got %q.", "POST", req.Method)
	}
	expectedURL, _ := url.Parse(client.getURL("/exec/" + execID + "/resize?h=10&w=20"))
	if gotPath := req.URL.RequestURI(); gotPath != expectedURL.RequestURI() {
		t.Errorf("ExecCreate: Wrong path in request. Want %q. Got %q.", expectedURL.Path, gotPath)
	}
}

func TestExecInspect(t *testing.T) {
	jsonExec := `{
	  "ID": "32adfeeec34250f9530ce1dafd40c6233832315e065ea6b362d745e2f63cde0e",
	  "Running": true,
	  "ExitCode": 0,
	  "ProcessConfig": {
	    "privileged": false,
	    "user": "",
	    "tty": true,
	    "entrypoint": "bash",
	    "arguments": []
	  },
	  "OpenStdin": true,
	  "OpenStderr": true,
	  "OpenStdout": true,
	  "Container": {
	    "State": {
	      "Running": true,
	      "Paused": false,
	      "Restarting": false,
	      "OOMKilled": false,
	      "Pid": 29392,
	      "ExitCode": 0,
	      "Error": "",
	      "StartedAt": "2015-01-21T17:08:59.634662178Z",
	      "FinishedAt": "0001-01-01T00:00:00Z"
	    },
	    "ID": "922cd0568714763dc725b24b7c9801016b2a3de68e2a1dc989bf5abf07740521",
	    "Created": "2015-01-21T17:08:59.46407212Z",
	    "Path": "/bin/bash",
	    "Args": [
	      "-lc",
	      "tsuru_unit_agent http://192.168.50.4:8080 689b30e0ab3adce374346de2e72512138e0e8b75 gtest /var/lib/tsuru/start && tail -f /dev/null"
	    ],
	    "Config": {
	      "Hostname": "922cd0568714",
	      "Domainname": "",
	      "User": "ubuntu",
	      "Memory": 0,
	      "MemorySwap": 0,
	      "CpuShares": 100,
	      "Cpuset": "",
	      "AttachStdin": false,
	      "AttachStdout": false,
	      "AttachStderr": false,
	      "PortSpecs": null,
	      "ExposedPorts": {
	        "8888/tcp": {}
	      },
	      "Tty": false,
	      "OpenStdin": false,
	      "StdinOnce": false,
	      "Env": [
	        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
	      ],
	      "Cmd": [
	        "/bin/bash",
	        "-lc",
	        "tsuru_unit_agent http://192.168.50.4:8080 689b30e0ab3adce374346de2e72512138e0e8b75 gtest /var/lib/tsuru/start && tail -f /dev/null"
	      ],
	      "Image": "tsuru/app-gtest",
	      "Volumes": null,
	      "WorkingDir": "",
	      "Entrypoint": null,
	      "NetworkDisabled": false,
	      "MacAddress": "",
	      "OnBuild": null
	    },
	    "Image": "a88060b8b54fde0f7168c86742d0ce83b80f3f10925d85c98fdad9ed00bef544",
	    "NetworkSettings": {
	      "IPAddress": "172.17.0.8",
	      "IPPrefixLen": 16,
	      "MacAddress": "02:42:ac:11:00:08",
	      "LinkLocalIPv6Address": "fe80::42:acff:fe11:8",
	      "LinkLocalIPv6PrefixLen": 64,
	      "GlobalIPv6Address": "",
	      "GlobalIPv6PrefixLen": 0,
	      "Gateway": "172.17.42.1",
	      "IPv6Gateway": "",
	      "Bridge": "docker0",
	      "PortMapping": null,
	      "Ports": {
	        "8888/tcp": [
	          {
	            "HostIp": "0.0.0.0",
	            "HostPort": "49156"
	          }
	        ]
	      }
	    },
	    "ResolvConfPath": "/var/lib/docker/containers/922cd0568714763dc725b24b7c9801016b2a3de68e2a1dc989bf5abf07740521/resolv.conf",
	    "HostnamePath": "/var/lib/docker/containers/922cd0568714763dc725b24b7c9801016b2a3de68e2a1dc989bf5abf07740521/hostname",
	    "HostsPath": "/var/lib/docker/containers/922cd0568714763dc725b24b7c9801016b2a3de68e2a1dc989bf5abf07740521/hosts",
	    "Name": "/c7e43b72288ee9d0270a",
	    "Driver": "aufs",
	    "ExecDriver": "native-0.2",
	    "MountLabel": "",
	    "ProcessLabel": "",
	    "AppArmorProfile": "",
	    "RestartCount": 0,
	    "UpdateDns": false,
	    "Volumes": {},
	    "VolumesRW": {}
	  }
	}`
	var expected ExecInspect
	err := json.Unmarshal([]byte(jsonExec), &expected)
	if err != nil {
		t.Fatal(err)
	}
	fakeRT := &FakeRoundTripper{message: jsonExec, status: http.StatusOK}
	client := newTestClient(fakeRT)
	expectedID := "32adfeeec34250f9530ce1dafd40c6233832315e065ea6b362d745e2f63cde0e"
	execObj, err := client.InspectExec(expectedID)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(*execObj, expected) {
		t.Errorf("ExecInspect: Expected %#v. Got %#v.", expected, *execObj)
	}
	req := fakeRT.requests[0]
	if req.Method != "GET" {
		t.Errorf("ExecInspect: wrong HTTP method. Want %q. Got %q.", "GET", req.Method)
	}
	expectedURL, _ := url.Parse(client.getURL("/exec/" + expectedID + "/json"))
	if gotPath := fakeRT.requests[0].URL.Path; gotPath != expectedURL.Path {
		t.Errorf("ExecInspect: Wrong path in request. Want %q. Got %q.", expectedURL.Path, gotPath)
	}
}
