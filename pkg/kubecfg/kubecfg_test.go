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

package kubecfg

import (
	"bytes"
	"io"
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

func validateAction(expectedAction, actualAction client.FakeAction, t *testing.T) {
	if expectedAction != actualAction {
		t.Errorf("Unexpected Action: %#v, expected: %#v", actualAction, expectedAction)
	}
}

func TestUpdateWithPods(t *testing.T) {
	fakeClient := client.Fake{
		Pods: api.PodList{
			Items: []api.Pod{
				{JSONBase: api.JSONBase{ID: "pod-1"}},
				{JSONBase: api.JSONBase{ID: "pod-2"}},
			},
		},
	}
	Update("foo", &fakeClient, 0)
	if len(fakeClient.Actions) != 5 {
		t.Errorf("Unexpected action list %#v", fakeClient.Actions)
	}
	validateAction(client.FakeAction{Action: "get-controller", Value: "foo"}, fakeClient.Actions[0], t)
	validateAction(client.FakeAction{Action: "list-pods"}, fakeClient.Actions[1], t)
	// Update deletes the pods, it relies on the replication controller to replace them.
	validateAction(client.FakeAction{Action: "delete-pod", Value: "pod-1"}, fakeClient.Actions[2], t)
	validateAction(client.FakeAction{Action: "delete-pod", Value: "pod-2"}, fakeClient.Actions[3], t)
	validateAction(client.FakeAction{Action: "list-pods"}, fakeClient.Actions[4], t)
}

func TestUpdateNoPods(t *testing.T) {
	fakeClient := client.Fake{}
	Update("foo", &fakeClient, 0)
	if len(fakeClient.Actions) != 2 {
		t.Errorf("Unexpected action list %#v", fakeClient.Actions)
	}
	validateAction(client.FakeAction{Action: "get-controller", Value: "foo"}, fakeClient.Actions[0], t)
	validateAction(client.FakeAction{Action: "list-pods"}, fakeClient.Actions[1], t)
}

func TestRunController(t *testing.T) {
	fakeClient := client.Fake{}
	name := "name"
	image := "foo/bar"
	replicas := 3
	RunController(image, name, replicas, &fakeClient, "8080:80", -1)
	if len(fakeClient.Actions) != 1 || fakeClient.Actions[0].Action != "create-controller" {
		t.Errorf("Unexpected actions: %#v", fakeClient.Actions)
	}
	controller := fakeClient.Actions[0].Value.(api.ReplicationController)
	if controller.ID != name ||
		controller.DesiredState.Replicas != replicas ||
		controller.DesiredState.PodTemplate.DesiredState.Manifest.Containers[0].Image != image {
		t.Errorf("Unexpected controller: %#v", controller)
	}
}

func TestRunControllerWithService(t *testing.T) {
	fakeClient := client.Fake{}
	name := "name"
	image := "foo/bar"
	replicas := 3
	RunController(image, name, replicas, &fakeClient, "", 8000)
	if len(fakeClient.Actions) != 2 ||
		fakeClient.Actions[0].Action != "create-controller" ||
		fakeClient.Actions[1].Action != "create-service" {
		t.Errorf("Unexpected actions: %#v", fakeClient.Actions)
	}
	controller := fakeClient.Actions[0].Value.(api.ReplicationController)
	if controller.ID != name ||
		controller.DesiredState.Replicas != replicas ||
		controller.DesiredState.PodTemplate.DesiredState.Manifest.Containers[0].Image != image {
		t.Errorf("Unexpected controller: %#v", controller)
	}
}

func TestStopController(t *testing.T) {
	fakeClient := client.Fake{}
	name := "name"
	StopController(name, &fakeClient)
	if len(fakeClient.Actions) != 2 {
		t.Errorf("Unexpected actions: %#v", fakeClient.Actions)
	}
	if fakeClient.Actions[0].Action != "get-controller" ||
		fakeClient.Actions[0].Value.(string) != name {
		t.Errorf("Unexpected Action: %#v", fakeClient.Actions[0])
	}
	controller := fakeClient.Actions[1].Value.(api.ReplicationController)
	if fakeClient.Actions[1].Action != "update-controller" ||
		controller.DesiredState.Replicas != 0 {
		t.Errorf("Unexpected Action: %#v", fakeClient.Actions[1])
	}
}

func TestResizeController(t *testing.T) {
	fakeClient := client.Fake{}
	name := "name"
	replicas := 17
	ResizeController(name, replicas, &fakeClient)
	if len(fakeClient.Actions) != 2 {
		t.Errorf("Unexpected actions: %#v", fakeClient.Actions)
	}
	if fakeClient.Actions[0].Action != "get-controller" ||
		fakeClient.Actions[0].Value.(string) != name {
		t.Errorf("Unexpected Action: %#v", fakeClient.Actions[0])
	}
	controller := fakeClient.Actions[1].Value.(api.ReplicationController)
	if fakeClient.Actions[1].Action != "update-controller" ||
		controller.DesiredState.Replicas != 17 {
		t.Errorf("Unexpected Action: %#v", fakeClient.Actions[1])
	}
}

func TestCloudCfgDeleteController(t *testing.T) {
	fakeClient := client.Fake{}
	name := "name"
	err := DeleteController(name, &fakeClient)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(fakeClient.Actions) != 2 {
		t.Errorf("Unexpected actions: %#v", fakeClient.Actions)
	}
	if fakeClient.Actions[0].Action != "get-controller" ||
		fakeClient.Actions[0].Value.(string) != name {
		t.Errorf("Unexpected Action: %#v", fakeClient.Actions[0])
	}
	if fakeClient.Actions[1].Action != "delete-controller" ||
		fakeClient.Actions[1].Value.(string) != name {
		t.Errorf("Unexpected Action: %#v", fakeClient.Actions[1])
	}
}

func TestCloudCfgDeleteControllerWithReplicas(t *testing.T) {
	fakeClient := client.Fake{
		Ctrl: api.ReplicationController{
			DesiredState: api.ReplicationControllerState{
				Replicas: 2,
			},
		},
	}
	name := "name"
	err := DeleteController(name, &fakeClient)
	if len(fakeClient.Actions) != 1 {
		t.Errorf("Unexpected actions: %#v", fakeClient.Actions)
	}
	if fakeClient.Actions[0].Action != "get-controller" ||
		fakeClient.Actions[0].Value.(string) != name {
		t.Errorf("Unexpected Action: %#v", fakeClient.Actions[0])
	}
	if err == nil {
		t.Errorf("Unexpected non-error.")
	}
}

func TestLoadAuthInfo(t *testing.T) {
	loadAuthInfoTests := []struct {
		authData string
		authInfo *client.AuthInfo
		r        io.Reader
	}{
		{
			`{"user": "user", "password": "pass"}`,
			&client.AuthInfo{User: "user", Password: "pass"},
			nil,
		},
		{
			"", nil, nil,
		},
		{
			"missing",
			&client.AuthInfo{User: "user", Password: "pass"},
			bytes.NewBufferString("user\npass"),
		},
	}
	for _, loadAuthInfoTest := range loadAuthInfoTests {
		tt := loadAuthInfoTest
		aifile, err := ioutil.TempFile("", "testAuthInfo")
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if tt.authData != "missing" {
			defer os.Remove(aifile.Name())
			defer aifile.Close()
			_, err = aifile.WriteString(tt.authData)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		} else {
			aifile.Close()
			os.Remove(aifile.Name())
		}
		authInfo, err := LoadAuthInfo(aifile.Name(), tt.r)
		if len(tt.authData) == 0 && tt.authData != "missing" {
			if err == nil {
				t.Error("LoadAuthInfo didn't fail on empty file")
			}
			continue
		}
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if !reflect.DeepEqual(authInfo, tt.authInfo) {
			t.Errorf("Expected %v, got %v", tt.authInfo, authInfo)
		}
	}
}

func TestMakePorts(t *testing.T) {
	var testCases = []struct {
		spec  string
		ports []api.Port
	}{
		{
			"8080:80,8081:8081,443:444",
			[]api.Port{
				{HostPort: 8080, ContainerPort: 80},
				{HostPort: 8081, ContainerPort: 8081},
				{HostPort: 443, ContainerPort: 444},
			},
		},
	}
	for _, tt := range testCases {
		ports := portsFromString(tt.spec)
		if !reflect.DeepEqual(ports, tt.ports) {
			t.Errorf("Expected %#v, got %#v", tt.ports, ports)
		}
	}
}
