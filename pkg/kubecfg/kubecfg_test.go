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
	if !reflect.DeepEqual(expectedAction, actualAction) {
		t.Errorf("Unexpected Action: %#v, expected: %#v", actualAction, expectedAction)
	}
}

func TestUpdateWithPods(t *testing.T) {
	fakeClient := client.Fake{
		PodsList: api.PodList{
			Items: []api.Pod{
				{ObjectMeta: api.ObjectMeta{Name: "pod-1"}},
				{ObjectMeta: api.ObjectMeta{Name: "pod-2"}},
			},
		},
	}
	Update(api.NewDefaultContext(), "foo", &fakeClient, 0, "")
	if len(fakeClient.Actions) != 5 {
		t.Fatalf("Unexpected action list %#v", fakeClient.Actions)
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
	Update(api.NewDefaultContext(), "foo", &fakeClient, 0, "")
	if len(fakeClient.Actions) != 2 {
		t.Errorf("Unexpected action list %#v", fakeClient.Actions)
	}
	validateAction(client.FakeAction{Action: "get-controller", Value: "foo"}, fakeClient.Actions[0], t)
	validateAction(client.FakeAction{Action: "list-pods"}, fakeClient.Actions[1], t)
}

func TestUpdateWithNewImage(t *testing.T) {
	fakeClient := client.Fake{
		PodsList: api.PodList{
			Items: []api.Pod{
				{ObjectMeta: api.ObjectMeta{Name: "pod-1"}},
				{ObjectMeta: api.ObjectMeta{Name: "pod-2"}},
			},
		},
		Ctrl: api.ReplicationController{
			DesiredState: api.ReplicationControllerState{
				PodTemplate: api.PodTemplate{
					DesiredState: api.PodState{
						Manifest: api.ContainerManifest{
							Containers: []api.Container{
								{Image: "fooImage:1"},
							},
						},
					},
				},
			},
		},
	}
	Update(api.NewDefaultContext(), "foo", &fakeClient, 0, "fooImage:2")
	if len(fakeClient.Actions) != 6 {
		t.Errorf("Unexpected action list %#v", fakeClient.Actions)
	}
	validateAction(client.FakeAction{Action: "get-controller", Value: "foo"}, fakeClient.Actions[0], t)

	newCtrl := api.Scheme.CopyOrDie(&fakeClient.Ctrl).(*api.ReplicationController)
	newCtrl.DesiredState.PodTemplate.DesiredState.Manifest.Containers[0].Image = "fooImage:2"
	validateAction(client.FakeAction{Action: "update-controller", Value: newCtrl}, fakeClient.Actions[1], t)

	validateAction(client.FakeAction{Action: "list-pods"}, fakeClient.Actions[2], t)
	// Update deletes the pods, it relies on the replication controller to replace them.
	validateAction(client.FakeAction{Action: "delete-pod", Value: "pod-1"}, fakeClient.Actions[3], t)
	validateAction(client.FakeAction{Action: "delete-pod", Value: "pod-2"}, fakeClient.Actions[4], t)
	validateAction(client.FakeAction{Action: "list-pods"}, fakeClient.Actions[5], t)
}

func TestRunController(t *testing.T) {
	fakeClient := client.Fake{}
	name := "name"
	image := "foo/bar"
	replicas := 3
	RunController(api.NewDefaultContext(), image, name, replicas, &fakeClient, "8080:80", -1)
	if len(fakeClient.Actions) != 1 || fakeClient.Actions[0].Action != "create-controller" {
		t.Errorf("Unexpected actions: %#v", fakeClient.Actions)
	}
	controller := fakeClient.Actions[0].Value.(*api.ReplicationController)
	if controller.Name != name ||
		controller.DesiredState.Replicas != replicas ||
		controller.DesiredState.PodTemplate.DesiredState.Manifest.Containers[0].Image != image {
		t.Errorf("Unexpected controller: %#v", controller)
	}
}

func TestRunControllerWithWrongArgs(t *testing.T) {
	fakeClient := client.Fake{}
	name := "name"
	image := "foo/bar"
	replicas := 3
	err := RunController(api.NewDefaultContext(), image, name, replicas, &fakeClient, "8080:", -1)
	if err == nil {
		t.Errorf("Unexpected non-error: %#v", fakeClient.Actions)
	}
	RunController(api.NewDefaultContext(), image, name, replicas, &fakeClient, "8080:80", -1)
	if len(fakeClient.Actions) != 1 || fakeClient.Actions[0].Action != "create-controller" {
		t.Errorf("Unexpected actions: %#v", fakeClient.Actions)
	}
	controller := fakeClient.Actions[0].Value.(*api.ReplicationController)
	if controller.Name != name ||
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
	RunController(api.NewDefaultContext(), image, name, replicas, &fakeClient, "", 8000)
	if len(fakeClient.Actions) != 2 ||
		fakeClient.Actions[0].Action != "create-controller" ||
		fakeClient.Actions[1].Action != "create-service" {
		t.Errorf("Unexpected actions: %#v", fakeClient.Actions)
	}
	controller := fakeClient.Actions[0].Value.(*api.ReplicationController)
	if controller.Name != name ||
		controller.DesiredState.Replicas != replicas ||
		controller.DesiredState.PodTemplate.DesiredState.Manifest.Containers[0].Image != image {
		t.Errorf("Unexpected controller: %#v", controller)
	}
}

func TestStopController(t *testing.T) {
	fakeClient := client.Fake{}
	name := "name"
	StopController(api.NewDefaultContext(), name, &fakeClient)
	if len(fakeClient.Actions) != 2 {
		t.Errorf("Unexpected actions: %#v", fakeClient.Actions)
	}
	if fakeClient.Actions[0].Action != "get-controller" ||
		fakeClient.Actions[0].Value.(string) != name {
		t.Errorf("Unexpected Action: %#v", fakeClient.Actions[0])
	}
	controller := fakeClient.Actions[1].Value.(*api.ReplicationController)
	if fakeClient.Actions[1].Action != "update-controller" ||
		controller.DesiredState.Replicas != 0 {
		t.Errorf("Unexpected Action: %#v", fakeClient.Actions[1])
	}
}

func TestResizeController(t *testing.T) {
	fakeClient := client.Fake{}
	name := "name"
	replicas := 17
	ResizeController(api.NewDefaultContext(), name, replicas, &fakeClient)
	if len(fakeClient.Actions) != 2 {
		t.Errorf("Unexpected actions: %#v", fakeClient.Actions)
	}
	if fakeClient.Actions[0].Action != "get-controller" ||
		fakeClient.Actions[0].Value.(string) != name {
		t.Errorf("Unexpected Action: %#v", fakeClient.Actions[0])
	}
	controller := fakeClient.Actions[1].Value.(*api.ReplicationController)
	if fakeClient.Actions[1].Action != "update-controller" ||
		controller.DesiredState.Replicas != 17 {
		t.Errorf("Unexpected Action: %#v", fakeClient.Actions[1])
	}
}

func TestCloudCfgDeleteController(t *testing.T) {
	fakeClient := client.Fake{}
	name := "name"
	err := DeleteController(api.NewDefaultContext(), name, &fakeClient)
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
	err := DeleteController(api.NewDefaultContext(), name, &fakeClient)
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

func TestLoadNamespaceInfo(t *testing.T) {
	loadNamespaceInfoTests := []struct {
		nsData string
		nsInfo *NamespaceInfo
	}{
		{
			`{"Namespace":"test"}`,
			&NamespaceInfo{Namespace: "test"},
		},
		{
			"", nil,
		},
		{
			"missing",
			&NamespaceInfo{Namespace: "default"},
		},
	}
	for _, loadNamespaceInfoTest := range loadNamespaceInfoTests {
		tt := loadNamespaceInfoTest
		nsfile, err := ioutil.TempFile("", "testNamespaceInfo")
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if tt.nsData != "missing" {
			defer os.Remove(nsfile.Name())
			defer nsfile.Close()
			_, err := nsfile.WriteString(tt.nsData)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		} else {
			nsfile.Close()
			os.Remove(nsfile.Name())
		}
		nsInfo, err := LoadNamespaceInfo(nsfile.Name())
		if len(tt.nsData) == 0 && tt.nsData != "missing" {
			if err == nil {
				t.Error("LoadNamespaceInfo didn't fail on an empty file")
			}
			continue
		}
		if tt.nsData != "missing" {
			if err != nil {
				t.Errorf("Unexpected error: %v, %v", tt.nsData, err)
			}
			if !reflect.DeepEqual(nsInfo, tt.nsInfo) {
				t.Errorf("Expected %v, got %v", tt.nsInfo, nsInfo)
			}
		}
	}
}

func TestLoadAuthInfo(t *testing.T) {
	loadAuthInfoTests := []struct {
		authData string
		authInfo *AuthInfo
		r        io.Reader
	}{
		{
			`{"user": "user", "password": "pass"}`,
			&AuthInfo{User: "user", Password: "pass"},
			nil,
		},
		{
			"", nil, nil,
		},
		{
			"missing",
			&AuthInfo{User: "user", Password: "pass"},
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
	var successTestCases = []struct {
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
		{
			"",
			[]api.Port{},
		},
	}
	for _, tt := range successTestCases {
		ports, err := portsFromString(tt.spec)
		if !reflect.DeepEqual(ports, tt.ports) {
			t.Errorf("Expected %#v, got %#v", tt.ports, ports)
		}
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
	}

	var failTestCases = []struct {
		spec string
	}{
		{"8080:"},
		{":80"},
		{":"},
	}
	for _, tt := range failTestCases {
		_, err := portsFromString(tt.spec)
		if err == nil {
			t.Errorf("Unexpected non-error")
		}
	}
}
