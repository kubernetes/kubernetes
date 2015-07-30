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

package kubectl

import (
	"fmt"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/testclient"
)

func TestReplicationControllerStop(t *testing.T) {
	name := "foo"
	ns := "default"
	fake := testclient.NewSimpleFake(&api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 0,
		},
	})
	reaper := ReplicationControllerReaper{fake, time.Millisecond, time.Millisecond}
	s, err := reaper.Stop(ns, name, 0, nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	expected := "foo stopped"
	if s != expected {
		t.Errorf("expected %s, got %s", expected, s)
	}
	actions := fake.Actions()
	if len(actions) != 7 {
		t.Errorf("unexpected actions: %v, expected 6 actions (get, list, get, update, get, get, delete)", fake.Actions)
	}
	for i, action := range []string{"get", "list", "get", "update", "get", "get", "delete"} {
		if actions[i].Action != action+"-replicationController" {
			t.Errorf("unexpected action: %+v, expected %s-replicationController", actions[i], action)
		}
	}
}

type noSuchPod struct {
	*testclient.FakePods
}

func (c *noSuchPod) Get(name string) (*api.Pod, error) {
	return nil, fmt.Errorf("%s does not exist", name)
}

type noDeleteService struct {
	*testclient.FakeServices
}

func (c *noDeleteService) Delete(service string) error {
	return fmt.Errorf("I'm afraid I can't do that, Dave")
}

type reaperFake struct {
	*testclient.Fake
	noSuchPod, noDeleteService bool
}

func (c *reaperFake) Pods(namespace string) client.PodInterface {
	pods := &testclient.FakePods{c.Fake, namespace}
	if c.noSuchPod {
		return &noSuchPod{pods}
	}
	return pods
}

func (c *reaperFake) Services(namespace string) client.ServiceInterface {
	services := &testclient.FakeServices{c.Fake, namespace}
	if c.noDeleteService {
		return &noDeleteService{services}
	}
	return services
}

func TestSimpleStop(t *testing.T) {
	tests := []struct {
		fake        *reaperFake
		kind        string
		actions     []string
		expectError bool
		test        string
	}{
		{
			fake: &reaperFake{
				Fake: &testclient.Fake{},
			},
			kind:        "Pod",
			actions:     []string{"get-pod", "delete-pod"},
			expectError: false,
			test:        "stop pod succeeds",
		},
		{
			fake: &reaperFake{
				Fake: &testclient.Fake{},
			},
			kind:        "Service",
			actions:     []string{"get-service", "delete-service"},
			expectError: false,
			test:        "stop service succeeds",
		},
		{
			fake: &reaperFake{
				Fake:      &testclient.Fake{},
				noSuchPod: true,
			},
			kind:        "Pod",
			actions:     []string{},
			expectError: true,
			test:        "stop pod fails, no pod",
		},
		{
			fake: &reaperFake{
				Fake:            &testclient.Fake{},
				noDeleteService: true,
			},
			kind:        "Service",
			actions:     []string{"get-service"},
			expectError: true,
			test:        "stop service fails, can't delete",
		},
	}
	for _, test := range tests {
		fake := test.fake
		reaper, err := ReaperFor(test.kind, fake)
		if err != nil {
			t.Errorf("unexpected error: %v (%s)", err, test.test)
		}
		s, err := reaper.Stop("default", "foo", 0, nil)
		if err != nil && !test.expectError {
			t.Errorf("unexpected error: %v (%s)", err, test.test)
		}
		if err == nil {
			if test.expectError {
				t.Errorf("unexpected non-error: %v (%s)", err, test.test)
			}
			if s != "foo stopped" {
				t.Errorf("unexpected return: %s (%s)", s, test.test)
			}
		}
		actions := fake.Actions()
		if len(test.actions) != len(actions) {
			t.Errorf("unexpected actions: %v; expected %v (%s)", fake.Actions, test.actions, test.test)
		}
		for i, action := range actions {
			testAction := test.actions[i]
			if action.Action != testAction {
				t.Errorf("unexpected action: %v; expected %v (%s)", action, testAction, test.test)
			}
		}
	}
}
