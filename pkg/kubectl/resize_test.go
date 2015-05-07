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
	"errors"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/testclient"
)

type ErrorReplicationControllers struct {
	testclient.FakeReplicationControllers
}

func (c *ErrorReplicationControllers) Update(controller *api.ReplicationController) (*api.ReplicationController, error) {
	return nil, errors.New("Replication controller update failure")
}

type ErrorReplicationControllerClient struct {
	testclient.Fake
}

func (c *ErrorReplicationControllerClient) ReplicationControllers(namespace string) client.ReplicationControllerInterface {
	return &ErrorReplicationControllers{testclient.FakeReplicationControllers{Fake: &c.Fake, Namespace: namespace}}
}

func TestReplicationControllerResizeRetry(t *testing.T) {
	fake := &ErrorReplicationControllerClient{Fake: testclient.Fake{}}
	resizer := ReplicationControllerResizer{NewResizerClient(fake)}
	preconditions := ResizePrecondition{-1, ""}
	count := uint(3)
	name := "foo"
	namespace := "default"

	resizeFunc := ResizeCondition(&resizer, &preconditions, namespace, name, count)
	pass, err := resizeFunc()
	if pass != false {
		t.Errorf("Expected an update failure to return pass = false, got pass = %v", pass)
	}
	if err != nil {
		t.Errorf("Did not expect an error on update failure, got %v", err)
	}
	preconditions = ResizePrecondition{3, ""}
	resizeFunc = ResizeCondition(&resizer, &preconditions, namespace, name, count)
	pass, err = resizeFunc()
	if err == nil {
		t.Errorf("Expected error on precondition failure")
	}
}

func TestReplicationControllerResize(t *testing.T) {
	fake := &testclient.Fake{}
	resizer := ReplicationControllerResizer{NewResizerClient(fake)}
	preconditions := ResizePrecondition{-1, ""}
	count := uint(3)
	name := "foo"
	resizer.Resize("default", name, count, &preconditions, nil, nil)

	if len(fake.Actions) != 2 {
		t.Errorf("unexpected actions: %v, expected 2 actions (get, update)", fake.Actions)
	}
	if fake.Actions[0].Action != "get-replicationController" || fake.Actions[0].Value != name {
		t.Errorf("unexpected action: %v, expected get-replicationController %s", fake.Actions[0], name)
	}
	if fake.Actions[1].Action != "update-replicationController" || fake.Actions[1].Value.(*api.ReplicationController).Spec.Replicas != int(count) {
		t.Errorf("unexpected action %v, expected update-replicationController with replicas = %d", fake.Actions[1], count)
	}
}

func TestReplicationControllerResizeFailsPreconditions(t *testing.T) {
	fake := testclient.NewSimpleFake(&api.ReplicationController{
		Spec: api.ReplicationControllerSpec{
			Replicas: 10,
		},
	})
	resizer := ReplicationControllerResizer{NewResizerClient(fake)}
	preconditions := ResizePrecondition{2, ""}
	count := uint(3)
	name := "foo"
	resizer.Resize("default", name, count, &preconditions, nil, nil)

	if len(fake.Actions) != 1 {
		t.Errorf("unexpected actions: %v, expected 2 actions (get, update)", fake.Actions)
	}
	if fake.Actions[0].Action != "get-replicationController" || fake.Actions[0].Value != name {
		t.Errorf("unexpected action: %v, expected get-replicationController %s", fake.Actions[0], name)
	}
}

func TestPreconditionValidate(t *testing.T) {
	tests := []struct {
		preconditions ResizePrecondition
		controller    api.ReplicationController
		expectError   bool
		test          string
	}{
		{
			preconditions: ResizePrecondition{-1, ""},
			expectError:   false,
			test:          "defaults",
		},
		{
			preconditions: ResizePrecondition{-1, ""},
			controller: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 10,
				},
			},
			expectError: false,
			test:        "defaults 2",
		},
		{
			preconditions: ResizePrecondition{0, ""},
			controller: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 0,
				},
			},
			expectError: false,
			test:        "size matches",
		},
		{
			preconditions: ResizePrecondition{-1, "foo"},
			controller: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 10,
				},
			},
			expectError: false,
			test:        "resource version matches",
		},
		{
			preconditions: ResizePrecondition{10, "foo"},
			controller: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 10,
				},
			},
			expectError: false,
			test:        "both match",
		},
		{
			preconditions: ResizePrecondition{10, "foo"},
			controller: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 20,
				},
			},
			expectError: true,
			test:        "size different",
		},
		{
			preconditions: ResizePrecondition{10, "foo"},
			controller: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					ResourceVersion: "bar",
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 10,
				},
			},
			expectError: true,
			test:        "version different",
		},
		{
			preconditions: ResizePrecondition{10, "foo"},
			controller: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					ResourceVersion: "bar",
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 20,
				},
			},
			expectError: true,
			test:        "both different",
		},
	}
	for _, test := range tests {
		err := test.preconditions.Validate(&test.controller)
		if err != nil && !test.expectError {
			t.Errorf("unexpected error: %v (%s)", err, test.test)
		}
		if err == nil && test.expectError {
			t.Errorf("unexpected non-error: %v (%s)", err, test.test)
		}
	}
}
