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

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
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

func TestReplicationControllerScaleRetry(t *testing.T) {
	fake := &ErrorReplicationControllerClient{Fake: testclient.Fake{}}
	scaler := ReplicationControllerScaler{NewScalerClient(fake)}
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo"
	namespace := "default"

	scaleFunc := ScaleCondition(&scaler, &preconditions, namespace, name, count)
	pass, err := scaleFunc()
	if pass != false {
		t.Errorf("Expected an update failure to return pass = false, got pass = %v", pass)
	}
	if err != nil {
		t.Errorf("Did not expect an error on update failure, got %v", err)
	}
	preconditions = ScalePrecondition{3, ""}
	scaleFunc = ScaleCondition(&scaler, &preconditions, namespace, name, count)
	pass, err = scaleFunc()
	if err == nil {
		t.Errorf("Expected error on precondition failure")
	}
}

func TestReplicationControllerScale(t *testing.T) {
	fake := &testclient.Fake{}
	scaler := ReplicationControllerScaler{NewScalerClient(fake)}
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo"
	scaler.Scale("default", name, count, &preconditions, nil, nil)

	actions := fake.Actions()
	if len(actions) != 2 {
		t.Errorf("unexpected actions: %v, expected 2 actions (get, update)", actions)
	}
	if action, ok := actions[0].(testclient.GetAction); !ok || action.GetResource() != "replicationcontrollers" || action.GetName() != name {
		t.Errorf("unexpected action: %v, expected get-replicationController %s", actions[0], name)
	}
	if action, ok := actions[1].(testclient.UpdateAction); !ok || action.GetResource() != "replicationcontrollers" || action.GetObject().(*api.ReplicationController).Spec.Replicas != int(count) {
		t.Errorf("unexpected action %v, expected update-replicationController with replicas = %d", actions[1], count)
	}
}

func TestReplicationControllerScaleFailsPreconditions(t *testing.T) {
	fake := testclient.NewSimpleFake(&api.ReplicationController{
		Spec: api.ReplicationControllerSpec{
			Replicas: 10,
		},
	})
	scaler := ReplicationControllerScaler{NewScalerClient(fake)}
	preconditions := ScalePrecondition{2, ""}
	count := uint(3)
	name := "foo"
	scaler.Scale("default", name, count, &preconditions, nil, nil)

	actions := fake.Actions()
	if len(actions) != 1 {
		t.Errorf("unexpected actions: %v, expected 2 actions (get, update)", actions)
	}
	if action, ok := actions[0].(testclient.GetAction); !ok || action.GetResource() != "replicationcontrollers" || action.GetName() != name {
		t.Errorf("unexpected action: %v, expected get-replicationController %s", actions[0], name)
	}
}

func TestPreconditionValidate(t *testing.T) {
	tests := []struct {
		preconditions ScalePrecondition
		controller    api.ReplicationController
		expectError   bool
		test          string
	}{
		{
			preconditions: ScalePrecondition{-1, ""},
			expectError:   false,
			test:          "defaults",
		},
		{
			preconditions: ScalePrecondition{-1, ""},
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
			preconditions: ScalePrecondition{0, ""},
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
			preconditions: ScalePrecondition{-1, "foo"},
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
			preconditions: ScalePrecondition{10, "foo"},
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
			preconditions: ScalePrecondition{10, "foo"},
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
			preconditions: ScalePrecondition{10, "foo"},
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
			preconditions: ScalePrecondition{10, "foo"},
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
