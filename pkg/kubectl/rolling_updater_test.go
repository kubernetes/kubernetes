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
	"bytes"
	"fmt"
	"io/ioutil"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/testclient"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"
)

type updaterFake struct {
	*testclient.Fake
	ctrl client.ReplicationControllerInterface
}

func (c *updaterFake) ReplicationControllers(namespace string) client.ReplicationControllerInterface {
	return c.ctrl
}

func fakeClientFor(namespace string, responses []fakeResponse) client.Interface {
	fake := testclient.Fake{}
	return &updaterFake{
		&fake,
		&fakeRc{
			&testclient.FakeReplicationControllers{
				Fake:      &fake,
				Namespace: namespace,
			},
			responses,
		},
	}
}

type fakeResponse struct {
	controller *api.ReplicationController
	err        error
}

type fakeRc struct {
	*testclient.FakeReplicationControllers
	responses []fakeResponse
}

func (c *fakeRc) Get(name string) (*api.ReplicationController, error) {
	action := testclient.FakeAction{Action: "get-controller", Value: name}
	if len(c.responses) == 0 {
		return nil, fmt.Errorf("Unexpected Action: %s", action)
	}
	c.Fake.Actions = append(c.Fake.Actions, action)
	result := c.responses[0]
	c.responses = c.responses[1:]
	return result.controller, result.err
}

func (c *fakeRc) Create(controller *api.ReplicationController) (*api.ReplicationController, error) {
	c.Fake.Actions = append(c.Fake.Actions, testclient.FakeAction{Action: "create-controller", Value: controller.ObjectMeta.Name})
	return controller, nil
}

func (c *fakeRc) Update(controller *api.ReplicationController) (*api.ReplicationController, error) {
	c.Fake.Actions = append(c.Fake.Actions, testclient.FakeAction{Action: "update-controller", Value: controller.ObjectMeta.Name})
	return controller, nil
}

func oldRc(replicas int) *api.ReplicationController {
	return &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: "foo-v1",
			UID:  "7764ae47-9092-11e4-8393-42010af018ff",
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: replicas,
			Selector: map[string]string{"version": "v1"},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Name:   "foo-v1",
					Labels: map[string]string{"version": "v1"},
				},
			},
		},
		Status: api.ReplicationControllerStatus{
			Replicas: replicas,
		},
	}
}

func newRc(replicas int, desired int) *api.ReplicationController {
	rc := oldRc(replicas)
	rc.Spec.Template = &api.PodTemplateSpec{
		ObjectMeta: api.ObjectMeta{
			Name:   "foo-v2",
			Labels: map[string]string{"version": "v2"},
		},
	}
	rc.Spec.Selector = map[string]string{"version": "v2"}
	rc.ObjectMeta = api.ObjectMeta{
		Name: "foo-v2",
		Annotations: map[string]string{
			desiredReplicasAnnotation: fmt.Sprintf("%d", desired),
			sourceIdAnnotation:        "foo-v1:7764ae47-9092-11e4-8393-42010af018ff",
		},
	}
	return rc
}

func TestUpdate(t *testing.T) {
	tests := []struct {
		oldRc, newRc *api.ReplicationController
		responses    []fakeResponse
		output       string
	}{
		{
			oldRc(1), newRc(1, 1),
			[]fakeResponse{
				// no existing newRc
				{nil, fmt.Errorf("not found")},
				// 3 gets for each resize
				{newRc(1, 1), nil},
				{newRc(1, 1), nil},
				{newRc(1, 1), nil},
				{newRc(1, 1), nil},
				{oldRc(0), nil},
				{oldRc(0), nil},
				{oldRc(0), nil},
				//				{oldRc(0), nil},
				// cleanup annotations
				{newRc(1, 1), nil},
				{newRc(1, 1), nil},
			},
			`Creating foo-v2
Updating foo-v1 replicas: 0, foo-v2 replicas: 1
Update succeeded. Deleting foo-v1
`,
		}, {
			oldRc(2), newRc(2, 2),
			[]fakeResponse{
				// no existing newRc
				{nil, fmt.Errorf("not found")},
				// 3 gets for each resize
				{newRc(1, 2), nil},
				{newRc(1, 2), nil},
				{newRc(1, 2), nil},
				{newRc(1, 2), nil},
				{oldRc(1), nil},
				{oldRc(1), nil},
				{oldRc(1), nil},
				//				{oldRc(1), nil},
				{newRc(2, 2), nil},
				{newRc(2, 2), nil},
				{newRc(2, 2), nil},
				{newRc(2, 2), nil},
				{oldRc(0), nil},
				{oldRc(0), nil},
				{oldRc(0), nil},
				//				{oldRc(0), nil},
				// cleanup annotations
				{newRc(2, 2), nil},
				{newRc(2, 2), nil},
			},
			`Creating foo-v2
Updating foo-v1 replicas: 1, foo-v2 replicas: 1
Updating foo-v1 replicas: 0, foo-v2 replicas: 2
Update succeeded. Deleting foo-v1
`,
		}, {
			oldRc(2), newRc(7, 7),
			[]fakeResponse{
				// no existing newRc
				{nil, fmt.Errorf("not found")},
				// 3 gets for each resize
				{newRc(1, 2), nil},
				{newRc(1, 2), nil},
				{newRc(1, 2), nil},
				{newRc(1, 2), nil},
				{oldRc(1), nil},
				{oldRc(1), nil},
				{oldRc(1), nil},
				{newRc(2, 2), nil},
				{newRc(2, 2), nil},
				{newRc(2, 2), nil},
				{newRc(2, 2), nil},
				{oldRc(0), nil},
				{oldRc(0), nil},
				{oldRc(0), nil},
				// final resize on newRc
				{newRc(7, 7), nil},
				{newRc(7, 7), nil},
				{newRc(7, 7), nil},
				// cleanup annotations
				{newRc(7, 7), nil},
				{newRc(7, 7), nil},
			},
			`Creating foo-v2
Updating foo-v1 replicas: 1, foo-v2 replicas: 1
Updating foo-v1 replicas: 0, foo-v2 replicas: 2
Resizing foo-v2 replicas: 2 -> 7
Update succeeded. Deleting foo-v1
`,
		}, {
			oldRc(7), newRc(2, 2),
			[]fakeResponse{
				// no existing newRc
				{nil, fmt.Errorf("not found")},
				// 3 gets for each update
				{newRc(1, 2), nil},
				{newRc(1, 2), nil},
				{newRc(1, 2), nil},
				{newRc(1, 2), nil},
				{oldRc(6), nil},
				{oldRc(6), nil},
				{oldRc(6), nil},
				{newRc(2, 2), nil},
				{newRc(2, 2), nil},
				{newRc(2, 2), nil},
				{newRc(2, 2), nil},
				{oldRc(5), nil},
				{oldRc(5), nil},
				{oldRc(5), nil},
				// stop oldRc
				{oldRc(0), nil},
				{oldRc(0), nil},
				// cleanup annotations
				{newRc(2, 2), nil},
				{newRc(2, 2), nil},
			},
			`Creating foo-v2
Updating foo-v1 replicas: 6, foo-v2 replicas: 1
Updating foo-v1 replicas: 5, foo-v2 replicas: 2
Stopping foo-v1 replicas: 5 -> 0
Update succeeded. Deleting foo-v1
`,
		},
	}

	for _, test := range tests {
		updater := RollingUpdater{
			NewRollingUpdaterClient(fakeClientFor("default", test.responses)),
			"default",
		}
		var buffer bytes.Buffer
		config := &RollingUpdaterConfig{
			Out:           &buffer,
			OldRc:         test.oldRc,
			NewRc:         test.newRc,
			UpdatePeriod:  0,
			Interval:      time.Millisecond,
			Timeout:       time.Millisecond,
			CleanupPolicy: DeleteRollingUpdateCleanupPolicy,
		}
		if err := updater.Update(config); err != nil {
			t.Errorf("Update failed: %v", err)
		}
		if buffer.String() != test.output {
			t.Errorf("Bad output. expected:\n%s\ngot:\n%s", test.output, buffer.String())
		}
	}
}

func PTestUpdateRecovery(t *testing.T) {
	// Test recovery from interruption
	rc := oldRc(2)
	rcExisting := newRc(1, 3)

	output := `Continuing update with existing controller foo-v2.
Updating foo-v1 replicas: 1, foo-v2 replicas: 2
Updating foo-v1 replicas: 0, foo-v2 replicas: 3
Update succeeded. Deleting foo-v1
`
	responses := []fakeResponse{
		// Existing newRc
		{rcExisting, nil},
		// 3 gets for each resize
		{newRc(2, 2), nil},
		{newRc(2, 2), nil},
		{newRc(2, 2), nil},
		{oldRc(1), nil},
		{oldRc(1), nil},
		{oldRc(1), nil},
		{newRc(3, 3), nil},
		{newRc(3, 3), nil},
		{newRc(3, 3), nil},
		{oldRc(0), nil},
		{oldRc(0), nil},
		{oldRc(0), nil},
		// cleanup annotations
		{newRc(3, 3), nil},
		{newRc(3, 3), nil},
	}
	updater := RollingUpdater{NewRollingUpdaterClient(fakeClientFor("default", responses)), "default"}

	var buffer bytes.Buffer
	config := &RollingUpdaterConfig{
		Out:           &buffer,
		OldRc:         rc,
		NewRc:         rcExisting,
		UpdatePeriod:  0,
		Interval:      time.Millisecond,
		Timeout:       time.Millisecond,
		CleanupPolicy: DeleteRollingUpdateCleanupPolicy,
	}
	if err := updater.Update(config); err != nil {
		t.Errorf("Update failed: %v", err)
	}
	if buffer.String() != output {
		t.Errorf("Output was not as expected. Expected:\n%s\nGot:\n%s", output, buffer.String())
	}
}

// TestRollingUpdater_preserveCleanup ensures that the old controller isn't
// deleted following a successful deployment.
func TestRollingUpdater_preserveCleanup(t *testing.T) {
	rc := oldRc(2)
	rcExisting := newRc(1, 3)

	updater := &RollingUpdater{
		ns: "default",
		c: &rollingUpdaterClientImpl{
			GetReplicationControllerFn: func(namespace, name string) (*api.ReplicationController, error) {
				switch name {
				case rc.Name:
					return rc, nil
				case rcExisting.Name:
					return rcExisting, nil
				default:
					return nil, fmt.Errorf("unexpected get call for %s/%s", namespace, name)
				}
			},
			UpdateReplicationControllerFn: func(namespace string, rc *api.ReplicationController) (*api.ReplicationController, error) {
				return rc, nil
			},
			CreateReplicationControllerFn: func(namespace string, rc *api.ReplicationController) (*api.ReplicationController, error) {
				t.Fatalf("unexpected call to create %s/rc:%#v", namespace, rc)
				return nil, nil
			},
			DeleteReplicationControllerFn: func(namespace, name string) error {
				t.Fatalf("unexpected call to delete %s/%s", namespace, name)
				return nil
			},
			ControllerHasDesiredReplicasFn: func(rc *api.ReplicationController) wait.ConditionFunc {
				return func() (done bool, err error) {
					return true, nil
				}
			},
		},
	}

	config := &RollingUpdaterConfig{
		Out:           ioutil.Discard,
		OldRc:         rc,
		NewRc:         rcExisting,
		UpdatePeriod:  0,
		Interval:      time.Millisecond,
		Timeout:       time.Millisecond,
		CleanupPolicy: PreserveRollingUpdateCleanupPolicy,
	}
	err := updater.Update(config)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

// rollingUpdaterClientImpl is a dynamic RollingUpdaterClient.
type rollingUpdaterClientImpl struct {
	GetReplicationControllerFn     func(namespace, name string) (*api.ReplicationController, error)
	UpdateReplicationControllerFn  func(namespace string, rc *api.ReplicationController) (*api.ReplicationController, error)
	CreateReplicationControllerFn  func(namespace string, rc *api.ReplicationController) (*api.ReplicationController, error)
	DeleteReplicationControllerFn  func(namespace, name string) error
	ControllerHasDesiredReplicasFn func(rc *api.ReplicationController) wait.ConditionFunc
}

func (c *rollingUpdaterClientImpl) GetReplicationController(namespace, name string) (*api.ReplicationController, error) {
	return c.GetReplicationControllerFn(namespace, name)
}

func (c *rollingUpdaterClientImpl) UpdateReplicationController(namespace string, rc *api.ReplicationController) (*api.ReplicationController, error) {
	return c.UpdateReplicationControllerFn(namespace, rc)
}

func (c *rollingUpdaterClientImpl) CreateReplicationController(namespace string, rc *api.ReplicationController) (*api.ReplicationController, error) {
	return c.CreateReplicationControllerFn(namespace, rc)
}

func (c *rollingUpdaterClientImpl) DeleteReplicationController(namespace, name string) error {
	return c.DeleteReplicationControllerFn(namespace, name)
}

func (c *rollingUpdaterClientImpl) ControllerHasDesiredReplicas(rc *api.ReplicationController) wait.ConditionFunc {
	return c.ControllerHasDesiredReplicasFn(rc)
}
