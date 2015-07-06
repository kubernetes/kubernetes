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
	"io"
	"io/ioutil"
	"net/http"
	"reflect"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/testclient"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
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
	c.Fake.Invokes(action, nil)
	result := c.responses[0]
	c.responses = c.responses[1:]
	return result.controller, result.err
}

func (c *fakeRc) Create(controller *api.ReplicationController) (*api.ReplicationController, error) {
	c.Fake.Invokes(testclient.FakeAction{Action: "create-controller", Value: controller.ObjectMeta.Name}, nil)
	return controller, nil
}

func (c *fakeRc) Update(controller *api.ReplicationController) (*api.ReplicationController, error) {
	c.Fake.Invokes(testclient.FakeAction{Action: "update-controller", Value: controller.ObjectMeta.Name}, nil)
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
	// Helpers
	Percent := func(p int) *int {
		return &p
	}
	var NilPercent *int
	// Scenarios
	tests := []struct {
		oldRc, newRc *api.ReplicationController
		accepted     bool
		percent      *int
		responses    []fakeResponse
		output       string
	}{
		{
			oldRc:    oldRc(1),
			newRc:    newRc(1, 1),
			accepted: true,
			percent:  NilPercent,
			responses: []fakeResponse{
				// no existing newRc
				{nil, fmt.Errorf("not found")},
				// scaling iteration
				{newRc(1, 1), nil},
				{oldRc(0), nil},
				// cleanup annotations
				{newRc(1, 1), nil},
				{newRc(1, 1), nil},
				{newRc(1, 1), nil},
			},
			output: `Creating foo-v2
Scaling up foo-v2 from 0 to 1, scaling down foo-v1 from 1 to 0 (scale up first by 1 each interval)
Scaling foo-v2 up to 1
Scaling foo-v1 down to 0
Update succeeded. Deleting foo-v1
`,
		}, {
			oldRc:    oldRc(1),
			newRc:    newRc(1, 1),
			accepted: true,
			percent:  NilPercent,
			responses: []fakeResponse{
				// no existing newRc
				{nil, fmt.Errorf("not found")},
				// scaling iteration
				{newRc(1, 1), nil},
				{oldRc(0), nil},
				// cleanup annotations
				{newRc(1, 1), nil},
				{newRc(1, 1), nil},
				{newRc(1, 1), nil},
			},
			output: `Creating foo-v2
Scaling up foo-v2 from 0 to 1, scaling down foo-v1 from 1 to 0 (scale up first by 1 each interval)
Scaling foo-v2 up to 1
Scaling foo-v1 down to 0
Update succeeded. Deleting foo-v1
`,
		}, {
			oldRc:    oldRc(2),
			newRc:    newRc(2, 2),
			accepted: true,
			percent:  NilPercent,
			responses: []fakeResponse{
				// no existing newRc
				{nil, fmt.Errorf("not found")},
				// scaling iteration
				{newRc(1, 2), nil},
				{oldRc(1), nil},
				// scaling iteration
				{newRc(2, 2), nil},
				{oldRc(0), nil},
				// cleanup annotations
				{newRc(2, 2), nil},
				{newRc(2, 2), nil},
				{newRc(1, 1), nil},
			},
			output: `Creating foo-v2
Scaling up foo-v2 from 0 to 2, scaling down foo-v1 from 2 to 0 (scale up first by 1 each interval)
Scaling foo-v2 up to 1
Scaling foo-v1 down to 1
Scaling foo-v2 up to 2
Scaling foo-v1 down to 0
Update succeeded. Deleting foo-v1
`,
		}, {
			oldRc:    oldRc(2),
			newRc:    newRc(7, 7),
			accepted: true,
			percent:  NilPercent,
			responses: []fakeResponse{
				// no existing newRc
				{nil, fmt.Errorf("not found")},
				// scaling iteration
				{newRc(1, 7), nil},
				{oldRc(1), nil},
				// scaling iteration
				{newRc(2, 7), nil},
				{oldRc(0), nil},
				// final scale on newRc
				{newRc(7, 7), nil},
				// cleanup annotations
				{newRc(7, 7), nil},
				{newRc(7, 7), nil},
				{newRc(7, 7), nil},
			},
			output: `Creating foo-v2
Scaling up foo-v2 from 0 to 7, scaling down foo-v1 from 2 to 0 (scale up first by 1 each interval)
Scaling foo-v2 up to 1
Scaling foo-v1 down to 1
Scaling foo-v2 up to 2
Scaling foo-v1 down to 0
Scaling foo-v2 up to 7
Update succeeded. Deleting foo-v1
`,
		}, {
			oldRc:    oldRc(7),
			newRc:    newRc(2, 2),
			accepted: true,
			percent:  NilPercent,
			responses: []fakeResponse{
				// no existing newRc
				{nil, fmt.Errorf("not found")},
				// scaling iteration
				{newRc(1, 2), nil},
				{oldRc(6), nil},
				// scaling iteration
				{newRc(2, 2), nil},
				{oldRc(0), nil},
				// cleanup annotations
				{newRc(2, 2), nil},
				{newRc(2, 2), nil},
				{newRc(2, 2), nil},
			},
			output: `Creating foo-v2
Scaling up foo-v2 from 0 to 2, scaling down foo-v1 from 7 to 0 (scale up first by 1 each interval)
Scaling foo-v2 up to 1
Scaling foo-v1 down to 6
Scaling foo-v2 up to 2
Scaling foo-v1 down to 0
Update succeeded. Deleting foo-v1
`,
		}, {
			oldRc:    oldRc(7),
			newRc:    newRc(2, 2),
			accepted: false,
			percent:  NilPercent,
			responses: []fakeResponse{
				// no existing newRc
				{nil, fmt.Errorf("not found")},
				// scaling iteration (only up occurs since the update is rejected)
				{newRc(1, 2), nil},
			},
			output: `Creating foo-v2
Scaling up foo-v2 from 0 to 2, scaling down foo-v1 from 7 to 0 (scale up first by 1 each interval)
Scaling foo-v2 up to 1
`,
		}, {
			oldRc:    oldRc(10),
			newRc:    newRc(10, 10),
			accepted: true,
			percent:  Percent(20),
			responses: []fakeResponse{
				// no existing newRc
				{nil, fmt.Errorf("not found")},
				// scaling iteration
				{newRc(2, 10), nil},
				{oldRc(8), nil},
				// scaling iteration
				{newRc(4, 10), nil},
				{oldRc(6), nil},
				// scaling iteration
				{newRc(6, 10), nil},
				{oldRc(4), nil},
				// scaling iteration
				{newRc(8, 10), nil},
				{oldRc(2), nil},
				// scaling iteration
				{newRc(10, 10), nil},
				{oldRc(0), nil},
				// cleanup annotations
				{newRc(10, 10), nil},
				{newRc(10, 10), nil},
				{newRc(10, 10), nil},
			},
			output: `Creating foo-v2
Scaling up foo-v2 from 0 to 10, scaling down foo-v1 from 10 to 0 (scale up first by 2 each interval)
Scaling foo-v2 up to 2
Scaling foo-v1 down to 8
Scaling foo-v2 up to 4
Scaling foo-v1 down to 6
Scaling foo-v2 up to 6
Scaling foo-v1 down to 4
Scaling foo-v2 up to 8
Scaling foo-v1 down to 2
Scaling foo-v2 up to 10
Scaling foo-v1 down to 0
Update succeeded. Deleting foo-v1
`,
		}, {
			oldRc:    oldRc(2),
			newRc:    newRc(6, 6),
			accepted: true,
			percent:  Percent(50),
			responses: []fakeResponse{
				// no existing newRc
				{nil, fmt.Errorf("not found")},
				// scaling iteration
				{newRc(3, 6), nil},
				{oldRc(0), nil},
				// scaling iteration
				{newRc(6, 6), nil},
				// cleanup annotations
				{newRc(6, 6), nil},
				{newRc(6, 6), nil},
				{newRc(6, 6), nil},
			},
			output: `Creating foo-v2
Scaling up foo-v2 from 0 to 6, scaling down foo-v1 from 2 to 0 (scale up first by 3 each interval)
Scaling foo-v2 up to 3
Scaling foo-v1 down to 0
Scaling foo-v2 up to 6
Update succeeded. Deleting foo-v1
`,
		}, {
			oldRc:    oldRc(10),
			newRc:    newRc(3, 3),
			accepted: true,
			percent:  Percent(50),
			responses: []fakeResponse{
				// no existing newRc
				{nil, fmt.Errorf("not found")},
				// scaling iteration
				{newRc(2, 3), nil},
				{oldRc(8), nil},
				// scaling iteration
				{newRc(3, 3), nil},
				{oldRc(0), nil},
				// cleanup annotations
				{newRc(3, 3), nil},
				{newRc(3, 3), nil},
				{newRc(3, 3), nil},
			},
			output: `Creating foo-v2
Scaling up foo-v2 from 0 to 3, scaling down foo-v1 from 10 to 0 (scale up first by 2 each interval)
Scaling foo-v2 up to 2
Scaling foo-v1 down to 8
Scaling foo-v2 up to 3
Scaling foo-v1 down to 0
Update succeeded. Deleting foo-v1
`,
		}, {
			oldRc:    oldRc(4),
			newRc:    newRc(4, 4),
			accepted: true,
			percent:  Percent(-50),
			responses: []fakeResponse{
				// no existing newRc
				{nil, fmt.Errorf("not found")},
				// scaling iteration
				{oldRc(2), nil},
				{newRc(2, 4), nil},
				// scaling iteration
				{oldRc(0), nil},
				{newRc(4, 4), nil},
				// cleanup annotations
				{newRc(4, 4), nil},
				{newRc(4, 4), nil},
				{newRc(4, 4), nil},
			},
			output: `Creating foo-v2
Scaling up foo-v2 from 0 to 4, scaling down foo-v1 from 4 to 0 (scale down first by 2 each interval)
Scaling foo-v1 down to 2
Scaling foo-v2 up to 2
Scaling foo-v1 down to 0
Scaling foo-v2 up to 4
Update succeeded. Deleting foo-v1
`,
		}, {
			oldRc:    oldRc(2),
			newRc:    newRc(4, 4),
			accepted: true,
			percent:  Percent(-50),
			responses: []fakeResponse{
				// no existing newRc
				{nil, fmt.Errorf("not found")},
				// scaling iteration
				{oldRc(0), nil},
				{newRc(4, 4), nil},
				// cleanup annotations
				{newRc(4, 4), nil},
				{newRc(4, 4), nil},
				{newRc(4, 4), nil},
			},
			output: `Creating foo-v2
Scaling up foo-v2 from 0 to 4, scaling down foo-v1 from 2 to 0 (scale down first by 2 each interval)
Scaling foo-v1 down to 0
Scaling foo-v2 up to 4
Update succeeded. Deleting foo-v1
`,
		}, {
			oldRc:    oldRc(4),
			newRc:    newRc(2, 2),
			accepted: true,
			percent:  Percent(-50),
			responses: []fakeResponse{
				// no existing newRc
				{nil, fmt.Errorf("not found")},
				// scaling iteration
				{oldRc(3), nil},
				{newRc(1, 2), nil},
				// scaling iteration
				{oldRc(2), nil},
				{newRc(2, 2), nil},
				// scaling iteration
				{oldRc(0), nil},
				// cleanup annotations
				{newRc(2, 2), nil},
				{newRc(2, 2), nil},
				{newRc(2, 2), nil},
			},
			output: `Creating foo-v2
Scaling up foo-v2 from 0 to 2, scaling down foo-v1 from 4 to 0 (scale down first by 1 each interval)
Scaling foo-v1 down to 3
Scaling foo-v2 up to 1
Scaling foo-v1 down to 2
Scaling foo-v2 up to 2
Scaling foo-v1 down to 0
Update succeeded. Deleting foo-v1
`,
		}, {
			oldRc:    oldRc(4),
			newRc:    newRc(4, 4),
			accepted: true,
			percent:  Percent(-100),
			responses: []fakeResponse{
				// no existing newRc
				{nil, fmt.Errorf("not found")},
				// scaling iteration
				{oldRc(0), nil},
				{newRc(4, 4), nil},
				// cleanup annotations
				{newRc(4, 4), nil},
				{newRc(4, 4), nil},
				{newRc(4, 4), nil},
			},
			output: `Creating foo-v2
Scaling up foo-v2 from 0 to 4, scaling down foo-v1 from 4 to 0 (scale down first by 4 each interval)
Scaling foo-v1 down to 0
Scaling foo-v2 up to 4
Update succeeded. Deleting foo-v1
`,
		},
	}

	for _, test := range tests {
		client := NewRollingUpdaterClient(fakeClientFor("default", test.responses))
		updater := RollingUpdater{
			c:  client,
			ns: "default",
			scaleAndWait: func(rc *api.ReplicationController, retry *RetryParams, wait *RetryParams) (*api.ReplicationController, error) {
				return client.GetReplicationController(rc.Namespace, rc.Name)
			},
		}
		var buffer bytes.Buffer
		acceptor := &testAcceptor{
			accept: func(rc *api.ReplicationController) error {
				if test.accepted {
					return nil
				}
				return fmt.Errorf("rejecting controller %s", rc.Name)
			},
		}
		config := &RollingUpdaterConfig{
			Out:            &buffer,
			OldRc:          test.oldRc,
			NewRc:          test.newRc,
			UpdatePeriod:   0,
			Interval:       time.Millisecond,
			Timeout:        time.Millisecond,
			CleanupPolicy:  DeleteRollingUpdateCleanupPolicy,
			UpdateAcceptor: acceptor,
			UpdatePercent:  test.percent,
		}
		err := updater.Update(config)
		if test.accepted && err != nil {
			t.Errorf("Update failed: %v", err)
		}
		if !test.accepted && err == nil {
			t.Errorf("Expected update to fail")
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
Scaling up foo-v2 from 1 to 3, scaling down foo-v1 from 2 to 0 (scale up first by 1 each interval)	
Scaling foo-v2 to 2
Scaling foo-v1 to 1
Scaling foo-v2 to 3
Scaling foo-v2 to 0
Update succeeded. Deleting foo-v1
`
	responses := []fakeResponse{
		// Existing newRc
		{rcExisting, nil},
		// scaling iteration
		{newRc(2, 2), nil},
		{oldRc(1), nil},
		// scaling iteration
		{newRc(3, 3), nil},
		{oldRc(0), nil},
		// cleanup annotations
		{newRc(3, 3), nil},
		{newRc(3, 3), nil},
		{newRc(3, 3), nil},
	}

	client := NewRollingUpdaterClient(fakeClientFor("default", responses))
	updater := RollingUpdater{
		c:  client,
		ns: "default",
		scaleAndWait: func(rc *api.ReplicationController, retry *RetryParams, wait *RetryParams) (*api.ReplicationController, error) {
			return client.GetReplicationController(rc.Namespace, rc.Name)
		},
	}

	var buffer bytes.Buffer
	config := &RollingUpdaterConfig{
		Out:            &buffer,
		OldRc:          rc,
		NewRc:          rcExisting,
		UpdatePeriod:   0,
		Interval:       time.Millisecond,
		Timeout:        time.Millisecond,
		CleanupPolicy:  DeleteRollingUpdateCleanupPolicy,
		UpdateAcceptor: DefaultUpdateAcceptor,
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

	client := &rollingUpdaterClientImpl{
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
	}
	updater := &RollingUpdater{
		ns:           "default",
		c:            client,
		scaleAndWait: scalerScaleAndWait(client, "default"),
	}

	config := &RollingUpdaterConfig{
		Out:            ioutil.Discard,
		OldRc:          rc,
		NewRc:          rcExisting,
		UpdatePeriod:   0,
		Interval:       time.Millisecond,
		Timeout:        time.Millisecond,
		CleanupPolicy:  PreserveRollingUpdateCleanupPolicy,
		UpdateAcceptor: DefaultUpdateAcceptor,
	}
	err := updater.Update(config)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRename(t *testing.T) {
	tests := []struct {
		namespace   string
		newName     string
		oldName     string
		err         error
		expectError bool
	}{
		{
			namespace: "default",
			newName:   "bar",
			oldName:   "foo",
		},
		{
			namespace:   "default",
			newName:     "bar",
			oldName:     "foo",
			err:         fmt.Errorf("Test Error"),
			expectError: true,
		},
	}
	for _, test := range tests {
		fakeClient := &rollingUpdaterClientImpl{
			CreateReplicationControllerFn: func(namespace string, rc *api.ReplicationController) (*api.ReplicationController, error) {
				if namespace != test.namespace {
					t.Errorf("unexepected namespace: %s, expected %s", namespace, test.namespace)
				}
				if rc.Name != test.newName {
					t.Errorf("unexepected name: %s, expected %s", rc.Name, test.newName)
				}
				return rc, test.err
			},
			DeleteReplicationControllerFn: func(namespace, name string) error {
				if namespace != test.namespace {
					t.Errorf("unexepected namespace: %s, expected %s", namespace, test.namespace)
				}
				if name != test.oldName {
					t.Errorf("unexepected name: %s, expected %s", name, test.oldName)
				}
				return nil
			},
		}
		err := Rename(fakeClient, &api.ReplicationController{ObjectMeta: api.ObjectMeta{Namespace: test.namespace, Name: test.oldName}}, test.newName)
		if err != nil && !test.expectError {
			t.Errorf("unexpected error: %v", err)
		}
		if err == nil && test.expectError {
			t.Errorf("unexpected non-error")
		}
	}
}

func TestFindSourceController(t *testing.T) {
	ctrl1 := api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
			Annotations: map[string]string{
				sourceIdAnnotation: "bar:1234",
			},
		},
	}
	ctrl2 := api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: "bar",
			Annotations: map[string]string{
				sourceIdAnnotation: "foo:12345",
			},
		},
	}
	ctrl3 := api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Annotations: map[string]string{
				sourceIdAnnotation: "baz:45667",
			},
		},
	}
	tests := []struct {
		list               *api.ReplicationControllerList
		expectedController *api.ReplicationController
		err                error
		name               string
		expectError        bool
	}{
		{
			list:        &api.ReplicationControllerList{},
			expectError: true,
		},
		{
			list: &api.ReplicationControllerList{
				Items: []api.ReplicationController{ctrl1},
			},
			name:        "foo",
			expectError: true,
		},
		{
			list: &api.ReplicationControllerList{
				Items: []api.ReplicationController{ctrl1},
			},
			name:               "bar",
			expectedController: &ctrl1,
		},
		{
			list: &api.ReplicationControllerList{
				Items: []api.ReplicationController{ctrl1, ctrl2},
			},
			name:               "bar",
			expectedController: &ctrl1,
		},
		{
			list: &api.ReplicationControllerList{
				Items: []api.ReplicationController{ctrl1, ctrl2},
			},
			name:               "foo",
			expectedController: &ctrl2,
		},
		{
			list: &api.ReplicationControllerList{
				Items: []api.ReplicationController{ctrl1, ctrl2, ctrl3},
			},
			name:               "baz",
			expectedController: &ctrl3,
		},
	}
	for _, test := range tests {
		fakeClient := rollingUpdaterClientImpl{
			ListReplicationControllersFn: func(namespace string, selector labels.Selector) (*api.ReplicationControllerList, error) {
				return test.list, test.err
			},
		}
		ctrl, err := FindSourceController(&fakeClient, "default", test.name)
		if test.expectError && err == nil {
			t.Errorf("unexpected non-error")
		}
		if !test.expectError && err != nil {
			t.Errorf("unexpected error")
		}
		if !reflect.DeepEqual(ctrl, test.expectedController) {
			t.Errorf("expected:\n%v\ngot:\n%v\n", test.expectedController, ctrl)
		}
	}
}

func TestUpdateExistingReplicationController(t *testing.T) {
	tests := []struct {
		rc              *api.ReplicationController
		name            string
		deploymentKey   string
		deploymentValue string

		expectedRc *api.ReplicationController
		expectErr  bool
	}{
		{
			rc: &api.ReplicationController{
				Spec: api.ReplicationControllerSpec{
					Template: &api.PodTemplateSpec{},
				},
			},
			name:            "foo",
			deploymentKey:   "dk",
			deploymentValue: "some-hash",

			expectedRc: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						"kubectl.kubernetes.io/next-controller-id": "foo",
					},
				},
				Spec: api.ReplicationControllerSpec{
					Selector: map[string]string{
						"dk": "some-hash",
					},
					Template: &api.PodTemplateSpec{
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{
								"dk": "some-hash",
							},
						},
					},
				},
			},
		},
		{
			rc: &api.ReplicationController{
				Spec: api.ReplicationControllerSpec{
					Template: &api.PodTemplateSpec{
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{
								"dk": "some-other-hash",
							},
						},
					},
					Selector: map[string]string{
						"dk": "some-other-hash",
					},
				},
			},
			name:            "foo",
			deploymentKey:   "dk",
			deploymentValue: "some-hash",

			expectedRc: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						"kubectl.kubernetes.io/next-controller-id": "foo",
					},
				},
				Spec: api.ReplicationControllerSpec{
					Selector: map[string]string{
						"dk": "some-other-hash",
					},
					Template: &api.PodTemplateSpec{
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{
								"dk": "some-other-hash",
							},
						},
					},
				},
			},
		},
	}
	for _, test := range tests {
		buffer := &bytes.Buffer{}
		fakeClient := fakeClientFor("default", []fakeResponse{})
		rc, err := UpdateExistingReplicationController(fakeClient, test.rc, "default", test.name, test.deploymentKey, test.deploymentValue, buffer)
		if !reflect.DeepEqual(rc, test.expectedRc) {
			t.Errorf("expected:\n%#v\ngot:\n%#v\n", test.expectedRc, rc)
		}
		if test.expectErr && err == nil {
			t.Errorf("unexpected non-error")
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	}
}

func TestUpdateWithRetries(t *testing.T) {
	codec := testapi.Codec()
	rc := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{Name: "rc",
			Labels: map[string]string{
				"foo": "bar",
			},
		},
		Spec: api.ReplicationControllerSpec{
			Selector: map[string]string{
				"foo": "bar",
			},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
		},
	}

	// Test end to end updating of the rc with retries. Essentially make sure the update handler
	// sees the right updates, failures in update/get are handled properly, and that the updated
	// rc with new resource version is returned to the caller. Without any of these rollingupdate
	// will fail cryptically.
	newRc := *rc
	newRc.ResourceVersion = "2"
	newRc.Spec.Selector["baz"] = "foobar"
	updates := []*http.Response{
		{StatusCode: 500, Body: objBody(codec, &api.ReplicationController{})},
		{StatusCode: 500, Body: objBody(codec, &api.ReplicationController{})},
		{StatusCode: 200, Body: objBody(codec, &newRc)},
	}
	gets := []*http.Response{
		{StatusCode: 500, Body: objBody(codec, &api.ReplicationController{})},
		{StatusCode: 200, Body: objBody(codec, rc)},
	}
	fakeClient := &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == testapi.ResourcePath("replicationcontrollers", "default", "rc") && m == "PUT":
				update := updates[0]
				updates = updates[1:]
				// We should always get an update with a valid rc even when the get fails. The rc should always
				// contain the update.
				if c, ok := readOrDie(t, req, codec).(*api.ReplicationController); !ok || !reflect.DeepEqual(rc, c) {
					t.Errorf("Unexpected update body, got %+v expected %+v", c, rc)
				} else if sel, ok := c.Spec.Selector["baz"]; !ok || sel != "foobar" {
					t.Errorf("Expected selector label update, got %+v", c.Spec.Selector)
				} else {
					delete(c.Spec.Selector, "baz")
				}
				return update, nil
			case p == testapi.ResourcePath("replicationcontrollers", "default", "rc") && m == "GET":
				get := gets[0]
				gets = gets[1:]
				return get, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	clientConfig := &client.Config{Version: testapi.Version()}
	client := client.NewOrDie(clientConfig)
	client.Client = fakeClient.Client

	if rc, err := updateWithRetries(
		client.ReplicationControllers("default"), rc, func(c *api.ReplicationController) {
			c.Spec.Selector["baz"] = "foobar"
		}); err != nil {
		t.Errorf("unexpected error: %v", err)
	} else if sel, ok := rc.Spec.Selector["baz"]; !ok || sel != "foobar" || rc.ResourceVersion != "2" {
		t.Errorf("Expected updated rc, got %+v", rc)
	}
	if len(updates) != 0 || len(gets) != 0 {
		t.Errorf("Remaining updates %+v gets %+v", updates, gets)
	}
}

func readOrDie(t *testing.T, req *http.Request, codec runtime.Codec) runtime.Object {
	data, err := ioutil.ReadAll(req.Body)
	if err != nil {
		t.Errorf("Error reading: %v", err)
		t.FailNow()
	}
	obj, err := codec.Decode(data)
	if err != nil {
		t.Errorf("error decoding: %v", err)
		t.FailNow()
	}
	return obj
}

func objBody(codec runtime.Codec, obj runtime.Object) io.ReadCloser {
	return ioutil.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(codec, obj))))
}

func TestAddDeploymentHash(t *testing.T) {
	buf := &bytes.Buffer{}
	codec := testapi.Codec()
	rc := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{Name: "rc"},
		Spec: api.ReplicationControllerSpec{
			Selector: map[string]string{
				"foo": "bar",
			},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
			},
		},
	}

	podList := &api.PodList{
		Items: []api.Pod{
			{ObjectMeta: api.ObjectMeta{Name: "foo"}},
			{ObjectMeta: api.ObjectMeta{Name: "bar"}},
			{ObjectMeta: api.ObjectMeta{Name: "baz"}},
		},
	}

	seen := util.StringSet{}
	updatedRc := false
	fakeClient := &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == testapi.ResourcePath("pods", "default", "") && m == "GET":
				if req.URL.RawQuery != "labelSelector=foo%3Dbar" {
					t.Errorf("Unexpected query string: %s", req.URL.RawQuery)
				}
				return &http.Response{StatusCode: 200, Body: objBody(codec, podList)}, nil
			case p == testapi.ResourcePath("pods", "default", "foo") && m == "PUT":
				seen.Insert("foo")
				obj := readOrDie(t, req, codec)
				podList.Items[0] = *(obj.(*api.Pod))
				return &http.Response{StatusCode: 200, Body: objBody(codec, &podList.Items[0])}, nil
			case p == testapi.ResourcePath("pods", "default", "bar") && m == "PUT":
				seen.Insert("bar")
				obj := readOrDie(t, req, codec)
				podList.Items[1] = *(obj.(*api.Pod))
				return &http.Response{StatusCode: 200, Body: objBody(codec, &podList.Items[1])}, nil
			case p == testapi.ResourcePath("pods", "default", "baz") && m == "PUT":
				seen.Insert("baz")
				obj := readOrDie(t, req, codec)
				podList.Items[2] = *(obj.(*api.Pod))
				return &http.Response{StatusCode: 200, Body: objBody(codec, &podList.Items[2])}, nil
			case p == testapi.ResourcePath("replicationcontrollers", "default", "rc") && m == "PUT":
				updatedRc = true
				return &http.Response{StatusCode: 200, Body: objBody(codec, rc)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	clientConfig := &client.Config{Version: testapi.Version()}
	client := client.NewOrDie(clientConfig)
	client.Client = fakeClient.Client

	if _, err := AddDeploymentKeyToReplicationController(rc, client, "dk", "hash", api.NamespaceDefault, buf); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	for _, pod := range podList.Items {
		if !seen.Has(pod.Name) {
			t.Errorf("Missing update for pod: %s", pod.Name)
		}
	}
	if !updatedRc {
		t.Errorf("Failed to update replication controller with new labels")
	}
}

// rollingUpdaterClientImpl is a dynamic RollingUpdaterClient.
type rollingUpdaterClientImpl struct {
	ListReplicationControllersFn   func(namespace string, selector labels.Selector) (*api.ReplicationControllerList, error)
	GetReplicationControllerFn     func(namespace, name string) (*api.ReplicationController, error)
	UpdateReplicationControllerFn  func(namespace string, rc *api.ReplicationController) (*api.ReplicationController, error)
	CreateReplicationControllerFn  func(namespace string, rc *api.ReplicationController) (*api.ReplicationController, error)
	DeleteReplicationControllerFn  func(namespace, name string) error
	ControllerHasDesiredReplicasFn func(rc *api.ReplicationController) wait.ConditionFunc
}

func (c *rollingUpdaterClientImpl) ListReplicationControllers(namespace string, selector labels.Selector) (*api.ReplicationControllerList, error) {
	return c.ListReplicationControllersFn(namespace, selector)
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

type testAcceptor struct {
	accept func(*api.ReplicationController) error
}

func (a *testAcceptor) Accept(rc *api.ReplicationController) error {
	return a.accept(rc)
}
