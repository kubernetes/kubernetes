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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/testclient"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
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
				// 3 gets for each scale
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
				// 3 gets for each scale
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
				// 3 gets for each scale
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
				// final scale on newRc
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
Scaling foo-v2 replicas: 2 -> 7
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
		// 3 gets for each scale
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
			ControllerHasDesiredReplicasFn: func(rc *api.ReplicationController, timeout time.Duration) error {
				return nil
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
			case p == "/api/v1beta3/namespaces/default/replicationcontrollers/rc" && m == "PUT":
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
			case p == "/api/v1beta3/namespaces/default/replicationcontrollers/rc" && m == "GET":
				get := gets[0]
				gets = gets[1:]
				return get, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	clientConfig := &client.Config{Version: latest.Version}
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
			case p == "/api/v1beta3/namespaces/default/pods" && m == "GET":
				if req.URL.RawQuery != "labelSelector=foo%3Dbar" {
					t.Errorf("Unexpected query string: %s", req.URL.RawQuery)
				}
				return &http.Response{StatusCode: 200, Body: objBody(codec, podList)}, nil
			case p == "/api/v1beta3/namespaces/default/pods/foo" && m == "PUT":
				seen.Insert("foo")
				obj := readOrDie(t, req, codec)
				podList.Items[0] = *(obj.(*api.Pod))
				return &http.Response{StatusCode: 200, Body: objBody(codec, &podList.Items[0])}, nil
			case p == "/api/v1beta3/namespaces/default/pods/bar" && m == "PUT":
				seen.Insert("bar")
				obj := readOrDie(t, req, codec)
				podList.Items[1] = *(obj.(*api.Pod))
				return &http.Response{StatusCode: 200, Body: objBody(codec, &podList.Items[1])}, nil
			case p == "/api/v1beta3/namespaces/default/pods/baz" && m == "PUT":
				seen.Insert("baz")
				obj := readOrDie(t, req, codec)
				podList.Items[2] = *(obj.(*api.Pod))
				return &http.Response{StatusCode: 200, Body: objBody(codec, &podList.Items[2])}, nil
			case p == "/api/v1beta3/namespaces/default/replicationcontrollers/rc" && m == "PUT":
				updatedRc = true
				return &http.Response{StatusCode: 200, Body: objBody(codec, rc)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	clientConfig := &client.Config{Version: latest.Version}
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
	ControllerHasDesiredReplicasFn func(rc *api.ReplicationController, timeout time.Duration) error
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

func (c *rollingUpdaterClientImpl) ControllerHasDesiredReplicas(rc *api.ReplicationController, timeout time.Duration) error {
	return c.ControllerHasDesiredReplicasFn(rc, timeout)
}
