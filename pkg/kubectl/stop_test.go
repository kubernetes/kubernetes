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
	"reflect"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/runtime"
)

func TestReplicationControllerStop(t *testing.T) {
	name := "foo"
	ns := "default"
	tests := []struct {
		Name            string
		Objs            []runtime.Object
		StopError       error
		StopMessage     string
		ExpectedActions []string
	}{
		{
			Name: "OnlyOneRC",
			Objs: []runtime.Object{
				&api.ReplicationController{ // GET
					ObjectMeta: api.ObjectMeta{
						Name:      name,
						Namespace: ns,
					},
					Spec: api.ReplicationControllerSpec{
						Replicas: 0,
						Selector: map[string]string{"k1": "v1"}},
				},
				&api.ReplicationControllerList{ // LIST
					Items: []api.ReplicationController{
						{
							ObjectMeta: api.ObjectMeta{
								Name:      name,
								Namespace: ns,
							},
							Spec: api.ReplicationControllerSpec{
								Replicas: 0,
								Selector: map[string]string{"k1": "v1"}},
						},
					},
				},
			},
			StopError:       nil,
			StopMessage:     "foo stopped",
			ExpectedActions: []string{"get", "list", "get", "update", "get", "get", "delete"},
		},
		{
			Name: "NoOverlapping",
			Objs: []runtime.Object{
				&api.ReplicationController{ // GET
					ObjectMeta: api.ObjectMeta{
						Name:      name,
						Namespace: ns,
					},
					Spec: api.ReplicationControllerSpec{
						Replicas: 0,
						Selector: map[string]string{"k1": "v1"}},
				},
				&api.ReplicationControllerList{ // LIST
					Items: []api.ReplicationController{
						{
							ObjectMeta: api.ObjectMeta{
								Name:      "baz",
								Namespace: ns,
							},
							Spec: api.ReplicationControllerSpec{
								Replicas: 0,
								Selector: map[string]string{"k3": "v3"}},
						},
						{
							ObjectMeta: api.ObjectMeta{
								Name:      name,
								Namespace: ns,
							},
							Spec: api.ReplicationControllerSpec{
								Replicas: 0,
								Selector: map[string]string{"k1": "v1"}},
						},
					},
				},
			},
			StopError:       nil,
			StopMessage:     "foo stopped",
			ExpectedActions: []string{"get", "list", "get", "update", "get", "get", "delete"},
		},
		{
			Name: "OverlappingError",
			Objs: []runtime.Object{

				&api.ReplicationController{ // GET
					ObjectMeta: api.ObjectMeta{
						Name:      name,
						Namespace: ns,
					},
					Spec: api.ReplicationControllerSpec{
						Replicas: 0,
						Selector: map[string]string{"k1": "v1"}},
				},
				&api.ReplicationControllerList{ // LIST
					Items: []api.ReplicationController{
						{
							ObjectMeta: api.ObjectMeta{
								Name:      "baz",
								Namespace: ns,
							},
							Spec: api.ReplicationControllerSpec{
								Replicas: 0,
								Selector: map[string]string{"k1": "v1", "k2": "v2"}},
						},
						{
							ObjectMeta: api.ObjectMeta{
								Name:      name,
								Namespace: ns,
							},
							Spec: api.ReplicationControllerSpec{
								Replicas: 0,
								Selector: map[string]string{"k1": "v1"}},
						},
					},
				},
			},
			StopError:       fmt.Errorf("Detected overlapping controllers for rc foo: baz, please manage deletion individually with --cascade=false."),
			StopMessage:     "",
			ExpectedActions: []string{"get", "list"},
		},

		{
			Name: "OverlappingButSafeDelete",
			Objs: []runtime.Object{

				&api.ReplicationController{ // GET
					ObjectMeta: api.ObjectMeta{
						Name:      name,
						Namespace: ns,
					},
					Spec: api.ReplicationControllerSpec{
						Replicas: 0,
						Selector: map[string]string{"k1": "v1", "k2": "v2"}},
				},
				&api.ReplicationControllerList{ // LIST
					Items: []api.ReplicationController{
						{
							ObjectMeta: api.ObjectMeta{
								Name:      "baz",
								Namespace: ns,
							},
							Spec: api.ReplicationControllerSpec{
								Replicas: 0,
								Selector: map[string]string{"k1": "v1", "k2": "v2", "k3": "v3"}},
						},
						{
							ObjectMeta: api.ObjectMeta{
								Name:      "zaz",
								Namespace: ns,
							},
							Spec: api.ReplicationControllerSpec{
								Replicas: 0,
								Selector: map[string]string{"k1": "v1"}},
						},
						{
							ObjectMeta: api.ObjectMeta{
								Name:      name,
								Namespace: ns,
							},
							Spec: api.ReplicationControllerSpec{
								Replicas: 0,
								Selector: map[string]string{"k1": "v1", "k2": "v2"}},
						},
					},
				},
			},

			StopError:       fmt.Errorf("Detected overlapping controllers for rc foo: baz,zaz, please manage deletion individually with --cascade=false."),
			StopMessage:     "",
			ExpectedActions: []string{"get", "list"},
		},

		{
			Name: "TwoExactMatchRCs",
			Objs: []runtime.Object{

				&api.ReplicationController{ // GET
					ObjectMeta: api.ObjectMeta{
						Name:      name,
						Namespace: ns,
					},
					Spec: api.ReplicationControllerSpec{
						Replicas: 0,
						Selector: map[string]string{"k1": "v1"}},
				},
				&api.ReplicationControllerList{ // LIST
					Items: []api.ReplicationController{
						{
							ObjectMeta: api.ObjectMeta{
								Name:      "zaz",
								Namespace: ns,
							},
							Spec: api.ReplicationControllerSpec{
								Replicas: 0,
								Selector: map[string]string{"k1": "v1"}},
						},
						{
							ObjectMeta: api.ObjectMeta{
								Name:      name,
								Namespace: ns,
							},
							Spec: api.ReplicationControllerSpec{
								Replicas: 0,
								Selector: map[string]string{"k1": "v1"}},
						},
					},
				},
			},

			StopError:       nil,
			StopMessage:     "foo stopped",
			ExpectedActions: []string{"get", "list", "delete"},
		},
	}

	for _, test := range tests {
		fake := testclient.NewSimpleFake(test.Objs...)
		reaper := ReplicationControllerReaper{fake, time.Millisecond, time.Millisecond}
		s, err := reaper.Stop(ns, name, 0, nil)
		if !reflect.DeepEqual(err, test.StopError) {
			t.Errorf("%s unexpected error: %v", test.Name, err)
			continue
		}

		if s != test.StopMessage {
			t.Errorf("%s expected '%s', got '%s'", test.Name, test.StopMessage, s)
			continue
		}
		actions := fake.Actions()
		if len(actions) != len(test.ExpectedActions) {
			t.Errorf("%s unexpected actions: %v, expected %d actions got %d", test.Name, actions, len(test.ExpectedActions), len(actions))
			continue
		}
		for i, verb := range test.ExpectedActions {
			if actions[i].GetResource() != "replicationcontrollers" {
				t.Errorf("%s unexpected action: %+v, expected %s-replicationController", test.Name, actions[i], verb)
			}
			if actions[i].GetVerb() != verb {
				t.Errorf("%s unexpected action: %+v, expected %s-replicationController", test.Name, actions[i], verb)
			}
		}
	}
}

func TestJobStop(t *testing.T) {
	name := "foo"
	ns := "default"
	zero := 0
	tests := []struct {
		Name            string
		Objs            []runtime.Object
		StopError       error
		StopMessage     string
		ExpectedActions []string
	}{
		{
			Name: "OnlyOneJob",
			Objs: []runtime.Object{
				&extensions.Job{ // GET
					ObjectMeta: api.ObjectMeta{
						Name:      name,
						Namespace: ns,
					},
					Spec: extensions.JobSpec{
						Parallelism: &zero,
						Selector: &extensions.PodSelector{
							MatchLabels: map[string]string{"k1": "v1"},
						},
					},
				},
				&extensions.JobList{ // LIST
					Items: []extensions.Job{
						{
							ObjectMeta: api.ObjectMeta{
								Name:      name,
								Namespace: ns,
							},
							Spec: extensions.JobSpec{
								Parallelism: &zero,
								Selector: &extensions.PodSelector{
									MatchLabels: map[string]string{"k1": "v1"},
								},
							},
						},
					},
				},
			},
			StopError:   nil,
			StopMessage: "foo stopped",
			ExpectedActions: []string{"get:jobs", "get:jobs", "update:jobs",
				"get:jobs", "get:jobs", "list:pods", "delete:jobs"},
		},
		{
			Name: "JobWithDeadPods",
			Objs: []runtime.Object{
				&extensions.Job{ // GET
					ObjectMeta: api.ObjectMeta{
						Name:      name,
						Namespace: ns,
					},
					Spec: extensions.JobSpec{
						Parallelism: &zero,
						Selector: &extensions.PodSelector{
							MatchLabels: map[string]string{"k1": "v1"},
						},
					},
				},
				&extensions.JobList{ // LIST
					Items: []extensions.Job{
						{
							ObjectMeta: api.ObjectMeta{
								Name:      name,
								Namespace: ns,
							},
							Spec: extensions.JobSpec{
								Parallelism: &zero,
								Selector: &extensions.PodSelector{
									MatchLabels: map[string]string{"k1": "v1"},
								},
							},
						},
					},
				},
				&api.PodList{ // LIST
					Items: []api.Pod{
						{
							ObjectMeta: api.ObjectMeta{
								Name:      "pod1",
								Namespace: ns,
								Labels:    map[string]string{"k1": "v1"},
							},
						},
					},
				},
			},
			StopError:   nil,
			StopMessage: "foo stopped",
			ExpectedActions: []string{"get:jobs", "get:jobs", "update:jobs",
				"get:jobs", "get:jobs", "list:pods", "delete:pods", "delete:jobs"},
		},
	}

	for _, test := range tests {
		fake := testclient.NewSimpleFake(test.Objs...)
		reaper := JobReaper{fake, time.Millisecond, time.Millisecond}
		s, err := reaper.Stop(ns, name, 0, nil)
		if !reflect.DeepEqual(err, test.StopError) {
			t.Errorf("%s unexpected error: %v", test.Name, err)
			continue
		}

		if s != test.StopMessage {
			t.Errorf("%s expected '%s', got '%s'", test.Name, test.StopMessage, s)
			continue
		}
		actions := fake.Actions()
		if len(actions) != len(test.ExpectedActions) {
			t.Errorf("%s unexpected actions: %v, expected %d actions got %d", test.Name, actions, len(test.ExpectedActions), len(actions))
			continue
		}
		for i, expAction := range test.ExpectedActions {
			action := strings.Split(expAction, ":")
			if actions[i].GetVerb() != action[0] {
				t.Errorf("%s unexpected verb: %+v, expected %s", test.Name, actions[i], expAction)
			}
			if actions[i].GetResource() != action[1] {
				t.Errorf("%s unexpected resource: %+v, expected %s", test.Name, actions[i], expAction)
			}
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
	pods := &testclient.FakePods{Fake: c.Fake, Namespace: namespace}
	if c.noSuchPod {
		return &noSuchPod{pods}
	}
	return pods
}

func (c *reaperFake) Services(namespace string) client.ServiceInterface {
	services := &testclient.FakeServices{Fake: c.Fake, Namespace: namespace}
	if c.noDeleteService {
		return &noDeleteService{services}
	}
	return services
}

func TestSimpleStop(t *testing.T) {
	tests := []struct {
		fake        *reaperFake
		kind        string
		actions     []testclient.Action
		expectError bool
		test        string
	}{
		{
			fake: &reaperFake{
				Fake: &testclient.Fake{},
			},
			kind: "Pod",
			actions: []testclient.Action{
				testclient.NewGetAction("pods", api.NamespaceDefault, "foo"),
				testclient.NewDeleteAction("pods", api.NamespaceDefault, "foo"),
			},
			expectError: false,
			test:        "stop pod succeeds",
		},
		{
			fake: &reaperFake{
				Fake: &testclient.Fake{},
			},
			kind: "Service",
			actions: []testclient.Action{
				testclient.NewGetAction("services", api.NamespaceDefault, "foo"),
				testclient.NewDeleteAction("services", api.NamespaceDefault, "foo"),
			},
			expectError: false,
			test:        "stop service succeeds",
		},
		{
			fake: &reaperFake{
				Fake:      &testclient.Fake{},
				noSuchPod: true,
			},
			kind:        "Pod",
			actions:     []testclient.Action{},
			expectError: true,
			test:        "stop pod fails, no pod",
		},
		{
			fake: &reaperFake{
				Fake:            &testclient.Fake{},
				noDeleteService: true,
			},
			kind: "Service",
			actions: []testclient.Action{
				testclient.NewGetAction("services", api.NamespaceDefault, "foo"),
			},
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
			if action != testAction {
				t.Errorf("unexpected action: %#v; expected %v (%s)", action, testAction, test.test)
			}
		}
	}
}
