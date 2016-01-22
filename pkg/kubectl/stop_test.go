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
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/runtime"
)

const (
	name = "foo"
)

func overlappingButSafe() *api.ReplicationControllerList {
	return &api.ReplicationControllerList{
		Items: []api.ReplicationController{
			{
				ObjectMeta: api.ObjectMeta{
					Name:      name,
					Namespace: api.NamespaceDefault,
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 1,
					Selector: map[string]string{"k1": "v1", "k2": "v2"}},
			},
			{
				ObjectMeta: api.ObjectMeta{
					Name:      "baz",
					Namespace: api.NamespaceDefault,
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 2,
					Selector: map[string]string{"k1": "v1", "k2": "v2", "k3": "v3"}},
			},
			{
				ObjectMeta: api.ObjectMeta{
					Name:      "zaz",
					Namespace: api.NamespaceDefault,
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 3,
					Selector: map[string]string{"k1": "v1"}},
			},
		},
	}
}

func exactMatches() *api.ReplicationControllerList {
	return &api.ReplicationControllerList{
		Items: []api.ReplicationController{
			{
				ObjectMeta: api.ObjectMeta{
					Name:      "zaz",
					Namespace: api.NamespaceDefault,
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 3,
					Selector: map[string]string{"k1": "v1"}},
			},
			{
				ObjectMeta: api.ObjectMeta{
					Name:      name,
					Namespace: api.NamespaceDefault,
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 3,
					Selector: map[string]string{"k1": "v1"}},
			},
		},
	}
}

func TestReplicationControllerStop(t *testing.T) {
	// test data
	toBeReaped := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 5,
			Selector: map[string]string{"k1": "v1"}},
	}
	reaped := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 0,
			Selector: map[string]string{"k1": "v1"}},
	}
	noOverlapping := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:      "baz",
			Namespace: api.NamespaceDefault,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 3,
			Selector: map[string]string{"k3": "v3"}},
	}
	overlapping := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:      "baz",
			Namespace: api.NamespaceDefault,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 2,
			Selector: map[string]string{"k1": "v1", "k2": "v2"}},
	}

	// tests
	tests := []struct {
		Name            string
		Fns             []testclient.ReactionFunc
		StopError       error
		ExpectedActions []testclient.Action
	}{
		{
			Name: "OnlyOneRC",
			Fns: []testclient.ReactionFunc{
				// GET rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, toBeReaped, nil
				},
				// LIST rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, &api.ReplicationControllerList{
						Items: []api.ReplicationController{*toBeReaped}}, nil
				},
				// GET rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, toBeReaped, nil
				},
				// UPDATE rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, reaped, nil
				},
				// GET rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, reaped, nil
				},
				// GET rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, reaped, nil
				},
				// DELETE rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, reaped, nil
				},
			},
			StopError: nil,
			ExpectedActions: []testclient.Action{
				testclient.NewGetAction("replicationcontrollers", api.NamespaceDefault, name),
				testclient.NewListAction("replicationcontrollers", api.NamespaceDefault, api.ListOptions{}),
				testclient.NewGetAction("replicationcontrollers", api.NamespaceDefault, name),
				testclient.NewUpdateAction("replicationcontrollers", api.NamespaceDefault, reaped),
				testclient.NewGetAction("replicationcontrollers", api.NamespaceDefault, name),
				testclient.NewGetAction("replicationcontrollers", api.NamespaceDefault, name),
				testclient.NewDeleteAction("replicationcontrollers", api.NamespaceDefault, name),
			},
		},
		{
			Name: "RCWithNoPods",
			Fns: []testclient.ReactionFunc{
				// GET rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, reaped, nil
				},
				// LIST rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, &api.ReplicationControllerList{
						Items: []api.ReplicationController{*reaped}}, nil
				},
				// GET rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, reaped, nil
				},
				// DELETE rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, reaped, nil
				},
			},
			StopError: nil,
			ExpectedActions: []testclient.Action{
				testclient.NewGetAction("replicationcontrollers", api.NamespaceDefault, name),
				testclient.NewListAction("replicationcontrollers", api.NamespaceDefault, api.ListOptions{}),
				testclient.NewGetAction("replicationcontrollers", api.NamespaceDefault, name),
				testclient.NewDeleteAction("replicationcontrollers", api.NamespaceDefault, name),
			},
		},
		{
			Name: "NoOverlapping",
			Fns: []testclient.ReactionFunc{
				// GET rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, toBeReaped, nil
				},
				// LIST rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, &api.ReplicationControllerList{
						Items: []api.ReplicationController{*toBeReaped, *noOverlapping}}, nil
				},
				// GET rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, toBeReaped, nil
				},
				// UPDATE rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, reaped, nil
				},
				// GET rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, reaped, nil
				},
				// GET rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, reaped, nil
				},
				// DELETE rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, reaped, nil
				},
			},
			StopError: nil,
			ExpectedActions: []testclient.Action{
				testclient.NewGetAction("replicationcontrollers", api.NamespaceDefault, name),
				testclient.NewListAction("replicationcontrollers", api.NamespaceDefault, api.ListOptions{}),
				testclient.NewGetAction("replicationcontrollers", api.NamespaceDefault, name),
				testclient.NewUpdateAction("replicationcontrollers", api.NamespaceDefault, reaped),
				testclient.NewGetAction("replicationcontrollers", api.NamespaceDefault, name),
				testclient.NewGetAction("replicationcontrollers", api.NamespaceDefault, name),
				testclient.NewDeleteAction("replicationcontrollers", api.NamespaceDefault, name),
			},
		},
		{
			Name: "OverlappingError",
			Fns: []testclient.ReactionFunc{
				// GET rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, toBeReaped, nil
				},
				// LIST rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, &api.ReplicationControllerList{
						Items: []api.ReplicationController{*toBeReaped, *overlapping}}, nil
				},
			},
			StopError: fmt.Errorf("Detected overlapping controllers for rc foo: baz, please manage deletion individually with --cascade=false."),
			ExpectedActions: []testclient.Action{
				testclient.NewGetAction("replicationcontrollers", api.NamespaceDefault, name),
				testclient.NewListAction("replicationcontrollers", api.NamespaceDefault, api.ListOptions{}),
			},
		},
		{
			Name: "OverlappingButSafeDelete",
			Fns: []testclient.ReactionFunc{
				// GET rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, &overlappingButSafe().Items[0], nil
				},
				// LIST rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, overlappingButSafe(), nil
				},
			},
			StopError: fmt.Errorf("Detected overlapping controllers for rc foo: baz,zaz, please manage deletion individually with --cascade=false."),
			ExpectedActions: []testclient.Action{
				testclient.NewGetAction("replicationcontrollers", api.NamespaceDefault, name),
				testclient.NewListAction("replicationcontrollers", api.NamespaceDefault, api.ListOptions{}),
			},
		},
		{
			Name: "TwoExactMatchRCs",
			Fns: []testclient.ReactionFunc{
				// GET rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, toBeReaped, nil
				},
				// LIST rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, exactMatches(), nil
				},
				// GET rc
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, reaped, nil
				},
			},
			StopError: nil,
			ExpectedActions: []testclient.Action{
				testclient.NewGetAction("replicationcontrollers", api.NamespaceDefault, name),
				testclient.NewListAction("replicationcontrollers", api.NamespaceDefault, api.ListOptions{}),
				testclient.NewDeleteAction("replicationcontrollers", api.NamespaceDefault, name),
			},
		},
	}

	for _, test := range tests {
		toBeReaped.Spec.Replicas = 5
		fake := &testclient.Fake{}
		for i, reaction := range test.Fns {
			fake.AddReactor(test.ExpectedActions[i].GetVerb(), test.ExpectedActions[i].GetResource(), reaction)
		}
		reaper := ReplicationControllerReaper{fake, time.Millisecond, time.Millisecond}
		err := reaper.Stop(api.NamespaceDefault, name, 0, nil)
		if !reflect.DeepEqual(err, test.StopError) {
			t.Errorf("%s: unexpected error: %v", test.Name, err)
			continue
		}

		actions := fake.Actions()
		if len(test.ExpectedActions) != len(actions) {
			t.Errorf("%s: unexpected actions:\n%v\nexpected\n%v\n", test.Name, actions, test.ExpectedActions)
		}
		for i, action := range actions {
			testAction := test.ExpectedActions[i]
			if !testAction.Matches(action.GetVerb(), action.GetResource()) {
				t.Errorf("%s: unexpected action: %#v; expected %v", test.Name, action, testAction)
			}
		}
	}
}

func TestJobStop(t *testing.T) {
	zero := 0
	one := 1
	toBeReaped := &extensions.Job{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Spec: extensions.JobSpec{
			Parallelism: &one,
			Selector: &extensions.LabelSelector{
				MatchLabels: map[string]string{"k1": "v1"},
			},
		},
	}
	reaped := &extensions.Job{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Spec: extensions.JobSpec{
			Parallelism: &zero,
			Selector: &extensions.LabelSelector{
				MatchLabels: map[string]string{"k1": "v1"},
			},
		},
	}

	tests := []struct {
		Name            string
		Fns             []testclient.ReactionFunc
		StopError       error
		ExpectedActions []testclient.Action
	}{
		{
			Name: "OnlyOneJob",
			Fns: []testclient.ReactionFunc{
				// GET job
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, toBeReaped, nil
				},
				// GET job
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, toBeReaped, nil
				},
				// UPDATE job
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					*toBeReaped.Spec.Parallelism = 0
					return true, toBeReaped, nil
				},
				// GET job
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, toBeReaped, nil
				},
				// GET job
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, toBeReaped, nil
				},
				// LIST pods
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, &api.PodList{}, nil
				},
				// DELETE job
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, toBeReaped, nil
				},
			},
			StopError: nil,
			ExpectedActions: []testclient.Action{
				testclient.NewGetAction("jobs", api.NamespaceDefault, name),
				testclient.NewGetAction("jobs", api.NamespaceDefault, name),
				testclient.NewUpdateAction("jobs", api.NamespaceDefault, toBeReaped),
				testclient.NewGetAction("jobs", api.NamespaceDefault, name),
				testclient.NewGetAction("jobs", api.NamespaceDefault, name),
				testclient.NewListAction("pods", api.NamespaceDefault, api.ListOptions{}),
				testclient.NewDeleteAction("jobs", api.NamespaceDefault, name),
			},
		},
		{
			Name: "JobWithDeadPods",
			Fns: []testclient.ReactionFunc{
				// GET job
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, toBeReaped, nil
				},
				// GET job
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, toBeReaped, nil
				},
				// UPDATE job
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					*toBeReaped.Spec.Parallelism = 0
					return true, toBeReaped, nil
				},
				// GET job
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, toBeReaped, nil
				},
				// GET job
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, toBeReaped, nil
				},
				// LIST pods
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, &api.PodList{
						Items: []api.Pod{
							{
								ObjectMeta: api.ObjectMeta{
									Name:      "pod1",
									Namespace: api.NamespaceDefault,
									Labels:    map[string]string{"k1": "v1"},
								},
							},
						},
					}, nil
				},
				// DELETE pod
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, nil, nil
				},
				// DELETE job
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, toBeReaped, nil
				},
			},
			StopError: nil,
			ExpectedActions: []testclient.Action{
				testclient.NewGetAction("jobs", api.NamespaceDefault, name),
				testclient.NewGetAction("jobs", api.NamespaceDefault, name),
				testclient.NewUpdateAction("jobs", api.NamespaceDefault, toBeReaped),
				testclient.NewGetAction("jobs", api.NamespaceDefault, name),
				testclient.NewGetAction("jobs", api.NamespaceDefault, name),
				testclient.NewListAction("pods", api.NamespaceDefault, api.ListOptions{}),
				testclient.NewDeleteAction("pods", api.NamespaceDefault, name),
				testclient.NewDeleteAction("jobs", api.NamespaceDefault, name),
			},
		},
		{
			Name: "JobWithNoParallelism",
			Fns: []testclient.ReactionFunc{
				// GET job
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, reaped, nil
				},
				// GET job
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, reaped, nil
				},
				// LIST pods
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, &api.PodList{}, nil
				},
				// DELETE job
				func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
					return true, reaped, nil
				},
			},
			StopError: nil,
			ExpectedActions: []testclient.Action{
				testclient.NewGetAction("jobs", api.NamespaceDefault, name),
				testclient.NewGetAction("jobs", api.NamespaceDefault, name),
				testclient.NewListAction("pods", api.NamespaceDefault, api.ListOptions{}),
				testclient.NewDeleteAction("jobs", api.NamespaceDefault, name),
			},
		},
	}

	for _, test := range tests {
		*toBeReaped.Spec.Parallelism = one
		fake := &testclient.Fake{}
		for i, reaction := range test.Fns {
			fake.AddReactor(test.ExpectedActions[i].GetVerb(), test.ExpectedActions[i].GetResource(), reaction)
		}
		reaper := JobReaper{fake, time.Millisecond, time.Millisecond}
		err := reaper.Stop(api.NamespaceDefault, name, 0, nil)
		if !reflect.DeepEqual(err, test.StopError) {
			t.Errorf("%s unexpected error: %v", test.Name, err)
			continue
		}

		actions := fake.Actions()
		if len(test.ExpectedActions) != len(actions) {
			t.Errorf("%s: unexpected actions:\n%v\nexpected\n%v\n", test.Name, actions, test.ExpectedActions)
		}
		for i, action := range actions {
			testAction := test.ExpectedActions[i]
			if !testAction.Matches(action.GetVerb(), action.GetResource()) {
				t.Errorf("%s: unexpected action: %#v; expected %v", test.Name, action, testAction)
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
		kind        unversioned.GroupKind
		actions     []testclient.Action
		expectError bool
		test        string
	}{
		{
			fake: &reaperFake{
				Fake: &testclient.Fake{},
			},
			kind: api.Kind("Pod"),
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
			kind: api.Kind("Service"),
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
			kind:        api.Kind("Pod"),
			actions:     []testclient.Action{},
			expectError: true,
			test:        "stop pod fails, no pod",
		},
		{
			fake: &reaperFake{
				Fake:            &testclient.Fake{},
				noDeleteService: true,
			},
			kind: api.Kind("Service"),
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
		err = reaper.Stop("default", "foo", 0, nil)
		if err != nil && !test.expectError {
			t.Errorf("unexpected error: %v (%s)", err, test.test)
		}
		if err == nil {
			if test.expectError {
				t.Errorf("unexpected non-error: %v (%s)", err, test.test)
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
