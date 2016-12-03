/*
Copyright 2014 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	testcore "k8s.io/kubernetes/pkg/client/testing/core"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/watch"
)

func TestReplicationControllerStop(t *testing.T) {
	name := "foo"
	ns := "default"
	tests := []struct {
		Name            string
		Objs            []runtime.Object
		StopError       error
		ExpectedActions []string
	}{
		{
			Name: "OnlyOneRC",
			Objs: []runtime.Object{
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
			ExpectedActions: []string{"get", "list", "get", "update", "get", "delete"},
		},
		{
			Name: "NoOverlapping",
			Objs: []runtime.Object{
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
			ExpectedActions: []string{"get", "list", "get", "update", "get", "delete"},
		},
		{
			Name: "OverlappingError",
			Objs: []runtime.Object{

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
			ExpectedActions: []string{"get", "list"},
		},

		{
			Name: "OverlappingButSafeDelete",
			Objs: []runtime.Object{

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
			ExpectedActions: []string{"get", "list"},
		},

		{
			Name: "TwoExactMatchRCs",
			Objs: []runtime.Object{

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
			ExpectedActions: []string{"get", "list", "delete"},
		},
	}

	for _, test := range tests {
		copiedForWatch, err := api.Scheme.Copy(test.Objs[0])
		if err != nil {
			t.Fatalf("%s unexpected error: %v", test.Name, err)
		}
		fake := fake.NewSimpleClientset(test.Objs...)
		fakeWatch := watch.NewFake()
		fake.PrependWatchReactor("replicationcontrollers", testcore.DefaultWatchReactor(fakeWatch, nil))

		go func() {
			fakeWatch.Add(copiedForWatch)
		}()

		reaper := ReplicationControllerReaper{fake.Core(), time.Millisecond, time.Millisecond}
		err = reaper.Stop(ns, name, 0, nil)
		if !reflect.DeepEqual(err, test.StopError) {
			t.Errorf("%s unexpected error: %v", test.Name, err)
			continue
		}

		actions := fake.Actions()
		if len(actions) != len(test.ExpectedActions) {
			t.Errorf("%s unexpected actions: %v, expected %d actions got %d", test.Name, actions, len(test.ExpectedActions), len(actions))
			continue
		}
		for i, verb := range test.ExpectedActions {
			if actions[i].GetResource().GroupResource() != api.Resource("replicationcontrollers") {
				t.Errorf("%s unexpected action: %+v, expected %s-replicationController", test.Name, actions[i], verb)
			}
			if actions[i].GetVerb() != verb {
				t.Errorf("%s unexpected action: %+v, expected %s-replicationController", test.Name, actions[i], verb)
			}
		}
	}
}

func TestReplicaSetStop(t *testing.T) {
	name := "foo"
	ns := "default"
	tests := []struct {
		Name            string
		Objs            []runtime.Object
		StopError       error
		ExpectedActions []string
	}{
		{
			Name: "OnlyOneRS",
			Objs: []runtime.Object{
				&extensions.ReplicaSetList{ // LIST
					Items: []extensions.ReplicaSet{
						{
							ObjectMeta: api.ObjectMeta{
								Name:      name,
								Namespace: ns,
							},
							Spec: extensions.ReplicaSetSpec{
								Replicas: 0,
								Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"k1": "v1"}},
							},
						},
					},
				},
			},
			StopError:       nil,
			ExpectedActions: []string{"get", "get", "update", "get", "get", "delete"},
		},
		{
			Name: "NoOverlapping",
			Objs: []runtime.Object{
				&extensions.ReplicaSetList{ // LIST
					Items: []extensions.ReplicaSet{
						{
							ObjectMeta: api.ObjectMeta{
								Name:      "baz",
								Namespace: ns,
							},
							Spec: extensions.ReplicaSetSpec{
								Replicas: 0,
								Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"k3": "v3"}},
							},
						},
						{
							ObjectMeta: api.ObjectMeta{
								Name:      name,
								Namespace: ns,
							},
							Spec: extensions.ReplicaSetSpec{
								Replicas: 0,
								Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"k1": "v1"}},
							},
						},
					},
				},
			},
			StopError:       nil,
			ExpectedActions: []string{"get", "get", "update", "get", "get", "delete"},
		},
		// TODO: Implement tests for overlapping replica sets, similar to replication controllers,
		// when the overlapping checks are implemented for replica sets.
	}

	for _, test := range tests {
		fake := fake.NewSimpleClientset(test.Objs...)
		reaper := ReplicaSetReaper{fake.Extensions(), time.Millisecond, time.Millisecond}
		err := reaper.Stop(ns, name, 0, nil)
		if !reflect.DeepEqual(err, test.StopError) {
			t.Errorf("%s unexpected error: %v", test.Name, err)
			continue
		}

		actions := fake.Actions()
		if len(actions) != len(test.ExpectedActions) {
			t.Errorf("%s unexpected actions: %v, expected %d actions got %d", test.Name, actions, len(test.ExpectedActions), len(actions))
			continue
		}
		for i, verb := range test.ExpectedActions {
			if actions[i].GetResource().GroupResource() != extensions.Resource("replicasets") {
				t.Errorf("%s unexpected action: %+v, expected %s-replicaSet", test.Name, actions[i], verb)
			}
			if actions[i].GetVerb() != verb {
				t.Errorf("%s unexpected action: %+v, expected %s-replicaSet", test.Name, actions[i], verb)
			}
		}
	}
}

func TestJobStop(t *testing.T) {
	name := "foo"
	ns := "default"
	zero := int32(0)
	tests := []struct {
		Name            string
		Objs            []runtime.Object
		StopError       error
		ExpectedActions []string
	}{
		{
			Name: "OnlyOneJob",
			Objs: []runtime.Object{
				&batch.JobList{ // LIST
					Items: []batch.Job{
						{
							ObjectMeta: api.ObjectMeta{
								Name:      name,
								Namespace: ns,
							},
							Spec: batch.JobSpec{
								Parallelism: &zero,
								Selector: &metav1.LabelSelector{
									MatchLabels: map[string]string{"k1": "v1"},
								},
							},
						},
					},
				},
			},
			StopError: nil,
			ExpectedActions: []string{"get:jobs", "get:jobs", "update:jobs",
				"get:jobs", "get:jobs", "list:pods", "delete:jobs"},
		},
		{
			Name: "JobWithDeadPods",
			Objs: []runtime.Object{
				&batch.JobList{ // LIST
					Items: []batch.Job{
						{
							ObjectMeta: api.ObjectMeta{
								Name:      name,
								Namespace: ns,
							},
							Spec: batch.JobSpec{
								Parallelism: &zero,
								Selector: &metav1.LabelSelector{
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
			StopError: nil,
			ExpectedActions: []string{"get:jobs", "get:jobs", "update:jobs",
				"get:jobs", "get:jobs", "list:pods", "delete:pods", "delete:jobs"},
		},
	}

	for _, test := range tests {
		fake := fake.NewSimpleClientset(test.Objs...)
		reaper := JobReaper{fake.Batch(), fake.Core(), time.Millisecond, time.Millisecond}
		err := reaper.Stop(ns, name, 0, nil)
		if !reflect.DeepEqual(err, test.StopError) {
			t.Errorf("%s unexpected error: %v", test.Name, err)
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
			if actions[i].GetResource().Resource != action[1] {
				t.Errorf("%s unexpected resource: %+v, expected %s", test.Name, actions[i], expAction)
			}
		}
	}
}

func TestDeploymentStop(t *testing.T) {
	name := "foo"
	ns := "default"
	deployment := extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: extensions.DeploymentSpec{
			Replicas: 0,
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"k1": "v1"}},
		},
		Status: extensions.DeploymentStatus{
			Replicas: 0,
		},
	}
	template := deploymentutil.GetNewReplicaSetTemplateInternal(&deployment)
	tests := []struct {
		Name            string
		Objs            []runtime.Object
		StopError       error
		ExpectedActions []string
	}{
		{
			Name: "SimpleDeployment",
			Objs: []runtime.Object{
				&extensions.Deployment{ // GET
					ObjectMeta: api.ObjectMeta{
						Name:      name,
						Namespace: ns,
					},
					Spec: extensions.DeploymentSpec{
						Replicas: 0,
						Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"k1": "v1"}},
					},
					Status: extensions.DeploymentStatus{
						Replicas: 0,
					},
				},
			},
			StopError: nil,
			ExpectedActions: []string{"get:deployments", "update:deployments",
				"get:deployments", "list:replicasets", "delete:deployments"},
		},
		{
			Name: "Deployment with single replicaset",
			Objs: []runtime.Object{
				&deployment, // GET
				&extensions.ReplicaSetList{ // LIST
					Items: []extensions.ReplicaSet{
						{
							ObjectMeta: api.ObjectMeta{
								Name:      name,
								Namespace: ns,
								Labels:    map[string]string{"k1": "v1"},
							},
							Spec: extensions.ReplicaSetSpec{
								Template: template,
							},
						},
					},
				},
			},
			StopError: nil,
			ExpectedActions: []string{"get:deployments", "update:deployments",
				"get:deployments", "list:replicasets", "get:replicasets",
				"get:replicasets", "update:replicasets", "get:replicasets",
				"get:replicasets", "delete:replicasets", "delete:deployments"},
		},
	}

	for _, test := range tests {
		fake := fake.NewSimpleClientset(test.Objs...)
		reaper := DeploymentReaper{fake.Extensions(), fake.Extensions(), time.Millisecond, time.Millisecond}
		err := reaper.Stop(ns, name, 0, nil)
		if !reflect.DeepEqual(err, test.StopError) {
			t.Errorf("%s unexpected error: %v", test.Name, err)
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
			if actions[i].GetResource().Resource != action[1] {
				t.Errorf("%s unexpected resource: %+v, expected %s", test.Name, actions[i], expAction)
			}
			if len(action) == 3 && actions[i].GetSubresource() != action[2] {
				t.Errorf("%s unexpected subresource: %+v, expected %s", test.Name, actions[i], expAction)
			}
		}
	}
}

type noSuchPod struct {
	coreclient.PodInterface
}

func (c *noSuchPod) Get(name string) (*api.Pod, error) {
	return nil, fmt.Errorf("%s does not exist", name)
}

type noDeleteService struct {
	coreclient.ServiceInterface
}

func (c *noDeleteService) Delete(service string, o *api.DeleteOptions) error {
	return fmt.Errorf("I'm afraid I can't do that, Dave")
}

type reaperFake struct {
	*fake.Clientset
	noSuchPod, noDeleteService bool
}

func (c *reaperFake) Core() coreclient.CoreInterface {
	return &reaperCoreFake{c.Clientset.Core(), c.noSuchPod, c.noDeleteService}
}

type reaperCoreFake struct {
	coreclient.CoreInterface
	noSuchPod, noDeleteService bool
}

func (c *reaperCoreFake) Pods(namespace string) coreclient.PodInterface {
	pods := c.CoreInterface.Pods(namespace)
	if c.noSuchPod {
		return &noSuchPod{pods}
	}
	return pods
}

func (c *reaperCoreFake) Services(namespace string) coreclient.ServiceInterface {
	services := c.CoreInterface.Services(namespace)
	if c.noDeleteService {
		return &noDeleteService{services}
	}
	return services
}

func pod() *api.Pod {
	return &api.Pod{ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "foo"}}
}

func service() *api.Service {
	return &api.Service{ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "foo"}}
}

func TestSimpleStop(t *testing.T) {
	tests := []struct {
		fake        *reaperFake
		kind        schema.GroupKind
		actions     []testcore.Action
		expectError bool
		test        string
	}{
		{
			fake: &reaperFake{
				Clientset: fake.NewSimpleClientset(pod()),
			},
			kind: api.Kind("Pod"),
			actions: []testcore.Action{
				testcore.NewGetAction(api.Resource("pods").WithVersion(""), api.NamespaceDefault, "foo"),
				testcore.NewDeleteAction(api.Resource("pods").WithVersion(""), api.NamespaceDefault, "foo"),
			},
			expectError: false,
			test:        "stop pod succeeds",
		},
		{
			fake: &reaperFake{
				Clientset: fake.NewSimpleClientset(service()),
			},
			kind: api.Kind("Service"),
			actions: []testcore.Action{
				testcore.NewGetAction(api.Resource("services").WithVersion(""), api.NamespaceDefault, "foo"),
				testcore.NewDeleteAction(api.Resource("services").WithVersion(""), api.NamespaceDefault, "foo"),
			},
			expectError: false,
			test:        "stop service succeeds",
		},
		{
			fake: &reaperFake{
				Clientset: fake.NewSimpleClientset(),
				noSuchPod: true,
			},
			kind:        api.Kind("Pod"),
			actions:     []testcore.Action{},
			expectError: true,
			test:        "stop pod fails, no pod",
		},
		{
			fake: &reaperFake{
				Clientset:       fake.NewSimpleClientset(service()),
				noDeleteService: true,
			},
			kind: api.Kind("Service"),
			actions: []testcore.Action{
				testcore.NewGetAction(api.Resource("services").WithVersion(""), api.NamespaceDefault, "foo"),
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
			t.Errorf("unexpected actions: %v; expected %v (%s)", actions, test.actions, test.test)
		}
		for i, action := range actions {
			testAction := test.actions[i]
			if action != testAction {
				t.Errorf("unexpected action: %#v; expected %v (%s)", action, testAction, test.test)
			}
		}
	}
}

func TestDeploymentNotFoundError(t *testing.T) {
	name := "foo"
	ns := "default"
	deployment := &extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: extensions.DeploymentSpec{
			Replicas: 0,
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"k1": "v1"}},
		},
		Status: extensions.DeploymentStatus{
			Replicas: 0,
		},
	}
	template := deploymentutil.GetNewReplicaSetTemplateInternal(deployment)

	fake := fake.NewSimpleClientset(
		deployment,
		&extensions.ReplicaSetList{Items: []extensions.ReplicaSet{
			{
				ObjectMeta: api.ObjectMeta{
					Name:      name,
					Namespace: ns,
				},
				Spec: extensions.ReplicaSetSpec{
					Template: template,
				},
			},
		},
		},
	)
	fake.AddReactor("get", "replicasets", func(action testcore.Action) (handled bool, ret runtime.Object, err error) {
		return true, nil, ScaleError{ActualError: errors.NewNotFound(api.Resource("replicaset"), "doesn't-matter")}
	})

	reaper := DeploymentReaper{fake.Extensions(), fake.Extensions(), time.Millisecond, time.Millisecond}
	if err := reaper.Stop(ns, name, 0, nil); err != nil {
		t.Fatalf("unexpected error: %#v", err)
	}
}
