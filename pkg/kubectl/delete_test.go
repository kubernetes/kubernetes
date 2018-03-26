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

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	kerrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/watch"
	fakescale "k8s.io/client-go/scale/fake"
	testcore "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
)

func TestReplicationControllerStop(t *testing.T) {
	name := "foo"
	ns := "default"
	tests := []struct {
		Name                      string
		Objs                      []runtime.Object
		ScaledDown                bool
		StopError                 error
		ExpectedActions           []string
		ScaleClientExpectedAction []string
	}{
		{
			Name: "OnlyOneRC",
			Objs: []runtime.Object{
				&api.ReplicationControllerList{ // LIST
					Items: []api.ReplicationController{
						{
							ObjectMeta: metav1.ObjectMeta{
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
			ScaledDown:                true,
			StopError:                 nil,
			ExpectedActions:           []string{"get", "list", "delete"},
			ScaleClientExpectedAction: []string{"get", "update", "get", "get"},
		},
		{
			Name: "NoOverlapping",
			Objs: []runtime.Object{
				&api.ReplicationControllerList{ // LIST
					Items: []api.ReplicationController{
						{
							ObjectMeta: metav1.ObjectMeta{
								Name:      "baz",
								Namespace: ns,
							},
							Spec: api.ReplicationControllerSpec{
								Replicas: 0,
								Selector: map[string]string{"k3": "v3"}},
						},
						{
							ObjectMeta: metav1.ObjectMeta{
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
			ScaledDown:                true,
			StopError:                 nil,
			ExpectedActions:           []string{"get", "list", "delete"},
			ScaleClientExpectedAction: []string{"get", "update", "get", "get"},
		},
		{
			Name: "OverlappingError",
			Objs: []runtime.Object{

				&api.ReplicationControllerList{ // LIST
					Items: []api.ReplicationController{
						{
							ObjectMeta: metav1.ObjectMeta{
								Name:      "baz",
								Namespace: ns,
							},
							Spec: api.ReplicationControllerSpec{
								Replicas: 0,
								Selector: map[string]string{"k1": "v1", "k2": "v2"}},
						},
						{
							ObjectMeta: metav1.ObjectMeta{
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
			ScaledDown:      false, // scale resource was not scaled down due to overlapping controllers
			StopError:       fmt.Errorf("Detected overlapping controllers for rc foo: baz, please manage deletion individually with --cascade=false."),
			ExpectedActions: []string{"get", "list"},
		},
		{
			Name: "OverlappingButSafeDelete",
			Objs: []runtime.Object{

				&api.ReplicationControllerList{ // LIST
					Items: []api.ReplicationController{
						{
							ObjectMeta: metav1.ObjectMeta{
								Name:      "baz",
								Namespace: ns,
							},
							Spec: api.ReplicationControllerSpec{
								Replicas: 0,
								Selector: map[string]string{"k1": "v1", "k2": "v2", "k3": "v3"}},
						},
						{
							ObjectMeta: metav1.ObjectMeta{
								Name:      "zaz",
								Namespace: ns,
							},
							Spec: api.ReplicationControllerSpec{
								Replicas: 0,
								Selector: map[string]string{"k1": "v1"}},
						},
						{
							ObjectMeta: metav1.ObjectMeta{
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
			ScaledDown:      false, // scale resource was not scaled down due to overlapping controllers
			StopError:       fmt.Errorf("Detected overlapping controllers for rc foo: baz,zaz, please manage deletion individually with --cascade=false."),
			ExpectedActions: []string{"get", "list"},
		},

		{
			Name: "TwoExactMatchRCs",
			Objs: []runtime.Object{

				&api.ReplicationControllerList{ // LIST
					Items: []api.ReplicationController{
						{
							ObjectMeta: metav1.ObjectMeta{
								Name:      "zaz",
								Namespace: ns,
							},
							Spec: api.ReplicationControllerSpec{
								Replicas: 0,
								Selector: map[string]string{"k1": "v1"}},
						},
						{
							ObjectMeta: metav1.ObjectMeta{
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
			ScaledDown:      false, // scale resource was not scaled down because there is still an additional replica
			StopError:       nil,
			ExpectedActions: []string{"get", "list", "delete"},
		},
	}

	for _, test := range tests {
		copiedForWatch := test.Objs[0].DeepCopyObject()
		scaleClient := createFakeScaleClient("replicationcontrollers", "foo", 3, nil)
		fake := fake.NewSimpleClientset(test.Objs...)
		fakeWatch := watch.NewFake()
		fake.PrependWatchReactor("replicationcontrollers", testcore.DefaultWatchReactor(fakeWatch, nil))

		go func() {
			fakeWatch.Add(copiedForWatch)
		}()

		reaper := ReplicationControllerReaper{fake.Core(), time.Millisecond, time.Millisecond, scaleClient}
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
			if actions[i].GetResource().GroupResource() != api.Resource("replicationcontrollers") {
				t.Errorf("%s unexpected action: %+v, expected %s-replicationController", test.Name, actions[i], verb)
			}
			if actions[i].GetVerb() != verb {
				t.Errorf("%s unexpected action: %+v, expected %s-replicationController", test.Name, actions[i], verb)
			}
		}
		if test.ScaledDown {
			scale, err := scaleClient.Scales(ns).Get(schema.GroupResource{Group: "", Resource: "replicationcontrollers"}, name)
			if err != nil {
				t.Error(err)
			}
			if scale.Spec.Replicas != 0 {
				t.Errorf("a scale subresource has unexpected number of replicas, got %d expected 0", scale.Spec.Replicas)
			}
			actions := scaleClient.Actions()
			if len(actions) != len(test.ScaleClientExpectedAction) {
				t.Errorf("%s unexpected actions: %v, expected %d actions got %d", test.Name, actions, len(test.ScaleClientExpectedAction), len(actions))
			}
			for i, verb := range test.ScaleClientExpectedAction {
				if actions[i].GetVerb() != verb {
					t.Errorf("%s unexpected action: %+v, expected %s", test.Name, actions[i].GetVerb(), verb)
				}
			}
		}
	}
}

func TestReplicaSetStop(t *testing.T) {
	name := "foo"
	ns := "default"
	tests := []struct {
		Name                      string
		Objs                      []runtime.Object
		DiscoveryResources        []*metav1.APIResourceList
		PathsResources            map[string]runtime.Object
		ScaledDown                bool
		StopError                 error
		ExpectedActions           []string
		ScaleClientExpectedAction []string
	}{
		{
			Name: "OnlyOneRS",
			Objs: []runtime.Object{
				&extensions.ReplicaSetList{ // LIST
					TypeMeta: metav1.TypeMeta{
						APIVersion: extensions.SchemeGroupVersion.String(),
					},
					Items: []extensions.ReplicaSet{
						{
							ObjectMeta: metav1.ObjectMeta{
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
			ScaledDown:                true,
			StopError:                 nil,
			ExpectedActions:           []string{"get", "delete"},
			ScaleClientExpectedAction: []string{"get", "update", "get", "get"},
		},
		{
			Name: "NoOverlapping",
			Objs: []runtime.Object{
				&extensions.ReplicaSetList{ // LIST
					Items: []extensions.ReplicaSet{
						{
							ObjectMeta: metav1.ObjectMeta{
								Name:      "baz",
								Namespace: ns,
							},
							Spec: extensions.ReplicaSetSpec{
								Replicas: 0,
								Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"k3": "v3"}},
							},
						},
						{
							ObjectMeta: metav1.ObjectMeta{
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
			ScaledDown:                true,
			StopError:                 nil,
			ExpectedActions:           []string{"get", "delete"},
			ScaleClientExpectedAction: []string{"get", "update", "get", "get"},
		},
		// TODO: Implement tests for overlapping replica sets, similar to replication controllers,
		// when the overlapping checks are implemented for replica sets.
	}

	for _, test := range tests {
		fake := fake.NewSimpleClientset(test.Objs...)
		scaleClient := createFakeScaleClient("replicasets", "foo", 3, nil)

		reaper := ReplicaSetReaper{fake.Extensions(), time.Millisecond, time.Millisecond, scaleClient, schema.GroupResource{Group: "extensions", Resource: "replicasets"}}
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
		if test.ScaledDown {
			scale, err := scaleClient.Scales(ns).Get(schema.GroupResource{Group: "extensions", Resource: "replicasets"}, name)
			if err != nil {
				t.Error(err)
			}
			if scale.Spec.Replicas != 0 {
				t.Errorf("a scale subresource has unexpected number of replicas, got %d expected 0", scale.Spec.Replicas)
			}
			actions := scaleClient.Actions()
			if len(actions) != len(test.ScaleClientExpectedAction) {
				t.Errorf("%s unexpected actions: %v, expected %d actions got %d", test.Name, actions, len(test.ScaleClientExpectedAction), len(actions))
			}
			for i, verb := range test.ScaleClientExpectedAction {
				if actions[i].GetVerb() != verb {
					t.Errorf("%s unexpected action: %+v, expected %s", test.Name, actions[i].GetVerb(), verb)
				}
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
							ObjectMeta: metav1.ObjectMeta{
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
							ObjectMeta: metav1.ObjectMeta{
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
							ObjectMeta: metav1.ObjectMeta{
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
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			UID:       uuid.NewUUID(),
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
	trueVar := true
	tests := []struct {
		Name                      string
		Objs                      []runtime.Object
		ScaledDown                bool
		StopError                 error
		ExpectedActions           []string
		ScaleClientExpectedAction []string
	}{
		{
			Name: "SimpleDeployment",
			Objs: []runtime.Object{
				&extensions.Deployment{ // GET
					ObjectMeta: metav1.ObjectMeta{
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
						// ReplicaSet owned by this Deployment.
						{
							ObjectMeta: metav1.ObjectMeta{
								Name:      name,
								Namespace: ns,
								Labels:    map[string]string{"k1": "v1"},
								OwnerReferences: []metav1.OwnerReference{
									{
										APIVersion: extensions.SchemeGroupVersion.String(),
										Kind:       "Deployment",
										Name:       deployment.Name,
										UID:        deployment.UID,
										Controller: &trueVar,
									},
								},
							},
							Spec: extensions.ReplicaSetSpec{},
						},
						// ReplicaSet owned by something else (should be ignored).
						{
							ObjectMeta: metav1.ObjectMeta{
								Name:      "rs2",
								Namespace: ns,
								Labels:    map[string]string{"k1": "v1"},
								OwnerReferences: []metav1.OwnerReference{
									{
										APIVersion: extensions.SchemeGroupVersion.String(),
										Kind:       "Deployment",
										Name:       "somethingelse",
										UID:        uuid.NewUUID(),
										Controller: &trueVar,
									},
								},
							},
							Spec: extensions.ReplicaSetSpec{},
						},
					},
				},
			},
			ScaledDown: true,
			StopError:  nil,
			ExpectedActions: []string{"get:deployments", "update:deployments",
				"get:deployments", "list:replicasets", "get:replicasets",
				"delete:replicasets", "delete:deployments"},
			ScaleClientExpectedAction: []string{"get", "update", "get", "get"},
		},
	}

	for _, test := range tests {
		scaleClient := createFakeScaleClient("deployments", "foo", 3, nil)

		fake := fake.NewSimpleClientset(test.Objs...)
		reaper := DeploymentReaper{fake.Extensions(), fake.Extensions(), time.Millisecond, time.Millisecond, scaleClient, schema.GroupResource{Group: "extensions", Resource: "deployments"}}
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
		if test.ScaledDown {
			scale, err := scaleClient.Scales(ns).Get(schema.GroupResource{Group: "extensions", Resource: "replicaset"}, name)
			if err != nil {
				t.Error(err)
			}
			if scale.Spec.Replicas != 0 {
				t.Errorf("a scale subresource has unexpected number of replicas, got %d expected 0", scale.Spec.Replicas)
			}
			actions := scaleClient.Actions()
			if len(actions) != len(test.ScaleClientExpectedAction) {
				t.Errorf("%s unexpected actions: %v, expected %d actions got %d", test.Name, actions, len(test.ScaleClientExpectedAction), len(actions))
			}
			for i, verb := range test.ScaleClientExpectedAction {
				if actions[i].GetVerb() != verb {
					t.Errorf("%s unexpected action: %+v, expected %s", test.Name, actions[i].GetVerb(), verb)
				}
			}
		}
	}
}

type noSuchPod struct {
	coreclient.PodInterface
}

func (c *noSuchPod) Get(name string, options metav1.GetOptions) (*api.Pod, error) {
	return nil, fmt.Errorf("%s does not exist", name)
}

type noDeletePod struct {
	coreclient.PodInterface
}

func (c *noDeletePod) Delete(name string, o *metav1.DeleteOptions) error {
	return fmt.Errorf("I'm afraid I can't do that, Dave")
}

type reaperFake struct {
	*fake.Clientset
	noSuchPod, noDeletePod bool
}

func (c *reaperFake) Core() coreclient.CoreInterface {
	return &reaperCoreFake{c.Clientset.Core(), c.noSuchPod, c.noDeletePod}
}

type reaperCoreFake struct {
	coreclient.CoreInterface
	noSuchPod, noDeletePod bool
}

func (c *reaperCoreFake) Pods(namespace string) coreclient.PodInterface {
	pods := c.CoreInterface.Pods(namespace)
	if c.noSuchPod {
		return &noSuchPod{pods}
	}
	if c.noDeletePod {
		return &noDeletePod{pods}
	}
	return pods
}

func newPod() *api.Pod {
	return &api.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "foo"}}
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
				Clientset: fake.NewSimpleClientset(newPod()),
			},
			kind: api.Kind("Pod"),
			actions: []testcore.Action{
				testcore.NewGetAction(api.Resource("pods").WithVersion(""), metav1.NamespaceDefault, "foo"),
				testcore.NewDeleteAction(api.Resource("pods").WithVersion(""), metav1.NamespaceDefault, "foo"),
			},
			expectError: false,
			test:        "stop pod succeeds",
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
				Clientset:   fake.NewSimpleClientset(newPod()),
				noDeletePod: true,
			},
			kind: api.Kind("Pod"),
			actions: []testcore.Action{
				testcore.NewGetAction(api.Resource("pods").WithVersion(""), metav1.NamespaceDefault, "foo"),
			},
			expectError: true,
			test:        "stop pod fails, can't delete",
		},
	}
	for _, test := range tests {
		fake := test.fake
		reaper, err := ReaperFor(test.kind, fake, nil)
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
		ObjectMeta: metav1.ObjectMeta{
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

	fake := fake.NewSimpleClientset(
		deployment,
		&extensions.ReplicaSetList{Items: []extensions.ReplicaSet{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      name,
					Namespace: ns,
				},
				Spec: extensions.ReplicaSetSpec{},
			},
		},
		},
	)
	fake.AddReactor("get", "replicasets", func(action testcore.Action) (handled bool, ret runtime.Object, err error) {
		return true, nil, ScaleError{ActualError: errors.NewNotFound(api.Resource("replicaset"), "doesn't-matter")}
	})

	reaper := DeploymentReaper{fake.Extensions(), fake.Extensions(), time.Millisecond, time.Millisecond, nil, schema.GroupResource{}}
	if err := reaper.Stop(ns, name, 0, nil); err != nil {
		t.Fatalf("unexpected error: %#v", err)
	}
}

func createFakeScaleClient(resource string, resourceName string, replicas int, errorsOnVerb map[string]*kerrors.StatusError) *fakescale.FakeScaleClient {
	shouldReturnAnError := func(verb string) (*kerrors.StatusError, bool) {
		if anError, anErrorExists := errorsOnVerb[verb]; anErrorExists {
			return anError, true
		}
		return &kerrors.StatusError{}, false
	}
	newReplicas := int32(replicas)
	scaleClient := &fakescale.FakeScaleClient{}
	scaleClient.AddReactor("get", resource, func(rawAction testcore.Action) (handled bool, ret runtime.Object, err error) {
		action := rawAction.(testcore.GetAction)
		if action.GetName() != resourceName {
			return true, nil, fmt.Errorf("expected = %s, got = %s", resourceName, action.GetName())
		}
		if anError, should := shouldReturnAnError("get"); should {
			return true, nil, anError
		}
		obj := &autoscalingv1.Scale{
			ObjectMeta: metav1.ObjectMeta{
				Name:      action.GetName(),
				Namespace: action.GetNamespace(),
			},
			Spec: autoscalingv1.ScaleSpec{
				Replicas: newReplicas,
			},
		}
		return true, obj, nil
	})
	scaleClient.AddReactor("update", resource, func(rawAction testcore.Action) (handled bool, ret runtime.Object, err error) {
		action := rawAction.(testcore.UpdateAction)
		obj := action.GetObject().(*autoscalingv1.Scale)
		if obj.Name != resourceName {
			return true, nil, fmt.Errorf("expected = %s, got = %s", resourceName, obj.Name)
		}
		if anError, should := shouldReturnAnError("update"); should {
			return true, nil, anError
		}
		newReplicas = obj.Spec.Replicas
		return true, &autoscalingv1.Scale{
			ObjectMeta: metav1.ObjectMeta{
				Name:      obj.Name,
				Namespace: action.GetNamespace(),
			},
			Spec: autoscalingv1.ScaleSpec{
				Replicas: newReplicas,
			},
		}, nil
	})
	return scaleClient
}
