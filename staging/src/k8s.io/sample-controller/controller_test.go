/*
Copyright 2017 The Kubernetes Authors.

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

package main

import (
	"context"
	"fmt"
	"reflect"
	"testing"
	"time"

	apps "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	kubeinformers "k8s.io/client-go/informers"
	k8sfake "k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"

	samplecontroller "k8s.io/sample-controller/pkg/apis/samplecontroller/v1alpha1"
	"k8s.io/sample-controller/pkg/generated/clientset/versioned/fake"
	informers "k8s.io/sample-controller/pkg/generated/informers/externalversions"
)

var (
	alwaysReady        = func() bool { return true }
	noResyncPeriodFunc = func() time.Duration { return 0 }
)

type fixture struct {
	t *testing.T

	client     *fake.Clientset
	kubeclient *k8sfake.Clientset
	// Objects to put in the store.
	fooLister        []*samplecontroller.Foo
	deploymentLister []*apps.Deployment
	// Actions expected to happen on the client.
	kubeactions []core.Action
	actions     []core.Action
	// Objects from here preloaded into NewSimpleFake.
	kubeobjects []runtime.Object
	objects     []runtime.Object
}

func newFixture(t *testing.T) *fixture {
	f := &fixture{}
	f.t = t
	f.objects = []runtime.Object{}
	f.kubeobjects = []runtime.Object{}
	return f
}

func newFoo(name string, replicas *int32) *samplecontroller.Foo {
	return &samplecontroller.Foo{
		TypeMeta: metav1.TypeMeta{APIVersion: samplecontroller.SchemeGroupVersion.String()},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: metav1.NamespaceDefault,
		},
		Spec: samplecontroller.FooSpec{
			DeploymentName: fmt.Sprintf("%s-deployment", name),
			Replicas:       replicas,
		},
	}
}

func (f *fixture) newController(ctx context.Context) (*Controller, informers.SharedInformerFactory, kubeinformers.SharedInformerFactory) {
	f.client = fake.NewSimpleClientset(f.objects...)
	f.kubeclient = k8sfake.NewSimpleClientset(f.kubeobjects...)

	i := informers.NewSharedInformerFactory(f.client, noResyncPeriodFunc())
	k8sI := kubeinformers.NewSharedInformerFactory(f.kubeclient, noResyncPeriodFunc())

	c := NewController(ctx, f.kubeclient, f.client,
		k8sI.Apps().V1().Deployments(), i.Samplecontroller().V1alpha1().Foos())

	c.foosSynced = alwaysReady
	c.deploymentsSynced = alwaysReady
	c.recorder = &record.FakeRecorder{}

	for _, f := range f.fooLister {
		i.Samplecontroller().V1alpha1().Foos().Informer().GetIndexer().Add(f)
	}

	for _, d := range f.deploymentLister {
		k8sI.Apps().V1().Deployments().Informer().GetIndexer().Add(d)
	}

	return c, i, k8sI
}

func (f *fixture) run(ctx context.Context, fooRef cache.ObjectName) {
	f.runController(ctx, fooRef, true, false)
}

func (f *fixture) runExpectError(ctx context.Context, fooRef cache.ObjectName) {
	f.runController(ctx, fooRef, true, true)
}

func (f *fixture) runController(ctx context.Context, fooRef cache.ObjectName, startInformers bool, expectError bool) {
	c, i, k8sI := f.newController(ctx)
	if startInformers {
		i.Start(ctx.Done())
		k8sI.Start(ctx.Done())
	}

	err := c.syncHandler(ctx, fooRef)
	if !expectError && err != nil {
		f.t.Errorf("error syncing foo: %v", err)
	} else if expectError && err == nil {
		f.t.Error("expected error syncing foo, got nil")
	}

	actions := filterInformerActions(f.client.Actions())
	for i, action := range actions {
		if len(f.actions) < i+1 {
			f.t.Errorf("%d unexpected actions: %+v", len(actions)-len(f.actions), actions[i:])
			break
		}

		expectedAction := f.actions[i]
		checkAction(expectedAction, action, f.t)
	}

	if len(f.actions) > len(actions) {
		f.t.Errorf("%d additional expected actions:%+v", len(f.actions)-len(actions), f.actions[len(actions):])
	}

	k8sActions := filterInformerActions(f.kubeclient.Actions())
	for i, action := range k8sActions {
		if len(f.kubeactions) < i+1 {
			f.t.Errorf("%d unexpected actions: %+v", len(k8sActions)-len(f.kubeactions), k8sActions[i:])
			break
		}

		expectedAction := f.kubeactions[i]
		checkAction(expectedAction, action, f.t)
	}

	if len(f.kubeactions) > len(k8sActions) {
		f.t.Errorf("%d additional expected actions:%+v", len(f.kubeactions)-len(k8sActions), f.kubeactions[len(k8sActions):])
	}
}

// checkAction verifies that expected and actual actions are equal and both have
// same attached resources
func checkAction(expected, actual core.Action, t *testing.T) {
	if !(expected.Matches(actual.GetVerb(), actual.GetResource().Resource) && actual.GetSubresource() == expected.GetSubresource()) {
		t.Errorf("Expected\n\t%#v\ngot\n\t%#v", expected, actual)
		return
	}

	if reflect.TypeOf(actual) != reflect.TypeOf(expected) {
		t.Errorf("Action has wrong type. Expected: %t. Got: %t", expected, actual)
		return
	}

	switch a := actual.(type) {
	case core.CreateActionImpl:
		e, _ := expected.(core.CreateActionImpl)
		expObject := e.GetObject()
		object := a.GetObject()

		if !reflect.DeepEqual(expObject, object) {
			t.Errorf("Action %s %s has wrong object\nDiff:\n %s",
				a.GetVerb(), a.GetResource().Resource, diff.ObjectGoPrintSideBySide(expObject, object))
		}
	case core.UpdateActionImpl:
		e, _ := expected.(core.UpdateActionImpl)
		expObject := e.GetObject()
		object := a.GetObject()

		if !reflect.DeepEqual(expObject, object) {
			t.Errorf("Action %s %s has wrong object\nDiff:\n %s",
				a.GetVerb(), a.GetResource().Resource, diff.ObjectGoPrintSideBySide(expObject, object))
		}
	case core.PatchActionImpl:
		e, _ := expected.(core.PatchActionImpl)
		expPatch := e.GetPatch()
		patch := a.GetPatch()

		if !reflect.DeepEqual(expPatch, patch) {
			t.Errorf("Action %s %s has wrong patch\nDiff:\n %s",
				a.GetVerb(), a.GetResource().Resource, diff.ObjectGoPrintSideBySide(expPatch, patch))
		}
	default:
		t.Errorf("Uncaptured Action %s %s, you should explicitly add a case to capture it",
			actual.GetVerb(), actual.GetResource().Resource)
	}
}

// filterInformerActions filters list and watch actions for testing resources.
// Since list and watch don't change resource state we can filter it to lower
// nose level in our tests.
func filterInformerActions(actions []core.Action) []core.Action {
	ret := []core.Action{}
	for _, action := range actions {
		if len(action.GetNamespace()) == 0 &&
			(action.Matches("list", "foos") ||
				action.Matches("watch", "foos") ||
				action.Matches("list", "deployments") ||
				action.Matches("watch", "deployments")) {
			continue
		}
		ret = append(ret, action)
	}

	return ret
}

func (f *fixture) expectCreateDeploymentAction(d *apps.Deployment) {
	f.kubeactions = append(f.kubeactions, core.NewCreateAction(schema.GroupVersionResource{Resource: "deployments"}, d.Namespace, d))
}

func (f *fixture) expectUpdateDeploymentAction(d *apps.Deployment) {
	f.kubeactions = append(f.kubeactions, core.NewUpdateAction(schema.GroupVersionResource{Resource: "deployments"}, d.Namespace, d))
}

func (f *fixture) expectUpdateFooStatusAction(foo *samplecontroller.Foo) {
	action := core.NewUpdateSubresourceAction(schema.GroupVersionResource{Resource: "foos"}, "status", foo.Namespace, foo)
	f.actions = append(f.actions, action)
}

func getRef(foo *samplecontroller.Foo, t *testing.T) cache.ObjectName {
	ref := cache.MetaObjectToName(foo)
	return ref
}

func TestCreatesDeployment(t *testing.T) {
	f := newFixture(t)
	foo := newFoo("test", ptr.To[int32](1))
	_, ctx := ktesting.NewTestContext(t)

	f.fooLister = append(f.fooLister, foo)
	f.objects = append(f.objects, foo)

	expDeployment := newDeployment(foo)
	f.expectCreateDeploymentAction(expDeployment)
	f.expectUpdateFooStatusAction(foo)

	f.run(ctx, getRef(foo, t))
}

func TestDoNothing(t *testing.T) {
	f := newFixture(t)
	foo := newFoo("test", ptr.To[int32](1))
	_, ctx := ktesting.NewTestContext(t)

	d := newDeployment(foo)

	f.fooLister = append(f.fooLister, foo)
	f.objects = append(f.objects, foo)
	f.deploymentLister = append(f.deploymentLister, d)
	f.kubeobjects = append(f.kubeobjects, d)

	f.expectUpdateFooStatusAction(foo)
	f.run(ctx, getRef(foo, t))
}

func TestUpdateDeployment(t *testing.T) {
	f := newFixture(t)
	foo := newFoo("test", ptr.To[int32](1))
	_, ctx := ktesting.NewTestContext(t)

	d := newDeployment(foo)

	// Update replicas
	foo.Spec.Replicas = ptr.To[int32](2)
	expDeployment := newDeployment(foo)

	f.fooLister = append(f.fooLister, foo)
	f.objects = append(f.objects, foo)
	f.deploymentLister = append(f.deploymentLister, d)
	f.kubeobjects = append(f.kubeobjects, d)

	f.expectUpdateFooStatusAction(foo)
	f.expectUpdateDeploymentAction(expDeployment)
	f.run(ctx, getRef(foo, t))
}

func TestNotControlledByUs(t *testing.T) {
	f := newFixture(t)
	foo := newFoo("test", ptr.To[int32](1))
	_, ctx := ktesting.NewTestContext(t)

	d := newDeployment(foo)

	d.ObjectMeta.OwnerReferences = []metav1.OwnerReference{}

	f.fooLister = append(f.fooLister, foo)
	f.objects = append(f.objects, foo)
	f.deploymentLister = append(f.deploymentLister, d)
	f.kubeobjects = append(f.kubeobjects, d)

	f.runExpectError(ctx, getRef(foo, t))
}
