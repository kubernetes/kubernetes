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

package etcd

import (
	"fmt"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	etcderrors "k8s.io/kubernetes/pkg/api/errors/etcd"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/securitycontext"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
	"k8s.io/kubernetes/pkg/util"
)

func newStorage(t *testing.T) (*REST, *BindingREST, *StatusREST, *tools.FakeEtcdClient) {
	etcdStorage, fakeClient := registrytest.NewEtcdStorage(t, "")
	storage := NewStorage(etcdStorage, false, nil)
	return storage.Pod, storage.Binding, storage.Status, fakeClient
}

func validNewPod() *api.Pod {
	grace := int64(30)
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,

			TerminationGracePeriodSeconds: &grace,
			Containers: []api.Container{
				{
					Name:            "foo",
					Image:           "test",
					ImagePullPolicy: api.PullAlways,

					TerminationMessagePath: api.TerminationMessagePathDefault,
					SecurityContext:        securitycontext.ValidSecurityContextWithContainerDefaults(),
				},
			},
		},
	}
}

func validChangedPod() *api.Pod {
	pod := validNewPod()
	pod.ResourceVersion = "1"
	pod.Labels = map[string]string{
		"foo": "bar",
	}
	return pod
}

func TestCreate(t *testing.T) {
	storage, _, _, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd)
	pod := validNewPod()
	pod.ObjectMeta = api.ObjectMeta{}
	// Make an invalid pod with an an incorrect label.
	invalidPod := validNewPod()
	invalidPod.Namespace = test.TestNamespace()
	invalidPod.Labels = map[string]string{
		"invalid/label/to/cause/validation/failure": "bar",
	}
	test.TestCreate(
		// valid
		pod,
		// invalid (empty contains list)
		&api.Pod{
			Spec: api.PodSpec{
				Containers: []api.Container{},
			},
		},
		// invalid (invalid labels)
		invalidPod,
	)
}

func TestUpdate(t *testing.T) {
	storage, _, _, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd)
	test.TestUpdate(
		// valid
		validNewPod(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*api.Pod)
			object.Labels = map[string]string{"a": "b"}
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, _, _, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd).ReturnDeletedObject()
	test.TestDelete(validNewPod())

	scheduledPod := validNewPod()
	scheduledPod.Spec.NodeName = "some-node"
	test.TestDeleteGraceful(scheduledPod, 30)
}

func TestCreateRegistryError(t *testing.T) {
	storage, _, _, fakeClient := newStorage(t)
	fakeClient.Err = fmt.Errorf("test error")

	pod := validNewPod()
	_, err := storage.Create(api.NewDefaultContext(), pod)
	if err != fakeClient.Err {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestCreateSetsFields(t *testing.T) {
	storage, _, _, fakeClient := newStorage(t)
	pod := validNewPod()
	_, err := storage.Create(api.NewDefaultContext(), pod)
	if err != fakeClient.Err {
		t.Fatalf("unexpected error: %v", err)
	}
	ctx := api.NewDefaultContext()
	object, err := storage.Get(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	actual := object.(*api.Pod)
	if actual.Name != pod.Name {
		t.Errorf("unexpected pod: %#v", actual)
	}
	if len(actual.UID) == 0 {
		t.Errorf("expected pod UID to be set: %#v", actual)
	}
}

func TestResourceLocation(t *testing.T) {
	expectedIP := "1.2.3.4"
	testCases := []struct {
		pod      api.Pod
		query    string
		location string
	}{
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Status:     api.PodStatus{PodIP: expectedIP},
			},
			query:    "foo",
			location: expectedIP,
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Status:     api.PodStatus{PodIP: expectedIP},
			},
			query:    "foo:12345",
			location: expectedIP + ":12345",
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "ctr"},
					},
				},
				Status: api.PodStatus{PodIP: expectedIP},
			},
			query:    "foo",
			location: expectedIP,
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "ctr", Ports: []api.ContainerPort{{ContainerPort: 9376}}},
					},
				},
				Status: api.PodStatus{PodIP: expectedIP},
			},
			query:    "foo",
			location: expectedIP + ":9376",
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "ctr", Ports: []api.ContainerPort{{ContainerPort: 9376}}},
					},
				},
				Status: api.PodStatus{PodIP: expectedIP},
			},
			query:    "foo:12345",
			location: expectedIP + ":12345",
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "ctr1"},
						{Name: "ctr2", Ports: []api.ContainerPort{{ContainerPort: 9376}}},
					},
				},
				Status: api.PodStatus{PodIP: expectedIP},
			},
			query:    "foo",
			location: expectedIP + ":9376",
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "ctr1", Ports: []api.ContainerPort{{ContainerPort: 9376}}},
						{Name: "ctr2", Ports: []api.ContainerPort{{ContainerPort: 1234}}},
					},
				},
				Status: api.PodStatus{PodIP: expectedIP},
			},
			query:    "foo",
			location: expectedIP + ":9376",
		},
	}

	ctx := api.NewDefaultContext()
	for _, tc := range testCases {
		storage, _, _, fakeClient := newStorage(t)
		key, _ := storage.KeyFunc(ctx, tc.pod.Name)
		key = etcdtest.AddPrefix(key)
		if _, err := fakeClient.Set(key, runtime.EncodeOrDie(testapi.Default.Codec(), &tc.pod), 0); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		redirector := rest.Redirector(storage)
		location, _, err := redirector.ResourceLocation(api.NewDefaultContext(), tc.query)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if location == nil {
			t.Errorf("Unexpected nil: %v", location)
		}

		if location.Scheme != "" {
			t.Errorf("Expected '%v', but got '%v'", "", location.Scheme)
		}
		if location.Host != tc.location {
			t.Errorf("Expected %v, but got %v", tc.location, location.Host)
		}
	}
}

func TestGet(t *testing.T) {
	storage, _, _, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd)
	test.TestGet(validNewPod())
}

func TestList(t *testing.T) {
	storage, _, _, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd)
	test.TestList(validNewPod())
}

func TestWatch(t *testing.T) {
	storage, _, _, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd)
	test.TestWatch(
		validNewPod(),
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{"metadata.name": "foo"},
		},
		// not matchin fields
		[]fields.Set{
			{"metadata.name": "bar"},
		},
	)
}

func TestEtcdCreate(t *testing.T) {
	storage, bindingStorage, _, fakeClient := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true
	key, _ := storage.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.ExpectNotFoundGet(key)
	_, err := storage.Create(ctx, validNewPod())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Suddenly, a wild scheduler appears:
	_, err = bindingStorage.Create(ctx, &api.Binding{
		ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "foo"},
		Target:     api.ObjectReference{Name: "machine"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	resp, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var pod api.Pod
	err = testapi.Default.Codec().DecodeInto([]byte(resp.Node.Value), &pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if pod.Name != "foo" {
		t.Errorf("Unexpected pod: %#v %s", pod, resp.Node.Value)
	}
}

// Ensure that when scheduler creates a binding for a pod that has already been deleted
// by the API server, API server returns not-found error.
func TestEtcdCreateBindingNoPod(t *testing.T) {
	storage, bindingStorage, _, fakeClient := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true

	key, _ := storage.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.ExpectNotFoundGet(key)
	// Assume that a pod has undergone the following:
	// - Create (apiserver)
	// - Schedule (scheduler)
	// - Delete (apiserver)
	_, err := bindingStorage.Create(ctx, &api.Binding{
		ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "foo"},
		Target:     api.ObjectReference{Name: "machine"},
	})
	if err == nil {
		t.Fatalf("Expected not-found-error but got nothing")
	}
	if !errors.IsNotFound(etcderrors.InterpretGetError(err, "Pod", "foo")) {
		t.Fatalf("Unexpected error returned: %#v", err)
	}

	_, err = storage.Get(ctx, "foo")
	if err == nil {
		t.Fatalf("Expected not-found-error but got nothing")
	}
	if !errors.IsNotFound(etcderrors.InterpretGetError(err, "Pod", "foo")) {
		t.Fatalf("Unexpected error: %v", err)
	}
}

func TestEtcdCreateFailsWithoutNamespace(t *testing.T) {
	storage, _, _, fakeClient := newStorage(t)
	fakeClient.TestIndex = true
	pod := validNewPod()
	pod.Namespace = ""
	_, err := storage.Create(api.NewContext(), pod)
	// Accept "namespace" or "Namespace".
	if err == nil || !strings.Contains(err.Error(), "amespace") {
		t.Fatalf("expected error that namespace was missing from context, got: %v", err)
	}
}

func TestEtcdCreateWithContainersNotFound(t *testing.T) {
	storage, bindingStorage, _, fakeClient := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true
	key, _ := storage.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.ExpectNotFoundGet(key)
	_, err := storage.Create(ctx, validNewPod())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Suddenly, a wild scheduler appears:
	_, err = bindingStorage.Create(ctx, &api.Binding{
		ObjectMeta: api.ObjectMeta{
			Namespace:   api.NamespaceDefault,
			Name:        "foo",
			Annotations: map[string]string{"label1": "value1"},
		},
		Target: api.ObjectReference{Name: "machine"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	resp, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var pod api.Pod
	err = testapi.Default.Codec().DecodeInto([]byte(resp.Node.Value), &pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if pod.Name != "foo" {
		t.Errorf("Unexpected pod: %#v %s", pod, resp.Node.Value)
	}
	if !(pod.Annotations != nil && pod.Annotations["label1"] == "value1") {
		t.Fatalf("Pod annotations don't match the expected: %v", pod.Annotations)
	}
}

func TestEtcdCreateWithConflict(t *testing.T) {
	storage, bindingStorage, _, fakeClient := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true
	key, _ := storage.KeyFunc(ctx, "foo")
	fakeClient.ExpectNotFoundGet(key)

	_, err := storage.Create(ctx, validNewPod())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Suddenly, a wild scheduler appears:
	binding := api.Binding{
		ObjectMeta: api.ObjectMeta{
			Namespace:   api.NamespaceDefault,
			Name:        "foo",
			Annotations: map[string]string{"label1": "value1"},
		},
		Target: api.ObjectReference{Name: "machine"},
	}
	_, err = bindingStorage.Create(ctx, &binding)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	_, err = bindingStorage.Create(ctx, &binding)
	if err == nil || !errors.IsConflict(err) {
		t.Fatalf("expected resource conflict error, not: %v", err)
	}
}

func TestEtcdCreateWithExistingContainers(t *testing.T) {
	storage, bindingStorage, _, fakeClient := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true
	key, _ := storage.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.ExpectNotFoundGet(key)
	_, err := storage.Create(ctx, validNewPod())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Suddenly, a wild scheduler appears:
	_, err = bindingStorage.Create(ctx, &api.Binding{
		ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "foo"},
		Target:     api.ObjectReference{Name: "machine"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	resp, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var pod api.Pod
	err = testapi.Default.Codec().DecodeInto([]byte(resp.Node.Value), &pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if pod.Name != "foo" {
		t.Errorf("Unexpected pod: %#v %s", pod, resp.Node.Value)
	}
}

func TestEtcdCreateBinding(t *testing.T) {
	ctx := api.NewDefaultContext()

	testCases := map[string]struct {
		binding api.Binding
		errOK   func(error) bool
	}{
		"noName": {
			binding: api.Binding{
				ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "foo"},
				Target:     api.ObjectReference{},
			},
			errOK: func(err error) bool { return errors.IsInvalid(err) },
		},
		"badKind": {
			binding: api.Binding{
				ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "foo"},
				Target:     api.ObjectReference{Name: "machine1", Kind: "unknown"},
			},
			errOK: func(err error) bool { return errors.IsInvalid(err) },
		},
		"emptyKind": {
			binding: api.Binding{
				ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "foo"},
				Target:     api.ObjectReference{Name: "machine2"},
			},
			errOK: func(err error) bool { return err == nil },
		},
		"kindNode": {
			binding: api.Binding{
				ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "foo"},
				Target:     api.ObjectReference{Name: "machine3", Kind: "Node"},
			},
			errOK: func(err error) bool { return err == nil },
		},
		"kindMinion": {
			binding: api.Binding{
				ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "foo"},
				Target:     api.ObjectReference{Name: "machine4", Kind: "Minion"},
			},
			errOK: func(err error) bool { return err == nil },
		},
	}
	for k, test := range testCases {
		storage, bindingStorage, _, fakeClient := newStorage(t)
		key, _ := storage.KeyFunc(ctx, "foo")
		key = etcdtest.AddPrefix(key)
		fakeClient.ExpectNotFoundGet(key)

		if _, err := storage.Create(ctx, validNewPod()); err != nil {
			t.Fatalf("%s: unexpected error: %v", k, err)
		}
		if _, err := bindingStorage.Create(ctx, &test.binding); !test.errOK(err) {
			t.Errorf("%s: unexpected error: %v", k, err)
		} else if err == nil {
			// If bind succeeded, verify Host field in pod's Spec.
			pod, err := storage.Get(ctx, validNewPod().ObjectMeta.Name)
			if err != nil {
				t.Errorf("%s: unexpected error: %v", k, err)
			} else if pod.(*api.Pod).Spec.NodeName != test.binding.Target.Name {
				t.Errorf("%s: expected: %v, got: %v", k, pod.(*api.Pod).Spec.NodeName, test.binding.Target.Name)
			}
		}
	}
}

func TestEtcdUpdateNotScheduled(t *testing.T) {
	storage, _, _, fakeClient := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true

	key, _ := storage.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(testapi.Default.Codec(), validNewPod()), 1)

	podIn := validChangedPod()
	_, _, err := storage.Update(ctx, podIn)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	response, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	podOut := &api.Pod{}
	testapi.Default.Codec().DecodeInto([]byte(response.Node.Value), podOut)
	if !api.Semantic.DeepEqual(podOut, podIn) {
		t.Errorf("objects differ: %v", util.ObjectDiff(podOut, podIn))
	}
}

func TestEtcdUpdateScheduled(t *testing.T) {
	storage, _, _, fakeClient := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true

	key, _ := storage.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(testapi.Default.Codec(), &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
		},
		Spec: api.PodSpec{
			NodeName: "machine",
			Containers: []api.Container{
				{
					Name:            "foobar",
					Image:           "foo:v1",
					SecurityContext: securitycontext.ValidSecurityContextWithContainerDefaults(),
				},
			},
		},
	}), 1)

	grace := int64(30)
	podIn := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "1",
			Labels: map[string]string{
				"foo": "bar",
			},
		},
		Spec: api.PodSpec{
			NodeName: "machine",
			Containers: []api.Container{
				{
					Name:                   "foobar",
					Image:                  "foo:v2",
					ImagePullPolicy:        api.PullIfNotPresent,
					TerminationMessagePath: api.TerminationMessagePathDefault,
					SecurityContext:        securitycontext.ValidSecurityContextWithContainerDefaults(),
				},
			},
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,

			TerminationGracePeriodSeconds: &grace,
		},
	}
	_, _, err := storage.Update(ctx, &podIn)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	response, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	var podOut api.Pod
	testapi.Default.Codec().DecodeInto([]byte(response.Node.Value), &podOut)
	if !api.Semantic.DeepEqual(podOut, podIn) {
		t.Errorf("expected: %#v, got: %#v", podOut, podIn)
	}

}

func TestEtcdUpdateStatus(t *testing.T) {
	storage, _, statusStorage, fakeClient := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true

	key, _ := storage.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	podStart := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
		},
		Spec: api.PodSpec{
			NodeName: "machine",
			Containers: []api.Container{
				{
					Image:           "foo:v1",
					SecurityContext: securitycontext.ValidSecurityContextWithContainerDefaults(),
				},
			},
		},
	}
	fakeClient.Set(key, runtime.EncodeOrDie(testapi.Default.Codec(), &podStart), 0)

	podIn := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "1",
			Labels: map[string]string{
				"foo": "bar",
			},
		},
		Spec: api.PodSpec{
			NodeName: "machine",
			Containers: []api.Container{
				{
					Image:                  "foo:v2",
					ImagePullPolicy:        api.PullIfNotPresent,
					TerminationMessagePath: api.TerminationMessagePathDefault,
				},
			},
		},
		Status: api.PodStatus{
			Phase:   api.PodRunning,
			PodIP:   "127.0.0.1",
			Message: "is now scheduled",
		},
	}

	expected := podStart
	expected.ResourceVersion = "2"
	grace := int64(30)
	expected.Spec.TerminationGracePeriodSeconds = &grace
	expected.Spec.RestartPolicy = api.RestartPolicyAlways
	expected.Spec.DNSPolicy = api.DNSClusterFirst
	expected.Spec.Containers[0].ImagePullPolicy = api.PullIfNotPresent
	expected.Spec.Containers[0].TerminationMessagePath = api.TerminationMessagePathDefault
	expected.Labels = podIn.Labels
	expected.Status = podIn.Status

	_, _, err := statusStorage.Update(ctx, &podIn)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	podOut, err := storage.Get(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !api.Semantic.DeepEqual(&expected, podOut) {
		t.Errorf("unexpected object: %s", util.ObjectDiff(&expected, podOut))
	}
}
