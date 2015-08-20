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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	etcderrors "k8s.io/kubernetes/pkg/api/errors/etcd"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/rest/resttest"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/pod"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/securitycontext"
	"k8s.io/kubernetes/pkg/storage"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
	"k8s.io/kubernetes/pkg/util"

	"github.com/coreos/go-etcd/etcd"
)

func newEtcdStorage(t *testing.T) (*tools.FakeEtcdClient, storage.Interface) {
	fakeEtcdClient := tools.NewFakeEtcdClient(t)
	fakeEtcdClient.TestIndex = true
	etcdStorage := etcdstorage.NewEtcdStorage(fakeEtcdClient, latest.Codec, etcdtest.PathPrefix())
	return fakeEtcdClient, etcdStorage
}

func newStorage(t *testing.T) (*REST, *BindingREST, *StatusREST, *tools.FakeEtcdClient, storage.Interface) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	storage := NewStorage(etcdStorage, nil)
	return storage.Pod, storage.Binding, storage.Status, fakeEtcdClient, etcdStorage
}

func validNewPod() *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
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

func TestStorage(t *testing.T) {
	storage, _, _, _, _ := newStorage(t)
	pod.NewRegistry(storage)
}

func TestCreate(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	storage := NewStorage(etcdStorage, nil).Pod
	test := resttest.New(t, storage, fakeEtcdClient.SetError)
	pod := validNewPod()
	pod.ObjectMeta = api.ObjectMeta{}
	test.TestCreate(
		// valid
		pod,
		// invalid
		&api.Pod{
			Spec: api.PodSpec{
				Containers: []api.Container{},
			},
		},
	)
}

func TestDelete(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	storage := NewStorage(etcdStorage, nil).Pod
	ctx := api.NewDefaultContext()
	key, _ := storage.Etcd.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	test := resttest.New(t, storage, fakeEtcdClient.SetError)

	createFn := func() runtime.Object {
		pod := validChangedPod()
		fakeEtcdClient.Data[key] = tools.EtcdResponseWithError{
			R: &etcd.Response{
				Node: &etcd.Node{
					Value:         runtime.EncodeOrDie(latest.Codec, pod),
					ModifiedIndex: 1,
				},
			},
		}
		return pod
	}
	gracefulSetFn := func() bool {
		if fakeEtcdClient.Data[key].R.Node == nil {
			return false
		}
		return fakeEtcdClient.Data[key].R.Node.TTL == 30
	}
	test.TestDelete(createFn, gracefulSetFn)
}

func expectPod(t *testing.T, out runtime.Object) (*api.Pod, bool) {
	pod, ok := out.(*api.Pod)
	if !ok || pod == nil {
		t.Errorf("Expected an api.Pod object, was %#v", out)
		return nil, false
	}
	return pod, true
}

func TestCreateRegistryError(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	fakeEtcdClient.Err = fmt.Errorf("test error")
	storage := NewStorage(etcdStorage, nil).Pod

	pod := validNewPod()
	_, err := storage.Create(api.NewDefaultContext(), pod)
	if err != fakeEtcdClient.Err {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestCreateSetsFields(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	storage := NewStorage(etcdStorage, nil).Pod
	pod := validNewPod()
	_, err := storage.Create(api.NewDefaultContext(), pod)
	if err != fakeEtcdClient.Err {
		t.Fatalf("unexpected error: %v", err)
	}
	ctx := api.NewDefaultContext()
	key, _ := storage.Etcd.KeyFunc(ctx, "foo")
	actual := &api.Pod{}
	if err := etcdStorage.Get(key, actual, false); err != nil {
		t.Fatalf("unexpected extraction error: %v", err)
	}
	if actual.Name != pod.Name {
		t.Errorf("unexpected pod: %#v", actual)
	}
	if len(actual.UID) == 0 {
		t.Errorf("expected pod UID to be set: %#v", actual)
	}
}

func TestPodDecode(t *testing.T) {
	_, etcdStorage := newEtcdStorage(t)
	storage := NewStorage(etcdStorage, nil).Pod
	expected := validNewPod()
	body, err := latest.Codec.Encode(expected)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	actual := storage.New()
	if err := latest.Codec.DecodeInto(body, actual); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !api.Semantic.DeepEqual(expected, actual) {
		t.Errorf("mismatch: %s", util.ObjectDiff(expected, actual))
	}
}

// TODO: remove, this is covered by RESTTest.TestCreate
func TestPodStorageValidatesCreate(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	fakeEtcdClient.Err = fmt.Errorf("test error")
	storage := NewStorage(etcdStorage, nil).Pod

	pod := validNewPod()
	pod.Labels = map[string]string{
		"invalid/label/to/cause/validation/failure": "bar",
	}
	c, err := storage.Create(api.NewDefaultContext(), pod)
	if c != nil {
		t.Errorf("Expected nil object")
	}
	if !errors.IsInvalid(err) {
		t.Errorf("Expected to get an invalid resource error, got %v", err)
	}
}

// TODO: remove, this is covered by RESTTest.TestCreate
func TestCreatePod(t *testing.T) {
	_, etcdStorage := newEtcdStorage(t)
	storage := NewStorage(etcdStorage, nil).Pod
	ctx := api.NewDefaultContext()
	key, _ := storage.Etcd.KeyFunc(ctx, "foo")

	pod := validNewPod()
	obj, err := storage.Create(api.NewDefaultContext(), pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if obj == nil {
		t.Fatalf("unexpected object: %#v", obj)
	}
	actual := &api.Pod{}
	if err := etcdStorage.Get(key, actual, false); err != nil {
		t.Fatalf("unexpected extraction error: %v", err)
	}
	if !api.HasObjectMetaSystemFieldValues(&actual.ObjectMeta) {
		t.Errorf("Expected ObjectMeta field values were populated: %#v", actual)
	}
}

// TODO: remove, this is covered by RESTTest.TestCreate
func TestCreateWithConflictingNamespace(t *testing.T) {
	_, etcdStorage := newEtcdStorage(t)
	storage := NewStorage(etcdStorage, nil).Pod

	pod := validNewPod()
	pod.Namespace = "not-default"

	obj, err := storage.Create(api.NewDefaultContext(), pod)
	if obj != nil {
		t.Error("Expected a nil obj, but we got a value")
	}
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	} else if strings.Contains(err.Error(), "Controller.Namespace does not match the provided context") {
		t.Errorf("Expected 'Pod.Namespace does not match the provided context' error, got '%v'", err.Error())
	}
}

func TestUpdateWithConflictingNamespace(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	storage := NewStorage(etcdStorage, nil).Pod
	ctx := api.NewDefaultContext()
	key, _ := storage.Etcd.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeEtcdClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: runtime.EncodeOrDie(latest.Codec, &api.Pod{
					ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "default"},
					Spec:       api.PodSpec{NodeName: "machine"},
				}),
				ModifiedIndex: 1,
			},
		},
	}

	pod := validChangedPod()
	pod.Namespace = "not-default"

	obj, created, err := storage.Update(api.NewDefaultContext(), pod)
	if obj != nil || created {
		t.Error("Expected a nil channel, but we got a value or created")
	}
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	} else if strings.Index(err.Error(), "the namespace of the provided object does not match the namespace sent on the request") == -1 {
		t.Errorf("Expected 'Pod.Namespace does not match the provided context' error, got '%v'", err.Error())
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
		fakeEtcdClient, etcdStorage := newEtcdStorage(t)
		storage := NewStorage(etcdStorage, nil).Pod
		key, _ := storage.Etcd.KeyFunc(ctx, "foo")
		key = etcdtest.AddPrefix(key)
		fakeEtcdClient.Data[key] = tools.EtcdResponseWithError{
			R: &etcd.Response{
				Node: &etcd.Node{
					Value: runtime.EncodeOrDie(latest.Codec, &tc.pod),
				},
			},
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

func TestDeletePod(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	fakeEtcdClient.ChangeIndex = 1
	storage := NewStorage(etcdStorage, nil).Pod
	ctx := api.NewDefaultContext()
	key, _ := storage.Etcd.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeEtcdClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: runtime.EncodeOrDie(latest.Codec, &api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name:      "foo",
						Namespace: api.NamespaceDefault,
					},
					Spec: api.PodSpec{NodeName: "machine"},
				}),
				ModifiedIndex: 1,
				CreatedIndex:  1,
			},
		},
	}
	_, err := storage.Delete(api.NewDefaultContext(), "foo", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEtcdGet(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	storage := NewStorage(etcdStorage, nil).Pod
	test := resttest.New(t, storage, fakeEtcdClient.SetError)
	pod := validNewPod()
	test.TestGet(pod)
}

func TestEtcdList(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	storage := NewStorage(etcdStorage, nil).Pod
	test := resttest.New(t, storage, fakeEtcdClient.SetError)
	key := etcdtest.AddPrefix(storage.Etcd.KeyRootFunc(test.TestContext()))
	pod := validNewPod()
	test.TestList(
		pod,
		func(objects []runtime.Object) []runtime.Object {
			return registrytest.SetObjectsForKey(fakeEtcdClient, key, objects)
		},
		func(resourceVersion uint64) {
			registrytest.SetResourceVersion(fakeEtcdClient, resourceVersion)
		})
}

func TestEtcdCreate(t *testing.T) {
	registry, bindingRegistry, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true
	key, _ := registry.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	_, err := registry.Create(ctx, validNewPod())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Suddenly, a wild scheduler appears:
	_, err = bindingRegistry.Create(ctx, &api.Binding{
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
	err = latest.Codec.DecodeInto([]byte(resp.Node.Value), &pod)
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
	registry, bindingRegistry, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true

	key, _ := registry.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	// Assume that a pod has undergone the following:
	// - Create (apiserver)
	// - Schedule (scheduler)
	// - Delete (apiserver)
	_, err := bindingRegistry.Create(ctx, &api.Binding{
		ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "foo"},
		Target:     api.ObjectReference{Name: "machine"},
	})
	if err == nil {
		t.Fatalf("Expected not-found-error but got nothing")
	}
	if !errors.IsNotFound(etcderrors.InterpretGetError(err, "Pod", "foo")) {
		t.Fatalf("Unexpected error returned: %#v", err)
	}

	_, err = registry.Get(ctx, "foo")
	if err == nil {
		t.Fatalf("Expected not-found-error but got nothing")
	}
	if !errors.IsNotFound(etcderrors.InterpretGetError(err, "Pod", "foo")) {
		t.Fatalf("Unexpected error: %v", err)
	}
}

func TestEtcdCreateFailsWithoutNamespace(t *testing.T) {
	registry, _, _, fakeClient, _ := newStorage(t)
	fakeClient.TestIndex = true
	pod := validNewPod()
	pod.Namespace = ""
	_, err := registry.Create(api.NewContext(), pod)
	// Accept "namespace" or "Namespace".
	if err == nil || !strings.Contains(err.Error(), "amespace") {
		t.Fatalf("expected error that namespace was missing from context, got: %v", err)
	}
}

func TestEtcdCreateAlreadyExisting(t *testing.T) {
	registry, _, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	key, _ := registry.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: runtime.EncodeOrDie(latest.Codec, &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}),
			},
		},
		E: nil,
	}
	_, err := registry.Create(ctx, validNewPod())
	if !errors.IsAlreadyExists(err) {
		t.Errorf("Unexpected error returned: %#v", err)
	}
}

func TestEtcdCreateWithContainersNotFound(t *testing.T) {
	registry, bindingRegistry, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true
	key, _ := registry.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	_, err := registry.Create(ctx, validNewPod())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Suddenly, a wild scheduler appears:
	_, err = bindingRegistry.Create(ctx, &api.Binding{
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
	err = latest.Codec.DecodeInto([]byte(resp.Node.Value), &pod)
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
	registry, bindingRegistry, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true
	key, _ := registry.KeyFunc(ctx, "foo")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}

	_, err := registry.Create(ctx, validNewPod())
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
	_, err = bindingRegistry.Create(ctx, &binding)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	_, err = bindingRegistry.Create(ctx, &binding)
	if err == nil || !errors.IsConflict(err) {
		t.Fatalf("expected resource conflict error, not: %v", err)
	}
}

func TestEtcdCreateWithExistingContainers(t *testing.T) {
	registry, bindingRegistry, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true
	key, _ := registry.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	_, err := registry.Create(ctx, validNewPod())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Suddenly, a wild scheduler appears:
	_, err = bindingRegistry.Create(ctx, &api.Binding{
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
	err = latest.Codec.DecodeInto([]byte(resp.Node.Value), &pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if pod.Name != "foo" {
		t.Errorf("Unexpected pod: %#v %s", pod, resp.Node.Value)
	}
}

func TestEtcdCreateBinding(t *testing.T) {
	registry, bindingRegistry, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true

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
		key, _ := registry.KeyFunc(ctx, "foo")
		key = etcdtest.AddPrefix(key)
		fakeClient.Data[key] = tools.EtcdResponseWithError{
			R: &etcd.Response{
				Node: nil,
			},
			E: tools.EtcdErrorNotFound,
		}
		if _, err := registry.Create(ctx, validNewPod()); err != nil {
			t.Fatalf("%s: unexpected error: %v", k, err)
		}
		if _, err := bindingRegistry.Create(ctx, &test.binding); !test.errOK(err) {
			t.Errorf("%s: unexpected error: %v", k, err)
		} else if err == nil {
			// If bind succeeded, verify Host field in pod's Spec.
			pod, err := registry.Get(ctx, validNewPod().ObjectMeta.Name)
			if err != nil {
				t.Errorf("%s: unexpected error: %v", k, err)
			} else if pod.(*api.Pod).Spec.NodeName != test.binding.Target.Name {
				t.Errorf("%s: expected: %v, got: %v", k, pod.(*api.Pod).Spec.NodeName, test.binding.Target.Name)
			}
		}
	}
}

func TestEtcdUpdateNotFound(t *testing.T) {
	registry, _, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true

	key, _ := registry.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}

	podIn := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "1",
			Labels: map[string]string{
				"foo": "bar",
			},
		},
	}
	_, _, err := registry.Update(ctx, &podIn)
	if err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestEtcdUpdateNotScheduled(t *testing.T) {
	registry, _, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true

	key, _ := registry.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, validNewPod()), 1)

	podIn := validChangedPod()
	_, _, err := registry.Update(ctx, podIn)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	response, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	podOut := &api.Pod{}
	latest.Codec.DecodeInto([]byte(response.Node.Value), podOut)
	if !api.Semantic.DeepEqual(podOut, podIn) {
		t.Errorf("objects differ: %v", util.ObjectDiff(podOut, podIn))
	}
}

func TestEtcdUpdateScheduled(t *testing.T) {
	registry, _, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true

	key, _ := registry.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Pod{
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
		},
	}
	_, _, err := registry.Update(ctx, &podIn)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	response, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	var podOut api.Pod
	latest.Codec.DecodeInto([]byte(response.Node.Value), &podOut)
	if !api.Semantic.DeepEqual(podOut, podIn) {
		t.Errorf("expected: %#v, got: %#v", podOut, podIn)
	}

}

func TestEtcdUpdateStatus(t *testing.T) {
	registry, _, status, fakeClient, etcdStorage := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true

	key, _ := registry.KeyFunc(ctx, "foo")
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
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &podStart), 1)

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
	expected.Spec.RestartPolicy = api.RestartPolicyAlways
	expected.Spec.DNSPolicy = api.DNSClusterFirst
	expected.Spec.Containers[0].ImagePullPolicy = api.PullIfNotPresent
	expected.Spec.Containers[0].TerminationMessagePath = api.TerminationMessagePathDefault
	expected.Labels = podIn.Labels
	expected.Status = podIn.Status

	_, _, err := status.Update(ctx, &podIn)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	var podOut api.Pod
	key, _ = registry.KeyFunc(ctx, "foo")
	if err := etcdStorage.Get(key, &podOut, false); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if !api.Semantic.DeepEqual(expected, podOut) {
		t.Errorf("unexpected object: %s", util.ObjectDiff(expected, podOut))
	}
}

func TestEtcdDeletePod(t *testing.T) {
	registry, _, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true

	key, _ := registry.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec:       api.PodSpec{NodeName: "machine"},
	}), 0)
	_, err := registry.Delete(ctx, "foo", api.NewDeleteOptions(0))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(fakeClient.DeletedKeys) != 1 {
		t.Errorf("Expected 1 delete, found %#v", fakeClient.DeletedKeys)
	} else if fakeClient.DeletedKeys[0] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[0], key)
	}
}

func TestEtcdDeletePodMultipleContainers(t *testing.T) {
	registry, _, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true
	key, _ := registry.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec:       api.PodSpec{NodeName: "machine"},
	}), 0)
	_, err := registry.Delete(ctx, "foo", api.NewDeleteOptions(0))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(fakeClient.DeletedKeys) != 1 {
		t.Errorf("Expected 1 delete, found %#v", fakeClient.DeletedKeys)
	}
	if fakeClient.DeletedKeys[0] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[0], key)
	}
}

func TestEtcdWatchPods(t *testing.T) {
	registry, _, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	watching, err := registry.Watch(ctx,
		labels.Everything(),
		fields.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	select {
	case _, ok := <-watching.ResultChan():
		if !ok {
			t.Errorf("watching channel should be open")
		}
	default:
	}
	fakeClient.WatchInjectError <- nil
	if _, ok := <-watching.ResultChan(); ok {
		t.Errorf("watching channel should be closed")
	}
	watching.Stop()
}

func TestEtcdWatchPodsMatch(t *testing.T) {
	registry, _, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	watching, err := registry.Watch(ctx,
		labels.SelectorFromSet(labels.Set{"name": "foo"}),
		fields.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
			Labels: map[string]string{
				"name": "foo",
			},
		},
	}
	podBytes, _ := latest.Codec.Encode(pod)
	fakeClient.WatchResponse <- &etcd.Response{
		Action: "create",
		Node: &etcd.Node{
			Value: string(podBytes),
		},
	}
	select {
	case _, ok := <-watching.ResultChan():
		if !ok {
			t.Errorf("watching channel should be open")
		}
	case <-time.After(time.Millisecond * 100):
		t.Error("unexpected timeout from result channel")
	}
	watching.Stop()
}

func TestEtcdWatchPodsNotMatch(t *testing.T) {
	registry, _, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	watching, err := registry.Watch(ctx,
		labels.SelectorFromSet(labels.Set{"name": "foo"}),
		fields.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "bar",
			Labels: map[string]string{
				"name": "bar",
			},
		},
	}
	podBytes, _ := latest.Codec.Encode(pod)
	fakeClient.WatchResponse <- &etcd.Response{
		Action: "create",
		Node: &etcd.Node{
			Value: string(podBytes),
		},
	}

	select {
	case <-watching.ResultChan():
		t.Error("unexpected result from result channel")
	case <-time.After(time.Millisecond * 100):
		// expected case
	}
}
