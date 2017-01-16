/*
Copyright 2015 The Kubernetes Authors.

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

package storage

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	genericapirequest "k8s.io/apiserver/pkg/request"
	"k8s.io/kubernetes/pkg/api"
	storeerr "k8s.io/kubernetes/pkg/api/errors/storage"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
	"k8s.io/kubernetes/pkg/util/intstr"
)

const defaultReplicas = 100

func newStorage(t *testing.T) (*DeploymentStorage, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, extensions.GroupName)
	restOptions := generic.RESTOptions{StorageConfig: etcdStorage, Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1, ResourcePrefix: "deployments"}
	deploymentStorage := NewStorage(restOptions)
	return &deploymentStorage, server
}

var namespace = "foo-namespace"
var name = "foo-deployment"

func validNewDeployment() *extensions.Deployment {
	return &extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: extensions.DeploymentSpec{
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "b"}},
			Strategy: extensions.DeploymentStrategy{
				Type: extensions.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &extensions.RollingUpdateDeployment{
					MaxSurge:       intstr.FromInt(1),
					MaxUnavailable: intstr.FromInt(1),
				},
			},
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"a": "b"},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:            "test",
							Image:           "test_image",
							ImagePullPolicy: api.PullIfNotPresent,
						},
					},
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
			Replicas: 7,
		},
		Status: extensions.DeploymentStatus{
			Replicas: 5,
		},
	}
}

var validDeployment = *validNewDeployment()

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Deployment.Store.DestroyFunc()
	test := registrytest.New(t, storage.Deployment.Store)
	deployment := validNewDeployment()
	deployment.ObjectMeta = api.ObjectMeta{}
	test.TestCreate(
		// valid
		deployment,
		// invalid (invalid selector)
		&extensions.Deployment{
			Spec: extensions.DeploymentSpec{
				Selector: &metav1.LabelSelector{MatchLabels: map[string]string{}},
				Template: validDeployment.Spec.Template,
			},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Deployment.Store.DestroyFunc()
	test := registrytest.New(t, storage.Deployment.Store)
	test.TestUpdate(
		// valid
		validNewDeployment(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.Deployment)
			object.Spec.Template.Spec.NodeSelector = map[string]string{"c": "d"}
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.Deployment)
			object.Name = ""
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.Deployment)
			object.Spec.Template.Spec.RestartPolicy = api.RestartPolicyOnFailure
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.Deployment)
			object.Spec.Selector = &metav1.LabelSelector{MatchLabels: map[string]string{}}
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Deployment.Store.DestroyFunc()
	test := registrytest.New(t, storage.Deployment.Store)
	test.TestDelete(validNewDeployment())
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Deployment.Store.DestroyFunc()
	test := registrytest.New(t, storage.Deployment.Store)
	test.TestGet(validNewDeployment())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Deployment.Store.DestroyFunc()
	test := registrytest.New(t, storage.Deployment.Store)
	test.TestList(validNewDeployment())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Deployment.Store.DestroyFunc()
	test := registrytest.New(t, storage.Deployment.Store)
	test.TestWatch(
		validNewDeployment(),
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"a": "c"},
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{"metadata.name": name},
		},
		// not matchin fields
		[]fields.Set{
			{"metadata.name": "bar"},
			{"name": name},
		},
	)
}

func TestScaleGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Deployment.Store.DestroyFunc()
	var deployment extensions.Deployment
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace)
	key := "/deployments/" + namespace + "/" + name
	if err := storage.Deployment.Storage.Create(ctx, key, &validDeployment, &deployment, 0); err != nil {
		t.Fatalf("error setting new deployment (key: %s) %v: %v", key, validDeployment, err)
	}

	want := &extensions.Scale{
		ObjectMeta: api.ObjectMeta{
			Name:              name,
			Namespace:         namespace,
			UID:               deployment.UID,
			ResourceVersion:   deployment.ResourceVersion,
			CreationTimestamp: deployment.CreationTimestamp,
		},
		Spec: extensions.ScaleSpec{
			Replicas: validDeployment.Spec.Replicas,
		},
		Status: extensions.ScaleStatus{
			Replicas: validDeployment.Status.Replicas,
			Selector: validDeployment.Spec.Selector,
		},
	}
	obj, err := storage.Scale.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error fetching scale for %s: %v", name, err)
	}
	got := obj.(*extensions.Scale)
	if !api.Semantic.DeepEqual(want, got) {
		t.Errorf("unexpected scale: %s", diff.ObjectDiff(want, got))
	}
}

func TestScaleUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Deployment.Store.DestroyFunc()
	var deployment extensions.Deployment
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace)
	key := "/deployments/" + namespace + "/" + name
	if err := storage.Deployment.Storage.Create(ctx, key, &validDeployment, &deployment, 0); err != nil {
		t.Fatalf("error setting new deployment (key: %s) %v: %v", key, validDeployment, err)
	}
	replicas := int32(12)
	update := extensions.Scale{
		ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespace},
		Spec: extensions.ScaleSpec{
			Replicas: replicas,
		},
	}

	if _, _, err := storage.Scale.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update, api.Scheme)); err != nil {
		t.Fatalf("error updating scale %v: %v", update, err)
	}
	obj, err := storage.Scale.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error fetching scale for %s: %v", name, err)
	}
	scale := obj.(*extensions.Scale)
	if scale.Spec.Replicas != replicas {
		t.Errorf("wrong replicas count expected: %d got: %d", replicas, deployment.Spec.Replicas)
	}

	update.ResourceVersion = deployment.ResourceVersion
	update.Spec.Replicas = 15

	if _, _, err = storage.Scale.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update, api.Scheme)); err != nil && !errors.IsConflict(err) {
		t.Fatalf("unexpected error, expecting an update conflict but got %v", err)
	}
}

func TestStatusUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Deployment.Store.DestroyFunc()
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace)
	key := "/deployments/" + namespace + "/" + name
	if err := storage.Deployment.Storage.Create(ctx, key, &validDeployment, nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	update := extensions.Deployment{
		ObjectMeta: validDeployment.ObjectMeta,
		Spec: extensions.DeploymentSpec{
			Replicas: defaultReplicas,
		},
		Status: extensions.DeploymentStatus{
			Replicas: defaultReplicas,
		},
	}

	if _, _, err := storage.Status.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update, api.Scheme)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err := storage.Deployment.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	deployment := obj.(*extensions.Deployment)
	if deployment.Spec.Replicas != 7 {
		t.Errorf("we expected .spec.replicas to not be updated but it was updated to %v", deployment.Spec.Replicas)
	}
	if deployment.Status.Replicas != defaultReplicas {
		t.Errorf("we expected .status.replicas to be updated to %d but it was %v", defaultReplicas, deployment.Status.Replicas)
	}
}

func TestEtcdCreateDeploymentRollback(t *testing.T) {
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace)

	testCases := map[string]struct {
		rollback extensions.DeploymentRollback
		errOK    func(error) bool
	}{
		"normal": {
			rollback: extensions.DeploymentRollback{
				Name:               name,
				UpdatedAnnotations: map[string]string{},
				RollbackTo:         extensions.RollbackConfig{Revision: 1},
			},
			errOK: func(err error) bool { return err == nil },
		},
		"noAnnotation": {
			rollback: extensions.DeploymentRollback{
				Name:       name,
				RollbackTo: extensions.RollbackConfig{Revision: 1},
			},
			errOK: func(err error) bool { return err == nil },
		},
		"noName": {
			rollback: extensions.DeploymentRollback{
				UpdatedAnnotations: map[string]string{},
				RollbackTo:         extensions.RollbackConfig{Revision: 1},
			},
			errOK: func(err error) bool { return err != nil },
		},
	}
	for k, test := range testCases {
		storage, server := newStorage(t)
		rollbackStorage := storage.Rollback

		if _, err := storage.Deployment.Create(ctx, validNewDeployment()); err != nil {
			t.Fatalf("%s: unexpected error: %v", k, err)
		}
		if _, err := rollbackStorage.Create(ctx, &test.rollback); !test.errOK(err) {
			t.Errorf("%s: unexpected error: %v", k, err)
		} else if err == nil {
			// If rollback succeeded, verify Rollback field of deployment
			d, err := storage.Deployment.Get(ctx, validNewDeployment().ObjectMeta.Name, &metav1.GetOptions{})
			if err != nil {
				t.Errorf("%s: unexpected error: %v", k, err)
			} else if !reflect.DeepEqual(*d.(*extensions.Deployment).Spec.RollbackTo, test.rollback.RollbackTo) {
				t.Errorf("%s: expected: %v, got: %v", k, *d.(*extensions.Deployment).Spec.RollbackTo, test.rollback.RollbackTo)
			}
		}
		storage.Deployment.Store.DestroyFunc()
		server.Terminate(t)
	}
}

// Ensure that when a deploymentRollback is created for a deployment that has already been deleted
// by the API server, API server returns not-found error.
func TestEtcdCreateDeploymentRollbackNoDeployment(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Deployment.Store.DestroyFunc()
	rollbackStorage := storage.Rollback
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace)

	_, err := rollbackStorage.Create(ctx, &extensions.DeploymentRollback{
		Name:               name,
		UpdatedAnnotations: map[string]string{},
		RollbackTo:         extensions.RollbackConfig{Revision: 1},
	})
	if err == nil {
		t.Fatalf("Expected not-found-error but got nothing")
	}
	if !errors.IsNotFound(storeerr.InterpretGetError(err, extensions.Resource("deployments"), name)) {
		t.Fatalf("Unexpected error returned: %#v", err)
	}

	_, err = storage.Deployment.Get(ctx, name, &metav1.GetOptions{})
	if err == nil {
		t.Fatalf("Expected not-found-error but got nothing")
	}
	if !errors.IsNotFound(storeerr.InterpretGetError(err, extensions.Resource("deployments"), name)) {
		t.Fatalf("Unexpected error: %v", err)
	}
}
