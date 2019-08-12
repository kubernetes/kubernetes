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
	"fmt"
	"net/http"
	"reflect"
	"testing"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/intstr"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	storeerr "k8s.io/apiserver/pkg/storage/errors"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

const defaultReplicas = 100

func newStorage(t *testing.T) (*DeploymentStorage, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, apps.GroupName)
	restOptions := generic.RESTOptions{StorageConfig: etcdStorage, Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1, ResourcePrefix: "deployments"}
	deploymentStorage, err := NewStorage(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return &deploymentStorage, server
}

var namespace = "foo-namespace"
var name = "foo-deployment"

func validNewDeployment() *apps.Deployment {
	return &apps.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: apps.DeploymentSpec{
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "b"}},
			Strategy: apps.DeploymentStrategy{
				Type: apps.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &apps.RollingUpdateDeployment{
					MaxSurge:       intstr.FromInt(1),
					MaxUnavailable: intstr.FromInt(1),
				},
			},
			Template: api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"a": "b"},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:                     "test",
							Image:                    "test_image",
							ImagePullPolicy:          api.PullIfNotPresent,
							TerminationMessagePolicy: api.TerminationMessageReadFile,
						},
					},
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
			Replicas: 7,
		},
		Status: apps.DeploymentStatus{
			Replicas: 5,
		},
	}
}

var validDeployment = *validNewDeployment()

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Deployment.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Deployment.Store)
	deployment := validNewDeployment()
	deployment.ObjectMeta = metav1.ObjectMeta{}
	test.TestCreate(
		// valid
		deployment,
		// invalid (invalid selector)
		&apps.Deployment{
			Spec: apps.DeploymentSpec{
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
	test := genericregistrytest.New(t, storage.Deployment.Store)
	test.TestUpdate(
		// valid
		validNewDeployment(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*apps.Deployment)
			object.Spec.Template.Spec.NodeSelector = map[string]string{"c": "d"}
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*apps.Deployment)
			object.Name = ""
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*apps.Deployment)
			object.Spec.Template.Spec.RestartPolicy = api.RestartPolicyOnFailure
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*apps.Deployment)
			object.Spec.Selector = &metav1.LabelSelector{MatchLabels: map[string]string{}}
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Deployment.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Deployment.Store)
	test.TestDelete(validNewDeployment())
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Deployment.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Deployment.Store)
	test.TestGet(validNewDeployment())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Deployment.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Deployment.Store)
	test.TestList(validNewDeployment())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Deployment.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Deployment.Store)
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
		// not matching fields
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
	var deployment apps.Deployment
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace)
	key := "/deployments/" + namespace + "/" + name
	if err := storage.Deployment.Storage.Create(ctx, key, &validDeployment, &deployment, 0, false); err != nil {
		t.Fatalf("error setting new deployment (key: %s) %v: %v", key, validDeployment, err)
	}

	selector, err := metav1.LabelSelectorAsSelector(validDeployment.Spec.Selector)
	if err != nil {
		t.Fatal(err)
	}
	want := &autoscaling.Scale{
		ObjectMeta: metav1.ObjectMeta{
			Name:              name,
			Namespace:         namespace,
			UID:               deployment.UID,
			ResourceVersion:   deployment.ResourceVersion,
			CreationTimestamp: deployment.CreationTimestamp,
		},
		Spec: autoscaling.ScaleSpec{
			Replicas: validDeployment.Spec.Replicas,
		},
		Status: autoscaling.ScaleStatus{
			Replicas: validDeployment.Status.Replicas,
			Selector: selector.String(),
		},
	}
	obj, err := storage.Scale.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error fetching scale for %s: %v", name, err)
	}
	got := obj.(*autoscaling.Scale)
	if !apiequality.Semantic.DeepEqual(want, got) {
		t.Errorf("unexpected scale: %s", diff.ObjectDiff(want, got))
	}
}

func TestScaleUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Deployment.Store.DestroyFunc()
	var deployment apps.Deployment
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace)
	key := "/deployments/" + namespace + "/" + name
	if err := storage.Deployment.Storage.Create(ctx, key, &validDeployment, &deployment, 0, false); err != nil {
		t.Fatalf("error setting new deployment (key: %s) %v: %v", key, validDeployment, err)
	}
	replicas := int32(12)
	update := autoscaling.Scale{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: autoscaling.ScaleSpec{
			Replicas: replicas,
		},
	}

	if _, _, err := storage.Scale.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("error updating scale %v: %v", update, err)
	}
	obj, err := storage.Scale.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error fetching scale for %s: %v", name, err)
	}
	scale := obj.(*autoscaling.Scale)
	if scale.Spec.Replicas != replicas {
		t.Errorf("wrong replicas count expected: %d got: %d", replicas, deployment.Spec.Replicas)
	}

	update.ResourceVersion = deployment.ResourceVersion
	update.Spec.Replicas = 15

	if _, _, err = storage.Scale.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil && !errors.IsConflict(err) {
		t.Fatalf("unexpected error, expecting an update conflict but got %v", err)
	}
}

func TestStatusUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Deployment.Store.DestroyFunc()
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace)
	key := "/deployments/" + namespace + "/" + name
	if err := storage.Deployment.Storage.Create(ctx, key, &validDeployment, nil, 0, false); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	update := apps.Deployment{
		ObjectMeta: validDeployment.ObjectMeta,
		Spec: apps.DeploymentSpec{
			Replicas: defaultReplicas,
		},
		Status: apps.DeploymentStatus{
			Replicas: defaultReplicas,
		},
	}

	if _, _, err := storage.Status.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err := storage.Deployment.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	deployment := obj.(*apps.Deployment)
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
		rollback apps.DeploymentRollback
		errOK    func(error) bool
	}{
		"normal": {
			rollback: apps.DeploymentRollback{
				Name:               name,
				UpdatedAnnotations: map[string]string{},
				RollbackTo:         apps.RollbackConfig{Revision: 1},
			},
			errOK: func(err error) bool { return err == nil },
		},
		"noAnnotation": {
			rollback: apps.DeploymentRollback{
				Name:       name,
				RollbackTo: apps.RollbackConfig{Revision: 1},
			},
			errOK: func(err error) bool { return err == nil },
		},
		"noName": {
			rollback: apps.DeploymentRollback{
				UpdatedAnnotations: map[string]string{},
				RollbackTo:         apps.RollbackConfig{Revision: 1},
			},
			errOK: func(err error) bool { return err != nil },
		},
	}
	for k, test := range testCases {
		storage, server := newStorage(t)
		rollbackStorage := storage.Rollback

		if _, err := storage.Deployment.Create(ctx, validNewDeployment(), rest.ValidateAllObjectFunc, &metav1.CreateOptions{}); err != nil {
			t.Fatalf("%s: unexpected error: %v", k, err)
		}
		rollbackRespStatus, err := rollbackStorage.Create(ctx, test.rollback.Name, &test.rollback, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
		if !test.errOK(err) {
			t.Errorf("%s: unexpected error: %v", k, err)
		} else if err == nil {
			// If rollback succeeded, verify Rollback response and Rollback field of deployment
			status, ok := rollbackRespStatus.(*metav1.Status)
			if !ok {
				t.Errorf("%s: unexpected response format", k)
			}
			if status.Code != http.StatusOK || status.Status != metav1.StatusSuccess {
				t.Errorf("%s: unexpected response, code: %d, status: %s", k, status.Code, status.Status)
			}
			d, err := storage.Deployment.Get(ctx, validNewDeployment().ObjectMeta.Name, &metav1.GetOptions{})
			if err != nil {
				t.Errorf("%s: unexpected error: %v", k, err)
			} else if !reflect.DeepEqual(*d.(*apps.Deployment).Spec.RollbackTo, test.rollback.RollbackTo) {
				t.Errorf("%s: expected: %v, got: %v", k, *d.(*apps.Deployment).Spec.RollbackTo, test.rollback.RollbackTo)
			}
		}
		storage.Deployment.Store.DestroyFunc()
		server.Terminate(t)
	}
}

func TestCreateDeploymentRollbackValidation(t *testing.T) {
	storage, server := newStorage(t)
	rollbackStorage := storage.Rollback
	rollback := apps.DeploymentRollback{
		Name:               name,
		UpdatedAnnotations: map[string]string{},
		RollbackTo:         apps.RollbackConfig{Revision: 1},
	}

	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace)

	if _, err := storage.Deployment.Create(ctx, validNewDeployment(), rest.ValidateAllObjectFunc, &metav1.CreateOptions{}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	validationError := fmt.Errorf("admission deny")
	alwaysDenyValidationFunc := func(obj runtime.Object) error { return validationError }
	_, err := rollbackStorage.Create(ctx, rollback.Name, &rollback, alwaysDenyValidationFunc, &metav1.CreateOptions{})

	if err == nil || validationError != err {
		t.Errorf("expected: %v, got: %v", validationError, err)
	}

	storage.Deployment.Store.DestroyFunc()
	server.Terminate(t)
}

// Ensure that when a deploymentRollback is created for a deployment that has already been deleted
// by the API server, API server returns not-found error.
func TestEtcdCreateDeploymentRollbackNoDeployment(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Deployment.Store.DestroyFunc()
	rollbackStorage := storage.Rollback
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace)

	_, err := rollbackStorage.Create(ctx, name, &apps.DeploymentRollback{
		Name:               name,
		UpdatedAnnotations: map[string]string{},
		RollbackTo:         apps.RollbackConfig{Revision: 1},
	}, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err == nil {
		t.Fatalf("Expected not-found-error but got nothing")
	}
	if !errors.IsNotFound(storeerr.InterpretGetError(err, apps.Resource("deployments"), name)) {
		t.Fatalf("Unexpected error returned: %#v", err)
	}

	_, err = storage.Deployment.Get(ctx, name, &metav1.GetOptions{})
	if err == nil {
		t.Fatalf("Expected not-found-error but got nothing")
	}
	if !errors.IsNotFound(storeerr.InterpretGetError(err, apps.Resource("deployments"), name)) {
		t.Fatalf("Unexpected error: %v", err)
	}
}

func TestShortNames(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Deployment.Store.DestroyFunc()
	expected := []string{"deploy"}
	registrytest.AssertShortNames(t, storage.Deployment, expected)
}

func TestCategories(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Deployment.Store.DestroyFunc()
	expected := []string{"all"}
	registrytest.AssertCategories(t, storage.Deployment, expected)
}
