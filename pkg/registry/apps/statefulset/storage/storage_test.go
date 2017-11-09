/*
Copyright 2016 The Kubernetes Authors.

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
	"testing"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/diff"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

// TODO: allow for global factory override
func newStorage(t *testing.T) (StatefulSetStorage, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, apps.GroupName)
	restOptions := generic.RESTOptions{StorageConfig: etcdStorage, Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1, ResourcePrefix: "statefulsets"}
	storage := NewStorage(restOptions)
	return storage, server
}

// createStatefulSet is a helper function that returns a StatefulSet with the updated resource version.
func createStatefulSet(storage *REST, ps apps.StatefulSet, t *testing.T) (apps.StatefulSet, error) {
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), ps.Namespace)
	obj, err := storage.Create(ctx, &ps, rest.ValidateAllObjectFunc, false)
	if err != nil {
		t.Errorf("Failed to create StatefulSet, %v", err)
	}
	newPS := obj.(*apps.StatefulSet)
	return *newPS, nil
}

func validNewStatefulSet() *apps.StatefulSet {
	return &apps.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: metav1.NamespaceDefault,
			Labels:    map[string]string{"a": "b"},
		},
		Spec: apps.StatefulSetSpec{
			PodManagementPolicy: apps.OrderedReadyPodManagement,
			Selector:            &metav1.LabelSelector{MatchLabels: map[string]string{"a": "b"}},
			Template: api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
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
			Replicas:       7,
			UpdateStrategy: apps.StatefulSetUpdateStrategy{Type: apps.RollingUpdateStatefulSetStrategyType},
		},
		Status: apps.StatefulSetStatus{},
	}
}

var validStatefulSet = *validNewStatefulSet()

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.StatefulSet.Store.DestroyFunc()
	test := registrytest.New(t, storage.StatefulSet.Store)
	ps := validNewStatefulSet()
	ps.ObjectMeta = metav1.ObjectMeta{}
	test.TestCreate(
		// valid
		ps,
		// TODO: Add an invalid case when we have validation.
	)
}

// TODO: Test updates to spec when we allow them.

func TestStatusUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.StatefulSet.Store.DestroyFunc()
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
	key := "/statefulsets/" + metav1.NamespaceDefault + "/foo"
	validStatefulSet := validNewStatefulSet()
	if err := storage.StatefulSet.Storage.Create(ctx, key, validStatefulSet, nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	update := apps.StatefulSet{
		ObjectMeta: validStatefulSet.ObjectMeta,
		Spec: apps.StatefulSetSpec{
			Replicas: 7,
		},
		Status: apps.StatefulSetStatus{
			Replicas: 7,
		},
	}

	if _, _, err := storage.Status.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err := storage.StatefulSet.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	ps := obj.(*apps.StatefulSet)
	if ps.Spec.Replicas != 7 {
		t.Errorf("we expected .spec.replicas to not be updated but it was updated to %v", ps.Spec.Replicas)
	}
	if ps.Status.Replicas != 7 {
		t.Errorf("we expected .status.replicas to be updated to %d but it was %v", 7, ps.Status.Replicas)
	}
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.StatefulSet.Store.DestroyFunc()
	test := registrytest.New(t, storage.StatefulSet.Store)
	test.TestGet(validNewStatefulSet())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.StatefulSet.Store.DestroyFunc()
	test := registrytest.New(t, storage.StatefulSet.Store)
	test.TestList(validNewStatefulSet())
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.StatefulSet.Store.DestroyFunc()
	test := registrytest.New(t, storage.StatefulSet.Store)
	test.TestDelete(validNewStatefulSet())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.StatefulSet.Store.DestroyFunc()
	test := registrytest.New(t, storage.StatefulSet.Store)
	test.TestWatch(
		validNewStatefulSet(),
		// matching labels
		[]labels.Set{
			{"a": "b"},
		},
		// not matching labels
		[]labels.Set{
			{"a": "c"},
			{"foo": "bar"},
		},

		// matching fields
		[]fields.Set{
			{"metadata.name": "foo"},
		},
		// not matching fields
		[]fields.Set{
			{"metadata.name": "bar"},
		},
	)
}

func TestCategories(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.StatefulSet.Store.DestroyFunc()
	expected := []string{"all"}
	registrytest.AssertCategories(t, storage.StatefulSet, expected)
}

func TestShortNames(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.StatefulSet.Store.DestroyFunc()
	expected := []string{"sts"}
	registrytest.AssertShortNames(t, storage.StatefulSet, expected)
}

func TestScaleGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.StatefulSet.Store.DestroyFunc()

	versions := []string{"v1beta1", "v1beta2", "v1"}
	for _, version := range versions {
		name := "foo-" + version
		var sts apps.StatefulSet
		ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
		requestInfo := genericapirequest.RequestInfo{
			APIGroup:   "apps",
			APIVersion: version,
			Resource:   "statefulsets",
		}
		ctx = genericapirequest.WithRequestInfo(ctx, &requestInfo)
		key := "/statefulsets/" + metav1.NamespaceDefault + "/" + name
		validStatefulSet.Name = name
		if err := storage.StatefulSet.Storage.Create(ctx, key, &validStatefulSet, &sts, 0); err != nil {
			t.Fatalf("error setting new statefulset (key: %s) %v: %v", key, validStatefulSet, err)
		}

		// for apps/v1beta1 and apps/v1beta2
		wantExtensions := &extensions.Scale{
			ObjectMeta: metav1.ObjectMeta{
				Name:              name,
				Namespace:         metav1.NamespaceDefault,
				UID:               sts.UID,
				ResourceVersion:   sts.ResourceVersion,
				CreationTimestamp: sts.CreationTimestamp,
			},
			Spec: extensions.ScaleSpec{
				Replicas: validStatefulSet.Spec.Replicas,
			},
			Status: extensions.ScaleStatus{
				Replicas: validStatefulSet.Status.Replicas,
				Selector: validStatefulSet.Spec.Selector,
			},
		}
		// for starting apps/v1
		wantAuto := &autoscaling.Scale{
			ObjectMeta: metav1.ObjectMeta{
				Name:              name,
				Namespace:         metav1.NamespaceDefault,
				UID:               sts.UID,
				ResourceVersion:   sts.ResourceVersion,
				CreationTimestamp: sts.CreationTimestamp,
			},
			Spec: autoscaling.ScaleSpec{
				Replicas: validStatefulSet.Spec.Replicas,
			},
			Status: autoscaling.ScaleStatus{
				Replicas: validStatefulSet.Status.Replicas,
				Selector: labels.SelectorFromSet(validStatefulSet.Spec.Selector.MatchLabels).String(),
			},
		}

		obj, err := storage.Scale.Get(ctx, name, &metav1.GetOptions{})
		if err != nil {
			t.Fatalf("error fetching scale for %s: %v", name, err)
		}
		switch version {
		case "v1beta1", "v1beta2":
			got := obj.(*extensions.Scale)
			if !apiequality.Semantic.DeepEqual(got, wantExtensions) {
				t.Errorf("unexpected scale: %s", diff.ObjectDiff(got, wantExtensions))
			}
		default:
			got := obj.(*autoscaling.Scale)
			if !apiequality.Semantic.DeepEqual(got, wantAuto) {
				t.Errorf("unexpected scale: %s", diff.ObjectDiff(got, wantAuto))
			}
		}
	}
}

func TestScaleUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.StatefulSet.Store.DestroyFunc()

	versions := []string{"v1beta1", "v1beta2", "v1"}
	for _, version := range versions {
		name := "foo-" + version
		var sts apps.StatefulSet
		ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
		requestInfo := genericapirequest.RequestInfo{
			APIGroup:   "apps",
			APIVersion: version,
			Resource:   "statefulsets",
		}
		ctx = genericapirequest.WithRequestInfo(ctx, &requestInfo)
		key := "/statefulsets/" + metav1.NamespaceDefault + "/" + name
		validStatefulSet.Name = name
		if err := storage.StatefulSet.Storage.Create(ctx, key, &validStatefulSet, &sts, 0); err != nil {
			t.Fatalf("error setting new statefulset (key: %s) %v: %v", key, validStatefulSet, err)
		}
		replicas := 12
		extensionsScale := extensions.Scale{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: metav1.NamespaceDefault,
			},
			Spec: extensions.ScaleSpec{
				Replicas: int32(replicas),
			},
		}
		autoScale := autoscaling.Scale{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: metav1.NamespaceDefault,
			},
			Spec: autoscaling.ScaleSpec{
				Replicas: int32(replicas),
			},
		}

		switch version {
		case "v1beta1", "v1beta2":
			if _, _, err := storage.Scale.Update(ctx, extensionsScale.Name, rest.DefaultUpdatedObjectInfo(&extensionsScale), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc); err != nil {
				t.Fatalf("error updating scale %v: %v", extensionsScale, err)
			}
		default:
			if _, _, err := storage.Scale.Update(ctx, autoScale.Name, rest.DefaultUpdatedObjectInfo(&autoScale), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc); err != nil {
				t.Fatalf("error updating scale %v: %v", autoScale, err)
			}
		}

		obj, err := storage.Scale.Get(ctx, name, &metav1.GetOptions{})
		if err != nil {
			t.Fatalf("error fetching scale for %s: %v", name, err)
		}

		switch version {
		case "v1beta1", "v1beta2":
			extensionsScale = *obj.(*extensions.Scale)
			if extensionsScale.Spec.Replicas != int32(replicas) {
				t.Errorf("wrong replicas count expected: %d got: %d", replicas, extensionsScale.Spec.Replicas)
			}
			extensionsScale.ResourceVersion = sts.ResourceVersion
			extensionsScale.Spec.Replicas = 15
			if _, _, err = storage.Scale.Update(ctx, extensionsScale.Name, rest.DefaultUpdatedObjectInfo(&extensionsScale), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc); err != nil && !errors.IsConflict(err) {
				t.Fatalf("unexpected error, expecting an update conflict but got %v", err)
			}
		default:
			autoScale = *obj.(*autoscaling.Scale)
			if autoScale.Spec.Replicas != int32(replicas) {
				t.Errorf("wrong replicas count expected: %d got: %d", replicas, autoScale.Spec.Replicas)
			}
			autoScale.ResourceVersion = sts.ResourceVersion
			autoScale.Spec.Replicas = 15
			if _, _, err = storage.Scale.Update(ctx, autoScale.Name, rest.DefaultUpdatedObjectInfo(&autoScale), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc); err != nil && !errors.IsConflict(err) {
				t.Fatalf("unexpected error, expecting an update conflict but got %v", err)
			}
		}
	}
}

// TODO: Test generation number.
