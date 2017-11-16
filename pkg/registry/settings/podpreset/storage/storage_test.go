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

package storage

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/settings"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, settings.GroupName)
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "podpresets",
	}
	return NewREST(restOptions), server
}

func validNewPodPreset(namespace string) *settings.PodPreset {
	return &settings.PodPreset{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "podPreset",
			Namespace: namespace,
			Labels:    map[string]string{"a": "b"},
		},
		Spec: settings.PodPresetSpec{
			Selector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"role": "frontend",
				},
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{
						Key:      "security",
						Operator: metav1.LabelSelectorOpIn,
						Values:   []string{"S2"},
					},
				},
			},
			Env: []api.EnvVar{
				{
					Name:  "DB_PORT",
					Value: "6379",
				},
			},
			EnvFrom: []api.EnvFromSource{
				{
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{Name: "abc"},
					},
				},
				{
					Prefix: "pre_",
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{Name: "abc"},
					},
				},
			},
			VolumeMounts: []api.VolumeMount{
				{
					MountPath: "/cache",
					Name:      "cache-volume",
				},
			},
			Volumes: []api.Volume{
				{
					Name: "cache-volume",
					VolumeSource: api.VolumeSource{
						EmptyDir: &api.EmptyDirVolumeSource{},
					},
				},
			},
		},
	}
}

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store, legacyscheme.Scheme)
	invalidPodPreset := validNewPodPreset(test.TestNamespace())
	invalidPodPreset.Spec.VolumeMounts[0].Name = "/cache/VolumeMounts"
	test.TestCreate(
		validNewPodPreset(test.TestNamespace()),
		// invalid cases
		invalidPodPreset,
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store, legacyscheme.Scheme)
	test.TestUpdate(
		// valid
		validNewPodPreset(test.TestNamespace()),
		// invalid updates
		func(obj runtime.Object) runtime.Object {
			pp := obj.(*settings.PodPreset)
			pp.Labels = map[string]string{"c": "d"}
			return pp
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store, legacyscheme.Scheme)
	test.TestDelete(validNewPodPreset(test.TestNamespace()))
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store, legacyscheme.Scheme)
	test.TestGet(validNewPodPreset(test.TestNamespace()))
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store, legacyscheme.Scheme)
	test.TestList(validNewPodPreset(test.TestNamespace()))
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store, legacyscheme.Scheme)
	test.TestWatch(
		validNewPodPreset(test.TestNamespace()),
		// matching labels
		[]labels.Set{},
		// not matching labels
		// not matching labels
		[]labels.Set{
			{"foo": "bar"},
		},

		// matching fields
		[]fields.Set{
			{"metadata.name": "podPreset"},
		},
		// not matching fields
		[]fields.Set{
			{"metadata.name": "bar"},
		},
	)
}
