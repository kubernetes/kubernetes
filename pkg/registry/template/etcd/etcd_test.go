/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
)

var namespace = "foo-namespace"
var name = "foo-template"

func newRest(t *testing.T) (*REST, *ProcessREST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, extensions.GroupName)
	restOptions := generic.RESTOptions{Storage: etcdStorage, Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1}
	rest, processRest := NewREST(restOptions)
	return rest, processRest, server
}

func validNewTemplate() *extensions.Template {
	return &extensions.Template{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: extensions.TemplateSpec{
			Objects:    []runtime.RawExtension{},
			Parameters: []extensions.Parameter{},
		},
	}
}

func TestCreate(t *testing.T) {
	rest, _, server := newRest(t)
	defer server.Terminate(t)
	test := registrytest.New(t, rest.Store)
	template := validNewTemplate()
	template.ObjectMeta = api.ObjectMeta{}

	invalid := []runtime.Object{}
	for _, t := range GetInvalidTemplates(t) {
		invalid = append(invalid, t)
	}
	test.TestCreate(
		// valid
		template,
		invalid...,
	)
}

func TestUpdate(t *testing.T) {
	rest, _, server := newRest(t)
	defer server.Terminate(t)
	test := registrytest.New(t, rest.Store)

	invalidUpdates := []registrytest.UpdateFunc{}
	for _, i := range GetInvalidTemplates(t) {
		t := *i
		invalidUpdates = append(invalidUpdates, func(obj runtime.Object) runtime.Object {
			object := &t
			return object
		})
	}
	test.TestUpdate(
		// valid object
		validNewTemplate(),
		// valid update
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.Template)
			return object
		},
		// Cases are already tested in the extensions/v1beta1/validation_test.go
		invalidUpdates...,
	)
}

func TestDelete(t *testing.T) {
	rest, _, server := newRest(t)
	defer server.Terminate(t)
	test := registrytest.New(t, rest.Store)
	test.TestDelete(validNewTemplate())
}

func TestGet(t *testing.T) {
	rest, _, server := newRest(t)
	defer server.Terminate(t)
	test := registrytest.New(t, rest.Store)
	test.TestGet(validNewTemplate())
}

func TestList(t *testing.T) {
	rest, _, server := newRest(t)
	defer server.Terminate(t)
	test := registrytest.New(t, rest.Store)
	test.TestList(validNewTemplate())
}

func TestWatch(t *testing.T) {
	rest, _, server := newRest(t)
	defer server.Terminate(t)
	test := registrytest.New(t, rest.Store)
	test.TestWatch(
		validNewTemplate(),
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

// Adapted from extensions/v1beta1/validation_test.go
func GetInvalidTemplates(t *testing.T) []*extensions.Template {
	invalid := []*extensions.Template{}
	for _, spec := range GetInvalidTemplateSpecss(t) {
		invalid = append(invalid, &extensions.Template{
			Spec:       *spec,
			ObjectMeta: api.ObjectMeta{},
		})
	}
	return invalid
}

// Copied from extensions/v1beta1/validation_test.go
func GetInvalidTemplateSpecss(t *testing.T) []*extensions.TemplateSpec {
	invalid := []*extensions.TemplateSpec{
		// Parameter with invalid name
		{
			Parameters: []extensions.Parameter{
				{
					Name:        "",
					Description: "parameter Name is empty.",
				},
			},
		},

		// Parameter with invalid name
		{
			Parameters: []extensions.Parameter{
				{
					Name: "NAME",
				},
				{
					Name:        "badNAME",
					Description: "parameter Name does not match [A-Z0-9_]+.",
				},
			},
		},

		// Same parameter defined multiple times
		{
			Parameters: []extensions.Parameter{
				{
					Name: "NAME",
				},
				{
					Name:        "NAME",
					Description: "same parameter Name specified multiple times.",
				},
			},
		},

		// Unknown Type
		{
			Parameters: []extensions.Parameter{
				{
					Name:        "NAME",
					Type:        "unknown",
					Description: "invalid Type.",
				},
			},
		},
	}

	// Boolean Type Validation
	invalid = append(invalid,
		// No Value and not Required
		&extensions.TemplateSpec{
			Parameters: []extensions.Parameter{
				{
					Name:        "NAME",
					Type:        "boolean",
					Description: "boolean missing default Value and not Required.",
				},
			},
		},
		// Unparseable
		&extensions.TemplateSpec{
			Parameters: []extensions.Parameter{
				{
					Name:        "NAME",
					Type:        "boolean",
					Value:       "5",
					Description: "boolean should not be able to parse '5'.",
				},
			},
		},
	)

	// Integer Type Validation
	invalid = append(invalid,
		&extensions.TemplateSpec{
			Parameters: []extensions.Parameter{
				{
					Name:        "NAME",
					Type:        "integer",
					Description: "integer missing default Value and not Required.",
				},
			},
		},
		&extensions.TemplateSpec{
			Parameters: []extensions.Parameter{
				{
					Name:        "NAME",
					Type:        "integer",
					Value:       "true",
					Description: "integer should not be able to parse 'true'",
				},
			},
		},
	)

	// Base64 Type Validation
	invalid = append(invalid,
		&extensions.TemplateSpec{
			Parameters: []extensions.Parameter{
				{
					Name:        "NAME",
					Type:        "base64",
					Value:       "aG VsbG8gd29ybGQ=",
					Description: "base64 data should not be parseable",
				},
			},
		},
	)
	return invalid
}
