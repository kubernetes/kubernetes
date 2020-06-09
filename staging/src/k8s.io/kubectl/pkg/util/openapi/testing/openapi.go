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

package testing

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kube-openapi/pkg/util/proto"
	"k8s.io/kube-openapi/pkg/util/proto/testing"
	"k8s.io/kubectl/pkg/util/openapi"
)

// FakeResources is a wrapper to directly load the openapi schema from a
// file, and get the schema for given GVK. This is only for test since
// it's assuming that the file is there and everything will go fine.
type FakeResources struct {
	fake testing.Fake
}

var _ openapi.Resources = &FakeResources{}

// NewFakeResources creates a new FakeResources.
func NewFakeResources(path string) *FakeResources {
	return &FakeResources{
		fake: testing.Fake{Path: path},
	}
}

// LookupResource will read the schema, parse it and return the
// resources. It doesn't return errors and will panic instead.
func (f *FakeResources) LookupResource(gvk schema.GroupVersionKind) proto.Schema {
	s, err := f.fake.OpenAPISchema()
	if err != nil {
		panic(err)
	}
	resources, err := openapi.NewOpenAPIData(s)
	if err != nil {
		panic(err)
	}
	return resources.LookupResource(gvk)
}

// EmptyResources implement a Resources that just doesn't have any resources.
type EmptyResources struct{}

var _ openapi.Resources = EmptyResources{}

// LookupResource will always return nil. It doesn't have any resources.
func (f EmptyResources) LookupResource(gvk schema.GroupVersionKind) proto.Schema {
	return nil
}

// CreateOpenAPISchemaFunc returns a function useful for the TestFactory.
func CreateOpenAPISchemaFunc(path string) func() (openapi.Resources, error) {
	return func() (openapi.Resources, error) {
		return NewFakeResources(path), nil
	}
}
