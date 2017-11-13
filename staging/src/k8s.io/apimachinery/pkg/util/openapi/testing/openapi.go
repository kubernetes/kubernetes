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
	"io/ioutil"
	"os"
	"sync"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/openapi"
	"k8s.io/kube-openapi/pkg/util/proto"

	yaml "gopkg.in/yaml.v2"

	"github.com/googleapis/gnostic/OpenAPIv2"
	"github.com/googleapis/gnostic/compiler"
)

// Getter is an interface used to download, open or create a fake
// openapi_v2.Document.
type Getter interface {
	OpenAPISchema() (*openapi_v2.Document, error)
}

// Fake opens and returns a openapi swagger from a file Path. It will
// parse only once and then return the same copy everytime.
type Fake struct {
	Path string

	once     sync.Once
	document *openapi_v2.Document
	err      error
}

var _ Getter = &Fake{}

// OpenAPISchema returns the openapi document and a potential error.
func (f *Fake) OpenAPISchema() (*openapi_v2.Document, error) {
	f.once.Do(func() {
		_, err := os.Stat(f.Path)
		if err != nil {
			f.err = err
			return
		}
		spec, err := ioutil.ReadFile(f.Path)
		if err != nil {
			f.err = err
			return
		}
		var info yaml.MapSlice
		err = yaml.Unmarshal(spec, &info)
		if err != nil {
			f.err = err
			return
		}
		f.document, f.err = openapi_v2.NewDocument(info, compiler.NewContext("$root", nil))
	})
	return f.document, f.err
}

// Empty is a Getter that returns nothing.
type Empty struct{}

var _ Getter = Empty{}

// OpenAPISchema returns an empty openapi document
func (Empty) OpenAPISchema() (*openapi_v2.Document, error) {
	return &openapi_v2.Document{}, nil
}

// FakeResources is a wrapper to directly load the openapi schema from a
// file, and get the schema for given GVK. This is only for test since
// it's assuming that the file is there and everything will go fine.
type FakeResources struct {
	Getter Getter
}

var _ openapi.Resources = &FakeResources{}

// LookupResource will read the schema, parse it and return the
// resources. It doesn't return errors and will panic instead.
func (f *FakeResources) LookupResource(gvk schema.GroupVersionKind) proto.Schema {
	s, err := f.Getter.OpenAPISchema()
	if err != nil {
		panic(err)
	}
	resources, err := openapi.NewOpenAPIData(s)
	if err != nil {
		panic(err)
	}
	return resources.LookupResource(gvk)
}
