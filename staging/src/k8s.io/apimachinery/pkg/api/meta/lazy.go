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

package meta

import (
	"sync"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// lazyObject defers loading the mapper and typer until necessary.
type lazyObject struct {
	loader func() (RESTMapper, runtime.ObjectTyper, error)

	lock   sync.Mutex
	loaded bool
	err    error
	mapper RESTMapper
	typer  runtime.ObjectTyper
}

// NewLazyObjectLoader handles unrecoverable errors when creating a RESTMapper / ObjectTyper by
// returning those initialization errors when the interface methods are invoked. This defers the
// initialization and any server calls until a client actually needs to perform the action.
func NewLazyObjectLoader(fn func() (RESTMapper, runtime.ObjectTyper, error)) (RESTMapper, runtime.ObjectTyper) {
	obj := &lazyObject{loader: fn}
	return obj, obj
}

// init lazily loads the mapper and typer, returning an error if initialization has failed.
func (o *lazyObject) init() error {
	o.lock.Lock()
	defer o.lock.Unlock()
	if o.loaded {
		return o.err
	}
	o.mapper, o.typer, o.err = o.loader()
	o.loaded = true
	return o.err
}

var _ RESTMapper = &lazyObject{}
var _ runtime.ObjectTyper = &lazyObject{}

func (o *lazyObject) KindFor(resource schema.GroupVersionResource) (schema.GroupVersionKind, error) {
	if err := o.init(); err != nil {
		return schema.GroupVersionKind{}, err
	}
	return o.mapper.KindFor(resource)
}

func (o *lazyObject) KindsFor(resource schema.GroupVersionResource) ([]schema.GroupVersionKind, error) {
	if err := o.init(); err != nil {
		return []schema.GroupVersionKind{}, err
	}
	return o.mapper.KindsFor(resource)
}

func (o *lazyObject) ResourceFor(input schema.GroupVersionResource) (schema.GroupVersionResource, error) {
	if err := o.init(); err != nil {
		return schema.GroupVersionResource{}, err
	}
	return o.mapper.ResourceFor(input)
}

func (o *lazyObject) ResourcesFor(input schema.GroupVersionResource) ([]schema.GroupVersionResource, error) {
	if err := o.init(); err != nil {
		return []schema.GroupVersionResource{}, err
	}
	return o.mapper.ResourcesFor(input)
}

func (o *lazyObject) RESTMapping(gk schema.GroupKind, versions ...string) (*RESTMapping, error) {
	if err := o.init(); err != nil {
		return nil, err
	}
	return o.mapper.RESTMapping(gk, versions...)
}

func (o *lazyObject) RESTMappings(gk schema.GroupKind, versions ...string) ([]*RESTMapping, error) {
	if err := o.init(); err != nil {
		return nil, err
	}
	return o.mapper.RESTMappings(gk, versions...)
}

func (o *lazyObject) ResourceSingularizer(resource string) (singular string, err error) {
	if err := o.init(); err != nil {
		return "", err
	}
	return o.mapper.ResourceSingularizer(resource)
}

func (o *lazyObject) ObjectKinds(obj runtime.Object) ([]schema.GroupVersionKind, bool, error) {
	if err := o.init(); err != nil {
		return nil, false, err
	}
	return o.typer.ObjectKinds(obj)
}

func (o *lazyObject) Recognizes(gvk schema.GroupVersionKind) bool {
	if err := o.init(); err != nil {
		return false
	}
	return o.typer.Recognizes(gvk)
}
