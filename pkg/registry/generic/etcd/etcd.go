/*
Copyright 2014 Google Inc. All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	etcderr "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// Etcd implements generic.Registry, backing it with etcd storage.
// It's intended to be embeddable, so that you can implement any
// non-generic functions if needed.
// You must supply a value for every field below before use; these are
// left public as it's meant to be overridable if need be.
type Etcd struct {
	// Called to make a new object, should return e.g., &api.Pod{}
	NewFunc func() runtime.Object

	// Called to make a new listing object, should return e.g., &api.PodList{}
	NewListFunc func() runtime.Object

	// Used for error reporting
	EndpointName string

	// Used for listing/watching; should not include trailing "/"
	KeyRoot string

	// Called for Create/Update/Get/Delete
	KeyFunc func(id string) string

	// Used for all etcd access functions
	Helper tools.EtcdHelper
}

// List returns a list of all the items matching m.
func (e *Etcd) List(ctx api.Context, m generic.Matcher) (runtime.Object, error) {
	list := e.NewListFunc()
	err := e.Helper.ExtractToList(e.KeyRoot, list)
	if err != nil {
		return nil, err
	}
	return generic.FilterList(list, m)
}

// Create inserts a new item.
func (e *Etcd) Create(ctx api.Context, id string, obj runtime.Object) error {
	err := e.Helper.CreateObj(e.KeyFunc(id), obj, 0)
	return etcderr.InterpretCreateError(err, e.EndpointName, id)
}

// Update updates the item.
func (e *Etcd) Update(ctx api.Context, id string, obj runtime.Object) error {
	// TODO: verify that SetObj checks ResourceVersion before succeeding.
	err := e.Helper.SetObj(e.KeyFunc(id), obj)
	return etcderr.InterpretUpdateError(err, e.EndpointName, id)
}

// Get retrieves the item from etcd.
func (e *Etcd) Get(ctx api.Context, id string) (runtime.Object, error) {
	obj := e.NewFunc()
	err := e.Helper.ExtractObj(e.KeyFunc(id), obj, false)
	if err != nil {
		return nil, etcderr.InterpretGetError(err, e.EndpointName, id)
	}
	return obj, nil
}

// Delete removes the item from etcd.
func (e *Etcd) Delete(ctx api.Context, id string) error {
	err := e.Helper.Delete(e.KeyFunc(id), false)
	return etcderr.InterpretDeleteError(err, e.EndpointName, id)
}

// Watch starts a watch for the items that m matches.
// TODO: Detect if m references a single object instead of a list.
func (e *Etcd) Watch(ctx api.Context, m generic.Matcher, resourceVersion uint64) (watch.Interface, error) {
	return e.Helper.WatchList(e.KeyRoot, resourceVersion, func(obj runtime.Object) bool {
		matches, err := m.Matches(obj)
		return err == nil && matches
	})
}
