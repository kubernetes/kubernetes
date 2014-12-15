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
	kubeerr "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
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
	KeyRootFunc func(ctx api.Context) string

	// Called for Create/Update/Get/Delete
	KeyFunc func(ctx api.Context, id string) (string, error)

	// Used for all etcd access functions
	Helper tools.EtcdHelper
}

// NamespaceKeyRootFunc is the default function for constructing etcd paths to resource directories enforcing namespace rules.
func NamespaceKeyRootFunc(ctx api.Context, prefix string) string {
	key := prefix
	ns, ok := api.NamespaceFrom(ctx)
	if ok && len(ns) > 0 {
		key = key + "/" + ns
	}
	return key
}

// NamespaceKeyFunc is the default function for constructing etcd paths to a resource relative to prefix enforcing namespace rules.
// If no namespace is on context, it errors.
func NamespaceKeyFunc(ctx api.Context, prefix string, id string) (string, error) {
	key := NamespaceKeyRootFunc(ctx, prefix)
	ns, ok := api.NamespaceFrom(ctx)
	if !ok || len(ns) == 0 {
		return "", kubeerr.NewBadRequest("Namespace parameter required.")
	}
	if len(id) == 0 {
		return "", kubeerr.NewBadRequest("Namespace parameter required.")
	}
	key = key + "/" + id
	return key, nil
}

// List returns a list of all the items matching m.
func (e *Etcd) List(ctx api.Context, m generic.Matcher) (runtime.Object, error) {
	list := e.NewListFunc()
	err := e.Helper.ExtractToList(e.KeyRootFunc(ctx), list)
	if err != nil {
		return nil, err
	}
	return generic.FilterList(list, m)
}

// Create inserts a new item.
func (e *Etcd) Create(ctx api.Context, id string, obj runtime.Object) error {
	key, err := e.KeyFunc(ctx, id)
	if err != nil {
		return err
	}
	err = e.Helper.CreateObj(key, obj, 0)
	return etcderr.InterpretCreateError(err, e.EndpointName, id)
}

// Update updates the item.
func (e *Etcd) Update(ctx api.Context, id string, obj runtime.Object) error {
	key, err := e.KeyFunc(ctx, id)
	if err != nil {
		return err
	}
	// TODO: verify that SetObj checks ResourceVersion before succeeding.
	err = e.Helper.SetObj(key, obj)
	return etcderr.InterpretUpdateError(err, e.EndpointName, id)
}

// Get retrieves the item from etcd.
func (e *Etcd) Get(ctx api.Context, id string) (runtime.Object, error) {
	obj := e.NewFunc()
	key, err := e.KeyFunc(ctx, id)
	if err != nil {
		return nil, err
	}
	err = e.Helper.ExtractObj(key, obj, false)
	if err != nil {
		return nil, etcderr.InterpretGetError(err, e.EndpointName, id)
	}
	return obj, nil
}

// Delete removes the item from etcd.
func (e *Etcd) Delete(ctx api.Context, id string) error {
	key, err := e.KeyFunc(ctx, id)
	if err != nil {
		return err
	}
	err = e.Helper.Delete(key, false)
	return etcderr.InterpretDeleteError(err, e.EndpointName, id)
}

// Watch starts a watch for the items that m matches.
// TODO: Detect if m references a single object instead of a list.
func (e *Etcd) Watch(ctx api.Context, m generic.Matcher, resourceVersion string) (watch.Interface, error) {
	version, err := tools.ParseWatchResourceVersion(resourceVersion, e.EndpointName)
	if err != nil {
		return nil, err
	}
	return e.Helper.WatchList(e.KeyRootFunc(ctx), version, func(obj runtime.Object) bool {
		matches, err := m.Matches(obj)
		return err == nil && matches
	})
}
