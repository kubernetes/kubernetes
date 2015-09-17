/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package registrytest

import (
	"sync"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

type NamespaceRegistry struct {
	mu        sync.Mutex
	List      api.NamespaceList
	Namespace *api.Namespace
	Err       error

	DeletedID string
	GottenID  string
	UpdatedID string
}

func (r *NamespaceRegistry) SetError(err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.Err = err
}

func (r *NamespaceRegistry) ListNamespaces(ctx api.Context, label labels.Selector) (*api.NamespaceList, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	ns, _ := api.NamespaceFrom(ctx)

	// Copy metadata from internal list into result
	res := new(api.NamespaceList)
	res.TypeMeta = r.List.TypeMeta
	res.ListMeta = r.List.ListMeta

	if ns != api.NamespaceAll {
		for _, Namespace := range r.List.Items {
			if ns == Namespace.Namespace {
				res.Items = append(res.Items, Namespace)
			}
		}
	} else {
		res.Items = append([]api.Namespace{}, r.List.Items...)
	}

	return res, r.Err
}

func (r *NamespaceRegistry) GetNamespace(ctx api.Context, id string) (*api.Namespace, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.GottenID = id
	return r.Namespace, r.Err
}

func (r *NamespaceRegistry) CreateNamespace(ctx api.Context, ns *api.Namespace) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.Namespace = new(api.Namespace)
	*r.Namespace = *ns
	r.List.Items = append(r.List.Items, *ns)
	return r.Err
}

func (r *NamespaceRegistry) DeleteNamespace(ctx api.Context, id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.DeletedID = id
	r.Namespace = nil
	return r.Err
}

func (r *NamespaceRegistry) UpdateNamespace(ctx api.Context, ns *api.Namespace) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.UpdatedID = ns.Name
	*r.Namespace = *ns
	return r.Err
}

func (r *NamespaceRegistry) WatchNamespaces(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	return nil, r.Err
}
