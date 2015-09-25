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

package autoprovision

import (
	"io"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

func init() {
	admission.RegisterPlugin("NamespaceAutoProvision", func(client client.Interface, config io.Reader) (admission.Interface, error) {
		return NewProvision(client), nil
	})
}

// provision is an implementation of admission.Interface.
// It looks at all incoming requests in a namespace context, and if the namespace does not exist, it creates one.
// It is useful in deployments that do not want to restrict creation of a namespace prior to its usage.
type provision struct {
	*admission.Handler
	client client.Interface
	store  cache.Store
}

func (p *provision) Admit(a admission.Attributes) (err error) {
	defaultVersion, kind, err := api.RESTMapper.VersionAndKindForResource(a.GetResource())
	if err != nil {
		return admission.NewForbidden(a, err)
	}
	mapping, err := api.RESTMapper.RESTMapping(kind, defaultVersion)
	if err != nil {
		return admission.NewForbidden(a, err)
	}
	if mapping.Scope.Name() != meta.RESTScopeNameNamespace {
		return nil
	}
	namespace := &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:      a.GetNamespace(),
			Namespace: "",
		},
		Status: api.NamespaceStatus{},
	}
	_, exists, err := p.store.Get(namespace)
	if err != nil {
		return admission.NewForbidden(a, err)
	}
	if exists {
		return nil
	}
	_, err = p.client.Namespaces().Create(namespace)
	if err != nil && !errors.IsAlreadyExists(err) {
		return admission.NewForbidden(a, err)
	}
	return nil
}

// NewProvision creates a new namespace provision admission control handler
func NewProvision(c client.Interface) admission.Interface {
	store := cache.NewStore(cache.MetaNamespaceKeyFunc)
	reflector := cache.NewReflector(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return c.Namespaces().List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(resourceVersion string) (watch.Interface, error) {
				return c.Namespaces().Watch(labels.Everything(), fields.Everything(), resourceVersion)
			},
		},
		&api.Namespace{},
		store,
		0,
	)
	reflector.Run()
	return createProvision(c, store)
}

func createProvision(c client.Interface, store cache.Store) admission.Interface {
	return &provision{
		Handler: admission.NewHandler(admission.Create),
		client:  c,
		store:   store,
	}
}
