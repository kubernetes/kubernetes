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

package exists

import (
	"fmt"
	"io"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/admission"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

func init() {
	admission.RegisterPlugin("NamespaceExists", func(client client.Interface, config io.Reader) (admission.Interface, error) {
		return NewExists(client), nil
	})
}

// exists is an implementation of admission.Interface.
// It rejects all incoming requests in a namespace context if the namespace does not exist.
// It is useful in deployments that want to enforce pre-declaration of a Namespace resource.
type exists struct {
	*admission.Handler
	client client.Interface
	store  cache.Store
}

func (e *exists) Admit(a admission.Attributes) (err error) {
	defaultVersion, kind, err := latest.RESTMapper.VersionAndKindForResource(a.GetResource())
	if err != nil {
		return admission.NewForbidden(a, err)
	}
	mapping, err := latest.RESTMapper.RESTMapping(kind, defaultVersion)
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
	_, exists, err := e.store.Get(namespace)
	if err != nil {
		return admission.NewForbidden(a, err)
	}
	if exists {
		return nil
	}

	// in case of latency in our caches, make a call direct to storage to verify that it truly exists or not
	_, err = e.client.Namespaces().Get(a.GetNamespace())
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("Namespace %s does not exist", a.GetNamespace()))
	}

	return nil
}

// NewExists creates a new namespace exists admission control handler
func NewExists(c client.Interface) admission.Interface {
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
		5*time.Minute,
	)
	reflector.Run()
	return &exists{
		client:  c,
		store:   store,
		Handler: admission.NewHandler(admission.Create, admission.Update, admission.Delete),
	}
}
