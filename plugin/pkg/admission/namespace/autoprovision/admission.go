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

package autoprovision

import (
	"io"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/admission"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
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
	client client.Interface
	store  cache.Store
}

func (p *provision) Admit(a admission.Attributes) (err error) {
	defaultVersion, kind, err := latest.RESTMapper.VersionAndKindForResource(a.GetResource())
	if err != nil {
		return err
	}
	mapping, err := latest.RESTMapper.RESTMapping(kind, defaultVersion)
	if err != nil {
		return err
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
		return err
	}
	if exists {
		return nil
	}
	_, err = p.client.Namespaces().Create(namespace)
	if err != nil {
		return err
	}
	return nil
}

func NewProvision(c client.Interface) admission.Interface {
	store := cache.NewStore(cache.MetaNamespaceKeyFunc)
	reflector := cache.NewReflector(
		&cache.ListWatch{
			Client:        c.(*client.Client),
			FieldSelector: labels.Everything(),
			Resource:      "namespaces",
		},
		&api.Namespace{},
		store,
	)
	reflector.Run()
	return &provision{
		client: c,
		store:  store,
	}
}
