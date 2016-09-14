/*
Copyright 2014 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"

	"fmt"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/controller/informers"
)

func init() {
	admission.RegisterPlugin("NamespaceAutoProvision", func(client clientset.Interface, config io.Reader) (admission.Interface, error) {
		return NewProvision(client), nil
	})
}

// provision is an implementation of admission.Interface.
// It looks at all incoming requests in a namespace context, and if the namespace does not exist, it creates one.
// It is useful in deployments that do not want to restrict creation of a namespace prior to its usage.
type provision struct {
	*admission.Handler
	client            clientset.Interface
	namespaceInformer cache.SharedIndexInformer
}

var _ = admission.WantsInformerFactory(&provision{})

func (p *provision) Admit(a admission.Attributes) (err error) {
	// if we're here, then we've already passed authentication, so we're allowed to do what we're trying to do
	// if we're here, then the API server has found a route, which means that if we have a non-empty namespace
	// its a namespaced resource.
	if len(a.GetNamespace()) == 0 || a.GetKind().GroupKind() == api.Kind("Namespace") {
		return nil
	}
	// we need to wait for our caches to warm
	if !p.WaitForReady() {
		return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
	}
	namespace := &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:      a.GetNamespace(),
			Namespace: "",
		},
		Status: api.NamespaceStatus{},
	}
	_, exists, err := p.namespaceInformer.GetStore().Get(namespace)
	if err != nil {
		return admission.NewForbidden(a, err)
	}
	if exists {
		return nil
	}
	_, err = p.client.Core().Namespaces().Create(namespace)
	if err != nil && !errors.IsAlreadyExists(err) {
		return admission.NewForbidden(a, err)
	}
	return nil
}

// NewProvision creates a new namespace provision admission control handler
func NewProvision(c clientset.Interface) admission.Interface {
	return &provision{
		Handler: admission.NewHandler(admission.Create),
		client:  c,
	}
}

func (p *provision) SetInformerFactory(f informers.SharedInformerFactory) {
	p.namespaceInformer = f.Namespaces().Informer()
	p.SetReadyFunc(p.namespaceInformer.HasSynced)
}

func (p *provision) Validate() error {
	if p.namespaceInformer == nil {
		return fmt.Errorf("missing namespaceInformer")
	}
	return nil
}
