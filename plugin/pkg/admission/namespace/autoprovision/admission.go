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
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/internalversion"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
)

func init() {
	admission.RegisterPlugin("NamespaceAutoProvision", func(config io.Reader) (admission.Interface, error) {
		return NewProvision(), nil
	})
}

// provision is an implementation of admission.Interface.
// It looks at all incoming requests in a namespace context, and if the namespace does not exist, it creates one.
// It is useful in deployments that do not want to restrict creation of a namespace prior to its usage.
type provision struct {
	*admission.Handler
	client          internalclientset.Interface
	namespaceLister corelisters.NamespaceLister
}

var _ = kubeapiserveradmission.WantsInternalKubeInformerFactory(&provision{})
var _ = kubeapiserveradmission.WantsInternalKubeClientSet(&provision{})

func (p *provision) Admit(a admission.Attributes) error {
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

	_, err := p.namespaceLister.Get(a.GetNamespace())
	if err == nil {
		return nil
	}

	if !errors.IsNotFound(err) {
		return admission.NewForbidden(a, err)
	}

	namespace := &api.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:      a.GetNamespace(),
			Namespace: "",
		},
		Status: api.NamespaceStatus{},
	}

	_, err = p.client.Core().Namespaces().Create(namespace)
	if err != nil && !errors.IsAlreadyExists(err) {
		return admission.NewForbidden(a, err)
	}

	return nil
}

// NewProvision creates a new namespace provision admission control handler
func NewProvision() admission.Interface {
	return &provision{
		Handler: admission.NewHandler(admission.Create),
	}
}

func (p *provision) SetInternalKubeClientSet(client internalclientset.Interface) {
	p.client = client
}

func (p *provision) SetInternalKubeInformerFactory(f informers.SharedInformerFactory) {
	namespaceInformer := f.Core().InternalVersion().Namespaces()
	p.namespaceLister = namespaceInformer.Lister()
	p.SetReadyFunc(namespaceInformer.Informer().HasSynced)
}

func (p *provision) Validate() error {
	if p.namespaceLister == nil {
		return fmt.Errorf("missing namespaceLister")
	}
	if p.client == nil {
		return fmt.Errorf("missing client")
	}
	return nil
}
