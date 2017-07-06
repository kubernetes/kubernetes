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

package exists

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

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register("NamespaceExists", func(config io.Reader) (admission.Interface, error) {
		return NewExists(), nil
	})
}

// exists is an implementation of admission.Interface.
// It rejects all incoming requests in a namespace context if the namespace does not exist.
// It is useful in deployments that want to enforce pre-declaration of a Namespace resource.
type exists struct {
	*admission.Handler
	client          internalclientset.Interface
	namespaceLister corelisters.NamespaceLister
}

var _ = kubeapiserveradmission.WantsInternalKubeInformerFactory(&exists{})
var _ = kubeapiserveradmission.WantsInternalKubeClientSet(&exists{})

func (e *exists) Admit(a admission.Attributes) error {
	// if we're here, then we've already passed authentication, so we're allowed to do what we're trying to do
	// if we're here, then the API server has found a route, which means that if we have a non-empty namespace
	// its a namespaced resource.
	if len(a.GetNamespace()) == 0 || a.GetKind().GroupKind() == api.Kind("Namespace") {
		return nil
	}

	// we need to wait for our caches to warm
	if !e.WaitForReady() {
		return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
	}
	_, err := e.namespaceLister.Get(a.GetNamespace())
	if err == nil {
		return nil
	}
	if !errors.IsNotFound(err) {
		return errors.NewInternalError(err)
	}

	// in case of latency in our caches, make a call direct to storage to verify that it truly exists or not
	_, err = e.client.Core().Namespaces().Get(a.GetNamespace(), metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			return err
		}
		return errors.NewInternalError(err)
	}

	return nil
}

// NewExists creates a new namespace exists admission control handler
func NewExists() admission.Interface {
	return &exists{
		Handler: admission.NewHandler(admission.Create, admission.Update, admission.Delete),
	}
}

func (e *exists) SetInternalKubeClientSet(client internalclientset.Interface) {
	e.client = client
}

func (e *exists) SetInternalKubeInformerFactory(f informers.SharedInformerFactory) {
	namespaceInformer := f.Core().InternalVersion().Namespaces()
	e.namespaceLister = namespaceInformer.Lister()
	e.SetReadyFunc(namespaceInformer.Informer().HasSynced)
}

func (e *exists) Validate() error {
	if e.namespaceLister == nil {
		return fmt.Errorf("missing namespaceLister")
	}
	if e.client == nil {
		return fmt.Errorf("missing client")
	}
	return nil
}
