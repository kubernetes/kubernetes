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
	"context"
	"fmt"
	"io"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	corev1listers "k8s.io/client-go/listers/core/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// PluginName indicates name of admission plugin.
const PluginName = "NamespaceAutoProvision"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewProvision(), nil
	})
}

// Provision is an implementation of admission.Interface.
// It looks at all incoming requests in a namespace context, and if the namespace does not exist, it creates one.
// It is useful in deployments that do not want to restrict creation of a namespace prior to its usage.
type Provision struct {
	*admission.Handler
	client          kubernetes.Interface
	namespaceLister corev1listers.NamespaceLister
}

var _ admission.MutationInterface = &Provision{}
var _ = genericadmissioninitializer.WantsExternalKubeInformerFactory(&Provision{})
var _ = genericadmissioninitializer.WantsExternalKubeClientSet(&Provision{})

// Admit makes an admission decision based on the request attributes
func (p *Provision) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	// Don't create a namespace if the request is for a dry-run.
	if a.IsDryRun() {
		return nil
	}

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

	namespace := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:      a.GetNamespace(),
			Namespace: "",
		},
		Status: corev1.NamespaceStatus{},
	}

	_, err = p.client.CoreV1().Namespaces().Create(context.TODO(), namespace, metav1.CreateOptions{})
	if err != nil && !errors.IsAlreadyExists(err) {
		return admission.NewForbidden(a, err)
	}

	return nil
}

// NewProvision creates a new namespace provision admission control handler
func NewProvision() *Provision {
	return &Provision{
		Handler: admission.NewHandler(admission.Create),
	}
}

// SetExternalKubeClientSet implements the WantsExternalKubeClientSet interface.
func (p *Provision) SetExternalKubeClientSet(client kubernetes.Interface) {
	p.client = client
}

// SetExternalKubeInformerFactory implements the WantsExternalKubeInformerFactory interface.
func (p *Provision) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	namespaceInformer := f.Core().V1().Namespaces()
	p.namespaceLister = namespaceInformer.Lister()
	p.SetReadyFunc(namespaceInformer.Informer().HasSynced)
}

// ValidateInitialization implements the InitializationValidator interface.
func (p *Provision) ValidateInitialization() error {
	if p.namespaceLister == nil {
		return fmt.Errorf("missing namespaceLister")
	}
	if p.client == nil {
		return fmt.Errorf("missing client")
	}
	return nil
}
