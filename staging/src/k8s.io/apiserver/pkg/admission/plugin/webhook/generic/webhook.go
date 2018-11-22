/*
Copyright 2018 The Kubernetes Authors.

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

package generic

import (
	"context"
	"fmt"
	"io"

	admissionv1beta1 "k8s.io/api/admission/v1beta1"
	"k8s.io/api/admissionregistration/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninit "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/config"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/namespace"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/rules"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
)

// Webhook is an abstract admission plugin with all the infrastructure to define Admit or Validate on-top.
type Webhook struct {
	*admission.Handler

	sourceFactory sourceFactory

	hookSource       Source
	clientManager    *webhook.ClientManager
	convertor        *convertor
	namespaceMatcher *namespace.Matcher
	dispatcher       Dispatcher
}

var (
	_ genericadmissioninit.WantsExternalKubeClientSet = &Webhook{}
	_ admission.Interface                             = &Webhook{}
)

type sourceFactory func(f informers.SharedInformerFactory) Source
type dispatcherFactory func(cm *webhook.ClientManager) Dispatcher

// NewWebhook creates a new generic admission webhook.
func NewWebhook(handler *admission.Handler, configFile io.Reader, sourceFactory sourceFactory, dispatcherFactory dispatcherFactory) (*Webhook, error) {
	kubeconfigFile, err := config.LoadConfig(configFile)
	if err != nil {
		return nil, err
	}

	cm, err := webhook.NewClientManager(admissionv1beta1.SchemeGroupVersion, admissionv1beta1.AddToScheme)
	if err != nil {
		return nil, err
	}
	authInfoResolver, err := webhook.NewDefaultAuthenticationInfoResolver(kubeconfigFile)
	if err != nil {
		return nil, err
	}
	// Set defaults which may be overridden later.
	cm.SetAuthenticationInfoResolver(authInfoResolver)
	cm.SetServiceResolver(webhook.NewDefaultServiceResolver())

	return &Webhook{
		Handler:          handler,
		sourceFactory:    sourceFactory,
		clientManager:    &cm,
		convertor:        &convertor{},
		namespaceMatcher: &namespace.Matcher{},
		dispatcher:       dispatcherFactory(&cm),
	}, nil
}

// SetAuthenticationInfoResolverWrapper sets the
// AuthenticationInfoResolverWrapper.
// TODO find a better way wire this, but keep this pull small for now.
func (a *Webhook) SetAuthenticationInfoResolverWrapper(wrapper webhook.AuthenticationInfoResolverWrapper) {
	a.clientManager.SetAuthenticationInfoResolverWrapper(wrapper)
}

// SetServiceResolver sets a service resolver for the webhook admission plugin.
// Passing a nil resolver does not have an effect, instead a default one will be used.
func (a *Webhook) SetServiceResolver(sr webhook.ServiceResolver) {
	a.clientManager.SetServiceResolver(sr)
}

// SetScheme sets a serializer(NegotiatedSerializer) which is derived from the scheme
func (a *Webhook) SetScheme(scheme *runtime.Scheme) {
	if scheme != nil {
		a.convertor.Scheme = scheme
	}
}

// SetExternalKubeClientSet implements the WantsExternalKubeInformerFactory interface.
// It sets external ClientSet for admission plugins that need it
func (a *Webhook) SetExternalKubeClientSet(client clientset.Interface) {
	a.namespaceMatcher.Client = client
}

// SetExternalKubeInformerFactory implements the WantsExternalKubeInformerFactory interface.
func (a *Webhook) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	namespaceInformer := f.Core().V1().Namespaces()
	a.namespaceMatcher.NamespaceLister = namespaceInformer.Lister()
	a.hookSource = a.sourceFactory(f)
	a.SetReadyFunc(func() bool {
		return namespaceInformer.Informer().HasSynced() && a.hookSource.HasSynced()
	})
}

// ValidateInitialization implements the InitializationValidator interface.
func (a *Webhook) ValidateInitialization() error {
	if a.hookSource == nil {
		return fmt.Errorf("kubernetes client is not properly setup")
	}
	if err := a.namespaceMatcher.Validate(); err != nil {
		return fmt.Errorf("namespaceMatcher is not properly setup: %v", err)
	}
	if err := a.clientManager.Validate(); err != nil {
		return fmt.Errorf("clientManager is not properly setup: %v", err)
	}
	if err := a.convertor.Validate(); err != nil {
		return fmt.Errorf("convertor is not properly setup: %v", err)
	}
	return nil
}

// ShouldCallHook makes a decision on whether to call the webhook or not by the attribute.
func (a *Webhook) ShouldCallHook(h *v1beta1.Webhook, attr admission.Attributes) (bool, *apierrors.StatusError) {
	var matches bool
	for _, r := range h.Rules {
		m := rules.Matcher{Rule: r, Attr: attr}
		if m.Matches() {
			matches = true
			break
		}
	}
	if !matches {
		return false, nil
	}

	return a.namespaceMatcher.MatchNamespaceSelector(h, attr)
}

// Dispatch is called by the downstream Validate or Admit methods.
func (a *Webhook) Dispatch(attr admission.Attributes) error {
	if rules.IsWebhookConfigurationResource(attr) {
		return nil
	}
	if !a.WaitForReady() {
		return admission.NewForbidden(attr, fmt.Errorf("not yet ready to handle request"))
	}
	hooks := a.hookSource.Webhooks()
	// TODO: Figure out if adding one second timeout make sense here.
	ctx := context.TODO()

	var relevantHooks []*v1beta1.Webhook
	for i := range hooks {
		call, err := a.ShouldCallHook(&hooks[i], attr)
		if err != nil {
			return err
		}
		if call {
			relevantHooks = append(relevantHooks, &hooks[i])
		}
	}

	if len(relevantHooks) == 0 {
		// no matching hooks
		return nil
	}

	// convert the object to the external version before sending it to the webhook
	versionedAttr := VersionedAttributes{
		Attributes: attr,
	}
	if oldObj := attr.GetOldObject(); oldObj != nil {
		out, err := a.convertor.ConvertToGVK(oldObj, attr.GetKind())
		if err != nil {
			return apierrors.NewInternalError(err)
		}
		versionedAttr.VersionedOldObject = out
	}
	if obj := attr.GetObject(); obj != nil {
		out, err := a.convertor.ConvertToGVK(obj, attr.GetKind())
		if err != nil {
			return apierrors.NewInternalError(err)
		}
		versionedAttr.VersionedObject = out
	}
	return a.dispatcher.Dispatch(ctx, &versionedAttr, relevantHooks)
}
