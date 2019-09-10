/*
Copyright 2019 The Kubernetes Authors.

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

package resourcerestriction

import (
	"context"
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/klog"
)

// PluginName indicates name of admission plugin.
const PluginName = "ResourceRestriction"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewResourceRestriction(), nil
	})
}

// ResourceRestriction is an implementation of admission.Interface.
// It looks at creation or update actions on cluster scoped resources and ensures
// labels are appropriately mapped based on the authorizer.
type ResourceRestriction struct {
	*admission.Handler
	client          kubernetes.Interface
	namespaceLister corev1listers.NamespaceLister
}

var _ admission.MutationInterface = &ResourceRestriction{}
var _ = genericadmissioninitializer.WantsExternalKubeInformerFactory(&ResourceRestriction{})
var _ = genericadmissioninitializer.WantsExternalKubeClientSet(&ResourceRestriction{})

// Admit makes an admission decision based on the request attributes
func (p *ResourceRestriction) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	// Don't modify if the request is for a dry-run.
	if a.IsDryRun() {
		return nil
	}

	// load user and find restrictions
	user, ok := request.UserFrom(ctx)
	if !ok {
		return nil
	}
	if len(user.GetExtra()["restrict.auth.k8s.io/default"]) == 0 && len(user.GetExtra()["restrict.auth.k8s.io/filter"]) == 0 {
		return nil
	}

	var defaultRestriction map[string]string
	switch extraPartDefaults := user.GetExtra()["restrict.auth.k8s.io/default"]; {
	case len(extraPartDefaults) > 1:
		return fmt.Errorf("ResourceRestriction requires zero or one default labels")
	case len(extraPartDefaults) == 1:
		reqs, err := labels.ParseToRequirements(extraPartDefaults[0])
		if err != nil {
			return fmt.Errorf("ResourceRestriction is unable to handle filters: %v", err)
		}
		for _, req := range reqs {
			if req.Operator() != selection.Equals {
				continue
			}
			if req.Values().Len() != 1 {
				continue
			}
			if defaultRestriction == nil {
				defaultRestriction = make(map[string]string)
			}
			defaultRestriction[req.Key()] = req.Values().List()[0]
		}
	}

	var filterRestriction labels.Selector = labels.Everything()
	switch extraPartRestrictions := user.GetExtra()["restrict.auth.k8s.io/filter"]; {
	case len(extraPartRestrictions) > 1:
		return fmt.Errorf("ResourceRestriction requires zero or one filters")
	case len(extraPartRestrictions) == 1:
		selector, err := labels.Parse(extraPartRestrictions[0])
		if err != nil {
			return fmt.Errorf("ResourceRestriction is unable to handle filters: %v", err)
		}
		filterRestriction = selector
	}

	if len(a.GetNamespace()) > 0 {
		// we need to wait for our caches to warm
		if !p.WaitForReady() {
			return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
		}

		ns, err := p.namespaceLister.Get(a.GetNamespace())
		if err == nil {
			return nil
		}
		if !errors.IsNotFound(err) {
			return admission.NewForbidden(a, err)
		}
		if !filterRestriction.Matches(labels.Set(ns.Labels)) {
			return admission.NewForbidden(a, fmt.Errorf("operations on this resource are forbidden"))
		}

		// TODO: limit setting filter labels here?

		return nil
	}

	if len(a.GetSubresource()) != 0 {
		// TODO: probably not safe, needs more thought, or possibly to be implemented in storage
		return nil
	}

	meta, ok := a.GetObject().(metav1.Object)
	if !ok {
		// ignore non-object resources
		klog.Infof("DEBUG: unexpected admission object for create: %T", a.GetObject())
		return nil
	}
	objLabels := labels.Set(meta.GetLabels())

	switch a.GetOperation() {
	case admission.Create:
		for k, v := range defaultRestriction {
			if _, ok := objLabels[k]; ok {
				continue
			}
			if objLabels == nil {
				objLabels = make(labels.Set)
				meta.SetLabels(objLabels)
			}
			objLabels[k] = v
		}
	}

	if !filterRestriction.Matches(objLabels) {
		return admission.NewForbidden(a, fmt.Errorf("operations on this resource are forbidden"))
	}

	return nil
}

// NewResourceRestriction creates a new namespace provision admission control handler
func NewResourceRestriction() *ResourceRestriction {
	return &ResourceRestriction{
		Handler: admission.NewHandler(admission.Create, admission.Update, admission.Delete, admission.Connect),
	}
}

// SetExternalKubeClientSet implements the WantsExternalKubeClientSet interface.
func (p *ResourceRestriction) SetExternalKubeClientSet(client kubernetes.Interface) {
	p.client = client
}

// SetExternalKubeInformerFactory implements the WantsExternalKubeInformerFactory interface.
func (p *ResourceRestriction) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	namespaceInformer := f.Core().V1().Namespaces()
	p.namespaceLister = namespaceInformer.Lister()
	p.SetReadyFunc(namespaceInformer.Informer().HasSynced)
}

// ValidateInitialization implements the InitializationValidator interface.
func (p *ResourceRestriction) ValidateInitialization() error {
	if p.namespaceLister == nil {
		return fmt.Errorf("missing namespaceLister")
	}
	if p.client == nil {
		return fmt.Errorf("missing client")
	}
	return nil
}
