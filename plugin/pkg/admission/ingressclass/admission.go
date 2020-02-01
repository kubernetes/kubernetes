/*
Copyright 2020 The Kubernetes Authors.

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

package ingressclass

import (
	"context"
	"fmt"
	"io"

	"k8s.io/klog"

	networkingv1 "k8s.io/api/networking/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	networkingv1listers "k8s.io/client-go/listers/networking/v1"
)

const (
	// PluginName is the name of this admission controller plugin
	PluginName = "IngressClass"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		plugin := newPlugin()
		return plugin, nil
	})
}

// claimDefaulterPlugin holds state for and implements the admission plugin.
type claimDefaulterPlugin struct {
	*admission.Handler
	lister networkingv1listers.IngressClassLister
}

var _ admission.Interface = &claimDefaulterPlugin{}
var _ admission.MutationInterface = &claimDefaulterPlugin{}
var _ = genericadmissioninitializer.WantsExternalKubeInformerFactory(&claimDefaulterPlugin{})

// newPlugin creates a new admission plugin.
func newPlugin() *claimDefaulterPlugin {
	return &claimDefaulterPlugin{
		Handler: admission.NewHandler(admission.Create),
	}
}

// SetExternalKubeInformerFactory sets a lister and readyFunc for this
// claimDefaulterPlugin using the provided SharedInformerFactory.
func (a *claimDefaulterPlugin) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	informer := f.Networking().V1().IngressClasses()
	a.lister = informer.Lister()
	a.SetReadyFunc(informer.Informer().HasSynced)
}

// ValidateInitialization ensures lister is set.
func (a *claimDefaulterPlugin) ValidateInitialization() error {
	if a.lister == nil {
		return fmt.Errorf("missing lister")
	}
	return nil
}

// Admit sets the default value of a Ingress's class if the user did not specify
// a class.
func (a *claimDefaulterPlugin) Admit(ctx context.Context, attr admission.Attributes, o admission.ObjectInterfaces) error {
	if attr.GetResource().GroupResource() != networkingv1.Resource("ingresses") {
		fmt.Println(attr.GetResource().GroupResource())
		fmt.Println(networkingv1.Resource("ingresses"))
		return nil
	}

	if len(attr.GetSubresource()) != 0 {
		return nil
	}

	ingress, ok := attr.GetObject().(*networkingv1.Ingress)
	// if we can't convert then we don't handle this object so just return
	if !ok {
		return nil
	}

	// Ingress Class field has been set, no need to set a default value.
	if ingress.Spec.Class != nil {
		return nil
	}

	// Ingress Class annotation has been set, no need to set a default value.
	if _, ok := ingress.Annotations[networkingv1.AnnotationIngressClass]; ok {
		return nil
	}

	klog.V(4).Infof("No class specified on Ingress %s", ingress.Name)

	defaultClass, err := getDefaultClass(a.lister)
	if err != nil {
		return admission.NewForbidden(attr, err)
	}

	// No default class specified, no need to set a default value.
	if defaultClass == nil {
		return nil
	}

	klog.V(4).Infof("Defaulting class for Ingress %s to %s", ingress.Name, defaultClass.Name)
	ingress.Spec.Class = &defaultClass.Name
	return nil
}

// getDefaultClass returns the default IngressClass from the store, or nil.
func getDefaultClass(lister networkingv1listers.IngressClassLister) (*networkingv1.IngressClass, error) {
	list, err := lister.List(labels.Everything())
	if err != nil {
		return nil, err
	}

	defaultClasses := []*networkingv1.IngressClass{}
	for _, class := range list {
		if class.Annotations[networkingv1.AnnotationIsDefaultIngressClass] == "true" {
			defaultClasses = append(defaultClasses, class)
		}
	}

	if len(defaultClasses) == 0 {
		return nil, nil
	}

	if len(defaultClasses) > 1 {
		klog.V(3).Infof("More than %d IngressClasses marked as default (only 1 allowed)", len(defaultClasses))
		return nil, errors.NewInternalError(fmt.Errorf("%d default IngressClasses were found, only 1 allowed", len(defaultClasses)))
	}

	return defaultClasses[0], nil
}
