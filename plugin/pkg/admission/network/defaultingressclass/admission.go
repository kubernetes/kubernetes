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

package defaultingressclass

import (
	"context"
	"fmt"
	"io"
	"sort"

	networkingv1 "k8s.io/api/networking/v1"
	networkingv1beta1 "k8s.io/api/networking/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	networkingv1listers "k8s.io/client-go/listers/networking/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/apis/networking"
)

const (
	// PluginName is the name of this admission controller plugin
	PluginName = "DefaultIngressClass"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		plugin := newPlugin()
		return plugin, nil
	})
}

// classDefaulterPlugin holds state for and implements the admission plugin.
type classDefaulterPlugin struct {
	*admission.Handler
	lister networkingv1listers.IngressClassLister
}

var _ admission.Interface = &classDefaulterPlugin{}
var _ admission.MutationInterface = &classDefaulterPlugin{}
var _ = genericadmissioninitializer.WantsExternalKubeInformerFactory(&classDefaulterPlugin{})

// newPlugin creates a new admission plugin.
func newPlugin() *classDefaulterPlugin {
	return &classDefaulterPlugin{
		Handler: admission.NewHandler(admission.Create),
	}
}

// SetExternalKubeInformerFactory sets a lister and readyFunc for this
// classDefaulterPlugin using the provided SharedInformerFactory.
func (a *classDefaulterPlugin) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	informer := f.Networking().V1().IngressClasses()
	a.lister = informer.Lister()
	a.SetReadyFunc(informer.Informer().HasSynced)
}

// ValidateInitialization ensures lister is set.
func (a *classDefaulterPlugin) ValidateInitialization() error {
	if a.lister == nil {
		return fmt.Errorf("missing lister")
	}
	return nil
}

// Admit sets the default value of a Ingress's class if the user did not specify
// a class.
func (a *classDefaulterPlugin) Admit(ctx context.Context, attr admission.Attributes, o admission.ObjectInterfaces) error {
	if attr.GetResource().GroupResource() != networkingv1.Resource("ingresses") {
		return nil
	}

	if len(attr.GetSubresource()) != 0 {
		return nil
	}

	ingress, ok := attr.GetObject().(*networking.Ingress)
	// if we can't convert then we don't handle this object so just return
	if !ok {
		klog.V(3).Infof("Expected Ingress resource, got: %v", attr.GetKind())
		return errors.NewInternalError(fmt.Errorf("Expected Ingress resource, got: %v", attr.GetKind()))
	}

	// IngressClassName field has been set, no need to set a default value.
	if ingress.Spec.IngressClassName != nil {
		return nil
	}

	// Ingress class annotation has been set, no need to set a default value.
	if _, ok := ingress.Annotations[networkingv1beta1.AnnotationIngressClass]; ok {
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
	ingress.Spec.IngressClassName = &defaultClass.Name
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
	sort.Slice(defaultClasses, func(i, j int) bool {
		if defaultClasses[i].CreationTimestamp.UnixNano() == defaultClasses[j].CreationTimestamp.UnixNano() {
			return defaultClasses[i].Name < defaultClasses[j].Name
		}
		return defaultClasses[i].CreationTimestamp.UnixNano() > defaultClasses[j].CreationTimestamp.UnixNano()
	})
	if len(defaultClasses) > 1 {
		klog.V(4).Infof("%d default IngressClasses were found, choosing the newest: %s", len(defaultClasses), defaultClasses[0].Name)
	}

	return defaultClasses[0], nil
}
