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

package conversion

import (
	"fmt"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsfeatures "k8s.io/apiextensions-apiserver/pkg/features"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/webhook"
)

// CRConverterFactory is the factory for all CR converters.
type CRConverterFactory struct {
	// webhookConverterFactory is the factory for webhook converters.
	// This field should not be used if CustomResourceWebhookConversion feature is disabled.
	webhookConverterFactory *webhookConverterFactory
}

// NewCRConverterFactory creates a new CRConverterFactory
func NewCRConverterFactory(serviceResolver webhook.ServiceResolver, authResolverWrapper webhook.AuthenticationInfoResolverWrapper) (*CRConverterFactory, error) {
	converterFactory := &CRConverterFactory{}
	if utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CustomResourceWebhookConversion) {
		webhookConverterFactory, err := newWebhookConverterFactory(serviceResolver, authResolverWrapper)
		if err != nil {
			return nil, err
		}
		converterFactory.webhookConverterFactory = webhookConverterFactory
	}
	return converterFactory, nil
}

// NewConverter returns a new CR converter based on the conversion settings in crd object.
func (m *CRConverterFactory) NewConverter(crd *apiextensions.CustomResourceDefinition) (safe, unsafe runtime.ObjectConvertor, err error) {
	validVersions := map[schema.GroupVersion]bool{}
	for _, version := range crd.Spec.Versions {
		validVersions[schema.GroupVersion{Group: crd.Spec.Group, Version: version.Name}] = true
	}

	switch crd.Spec.Conversion.Strategy {
	case apiextensions.NoneConverter:
		unsafe = &crConverter{
			clusterScoped: crd.Spec.Scope == apiextensions.ClusterScoped,
			delegate: &nopConverter{
				validVersions: validVersions,
			},
		}
		return &safeConverterWrapper{unsafe}, unsafe, nil
	case apiextensions.WebhookConverter:
		if !utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CustomResourceWebhookConversion) {
			return nil, nil, fmt.Errorf("webhook conversion is disabled on this cluster")
		}
		unsafe, err := m.webhookConverterFactory.NewWebhookConverter(validVersions, crd)
		if err != nil {
			return nil, nil, err
		}
		return &safeConverterWrapper{unsafe}, unsafe, nil
	}

	return nil, nil, fmt.Errorf("unknown conversion strategy %q for CRD %s", crd.Spec.Conversion.Strategy, crd.Name)
}

var _ runtime.ObjectConvertor = &crConverter{}

// crConverter extends the delegate with generic CR conversion behaviour. The delegate will implement the
// user defined conversion strategy given in the CustomResourceDefinition.
type crConverter struct {
	delegate      runtime.ObjectConvertor
	clusterScoped bool
}

func (c *crConverter) ConvertFieldLabel(gvk schema.GroupVersionKind, label, value string) (string, string, error) {
	// We currently only support metadata.namespace and metadata.name.
	switch {
	case label == "metadata.name":
		return label, value, nil
	case !c.clusterScoped && label == "metadata.namespace":
		return label, value, nil
	default:
		return "", "", fmt.Errorf("field label not supported: %s", label)
	}
}

func (c *crConverter) Convert(in, out, context interface{}) error {
	return c.delegate.Convert(in, out, context)
}

// ConvertToVersion converts in object to the given gvk in place and returns the same `in` object.
func (c *crConverter) ConvertToVersion(in runtime.Object, target runtime.GroupVersioner) (runtime.Object, error) {
	// Run the converter on the list items instead of list itself
	if list, ok := in.(*unstructured.UnstructuredList); ok {
		for i := range list.Items {
			obj, err := c.delegate.ConvertToVersion(&list.Items[i], target)
			if err != nil {
				return nil, err
			}

			u, ok := obj.(*unstructured.Unstructured)
			if !ok {
				return nil, fmt.Errorf("output type %T in not valid for unstructured conversion", obj)
			}
			list.Items[i] = *u
		}
		return list, nil
	}

	return c.delegate.ConvertToVersion(in, target)
}

// safeConverterWrapper is a wrapper over an unsafe object converter that makes copy of the input and then delegate to the unsafe converter.
type safeConverterWrapper struct {
	unsafe runtime.ObjectConvertor
}

var _ runtime.ObjectConvertor = &nopConverter{}

// ConvertFieldLabel delegate the call to the unsafe converter.
func (c *safeConverterWrapper) ConvertFieldLabel(gvk schema.GroupVersionKind, label, value string) (string, string, error) {
	return c.unsafe.ConvertFieldLabel(gvk, label, value)
}

// Convert makes a copy of in object and then delegate the call to the unsafe converter.
func (c *safeConverterWrapper) Convert(in, out, context interface{}) error {
	inObject, ok := in.(runtime.Object)
	if !ok {
		return fmt.Errorf("input type %T in not valid for object conversion", in)
	}
	return c.unsafe.Convert(inObject.DeepCopyObject(), out, context)
}

// ConvertToVersion makes a copy of in object and then delegate the call to the unsafe converter.
func (c *safeConverterWrapper) ConvertToVersion(in runtime.Object, target runtime.GroupVersioner) (runtime.Object, error) {
	return c.unsafe.ConvertToVersion(in.DeepCopyObject(), target)
}
