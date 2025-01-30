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
	"strings"

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsfeatures "k8s.io/apiextensions-apiserver/pkg/features"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/webhook"
	typedscheme "k8s.io/client-go/kubernetes/scheme"
)

// CRConverterFactory is the factory for all CR converters.
type CRConverterFactory struct {
	// webhookConverterFactory is the factory for webhook converters.
	// This field should not be used if CustomResourceWebhookConversion feature is disabled.
	webhookConverterFactory *webhookConverterFactory
}

// converterMetricFactorySingleton protects us from reregistration of metrics on repeated
// apiextensions-apiserver runs.
var converterMetricFactorySingleton = newConverterMetricFactory()

// NewCRConverterFactory creates a new CRConverterFactory
func NewCRConverterFactory(serviceResolver webhook.ServiceResolver, authResolverWrapper webhook.AuthenticationInfoResolverWrapper) (*CRConverterFactory, error) {
	converterFactory := &CRConverterFactory{}
	webhookConverterFactory, err := newWebhookConverterFactory(serviceResolver, authResolverWrapper)
	if err != nil {
		return nil, err
	}
	converterFactory.webhookConverterFactory = webhookConverterFactory
	return converterFactory, nil
}

// NewConverter returns a new CR converter based on the conversion settings in crd object.
func (m *CRConverterFactory) NewConverter(crd *apiextensionsv1.CustomResourceDefinition) (safe, unsafe runtime.ObjectConvertor, err error) {
	validVersions := map[schema.GroupVersion]bool{}
	for _, version := range crd.Spec.Versions {
		validVersions[schema.GroupVersion{Group: crd.Spec.Group, Version: version.Name}] = true
	}

	var converter crConverterInterface
	switch crd.Spec.Conversion.Strategy {
	case apiextensionsv1.NoneConverter:
		converter = &nopConverter{}
	case apiextensionsv1.WebhookConverter:
		converter, err = m.webhookConverterFactory.NewWebhookConverter(crd)
		if err != nil {
			return nil, nil, err
		}
		converter, err = converterMetricFactorySingleton.addMetrics(crd.Name, converter)
		if err != nil {
			return nil, nil, err
		}
	default:
		return nil, nil, fmt.Errorf("unknown conversion strategy %q for CRD %s", crd.Spec.Conversion.Strategy, crd.Name)
	}

	// Determine whether we should expect to be asked to "convert" autoscaling/v1 Scale types
	convertScale := false
	selectableFields := map[schema.GroupVersion]sets.Set[string]{}
	for _, version := range crd.Spec.Versions {
		gv := schema.GroupVersion{Group: crd.Spec.Group, Version: version.Name}
		if version.Subresources != nil && version.Subresources.Scale != nil {
			convertScale = true
		}
		if utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CustomResourceFieldSelectors) {
			fieldPaths := sets.New[string]()
			for _, sf := range version.SelectableFields {
				fieldPaths.Insert(strings.TrimPrefix(sf.JSONPath, "."))
			}
			selectableFields[gv] = fieldPaths
		}
	}

	unsafe = &crConverter{
		convertScale:     convertScale,
		validVersions:    validVersions,
		clusterScoped:    crd.Spec.Scope == apiextensionsv1.ClusterScoped,
		converter:        converter,
		selectableFields: selectableFields,
	}
	return &safeConverterWrapper{unsafe}, unsafe, nil
}

// crConverterInterface is the interface all cr converters must implement
type crConverterInterface interface {
	// Convert converts in object to the given gvk and returns the converted object.
	// Note that the function may mutate in object and return it. A safe wrapper will make sure
	// a safe converter will be returned.
	Convert(in runtime.Object, targetGVK schema.GroupVersion) (runtime.Object, error)
}

// crConverter extends the delegate converter with generic CR conversion behaviour. The delegate will implement the
// user defined conversion strategy given in the CustomResourceDefinition.
type crConverter struct {
	convertScale     bool
	converter        crConverterInterface
	validVersions    map[schema.GroupVersion]bool
	clusterScoped    bool
	selectableFields map[schema.GroupVersion]sets.Set[string]
}

func (c *crConverter) ConvertFieldLabel(gvk schema.GroupVersionKind, label, value string) (string, string, error) {
	switch {
	case label == "metadata.name":
		return label, value, nil
	case !c.clusterScoped && label == "metadata.namespace":
		return label, value, nil
	default:
		if utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CustomResourceFieldSelectors) {
			groupFields := c.selectableFields[gvk.GroupVersion()]
			if groupFields != nil && groupFields.Has(label) {
				return label, value, nil
			}
		}
		return "", "", fmt.Errorf("field label not supported: %s", label)
	}
}

func (c *crConverter) Convert(in, out, context interface{}) error {
	// Special-case typed scale conversion if this custom resource supports a scale endpoint
	if c.convertScale {
		_, isInScale := in.(*autoscalingv1.Scale)
		_, isOutScale := out.(*autoscalingv1.Scale)
		if isInScale || isOutScale {
			return typedscheme.Scheme.Convert(in, out, context)
		}
	}

	unstructIn, ok := in.(*unstructured.Unstructured)
	if !ok {
		return fmt.Errorf("input type %T in not valid for unstructured conversion to %T", in, out)
	}

	unstructOut, ok := out.(*unstructured.Unstructured)
	if !ok {
		return fmt.Errorf("output type %T in not valid for unstructured conversion from %T", out, in)
	}

	outGVK := unstructOut.GroupVersionKind()
	converted, err := c.ConvertToVersion(unstructIn, outGVK.GroupVersion())
	if err != nil {
		return err
	}
	unstructuredConverted, ok := converted.(runtime.Unstructured)
	if !ok {
		// this should not happened
		return fmt.Errorf("CR conversion failed")
	}
	unstructOut.SetUnstructuredContent(unstructuredConverted.UnstructuredContent())
	return nil
}

// ConvertToVersion converts in object to the given gvk in place and returns the same `in` object.
// The in object can be a single object or a UnstructuredList. CRD storage implementation creates an
// UnstructuredList with the request's GV, populates it from storage, then calls conversion to convert
// the individual items. This function assumes it never gets a v1.List.
func (c *crConverter) ConvertToVersion(in runtime.Object, target runtime.GroupVersioner) (runtime.Object, error) {
	fromGVK := in.GetObjectKind().GroupVersionKind()
	toGVK, ok := target.KindForGroupVersionKinds([]schema.GroupVersionKind{fromGVK})
	if !ok {
		// TODO: should this be a typed error?
		return nil, fmt.Errorf("%v is unstructured and is not suitable for converting to %q", fromGVK.String(), target)
	}
	// Special-case typed scale conversion if this custom resource supports a scale endpoint
	if c.convertScale {
		if _, isInScale := in.(*autoscalingv1.Scale); isInScale {
			return typedscheme.Scheme.ConvertToVersion(in, target)
		}
	}

	if !c.validVersions[toGVK.GroupVersion()] {
		return nil, fmt.Errorf("request to convert CR to an invalid group/version: %s", toGVK.GroupVersion().String())
	}
	// Note that even if the request is for a list, the GV of the request UnstructuredList is what
	// is expected to convert to. As mentioned in the function's document, it is not expected to
	// get a v1.List.
	if !c.validVersions[fromGVK.GroupVersion()] {
		return nil, fmt.Errorf("request to convert CR from an invalid group/version: %s", fromGVK.GroupVersion().String())
	}
	// Check list item's apiVersion
	if list, ok := in.(*unstructured.UnstructuredList); ok {
		for i := range list.Items {
			expectedGV := list.Items[i].GroupVersionKind().GroupVersion()
			if !c.validVersions[expectedGV] {
				return nil, fmt.Errorf("request to convert CR list failed, list index %d has invalid group/version: %s", i, expectedGV.String())
			}
		}
	}
	return c.converter.Convert(in, toGVK.GroupVersion())
}

// safeConverterWrapper is a wrapper over an unsafe object converter that makes copy of the input and then delegate to the unsafe converter.
type safeConverterWrapper struct {
	unsafe runtime.ObjectConvertor
}

var _ runtime.ObjectConvertor = &safeConverterWrapper{}

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
