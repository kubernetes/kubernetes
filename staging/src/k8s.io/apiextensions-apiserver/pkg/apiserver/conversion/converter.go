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
	apivalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	metav1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/webhook"
	typedscheme "k8s.io/client-go/kubernetes/scheme"
)

// Factory is able to create a new CRConverter for crd.
type Factory interface {
	// NewConverter returns a CRConverter capable of converting crd's versions.
	//
	// For proper conversion, the returned CRConverter must be used via NewDelegatingConverter.
	//
	// When implementing a CRConverter, you do not need to: test for valid API versions or no-op
	// conversions, handle field selector logic, or handle scale conversions; these are all handled
	// via NewDelegatingConverter.
	NewConverter(crd *apiextensionsv1.CustomResourceDefinition) (CRConverter, error)
}

// CRConverterFactory is the factory for all CR converters.
type CRConverterFactory struct {
	// webhookConverterFactory is the factory for webhook converters.
	// This field should not be used if CustomResourceWebhookConversion feature is disabled.
	webhookConverterFactory *webhookConverterFactory
}

// converterMetricFactorySingleton protects us from reregistration of metrics on repeated
// apiextensions-apiserver runs.
var converterMetricFactorySingleton = newConverterMetricFactory()

// NewCRConverterFactory creates a new CRConverterFactory that supports none and webhook conversion strategies.
func NewCRConverterFactory(serviceResolver webhook.ServiceResolver, authResolverWrapper webhook.AuthenticationInfoResolverWrapper) (*CRConverterFactory, error) {
	converterFactory := &CRConverterFactory{}
	webhookConverterFactory, err := newWebhookConverterFactory(serviceResolver, authResolverWrapper)
	if err != nil {
		return nil, err
	}
	converterFactory.webhookConverterFactory = webhookConverterFactory
	return converterFactory, nil
}

// NewConverter creates a new CRConverter based on the crd's conversion strategy. Supported strategies are none and
// webhook.
func (f *CRConverterFactory) NewConverter(crd *apiextensionsv1.CustomResourceDefinition) (CRConverter, error) {
	switch crd.Spec.Conversion.Strategy {
	case apiextensionsv1.NoneConverter:
		return NewNOPConverter(), nil
	case apiextensionsv1.WebhookConverter:
		converter, err := f.webhookConverterFactory.NewWebhookConverter(crd)
		if err != nil {
			return nil, err
		}
		return converterMetricFactorySingleton.addMetrics(crd.Name, converter)
	}

	return nil, fmt.Errorf("unknown conversion strategy %q for CRD %s", crd.Spec.Conversion.Strategy, crd.Name)
}

// NewDelegatingConverter returns new safe and unsafe converters based on the conversion settings in
// crd. These converters contain logic common to all converters, and they delegate the actual
// specific version-to-version conversion logic to the delegate.
func NewDelegatingConverter(crd *apiextensionsv1.CustomResourceDefinition, delegate CRConverter) (safe, unsafe runtime.ObjectConvertor, err error) {
	validVersions := map[schema.GroupVersion]bool{}
	for _, version := range crd.Spec.Versions {
		validVersions[schema.GroupVersion{Group: crd.Spec.Group, Version: version.Name}] = true
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

	unsafe = &delegatingCRConverter{
		convertScale:     convertScale,
		validVersions:    validVersions,
		clusterScoped:    crd.Spec.Scope == apiextensionsv1.ClusterScoped,
		converter:        delegate,
		selectableFields: selectableFields,
		// If this is a wildcard partial metadata CRD variant, we don't require that the CRD serves the appropriate
		// version, because the schema does not matter.
		requireValidVersion: !strings.HasSuffix(string(crd.UID), ".wildcard.partial-metadata"),
	}

	return &safeConverterWrapper{unsafe}, unsafe, nil
}

// CRConverter is the interface all CR converters must implement
type CRConverter interface {
	// Convert converts in object to the given gvk and returns the converted object.
	// Note that the function may mutate in object and return it. A safe wrapper will make sure
	// a safe converter will be returned.
	Convert(in *unstructured.UnstructuredList, targetGV schema.GroupVersion) (*unstructured.UnstructuredList, error)
}

// CRConverterFunc wraps a CR conversion func into a CRConverter.
type CRConverterFunc func(in *unstructured.UnstructuredList, targetGVK schema.GroupVersion) (*unstructured.UnstructuredList, error)

func (fn CRConverterFunc) Convert(in *unstructured.UnstructuredList, targetGV schema.GroupVersion) (*unstructured.UnstructuredList, error) {
	return fn(in, targetGV)
}

// delegatingCRConverter extends the delegate converter with generic CR conversion behaviour. The delegate will implement the
// user defined conversion strategy given in the CustomResourceDefinition.
type delegatingCRConverter struct {
	convertScale     bool
	converter        CRConverter
	validVersions    map[schema.GroupVersion]bool
	clusterScoped    bool
	selectableFields map[schema.GroupVersion]sets.Set[string]

	// If true, require that the CRD serves the appropriate version
	requireValidVersion bool
}

func (c *delegatingCRConverter) ConvertFieldLabel(gvk schema.GroupVersionKind, label, value string) (string, string, error) {
	// We currently only support metadata.namespace and metadata.name.
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

func (c *delegatingCRConverter) Convert(in, out, context interface{}) error {
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
func (c *delegatingCRConverter) ConvertToVersion(in runtime.Object, target runtime.GroupVersioner) (runtime.Object, error) {
	// Special-case typed scale conversion if this custom resource supports a scale endpoint
	if c.convertScale {
		if _, isInScale := in.(*autoscalingv1.Scale); isInScale {
			return typedscheme.Scheme.ConvertToVersion(in, target)
		}
	}

	fromGVK := in.GetObjectKind().GroupVersionKind()
	toGVK, ok := target.KindForGroupVersionKinds([]schema.GroupVersionKind{fromGVK})
	if !ok {
		// TODO: should this be a typed error?
		return nil, fmt.Errorf("%v is unstructured and is not suitable for converting to %q", fromGVK.String(), target)
	}

	isList := false
	var list *unstructured.UnstructuredList
	switch t := in.(type) {
	case *unstructured.Unstructured:
		list = &unstructured.UnstructuredList{Items: []unstructured.Unstructured{*t}}
	case *unstructured.UnstructuredList:
		list = t
		isList = true
	default:
		return nil, fmt.Errorf("unexpected type %T", in)
	}

	desiredAPIVersion := toGVK.GroupVersion().String()
	if c.requireValidVersion && !c.validVersions[toGVK.GroupVersion()] {
		return nil, fmt.Errorf("request to convert CR to an invalid group/version: %s", desiredAPIVersion)
	}
	// Note that even if the request is for a list, the GV of the request UnstructuredList is what
	// is expected to convert to. As mentioned in the function's document, it is not expected to
	// get a v1.List.
	if c.requireValidVersion && !c.validVersions[fromGVK.GroupVersion()] {
		return nil, fmt.Errorf("request to convert CR from an invalid group/version: %s", fromGVK.GroupVersion().String())
	}

	objectsToConvert, err := getObjectsToConvert(list, desiredAPIVersion, c.validVersions, c.requireValidVersion)
	if err != nil {
		return nil, err
	}

	objCount := len(objectsToConvert)
	if objCount == 0 {
		// no objects needed conversion
		if !isList {
			// for a single item, return as-is
			return in, nil
		}
		// for a list, set the version of the top-level list object (all individual objects are already in the correct version)
		list.SetAPIVersion(desiredAPIVersion)
		return list, nil
	}

	// A smoke test in API machinery calls the converter on empty objects during startup. The test is initiated here:
	// https://github.com/kubernetes/kubernetes/blob/dbb448bbdcb9e440eee57024ffa5f1698956a054/staging/src/k8s.io/apiserver/pkg/storage/cacher/cacher.go#L201
	if isEmptyUnstructuredObject(in) {
		converted, err := NewNOPConverter().Convert(list, toGVK.GroupVersion())
		if err != nil {
			return nil, err
		}
		if !isList {
			return &converted.Items[0], nil
		}
		return converted, nil
	}

	// Deepcopy ObjectMeta because the converter might mutate the objects, and we
	// need the original ObjectMeta furthermore for validation and restoration.
	for i := range objectsToConvert {
		original := objectsToConvert[i].Object
		objectsToConvert[i].Object = make(map[string]interface{}, len(original))
		for k, v := range original {
			if k == "metadata" {
				v = runtime.DeepCopyJSONValue(v)
			}
			objectsToConvert[i].Object[k] = v
		}
	}

	// Do the (potentially mutating) conversion.
	convertedObjects, err := c.converter.Convert(&unstructured.UnstructuredList{
		Object: list.Object,
		Items:  objectsToConvert,
	}, toGVK.GroupVersion())
	if err != nil {
		return nil, fmt.Errorf("conversion for %v failed: %w", in.GetObjectKind().GroupVersionKind(), err)
	}
	if len(convertedObjects.Items) != len(objectsToConvert) {
		return nil, fmt.Errorf("conversion for %v returned %d objects, expected %d", in.GetObjectKind().GroupVersionKind(), len(convertedObjects.Items), len(objectsToConvert))
	}

	// Fill back in the converted objects from the response at the right spots.
	// The response list might be sparse because objects had the right version already.
	convertedList := list
	convertedList.SetAPIVersion(desiredAPIVersion)
	convertedIndex := 0
	for i := range convertedList.Items {
		original := &convertedList.Items[i]
		if original.GetAPIVersion() == desiredAPIVersion {
			// This item has not been sent for conversion, and therefore does not show up in the response.
			// convertedList has the right item already.
			continue
		}
		converted := &convertedObjects.Items[convertedIndex]
		convertedIndex++
		if expected, got := toGVK.GroupVersion(), converted.GetObjectKind().GroupVersionKind().GroupVersion(); expected != got {
			return nil, fmt.Errorf("conversion for %v returned invalid converted object at index %v: invalid groupVersion (expected %v, received %v)", in.GetObjectKind().GroupVersionKind(), convertedIndex, expected, got)
		}
		if expected, got := original.GetObjectKind().GroupVersionKind().Kind, converted.GetObjectKind().GroupVersionKind().Kind; expected != got {
			return nil, fmt.Errorf("conversion for %v returned invalid converted object at index %v: invalid kind (expected %v, received %v)", in.GetObjectKind().GroupVersionKind(), convertedIndex, expected, got)
		}
		if err := validateConvertedObject(original, converted); err != nil {
			return nil, fmt.Errorf("conversion for %v returned invalid converted object at index %v: %v", in.GetObjectKind().GroupVersionKind(), convertedIndex, err)
		}
		if err := restoreObjectMeta(original, converted); err != nil {
			return nil, fmt.Errorf("conversion for %v returned invalid metadata in object at index %v: %v", in.GetObjectKind().GroupVersionKind(), convertedIndex, err)
		}
		convertedList.Items[i] = *converted
	}

	if isList {
		return convertedList, nil
	}

	return &convertedList.Items[0], nil
}

func getObjectsToConvert(
	list *unstructured.UnstructuredList,
	desiredAPIVersion string,
	validVersions map[schema.GroupVersion]bool,
	requireValidVersion bool,
) ([]unstructured.Unstructured, error) {
	var objectsToConvert []unstructured.Unstructured
	for i := range list.Items {
		expectedGV := list.Items[i].GroupVersionKind().GroupVersion()
		if requireValidVersion && !validVersions[expectedGV] {
			return nil, fmt.Errorf("request to convert CR list failed, list index %d has invalid group/version: %s", i, expectedGV.String())
		}

		// Only sent item for conversion, if the apiVersion is different
		if list.Items[i].GetAPIVersion() != desiredAPIVersion {
			objectsToConvert = append(objectsToConvert, list.Items[i])
		}
	}
	return objectsToConvert, nil
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

// isEmptyUnstructuredObject returns true if in is an empty unstructured object, i.e. an unstructured object that does
// not have any field except apiVersion and kind.
func isEmptyUnstructuredObject(in runtime.Object) bool {
	u, ok := in.(*unstructured.Unstructured)
	if !ok {
		return false
	}
	if len(u.Object) != 2 {
		return false
	}
	if _, ok := u.Object["kind"]; !ok {
		return false
	}
	if _, ok := u.Object["apiVersion"]; !ok {
		return false
	}
	return true
}

// validateConvertedObject checks that ObjectMeta fields match, with the exception of
// labels and annotations.
func validateConvertedObject(in, out *unstructured.Unstructured) error {
	if e, a := in.GetKind(), out.GetKind(); e != a {
		return fmt.Errorf("must have the same kind: %v != %v", e, a)
	}
	if e, a := in.GetName(), out.GetName(); e != a {
		return fmt.Errorf("must have the same name: %v != %v", e, a)
	}
	if e, a := in.GetNamespace(), out.GetNamespace(); e != a {
		return fmt.Errorf("must have the same namespace: %v != %v", e, a)
	}
	if e, a := in.GetUID(), out.GetUID(); e != a {
		return fmt.Errorf("must have the same UID: %v != %v", e, a)
	}
	return nil
}

// restoreObjectMeta copies metadata from original into converted, while preserving labels and annotations from converted.
func restoreObjectMeta(original, converted *unstructured.Unstructured) error {
	cm, found := converted.Object["metadata"]
	om, previouslyFound := original.Object["metadata"]
	switch {
	case !found && !previouslyFound:
		return nil
	case previouslyFound && !found:
		return fmt.Errorf("missing metadata in converted object")
	case !previouslyFound && found:
		om = map[string]interface{}{}
	}

	convertedMeta, ok := cm.(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid metadata of type %T in converted object", cm)
	}
	originalMeta, ok := om.(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid metadata of type %T in input object", om)
	}

	result := converted
	if previouslyFound {
		result.Object["metadata"] = originalMeta
	} else {
		result.Object["metadata"] = map[string]interface{}{}
	}
	resultMeta := result.Object["metadata"].(map[string]interface{})

	for _, fld := range []string{"labels", "annotations"} {
		obj, found := convertedMeta[fld]
		if !found || obj == nil {
			delete(resultMeta, fld)
			continue
		}

		convertedField, ok := obj.(map[string]interface{})
		if !ok {
			return fmt.Errorf("invalid metadata.%s of type %T in converted object", fld, obj)
		}
		originalField, ok := originalMeta[fld].(map[string]interface{})
		if !ok && originalField[fld] != nil {
			return fmt.Errorf("invalid metadata.%s of type %T in original object", fld, originalMeta[fld])
		}

		somethingChanged := len(originalField) != len(convertedField)
		for k, v := range convertedField {
			if _, ok := v.(string); !ok {
				return fmt.Errorf("metadata.%s[%s] must be a string, but is %T in converted object", fld, k, v)
			}
			if originalField[k] != interface{}(v) {
				somethingChanged = true
			}
		}

		if somethingChanged {
			stringMap := make(map[string]string, len(convertedField))
			for k, v := range convertedField {
				stringMap[k] = v.(string)
			}
			var errs field.ErrorList
			if fld == "labels" {
				errs = metav1validation.ValidateLabels(stringMap, field.NewPath("metadata", "labels"))
			} else {
				errs = apivalidation.ValidateAnnotations(stringMap, field.NewPath("metadata", "annotation"))
			}
			if len(errs) > 0 {
				return errs.ToAggregate()
			}
		}

		resultMeta[fld] = convertedField
	}

	return nil
}
