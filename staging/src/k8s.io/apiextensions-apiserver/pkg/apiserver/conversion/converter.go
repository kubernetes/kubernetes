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
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// NewCRDConverter returns a new CRD converter based on the conversion settings in crd object.
func NewCRDConverter(crd *apiextensions.CustomResourceDefinition) (safe, unsafe runtime.ObjectConvertor) {
	validVersions := map[schema.GroupVersion]bool{}
	for _, version := range crd.Spec.Versions {
		validVersions[schema.GroupVersion{Group: crd.Spec.Group, Version: version.Name}] = true
	}

	// The only converter right now is nopConverter. More converters will be returned based on the
	// CRD object when they introduced.
	unsafe = &crdConverter{
		clusterScoped: crd.Spec.Scope == apiextensions.ClusterScoped,
		delegate: &nopConverter{
			validVersions: validVersions,
		},
	}
	return &safeConverterWrapper{unsafe}, unsafe
}

var _ runtime.ObjectConvertor = &crdConverter{}

// crdConverter extends the delegate with generic CRD conversion behaviour. The delegate will implement the
// user defined conversion strategy given in the CustomResourceDefinition.
type crdConverter struct {
	delegate      runtime.ObjectConvertor
	clusterScoped bool
}

func (c *crdConverter) ConvertFieldLabel(gvk schema.GroupVersionKind, label, value string) (string, string, error) {
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

func (c *crdConverter) Convert(in, out, context interface{}) error {
	return c.delegate.Convert(in, out, context)
}

// ConvertToVersion converts in object to the given gvk in place and returns the same `in` object.
func (c *crdConverter) ConvertToVersion(in runtime.Object, target runtime.GroupVersioner) (runtime.Object, error) {
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
