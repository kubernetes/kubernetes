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
	converter := &nopConverter{}

	unsafe = &crdConverter{
		clusterScoped: crd.Spec.Scope == apiextensions.ClusterScoped,
		validVersions: validVersions,
		converter:     converter,
		safeConverter: false,
	}
	safe = &crdConverter{
		clusterScoped: crd.Spec.Scope == apiextensions.ClusterScoped,
		validVersions: validVersions,
		converter:     converter,
		safeConverter: true,
	}
	return safe, unsafe
}

// crdConverterInterface is the interface all CRD converters must implement.
type crdConverterInterface interface {
	// ConvertCustomResource converts obj to given version in place. The listHandled return value is optimization
	// for converters that want to convert the whole UnstructuredList including its items (e.g. webhooks).
	ConvertCustomResource(obj runtime.Unstructured, target runtime.GroupVersioner) (listHandled bool, err error)
}

// crdConverter is an implementation of common converter functionalities for CRDs. It calls into
// the converter interface for actual conversion.
type crdConverter struct {
	clusterScoped bool
	// validVersions is list of valid versions for this CRD. both in and out objects will be validated toward this list first.
	validVersions map[schema.GroupVersion]bool
	// converter is the actual converter that converts an object in place to the given version.
	converter crdConverterInterface
	// safeConverter is the flag to set this converter to safe or unsafe one. If true, no input to the converter would be mutated.
	safeConverter bool
}

var _ runtime.ObjectConvertor = &crdConverter{}

func (c *crdConverter) ConvertFieldLabel(version, kind, label, value string) (string, string, error) {
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
	unstructIn, ok := in.(runtime.Unstructured)
	if !ok {
		return fmt.Errorf("input type %T in not valid for unstructured conversion", in)
	}

	unstructOut, ok := out.(runtime.Unstructured)
	if !ok {
		return fmt.Errorf("output type %T in not valid for unstructured conversion", out)
	}

	outGVK := unstructOut.GetObjectKind().GroupVersionKind()
	if !c.validVersions[outGVK.GroupVersion()] {
		return fmt.Errorf("request to convert CRD from an invalid group/version: %s", outGVK.String())
	}
	inGVK := unstructIn.GetObjectKind().GroupVersionKind()
	if !c.validVersions[inGVK.GroupVersion()] {
		return fmt.Errorf("request to convert CRD to an invalid group/version: %s", inGVK.String())
	}

	unstructOut.SetUnstructuredContent(unstructIn.UnstructuredContent())
	_, err := c.ConvertToVersion(unstructOut, outGVK.GroupVersion())
	return err
}

// ConvertToVersion converts in object to the given gvk in place and returns the same `in` object.
func (c *crdConverter) ConvertToVersion(in runtime.Object, target runtime.GroupVersioner) (runtime.Object, error) {
	var err error
	if c.safeConverter {
		in = in.DeepCopyObject()
	}
	listHandled, err := c.validateAndConvert(in, target)
	if err != nil {
		return nil, err
	}
	if listHandled {
		return in, nil
	}

	// Run the converter on the list items instead of list itself
	if list, ok := in.(*unstructured.UnstructuredList); ok {
		err = list.EachListItem(func(item runtime.Object) error {
			_, err := c.validateAndConvert(item, target)
			return err
		})
	}
	return in, nil
}

func (c *crdConverter) validateAndConvert(obj runtime.Object, target runtime.GroupVersioner) (bool, error) {
	sourceKind := obj.GetObjectKind().GroupVersionKind()
	if !c.validVersions[sourceKind.GroupVersion()] {
		return false, fmt.Errorf("request to convert CRD to an invalid group/version: %s", sourceKind.String())
	}
	targetKind, ok := target.KindForGroupVersionKinds([]schema.GroupVersionKind{sourceKind})
	if !ok {
		return false, fmt.Errorf("%v is not suitable for converting to %q", sourceKind, target)
	}
	if targetKind == sourceKind {
		// No need for conversion. Source and Target are the same GVK.
		return false, nil
	}
	unstructObj, ok := obj.(runtime.Unstructured)
	if !ok {
		return false, fmt.Errorf("input type %T in not valid for unstructured conversion", obj)
	}
	return c.converter.ConvertCustomResource(unstructObj, target)
}
