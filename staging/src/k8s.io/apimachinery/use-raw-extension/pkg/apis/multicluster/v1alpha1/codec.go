package v1alpha1

import (
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
)

var _ runtime.NestedObjectDecoder = &Work{}
var _ runtime.NestedObjectEncoder = &Work{}

// DecodeNestedObjects decodes the object as a runtime.Unknown with JSON content.
func (c *Work) DecodeNestedObjects(d runtime.Decoder) error {
	for i := range c.Spec.Workload.Manifests {
		if c.Spec.Workload.Manifests[i].Object != nil {
			continue
		}

		// you can pass different schemes here so that you can have properly typed external versions of core resources for instance.
		// then you can do multi-pass checking for decoding options
		DecodeNestedRawExtensionOrUnknown(unstructured.UnstructuredJSONScheme, &c.Spec.Workload.Manifests[i])
	}
	return nil
}
func (c *Work) EncodeNestedObjects(e runtime.Encoder) error {
	for i := range c.Spec.Workload.Manifests {
		if err := EncodeNestedRawExtension(unstructured.UnstructuredJSONScheme, &c.Spec.Workload.Manifests[i]); err != nil {
			return err
		}
	}
	return nil
}

// DecodeNestedRawExtensionOrUnknown
func DecodeNestedRawExtensionOrUnknown(d runtime.Decoder, ext *runtime.RawExtension) {
	if ext.Raw == nil || ext.Object != nil {
		return
	}
	obj, gvk, err := d.Decode(ext.Raw, nil, nil)
	if err != nil {
		unk := &runtime.Unknown{Raw: ext.Raw}
		if runtime.IsNotRegisteredError(err) {
			if _, gvk, err := d.Decode(ext.Raw, nil, unk); err == nil {
				unk.APIVersion = gvk.GroupVersion().String()
				unk.Kind = gvk.Kind
				ext.Object = unk
				return
			}
		}
		// TODO: record mime-type with the object
		if gvk != nil {
			unk.APIVersion = gvk.GroupVersion().String()
			unk.Kind = gvk.Kind
		}
		obj = unk
	}
	ext.Object = obj
}

// EncodeNestedRawExtension will encode the object in the RawExtension (if not nil) or
// return an error.
func EncodeNestedRawExtension(e runtime.Encoder, ext *runtime.RawExtension) error {
	if ext.Raw != nil || ext.Object == nil {
		return nil
	}
	data, err := runtime.Encode(e, ext.Object)
	if err != nil {
		return err
	}
	ext.Raw = data
	return nil
}
