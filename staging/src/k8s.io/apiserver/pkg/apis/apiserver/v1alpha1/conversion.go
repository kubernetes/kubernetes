/*
Copyright 2017 The Kubernetes Authors.

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

package v1alpha1

import (
	runtime "k8s.io/apimachinery/pkg/runtime"
)

var _ runtime.NestedObjectDecoder = &AdmissionConfiguration{}

// DecodeNestedObjects handles encoding RawExtensions on the AdmissionConfiguration, ensuring the
// objects are decoded with the provided decoder.
func (c *AdmissionConfiguration) DecodeNestedObjects(d runtime.Decoder) error {
	// decoding failures result in a runtime.Unknown object being created in Object and passed
	// to conversion
	for k, v := range c.Plugins {
		decodeNestedRawExtensionOrUnknown(d, &v.Configuration)
		c.Plugins[k] = v
	}
	return nil
}

var _ runtime.NestedObjectEncoder = &AdmissionConfiguration{}

// EncodeNestedObjects handles encoding RawExtensions on the AdmissionConfiguration, ensuring the
// objects are encoded with the provided encoder.
func (c *AdmissionConfiguration) EncodeNestedObjects(e runtime.Encoder) error {
	for k, v := range c.Plugins {
		if err := encodeNestedRawExtension(e, &v.Configuration); err != nil {
			return err
		}
		c.Plugins[k] = v
	}
	return nil
}

// decodeNestedRawExtensionOrUnknown decodes the raw extension into an object once.  If called
// On a RawExtension that has already been decoded (has an object), it will not run again.
func decodeNestedRawExtensionOrUnknown(d runtime.Decoder, ext *runtime.RawExtension) {
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

func encodeNestedRawExtension(e runtime.Encoder, ext *runtime.RawExtension) error {
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
