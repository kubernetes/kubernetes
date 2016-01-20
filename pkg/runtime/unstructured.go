/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package runtime

import (
	"encoding/json"
	"fmt"
	"net/url"
	"reflect"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/conversion"
)

// UnstructuredJSONScheme is capable of converting JSON data into the Unstructured
// type, which can be used for generic access to objects without a predefined scheme.
var UnstructuredJSONScheme ObjectDecoder = unstructuredJSONScheme{}

type unstructuredJSONScheme struct{}

var _ Decoder = unstructuredJSONScheme{}
var _ ObjectDecoder = unstructuredJSONScheme{}

// Recognizes returns true for any version or kind that is specified (internal
// versions are specifically excluded).
func (unstructuredJSONScheme) Recognizes(gvk unversioned.GroupVersionKind) bool {
	return !gvk.GroupVersion().IsEmpty() && len(gvk.Kind) > 0
}

func (s unstructuredJSONScheme) Decode(data []byte) (Object, error) {
	unstruct := &Unstructured{}
	if err := DecodeInto(s, data, unstruct); err != nil {
		return nil, err
	}
	return unstruct, nil
}

func (unstructuredJSONScheme) DecodeInto(data []byte, obj Object) error {
	unstruct, ok := obj.(*Unstructured)
	if !ok {
		return fmt.Errorf("the unstructured JSON scheme does not recognize %v", reflect.TypeOf(obj))
	}

	m := make(map[string]interface{})
	if err := json.Unmarshal(data, &m); err != nil {
		return err
	}
	if v, ok := m["kind"]; ok {
		if s, ok := v.(string); ok {
			unstruct.Kind = s
		}
	}
	if v, ok := m["apiVersion"]; ok {
		if s, ok := v.(string); ok {
			unstruct.APIVersion = s
		}
	}
	if len(unstruct.APIVersion) == 0 {
		return conversion.NewMissingVersionErr(string(data))
	}
	if len(unstruct.Kind) == 0 {
		return conversion.NewMissingKindErr(string(data))
	}
	unstruct.Object = m
	return nil
}

func (unstructuredJSONScheme) DecodeIntoWithSpecifiedVersionKind(data []byte, obj Object, gvk unversioned.GroupVersionKind) error {
	return nil
}

func (unstructuredJSONScheme) DecodeToVersion(data []byte, gv unversioned.GroupVersion) (Object, error) {
	return nil, nil
}

func (unstructuredJSONScheme) DecodeParametersInto(paramaters url.Values, obj Object) error {
	return nil
}

func (unstructuredJSONScheme) DataKind(data []byte) (unversioned.GroupVersionKind, error) {
	obj := TypeMeta{}
	if err := json.Unmarshal(data, &obj); err != nil {
		return unversioned.GroupVersionKind{}, err
	}
	if len(obj.APIVersion) == 0 {
		return unversioned.GroupVersionKind{}, conversion.NewMissingVersionErr(string(data))
	}
	if len(obj.Kind) == 0 {
		return unversioned.GroupVersionKind{}, conversion.NewMissingKindErr(string(data))
	}

	gv, err := unversioned.ParseGroupVersion(obj.APIVersion)
	if err != nil {
		return unversioned.GroupVersionKind{}, err
	}

	return gv.WithKind(obj.Kind), nil
}
