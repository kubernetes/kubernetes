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

package v1beta1

import (
	"bytes"
	"errors"

	cbor "k8s.io/apimachinery/pkg/runtime/serializer/cbor/direct"
	"k8s.io/apimachinery/pkg/util/json"
)

var jsTrue = []byte("true")
var jsFalse = []byte("false")

// The CBOR parsing related constants and functions below are not exported so they can be
// easily removed at a future date when the CBOR library provides equivalent functionality.

type cborMajorType int

const (
	// https://www.rfc-editor.org/rfc/rfc8949.html#section-3.1
	cborUnsignedInteger cborMajorType = 0
	cborNegativeInteger cborMajorType = 1
	cborByteString      cborMajorType = 2
	cborTextString      cborMajorType = 3
	cborArray           cborMajorType = 4
	cborMap             cborMajorType = 5
	cborTag             cborMajorType = 6
	cborOther           cborMajorType = 7
)

const (
	cborFalseValue = 0xf4
	cborTrueValue  = 0xf5
	cborNullValue  = 0xf6
)

func cborType(b byte) cborMajorType {
	return cborMajorType(b >> 5)
}

func (s JSONSchemaPropsOrBool) MarshalJSON() ([]byte, error) {
	if s.Schema != nil {
		return json.Marshal(s.Schema)
	}

	if s.Schema == nil && !s.Allows {
		return jsFalse, nil
	}
	return jsTrue, nil
}

func (s *JSONSchemaPropsOrBool) UnmarshalJSON(data []byte) error {
	var nw JSONSchemaPropsOrBool
	switch {
	case len(data) == 0:
	case data[0] == '{':
		var sch JSONSchemaProps
		if err := json.Unmarshal(data, &sch); err != nil {
			return err
		}
		nw.Allows = true
		nw.Schema = &sch
	case len(data) == 4 && string(data) == "true":
		nw.Allows = true
	case len(data) == 5 && string(data) == "false":
		nw.Allows = false
	default:
		return errors.New("boolean or JSON schema expected")
	}
	*s = nw
	return nil
}

func (s JSONSchemaPropsOrBool) MarshalCBOR() ([]byte, error) {
	if s.Schema != nil {
		return cbor.Marshal(s.Schema)
	}
	return cbor.Marshal(s.Allows)
}

func (s *JSONSchemaPropsOrBool) UnmarshalCBOR(data []byte) error {
	switch {
	case len(data) == 0:
		// ideally we would avoid modifying *s here, but we are matching the behavior of UnmarshalJSON
		*s = JSONSchemaPropsOrBool{}
		return nil
	case cborType(data[0]) == cborMap:
		var p JSONSchemaProps
		if err := cbor.Unmarshal(data, &p); err != nil {
			return err
		}
		*s = JSONSchemaPropsOrBool{Allows: true, Schema: &p}
		return nil
	case data[0] == cborTrueValue:
		*s = JSONSchemaPropsOrBool{Allows: true}
		return nil
	case data[0] == cborFalseValue:
		*s = JSONSchemaPropsOrBool{Allows: false}
		return nil
	default:
		// ideally, this case would not also capture a null input value,
		// but we are matching the behavior of the UnmarshalJSON
		return errors.New("boolean or JSON schema expected")
	}
}

func (s JSONSchemaPropsOrStringArray) MarshalJSON() ([]byte, error) {
	if len(s.Property) > 0 {
		return json.Marshal(s.Property)
	}
	if s.Schema != nil {
		return json.Marshal(s.Schema)
	}
	return []byte("null"), nil
}

func (s *JSONSchemaPropsOrStringArray) UnmarshalJSON(data []byte) error {
	var first byte
	if len(data) > 1 {
		first = data[0]
	}
	var nw JSONSchemaPropsOrStringArray
	if first == '{' {
		var sch JSONSchemaProps
		if err := json.Unmarshal(data, &sch); err != nil {
			return err
		}
		nw.Schema = &sch
	}
	if first == '[' {
		if err := json.Unmarshal(data, &nw.Property); err != nil {
			return err
		}
	}
	*s = nw
	return nil
}

func (s JSONSchemaPropsOrStringArray) MarshalCBOR() ([]byte, error) {
	if len(s.Property) > 0 {
		return cbor.Marshal(s.Property)
	}
	if s.Schema != nil {
		return cbor.Marshal(s.Schema)
	}
	return cbor.Marshal(nil)
}

func (s *JSONSchemaPropsOrStringArray) UnmarshalCBOR(data []byte) error {
	if len(data) > 0 && cborType(data[0]) == cborArray {
		var a []string
		if err := cbor.Unmarshal(data, &a); err != nil {
			return err
		}
		*s = JSONSchemaPropsOrStringArray{Property: a}
		return nil
	}
	if len(data) > 0 && cborType(data[0]) == cborMap {
		var p JSONSchemaProps
		if err := cbor.Unmarshal(data, &p); err != nil {
			return err
		}
		*s = JSONSchemaPropsOrStringArray{Schema: &p}
		return nil
	}
	// At this point we either have: empty data, a null value, or an
	// unexpected type. In order to match the behavior of the existing
	// UnmarshalJSON, no error is returned and *s is overwritten here.
	*s = JSONSchemaPropsOrStringArray{}
	return nil
}

func (s JSONSchemaPropsOrArray) MarshalJSON() ([]byte, error) {
	if len(s.JSONSchemas) > 0 {
		return json.Marshal(s.JSONSchemas)
	}
	return json.Marshal(s.Schema)
}

func (s *JSONSchemaPropsOrArray) UnmarshalJSON(data []byte) error {
	var nw JSONSchemaPropsOrArray
	var first byte
	if len(data) > 1 {
		first = data[0]
	}
	if first == '{' {
		var sch JSONSchemaProps
		if err := json.Unmarshal(data, &sch); err != nil {
			return err
		}
		nw.Schema = &sch
	}
	if first == '[' {
		if err := json.Unmarshal(data, &nw.JSONSchemas); err != nil {
			return err
		}
	}
	*s = nw
	return nil
}

func (s JSONSchemaPropsOrArray) MarshalCBOR() ([]byte, error) {
	if len(s.JSONSchemas) > 0 {
		return cbor.Marshal(s.JSONSchemas)
	}
	return cbor.Marshal(s.Schema)
}

func (s *JSONSchemaPropsOrArray) UnmarshalCBOR(data []byte) error {
	if len(data) > 0 && cborType(data[0]) == cborMap {
		var p JSONSchemaProps
		if err := cbor.Unmarshal(data, &p); err != nil {
			return err
		}
		*s = JSONSchemaPropsOrArray{Schema: &p}
		return nil
	}
	if len(data) > 0 && cborType(data[0]) == cborArray {
		var a []JSONSchemaProps
		if err := cbor.Unmarshal(data, &a); err != nil {
			return err
		}
		*s = JSONSchemaPropsOrArray{JSONSchemas: a}
		return nil
	}
	// At this point we either have: empty data, a null value, or an
	// unexpected type. In order to match the behavior of the existing
	// UnmarshalJSON, no error is returned and *s is overwritten here.
	*s = JSONSchemaPropsOrArray{}
	return nil
}

func (s JSON) MarshalJSON() ([]byte, error) {
	if len(s.Raw) > 0 {
		return s.Raw, nil
	}
	return []byte("null"), nil

}

func (s *JSON) UnmarshalJSON(data []byte) error {
	if len(data) > 0 && !bytes.Equal(data, nullLiteral) {
		s.Raw = append(s.Raw[0:0], data...)
	}
	return nil
}

func (s JSON) MarshalCBOR() ([]byte, error) {
	// Note that non-semantic whitespace is lost during the transcoding performed here.
	// We do not forsee this to be a problem given the current known uses of this type.
	// Other limitations that arise when roundtripping JSON via dynamic clients also apply
	// here, for example: insignificant whitespace handling, number handling, and map key ordering.
	if len(s.Raw) == 0 {
		return []byte{cborNullValue}, nil
	}
	var u any
	if err := json.Unmarshal(s.Raw, &u); err != nil {
		return nil, err
	}
	return cbor.Marshal(u)
}

func (s *JSON) UnmarshalCBOR(data []byte) error {
	if len(data) == 0 || data[0] == cborNullValue {
		return nil
	}
	var u any
	if err := cbor.Unmarshal(data, &u); err != nil {
		return err
	}
	raw, err := json.Marshal(u)
	if err != nil {
		return err
	}
	s.Raw = raw
	return nil
}
