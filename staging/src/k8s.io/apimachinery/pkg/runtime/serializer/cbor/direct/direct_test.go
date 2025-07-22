/*
Copyright 2024 The Kubernetes Authors.

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

package direct_test

import (
	"encoding"
	"encoding/json"
	"fmt"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/serializer/cbor/direct"
)

var _ json.Marshaler = CustomJSONMarshaler{}

type CustomJSONMarshaler struct{}

func (CustomJSONMarshaler) MarshalJSON() ([]byte, error) {
	panic("unimplemented")
}

var _ json.Unmarshaler = CustomJSONUnmarshaler{}

type CustomJSONUnmarshaler struct{}

func (CustomJSONUnmarshaler) UnmarshalJSON([]byte) error {
	panic("unimplemented")
}

var _ encoding.TextMarshaler = CustomTextMarshaler{}

type CustomTextMarshaler struct{}

func (CustomTextMarshaler) MarshalText() ([]byte, error) {
	panic("unimplemented")
}

var _ encoding.TextUnmarshaler = CustomTextUnmarshaler{}

type CustomTextUnmarshaler struct{}

func (CustomTextUnmarshaler) UnmarshalText([]byte) error {
	panic("unimplemented")
}

func TestRejectsCustom(t *testing.T) {
	for _, tc := range []struct {
		value interface{}
		iface reflect.Type
	}{
		{value: CustomJSONMarshaler{}, iface: reflect.TypeFor[json.Marshaler]()},
		{value: CustomJSONUnmarshaler{}, iface: reflect.TypeFor[json.Unmarshaler]()},
		{value: CustomTextMarshaler{}, iface: reflect.TypeFor[encoding.TextMarshaler]()},
		{value: CustomTextUnmarshaler{}, iface: reflect.TypeFor[encoding.TextUnmarshaler]()},
	} {
		t.Run(fmt.Sprintf("%T", tc.value), func(t *testing.T) {
			want := fmt.Sprintf("unable to serialize %T: %T implements %s without corresponding cbor interface", tc.value, tc.value, tc.iface.String())
			if _, err := direct.Marshal(tc.value); err == nil || err.Error() != want {
				t.Errorf("want error: %q, got: %v", want, err)
			}
			if err := direct.Unmarshal(nil, tc.value); err == nil || err.Error() != want {
				t.Errorf("want error: %q, got: %v", want, err)
			}
		})
	}
}
