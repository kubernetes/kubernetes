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

package arraystr

import (
	"encoding/json"
	"fmt"

	openapi "k8s.io/kube-openapi/pkg/common"

	"github.com/go-openapi/spec"
	"github.com/google/gofuzz"
)

// StringArrayOrString is a type that can hold an array of strings or a string.
// When used in JSON or YAML marshalling and unmarshalling, it produces or consumes
// the inner type.  This allows you to have, for example, a JSON field that can
// accept a name or a list of names. A one-element list is different from a string.
// An empty string and an empty list are the same.
//
// +protobuf=true
// +protobuf.options.(gogoproto.goproto_stringer)=false
// +k8s:openapi-gen=true
type StringArrayOrString struct {
	Type     Type     `protobuf:"varint,1,opt,name=type,casttype=Type"`
	ArrayVal []string `protobuf:"varint,2,opt,name=intVal"`
	StrVal   string   `protobuf:"bytes,3,opt,name=strVal"`
}

// Type represents the stored type of StringArrayOrString.
type Type int

const (
	String Type = iota // The StringArrayOrString holds a string.
	Array              // The StringArrayOrString holds an string array.
)

// FromStringArray creates an StringArrayOrString object with an []string value.
func FromStringArray(val []string) StringArrayOrString {
	if len(val) == 0 {
		return StringArrayOrString{Type: String}
	}
	a := make([]string, len(val))
	for i := range val {
		a[i] = val[i]
	}
	return StringArrayOrString{Type: Array, ArrayVal: a}
}

// FromString creates an StringArrayOrString object with a string value.
func FromString(val string) StringArrayOrString {
	return StringArrayOrString{Type: String, StrVal: val}
}

// UnmarshalJSON implements the json.Unmarshaller interface.
func (arraystr *StringArrayOrString) UnmarshalJSON(value []byte) error {
	if value[0] == '"' {
		arraystr.Type = String
		return json.Unmarshal(value, &arraystr.StrVal)
	}
	var a []interface{}
	err := json.Unmarshal(value, &a)
	if err != nil {
		return err
	}
	arraystr.StrVal = ""
	if len(a) == 0 {
		*arraystr = StringArrayOrString{
			Type:   String,
			StrVal: "",
		}
		return nil
	}
	sa := make([]string, len(a))
	for i := range a {
		if x, ok := a[i].(string); !ok {
			return fmt.Errorf("array element %v expected to be a string", a[i])
		} else {
			sa[i] = x
		}
	}
	*arraystr = StringArrayOrString{
		Type:     Array,
		ArrayVal: sa,
	}
	return nil
}

func (arraystr *StringArrayOrString) IsZero() bool {
	return arraystr == nil || (arraystr.Type == String && len(arraystr.StrVal) == 0)
}

// String returns the string value, or the Itoa of the int value.
func (arraystr *StringArrayOrString) String() string {
	if arraystr.Type == String {
		return arraystr.StrVal
	}
	return fmt.Sprintf("%v", arraystr.ArrayVal)
}

// ArrayValue returns the array value if type Array, or if
// it is a String, will return a one-element slice.
func (arraystr *StringArrayOrString) ArrayValue() []string {
	if arraystr.Type == String {
		if arraystr.StrVal == "" {
			return nil
		}
		return []string{arraystr.StrVal}
	}
	return arraystr.ArrayVal
}

// MarshalJSON implements the json.Marshaller interface.
func (arraystr StringArrayOrString) MarshalJSON() ([]byte, error) {
	switch arraystr.Type {
	case Array:
		return json.Marshal(arraystr.ArrayVal)
	case String:
		return json.Marshal(arraystr.StrVal)
	default:
		return []byte{}, fmt.Errorf("impossible StringArrayOrString.Type")
	}
}

func (arraystr StringArrayOrString) OpenAPIDefinition() openapi.OpenAPIDefinition {
	return openapi.OpenAPIDefinition{
		Schema: spec.Schema{
			SchemaProps: spec.SchemaProps{
				Type:   []string{"string"}, // TODO: what to do here? AnyOf?
				Format: "string-array-or-string",
			},
		},
	}
}

func (arraystr *StringArrayOrString) Fuzz(c fuzz.Continue) {
	if arraystr == nil {
		return
	}
	if c.RandBool() {
		arraystr.Type = Array
		c.Fuzz(&arraystr.ArrayVal)
		arraystr.StrVal = ""
	} else {
		arraystr.Type = String
		arraystr.ArrayVal = nil
		c.Fuzz(&arraystr.StrVal)
	}
}
