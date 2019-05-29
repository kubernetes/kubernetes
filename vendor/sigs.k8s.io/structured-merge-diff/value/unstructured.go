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

package value

import (
	"encoding/json"
	"fmt"

	"gopkg.in/yaml.v2"
)

// FromYAML is a helper function for reading a YAML document; it attempts to
// preserve order of keys within maps/structs. This is as a convenience to
// humans keeping YAML documents, not because there is a behavior difference.
//
// Known bug: objects with top-level arrays don't parse correctly.
func FromYAML(input []byte) (Value, error) {
	var decoded interface{}

	if len(input) == 4 && string(input) == "null" {
		// Special case since the yaml package doesn't accurately
		// preserve this.
		return Value{Null: true}, nil
	}

	// This attempts to enable order sensitivity; note the yaml package is
	// broken for documents that have root-level arrays, hence the two-step
	// approach. TODO: This is a horrific hack. Is it worth it?
	var ms yaml.MapSlice
	if err := yaml.Unmarshal(input, &ms); err == nil {
		decoded = ms
	} else if err := yaml.Unmarshal(input, &decoded); err != nil {
		return Value{}, err
	}

	v, err := FromUnstructured(decoded)
	if err != nil {
		return Value{}, fmt.Errorf("failed to interpret (%v):\n%s", err, input)
	}
	return v, nil
}

// FromJSON is a helper function for reading a JSON document
func FromJSON(input []byte) (Value, error) {
	var decoded interface{}

	if err := json.Unmarshal(input, &decoded); err != nil {
		return Value{}, err
	}

	v, err := FromUnstructured(decoded)
	if err != nil {
		return Value{}, fmt.Errorf("failed to interpret (%v):\n%s", err, input)
	}
	return v, nil
}

// FromUnstructured will convert a go interface to a Value.
// It's most commonly expected to be used with map[string]interface{} as the
// input. `in` must not have any structures with cycles in them.
// yaml.MapSlice may be used for order-preservation.
func FromUnstructured(in interface{}) (Value, error) {
	if in == nil {
		return Value{Null: true}, nil
	}
	switch t := in.(type) {
	case map[interface{}]interface{}:
		m := Map{}
		for rawKey, rawVal := range t {
			k, ok := rawKey.(string)
			if !ok {
				return Value{}, fmt.Errorf("key %#v: not a string", k)
			}
			v, err := FromUnstructured(rawVal)
			if err != nil {
				return Value{}, fmt.Errorf("key %v: %v", k, err)
			}
			m.Set(k, v)
		}
		return Value{MapValue: &m}, nil
	case map[string]interface{}:
		m := Map{}
		for k, rawVal := range t {
			v, err := FromUnstructured(rawVal)
			if err != nil {
				return Value{}, fmt.Errorf("key %v: %v", k, err)
			}
			m.Set(k, v)
		}
		return Value{MapValue: &m}, nil
	case yaml.MapSlice:
		m := Map{}
		for _, item := range t {
			k, ok := item.Key.(string)
			if !ok {
				return Value{}, fmt.Errorf("key %#v is not a string", item.Key)
			}
			v, err := FromUnstructured(item.Value)
			if err != nil {
				return Value{}, fmt.Errorf("key %v: %v", k, err)
			}
			m.Set(k, v)
		}
		return Value{MapValue: &m}, nil
	case []interface{}:
		l := List{}
		for i, rawVal := range t {
			v, err := FromUnstructured(rawVal)
			if err != nil {
				return Value{}, fmt.Errorf("index %v: %v", i, err)
			}
			l.Items = append(l.Items, v)
		}
		return Value{ListValue: &l}, nil
	case int:
		n := Int(t)
		return Value{IntValue: &n}, nil
	case int8:
		n := Int(t)
		return Value{IntValue: &n}, nil
	case int16:
		n := Int(t)
		return Value{IntValue: &n}, nil
	case int32:
		n := Int(t)
		return Value{IntValue: &n}, nil
	case int64:
		n := Int(t)
		return Value{IntValue: &n}, nil
	case uint:
		n := Int(t)
		return Value{IntValue: &n}, nil
	case uint8:
		n := Int(t)
		return Value{IntValue: &n}, nil
	case uint16:
		n := Int(t)
		return Value{IntValue: &n}, nil
	case uint32:
		n := Int(t)
		return Value{IntValue: &n}, nil
	case float32:
		f := Float(t)
		return Value{FloatValue: &f}, nil
	case float64:
		f := Float(t)
		return Value{FloatValue: &f}, nil
	case string:
		return StringValue(t), nil
	case bool:
		return BooleanValue(t), nil
	default:
		return Value{}, fmt.Errorf("type unimplemented: %t", in)
	}
}

// ToYAML is a helper function for producing a YAML document; it attempts to
// preserve order of keys within maps/structs. This is as a convenience to
// humans keeping YAML documents, not because there is a behavior difference.
func (v *Value) ToYAML() ([]byte, error) {
	return yaml.Marshal(v.ToUnstructured(true))
}

// ToJSON is a helper function for producing a JSon document.
func (v *Value) ToJSON() ([]byte, error) {
	return json.Marshal(v.ToUnstructured(false))
}

// ToUnstructured will convert the Value into a go-typed object.
// If preserveOrder is true, then maps will be converted to the yaml.MapSlice
// type. Otherwise, map[string]interface{} must be used-- this destroys
// ordering information and is not recommended if the result of this will be
// serialized. Other types:
// * list -> []interface{}
// * others -> corresponding go type, wrapped in an interface{}
//
// Of note, floats and ints will always come out as float64 and int64,
// respectively.
func (v *Value) ToUnstructured(preserveOrder bool) interface{} {
	switch {
	case v.FloatValue != nil:
		f := float64(*v.FloatValue)
		return f
	case v.IntValue != nil:
		i := int64(*v.IntValue)
		return i
	case v.StringValue != nil:
		return string(*v.StringValue)
	case v.BooleanValue != nil:
		return bool(*v.BooleanValue)
	case v.ListValue != nil:
		out := []interface{}{}
		for _, item := range v.ListValue.Items {
			out = append(out, item.ToUnstructured(preserveOrder))
		}
		return out
	case v.MapValue != nil:
		m := v.MapValue
		if preserveOrder {
			ms := make(yaml.MapSlice, len(m.Items))
			for i := range m.Items {
				ms[i] = yaml.MapItem{
					Key:   m.Items[i].Name,
					Value: m.Items[i].Value.ToUnstructured(preserveOrder),
				}
			}
			return ms
		}
		// This case is unavoidably lossy.
		out := map[string]interface{}{}
		for i := range m.Items {
			out[m.Items[i].Name] = m.Items[i].Value.ToUnstructured(preserveOrder)
		}
		return out
	default:
		fallthrough
	case v.Null == true:
		return nil
	}
}
