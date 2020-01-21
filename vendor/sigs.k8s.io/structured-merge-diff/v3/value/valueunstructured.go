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
	"fmt"
	"sync"
)

var viPool = sync.Pool{
	New: func() interface{} {
		return &valueUnstructured{}
	},
}

// NewValueInterface creates a Value backed by an "interface{}" type,
// typically an unstructured object in Kubernetes world.
// interface{} must be one of: map[string]interface{}, map[interface{}]interface{}, []interface{}, int types, float types,
// string or boolean. Nested interface{} must also be one of these types.
func NewValueInterface(v interface{}) Value {
	vi := viPool.Get().(*valueUnstructured)
	vi.Value = v
	return Value(vi)
}

type valueUnstructured struct {
	Value interface{}
}

func (v valueUnstructured) IsMap() bool {
	if _, ok := v.Value.(map[string]interface{}); ok {
		return true
	}
	if _, ok := v.Value.(map[interface{}]interface{}); ok {
		return true
	}
	return false
}

func (v valueUnstructured) AsMap() Map {
	if v.Value == nil {
		panic("invalid nil")
	}
	switch t := v.Value.(type) {
	case map[string]interface{}:
		return mapUnstructuredString(t)
	case map[interface{}]interface{}:
		return mapUnstructuredInterface(t)
	}
	panic(fmt.Errorf("not a map: %#v", v))
}

func (v valueUnstructured) IsList() bool {
	if v.Value == nil {
		return false
	}
	_, ok := v.Value.([]interface{})
	return ok
}

func (v valueUnstructured) AsList() List {
	return listUnstructured(v.Value.([]interface{}))
}

func (v valueUnstructured) IsFloat() bool {
	if v.Value == nil {
		return false
	} else if _, ok := v.Value.(float64); ok {
		return true
	} else if _, ok := v.Value.(float32); ok {
		return true
	}
	return false
}

func (v valueUnstructured) AsFloat() float64 {
	if f, ok := v.Value.(float32); ok {
		return float64(f)
	}
	return v.Value.(float64)
}

func (v valueUnstructured) IsInt() bool {
	if v.Value == nil {
		return false
	} else if _, ok := v.Value.(int); ok {
		return true
	} else if _, ok := v.Value.(int8); ok {
		return true
	} else if _, ok := v.Value.(int16); ok {
		return true
	} else if _, ok := v.Value.(int32); ok {
		return true
	} else if _, ok := v.Value.(int64); ok {
		return true
	} else if _, ok := v.Value.(uint); ok {
		return true
	} else if _, ok := v.Value.(uint8); ok {
		return true
	} else if _, ok := v.Value.(uint16); ok {
		return true
	} else if _, ok := v.Value.(uint32); ok {
		return true
	}
	return false
}

func (v valueUnstructured) AsInt() int64 {
	if i, ok := v.Value.(int); ok {
		return int64(i)
	} else if i, ok := v.Value.(int8); ok {
		return int64(i)
	} else if i, ok := v.Value.(int16); ok {
		return int64(i)
	} else if i, ok := v.Value.(int32); ok {
		return int64(i)
	} else if i, ok := v.Value.(uint); ok {
		return int64(i)
	} else if i, ok := v.Value.(uint8); ok {
		return int64(i)
	} else if i, ok := v.Value.(uint16); ok {
		return int64(i)
	} else if i, ok := v.Value.(uint32); ok {
		return int64(i)
	}
	return v.Value.(int64)
}

func (v valueUnstructured) IsString() bool {
	if v.Value == nil {
		return false
	}
	_, ok := v.Value.(string)
	return ok
}

func (v valueUnstructured) AsString() string {
	return v.Value.(string)
}

func (v valueUnstructured) IsBool() bool {
	if v.Value == nil {
		return false
	}
	_, ok := v.Value.(bool)
	return ok
}

func (v valueUnstructured) AsBool() bool {
	return v.Value.(bool)
}

func (v valueUnstructured) IsNull() bool {
	return v.Value == nil
}

func (v *valueUnstructured) Recycle() {
	viPool.Put(v)
}

func (v valueUnstructured) Unstructured() interface{} {
	return v.Value
}
