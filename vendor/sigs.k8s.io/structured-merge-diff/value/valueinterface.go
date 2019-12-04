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
		return &valueInterface{}
	},
}

// NewValueInterface creates a Value backed by an "interface{}" type,
// typically an unstructured object in Kubernetes world.
func NewValueInterface(v interface{}) Value {
	vi := viPool.Get().(*valueInterface)
	vi.Value = v
	return Value(vi)
}

type valueInterface struct {
	Value interface{}
}

func (v valueInterface) IsMap() bool {
	if _, ok := v.Value.(map[string]interface{}); ok {
		return true
	}
	if _, ok := v.Value.(map[interface{}]interface{}); ok {
		return true
	}
	return false
}

func (v valueInterface) Map() Map {
	if v.Value == nil {
		return mapString(nil)
	}
	switch t := v.Value.(type) {
	case map[string]interface{}:
		return mapString(t)
	case map[interface{}]interface{}:
		return mapInterface(t)
	}
	panic(fmt.Errorf("not a map: %#v", v))
}

func (v valueInterface) IsList() bool {
	if v.Value == nil {
		return false
	}
	_, ok := v.Value.([]interface{})
	return ok
}

func (v valueInterface) List() List {
	return listInterface(v.Value.([]interface{}))
}

func (v valueInterface) IsFloat() bool {
	if v.Value == nil {
		return false
	} else if _, ok := v.Value.(float64); ok {
		return true
	} else if _, ok := v.Value.(float32); ok {
		return true
	}
	return false
}

func (v valueInterface) Float() float64 {
	if f, ok := v.Value.(float32); ok {
		return float64(f)
	}
	return v.Value.(float64)
}

func (v valueInterface) IsInt() bool {
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
	}
	return false
}

func (v valueInterface) Int() int64 {
	if i, ok := v.Value.(int); ok {
		return int64(i)
	} else if i, ok := v.Value.(int8); ok {
		return int64(i)
	} else if i, ok := v.Value.(int16); ok {
		return int64(i)
	} else if i, ok := v.Value.(int32); ok {
		return int64(i)
	}
	return v.Value.(int64)
}

func (v valueInterface) IsString() bool {
	if v.Value == nil {
		return false
	}
	_, ok := v.Value.(string)
	return ok
}

func (v valueInterface) String() string {
	return v.Value.(string)
}

func (v valueInterface) IsBool() bool {
	if v.Value == nil {
		return false
	}
	_, ok := v.Value.(bool)
	return ok
}

func (v valueInterface) Bool() bool {
	return v.Value.(bool)
}

func (v valueInterface) IsNull() bool {
	return v.Value == nil
}

func (v *valueInterface) Recycle() {
	viPool.Put(v)
}

func (v valueInterface) Interface() interface{} {
	return v.Value
}
