/*
Copyright 2019 The Kubernetes Authors.

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

type mapUnstructuredInterface map[interface{}]interface{}

func (m mapUnstructuredInterface) Set(key string, val Value) {
	m[key] = val.Unstructured()
}

func (m mapUnstructuredInterface) Get(key string) (Value, bool) {
	if v, ok := m[key]; !ok {
		return nil, false
	} else {
		return NewValueInterface(v), true
	}
}

func (m mapUnstructuredInterface) Has(key string) bool {
	_, ok := m[key]
	return ok
}

func (m mapUnstructuredInterface) Delete(key string) {
	delete(m, key)
}

func (m mapUnstructuredInterface) Iterate(fn func(key string, value Value) bool) bool {
	for k, v := range m {
		if ks, ok := k.(string); !ok {
			continue
		} else {
			vv := NewValueInterface(v)
			if !fn(ks, vv) {
				vv.Recycle()
				return false
			}
			vv.Recycle()
		}
	}
	return true
}

func (m mapUnstructuredInterface) Length() int {
	return len(m)
}

func (m mapUnstructuredInterface) Equals(other Map) bool {
	if m.Length() != other.Length() {
		return false
	}
	for k, v := range m {
		ks, ok := k.(string)
		if !ok {
			return false
		}
		vo, ok := other.Get(ks)
		if !ok {
			return false
		}
		vv := NewValueInterface(v)
		if !Equals(vv, vo) {
			vv.Recycle()
			vo.Recycle()
			return false
		}
		vo.Recycle()
		vv.Recycle()
	}
	return true
}

type mapUnstructuredString map[string]interface{}

func (m mapUnstructuredString) Set(key string, val Value) {
	m[key] = val.Unstructured()
}

func (m mapUnstructuredString) Get(key string) (Value, bool) {
	if v, ok := m[key]; !ok {
		return nil, false
	} else {
		return NewValueInterface(v), true
	}
}

func (m mapUnstructuredString) Has(key string) bool {
	_, ok := m[key]
	return ok
}

func (m mapUnstructuredString) Delete(key string) {
	delete(m, key)
}

func (m mapUnstructuredString) Iterate(fn func(key string, value Value) bool) bool {
	for k, v := range m {
		vv := NewValueInterface(v)
		if !fn(k, vv) {
			vv.Recycle()
			return false
		}
		vv.Recycle()
	}
	return true
}

func (m mapUnstructuredString) Length() int {
	return len(m)
}

func (m mapUnstructuredString) Equals(other Map) bool {
	if m.Length() != other.Length() {
		return false
	}
	for k, v := range m {
		vo, ok := other.Get(k)
		if !ok {
			return false
		}
		vv := NewValueInterface(v)
		if !Equals(vv, vo) {
			vo.Recycle()
			vv.Recycle()
			return false
		}
		vo.Recycle()
		vv.Recycle()
	}
	return true
}
