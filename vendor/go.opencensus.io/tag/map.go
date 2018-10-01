// Copyright 2017, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package tag

import (
	"bytes"
	"context"
	"fmt"
	"sort"
)

// Tag is a key value pair that can be propagated on wire.
type Tag struct {
	Key   Key
	Value string
}

// Map is a map of tags. Use New to create a context containing
// a new Map.
type Map struct {
	m map[Key]string
}

// Value returns the value for the key if a value for the key exists.
func (m *Map) Value(k Key) (string, bool) {
	if m == nil {
		return "", false
	}
	v, ok := m.m[k]
	return v, ok
}

func (m *Map) String() string {
	if m == nil {
		return "nil"
	}
	keys := make([]Key, 0, len(m.m))
	for k := range m.m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i].Name() < keys[j].Name() })

	var buffer bytes.Buffer
	buffer.WriteString("{ ")
	for _, k := range keys {
		buffer.WriteString(fmt.Sprintf("{%v %v}", k.name, m.m[k]))
	}
	buffer.WriteString(" }")
	return buffer.String()
}

func (m *Map) insert(k Key, v string) {
	if _, ok := m.m[k]; ok {
		return
	}
	m.m[k] = v
}

func (m *Map) update(k Key, v string) {
	if _, ok := m.m[k]; ok {
		m.m[k] = v
	}
}

func (m *Map) upsert(k Key, v string) {
	m.m[k] = v
}

func (m *Map) delete(k Key) {
	delete(m.m, k)
}

func newMap() *Map {
	return &Map{m: make(map[Key]string)}
}

// Mutator modifies a tag map.
type Mutator interface {
	Mutate(t *Map) (*Map, error)
}

// Insert returns a mutator that inserts a
// value associated with k. If k already exists in the tag map,
// mutator doesn't update the value.
func Insert(k Key, v string) Mutator {
	return &mutator{
		fn: func(m *Map) (*Map, error) {
			if !checkValue(v) {
				return nil, errInvalidValue
			}
			m.insert(k, v)
			return m, nil
		},
	}
}

// Update returns a mutator that updates the
// value of the tag associated with k with v. If k doesn't
// exists in the tag map, the mutator doesn't insert the value.
func Update(k Key, v string) Mutator {
	return &mutator{
		fn: func(m *Map) (*Map, error) {
			if !checkValue(v) {
				return nil, errInvalidValue
			}
			m.update(k, v)
			return m, nil
		},
	}
}

// Upsert returns a mutator that upserts the
// value of the tag associated with k with v. It inserts the
// value if k doesn't exist already. It mutates the value
// if k already exists.
func Upsert(k Key, v string) Mutator {
	return &mutator{
		fn: func(m *Map) (*Map, error) {
			if !checkValue(v) {
				return nil, errInvalidValue
			}
			m.upsert(k, v)
			return m, nil
		},
	}
}

// Delete returns a mutator that deletes
// the value associated with k.
func Delete(k Key) Mutator {
	return &mutator{
		fn: func(m *Map) (*Map, error) {
			m.delete(k)
			return m, nil
		},
	}
}

// New returns a new context that contains a tag map
// originated from the incoming context and modified
// with the provided mutators.
func New(ctx context.Context, mutator ...Mutator) (context.Context, error) {
	m := newMap()
	orig := FromContext(ctx)
	if orig != nil {
		for k, v := range orig.m {
			if !checkKeyName(k.Name()) {
				return ctx, fmt.Errorf("key:%q: %v", k, errInvalidKeyName)
			}
			if !checkValue(v) {
				return ctx, fmt.Errorf("key:%q value:%q: %v", k.Name(), v, errInvalidValue)
			}
			m.insert(k, v)
		}
	}
	var err error
	for _, mod := range mutator {
		m, err = mod.Mutate(m)
		if err != nil {
			return ctx, err
		}
	}
	return NewContext(ctx, m), nil
}

// Do is similar to pprof.Do: a convenience for installing the tags
// from the context as Go profiler labels. This allows you to
// correlated runtime profiling with stats.
//
// It converts the key/values from the given map to Go profiler labels
// and calls pprof.Do.
//
// Do is going to do nothing if your Go version is below 1.9.
func Do(ctx context.Context, f func(ctx context.Context)) {
	do(ctx, f)
}

type mutator struct {
	fn func(t *Map) (*Map, error)
}

func (m *mutator) Mutate(t *Map) (*Map, error) {
	return m.fn(t)
}
