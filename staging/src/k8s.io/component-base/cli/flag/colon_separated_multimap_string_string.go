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

package flag

import (
	"fmt"
	"sort"
	"strings"
)

// ColonSeparatedMultimapStringString supports setting a map[string][]string from an encoding
// that separates keys from values with ':' and separates key-value pairs with ','.
// A key can be repeated multiple times, in which case the values are appended to a
// slice of strings associated with that key. Items in the list associated with a given
// key will appear in the order provided.
// For example: `a:hello,b:again,c:world,b:beautiful` results in `{"a": ["hello"], "b": ["again", "beautiful"], "c": ["world"]}`
// The first call to Set will clear the map before adding entries; subsequent calls will simply append to the map.
// This makes it possible to override default values with a command-line option rather than appending to defaults,
// while still allowing the distribution of key-value pairs across multiple flag invocations.
// For example: `--flag "a:hello" --flag "b:again" --flag "b:beautiful" --flag "c:world"` results in `{"a": ["hello"], "b": ["again", "beautiful"], "c": ["world"]}`
type ColonSeparatedMultimapStringString struct {
	Multimap             *map[string][]string
	initialized          bool // set to true after the first Set call
	allowDefaultEmptyKey bool
}

// NewColonSeparatedMultimapStringString takes a pointer to a map[string][]string and returns the
// ColonSeparatedMultimapStringString flag parsing shim for that map.
func NewColonSeparatedMultimapStringString(m *map[string][]string) *ColonSeparatedMultimapStringString {
	return &ColonSeparatedMultimapStringString{Multimap: m}
}

// NewColonSeparatedMultimapStringStringAllowDefaultEmptyKey takes a pointer to a map[string][]string and returns the
// ColonSeparatedMultimapStringString flag parsing shim for that map. It allows default empty key with no colon in the flag.
func NewColonSeparatedMultimapStringStringAllowDefaultEmptyKey(m *map[string][]string) *ColonSeparatedMultimapStringString {
	return &ColonSeparatedMultimapStringString{Multimap: m, allowDefaultEmptyKey: true}
}

// Set implements github.com/spf13/pflag.Value
func (m *ColonSeparatedMultimapStringString) Set(value string) error {
	if m.Multimap == nil {
		return fmt.Errorf("no target (nil pointer to map[string][]string)")
	}
	if !m.initialized || *m.Multimap == nil {
		// clear default values, or allocate if no existing map
		*m.Multimap = make(map[string][]string)
		m.initialized = true
	}
	for _, pair := range strings.Split(value, ",") {
		if len(pair) == 0 {
			continue
		}
		kv := strings.SplitN(pair, ":", 2)
		var k, v string
		if m.allowDefaultEmptyKey && len(kv) == 1 {
			v = strings.TrimSpace(kv[0])
		} else {
			if len(kv) != 2 {
				return fmt.Errorf("malformed pair, expect string:string")
			}
			k = strings.TrimSpace(kv[0])
			v = strings.TrimSpace(kv[1])
		}
		(*m.Multimap)[k] = append((*m.Multimap)[k], v)
	}
	return nil
}

// String implements github.com/spf13/pflag.Value
func (m *ColonSeparatedMultimapStringString) String() string {
	type kv struct {
		k string
		v string
	}
	kvs := make([]kv, 0, len(*m.Multimap))
	for k, vs := range *m.Multimap {
		for i := range vs {
			kvs = append(kvs, kv{k: k, v: vs[i]})
		}
	}
	// stable sort by keys, order of values should be preserved
	sort.SliceStable(kvs, func(i, j int) bool {
		return kvs[i].k < kvs[j].k
	})
	pairs := make([]string, 0, len(kvs))
	for i := range kvs {
		pairs = append(pairs, fmt.Sprintf("%s:%s", kvs[i].k, kvs[i].v))
	}
	return strings.Join(pairs, ",")
}

// Type implements github.com/spf13/pflag.Value
func (m *ColonSeparatedMultimapStringString) Type() string {
	return "colonSeparatedMultimapStringString"
}

// Empty implements OmitEmpty
func (m *ColonSeparatedMultimapStringString) Empty() bool {
	return len(*m.Multimap) == 0
}
