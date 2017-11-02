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
type ColonSeparatedMultimapStringString map[string][]string

// Set implements github.com/spf13/pflag.Value
func (m ColonSeparatedMultimapStringString) Set(value string) error {
	// clear old values
	for k := range m {
		delete(m, k)
	}
	for _, pair := range strings.Split(value, ",") {
		if len(pair) == 0 {
			continue
		}
		kv := strings.SplitN(pair, ":", 2)
		if len(kv) != 2 {
			return fmt.Errorf("malformed pair, expect string:string")
		}
		k := strings.TrimSpace(kv[0])
		v := strings.TrimSpace(kv[1])
		m[k] = append(m[k], v)
	}
	return nil
}

// String implements github.com/spf13/pflag.Value
func (m ColonSeparatedMultimapStringString) String() string {
	type kv struct {
		k string
		v string
	}
	kvs := make([]kv, 0, len(m))
	for k, vs := range m {
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
func (m ColonSeparatedMultimapStringString) Type() string {
	return "colonSeparatedMultimapStringString"
}
