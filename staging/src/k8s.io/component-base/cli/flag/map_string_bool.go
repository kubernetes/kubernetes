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
	"strconv"
	"strings"
)

// MapStringBool can be set from the command line with the format `--flag "string=bool"`.
// Multiple comma-separated key-value pairs in a single invocation are supported. For example: `--flag "a=true,b=false"`.
// Multiple flag invocations are supported. For example: `--flag "a=true" --flag "b=false"`.
type MapStringBool struct {
	Map         *map[string]bool
	initialized bool
}

// NewMapStringBool takes a pointer to a map[string]string and returns the
// MapStringBool flag parsing shim for that map
func NewMapStringBool(m *map[string]bool) *MapStringBool {
	return &MapStringBool{Map: m}
}

// String implements github.com/spf13/pflag.Value
func (m *MapStringBool) String() string {
	if m == nil || m.Map == nil {
		return ""
	}
	pairs := []string{}
	for k, v := range *m.Map {
		pairs = append(pairs, fmt.Sprintf("%s=%t", k, v))
	}
	sort.Strings(pairs)
	return strings.Join(pairs, ",")
}

// Set implements github.com/spf13/pflag.Value
func (m *MapStringBool) Set(value string) error {
	if m.Map == nil {
		return fmt.Errorf("no target (nil pointer to map[string]bool)")
	}
	if !m.initialized || *m.Map == nil {
		// clear default values, or allocate if no existing map
		*m.Map = make(map[string]bool)
		m.initialized = true
	}
	for _, s := range strings.Split(value, ",") {
		if len(s) == 0 {
			continue
		}
		arr := strings.SplitN(s, "=", 2)
		if len(arr) != 2 {
			return fmt.Errorf("malformed pair, expect string=bool")
		}
		k := strings.TrimSpace(arr[0])
		v := strings.TrimSpace(arr[1])
		boolValue, err := strconv.ParseBool(v)
		if err != nil {
			return fmt.Errorf("invalid value of %s: %s, err: %v", k, v, err)
		}
		(*m.Map)[k] = boolValue
	}
	return nil
}

// Type implements github.com/spf13/pflag.Value
func (*MapStringBool) Type() string {
	return "mapStringBool"
}

// Empty implements OmitEmpty
func (m *MapStringBool) Empty() bool {
	return len(*m.Map) == 0
}
