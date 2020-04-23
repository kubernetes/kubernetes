/*
Copyright 2020 The Kubernetes Authors.

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

// BracketSeparatedSliceMapStringString can be set from the command line with the format `--flag {key=value, ...}, {...}`.
// Multiple comma-separated key-value pairs in brackets (`{}`) in a single invocation are supported. For example: `--flag {key=value, key=value, ...}`.
// Multiple bracket-separated list of key-value pairs in a single invocation are supported. For example: `--flag {key=value, key=value}, {key=value, key=value}`.
type BracketSeparatedSliceMapStringString struct {
	Value       *[]map[string]string
	initialized bool // set to true after the first Set call
}

// NewBracketSeparatedSliceMapStringString takes a pointer to a []map[string]string and returns the
// BracketSeparatedSliceMapStringString flag parsing shim for that map
func NewBracketSeparatedSliceMapStringString(m *[]map[string]string) *BracketSeparatedSliceMapStringString {
	return &BracketSeparatedSliceMapStringString{Value: m}
}

// Set implements github.com/spf13/pflag.Value
func (m *BracketSeparatedSliceMapStringString) Set(value string) error {
	if m.Value == nil {
		return fmt.Errorf("no target (nil pointer to []map[string]string)")
	}
	if !m.initialized || *m.Value == nil {
		*m.Value = make([]map[string]string, 0)
		m.initialized = true
	}

	value = strings.TrimSpace(value)

	for _, split := range strings.Split(value, ",{") {
		split = strings.TrimLeft(split, "{")
		split = strings.TrimRight(split, "}")

		if len(split) == 0 {
			continue
		}

		// now we have "numa-node=1,memory-type=memory,limit=1Gi"
		tmpRawMap := make(map[string]string)

		tmpMap := NewMapStringString(&tmpRawMap)

		if err := tmpMap.Set(split); err != nil {
			return fmt.Errorf("could not parse String: (%s): %v", value, err)
		}

		*m.Value = append(*m.Value, tmpRawMap)
	}

	return nil
}

// String implements github.com/spf13/pflag.Value
func (m *BracketSeparatedSliceMapStringString) String() string {
	if m == nil || m.Value == nil {
		return ""
	}

	var slices []string

	for _, configMap := range *m.Value {
		var tmpPairs []string

		var keys []string
		for key := range configMap {
			keys = append(keys, key)
		}
		sort.Strings(keys)

		for _, key := range keys {
			tmpPairs = append(tmpPairs, fmt.Sprintf("%s=%s", key, configMap[key]))
		}

		if len(tmpPairs) != 0 {
			slices = append(slices, "{"+strings.Join(tmpPairs, ",")+"}")
		}
	}
	sort.Strings(slices)
	return strings.Join(slices, ",")
}

// Type implements github.com/spf13/pflag.Value
func (*BracketSeparatedSliceMapStringString) Type() string {
	return "BracketSeparatedSliceMapStringString"
}

// Empty implements OmitEmpty
func (m *BracketSeparatedSliceMapStringString) Empty() bool {
	return !m.initialized || m.Value == nil || len(*m.Value) == 0
}
