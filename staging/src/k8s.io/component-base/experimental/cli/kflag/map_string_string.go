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

// This file is generated. DO NOT EDIT.

package kflag

import(
	"fmt"
	"sort"
	"strings"

	"github.com/spf13/pflag"
)

// MapStringStringValue contains the scratch space for a registered map[string]string flag.
// Values can be applied from this scratch space to a target using the Set, Merge, or Apply methods.
type MapStringStringValue struct {
	name string
	value map[string]string
	fs *pflag.FlagSet
}

// MapStringStringVar registers a flag for type map[string]string against the FlagSet, and returns a struct
// of type MapStringStringValue that contains the scratch space the flag will be parsed into.
func (fs *FlagSet) MapStringStringVar(name string, def map[string]string, sep string, usage string) *MapStringStringValue {
	val := &MapStringStringValue{
		name:  name,
		value: make(map[string]string),
		fs:    fs.fs,
	}
	for k, v := range def {
		val.value[k] = v
	}
	fs.fs.Var(NewMapStringString(&val.value, sep), name, usage)
	return val
}

// Set copies the map over the target if the flag was detected.
// It completely overwrites any existing target.
func (v *MapStringStringValue) Set(target *map[string]string) {
	if v.fs.Changed(v.name) {
		*target = make(map[string]string)
		for k, v := range v.value {
			(*target)[k] = v
		}
	}
}

// Merge copies the map keys/values piecewise into the target if the flag was detected.
func (v *MapStringStringValue) Merge(target *map[string]string) {
	if v.fs.Changed(v.name) {
		if *target == nil {
			*target = make(map[string]string)
		}
		for k, v := range v.value {
			(*target)[k] = v
		}
	}
}

// Apply calls the user-provided apply function with the map if the flag was detected.
func (v *MapStringStringValue) Apply(apply func(value map[string]string)) {
	if v.fs.Changed(v.name) {
		apply(v.value)
	}
}

// TODO(mtaufen): I just copied the below from map_string_string.go, but we can probably
// simplify, e.g. by deduplicating the map values between the below and the scratch space.

// TODO(mtaufen): Consider making all of the below types/methods private, since they should
//  be able to just hide behind this shim now.

// TODO(mtaufen): Consider not exposing sep as an option, and just use = by default and
// manually implement around the couple edge cases (langle_separated_map_string_string.go)

// MapStringString can be set from the command line with the format --flag "string=string".
// Multiple flag invocations are supported. For example: --flag "a=foo" --flag "b=bar". If this is desired
// to be the only type invocation NoSplit should be set to true.
// Multiple comma-separated key-value pairs in a single invocation are supported if NoSplit
// is set to false. For example: --flag "a=foo,b=bar".
type MapStringString struct {
	Map         *map[string]string
	initialized bool
	NoSplit     bool
	sep         string
}

// NewMapStringString takes a pointer to a map[string]string and returns the
// MapStringString flag parsing shim for that map.
func NewMapStringString(m *map[string]string, sep string) *MapStringString {
	return &MapStringString{Map: m, sep: sep}
}

// TODO(mtaufen): figure out how we want to handle "NoSplit".
// Do we want to provide a separate Var constructor for it?
// Do we want to give it a clearer name?
// Do we really want it to be public in the MapStringString struct?

// NewMapStringStringNoSplit takes a pointer to a map[string]string and sets NoSplit
// value to true and returns the MapStringString flag parsing shim for that map.
func NewMapStringStringNoSplit(m *map[string]string, sep string) *MapStringString {
	return &MapStringString{
		Map:     m,
		NoSplit: true,
		sep:     sep,
	}
}

// String implements github.com/spf13/pflag.Value
func (m *MapStringString) String() string {
	if m == nil || m.Map == nil {
		return ""
	}
	pairs := []string{}
	for k, v := range *m.Map {
		pairs = append(pairs, fmt.Sprintf("%s%s%s", k, m.sep, v))
	}
	sort.Strings(pairs)
	return strings.Join(pairs, ",")
}

// Set implements github.com/spf13/pflag.Value
func (m *MapStringString) Set(value string) error {
	if m.Map == nil {
		return fmt.Errorf("no target (nil pointer to map[string]string)")
	}
	if !m.initialized || *m.Map == nil {
		// clear default values, or allocate if no existing map
		*m.Map = make(map[string]string)
		m.initialized = true
	}

	// account for comma-separated key-value pairs in a single invocation
	if !m.NoSplit {
		for _, s := range strings.Split(value, ",") {
			if len(s) == 0 {
				continue
			}
			arr := strings.SplitN(s, m.sep, 2)
			if len(arr) != 2 {
				return fmt.Errorf("malformed pair, expect string=string")
			}
			k := strings.TrimSpace(arr[0])
			v := strings.TrimSpace(arr[1])
			(*m.Map)[k] = v
		}
		return nil
	}

	// account for only one key-value pair in a single invocation
	arr := strings.SplitN(value, m.sep, 2)
	if len(arr) != 2 {
		return fmt.Errorf("malformed pair, expect string=string")
	}
	k := strings.TrimSpace(arr[0])
	v := strings.TrimSpace(arr[1])
	(*m.Map)[k] = v
	return nil

}

// Type implements github.com/spf13/pflag.Value
func (*MapStringString) Type() string {
	return "MapStringString"
}

// Empty implements OmitEmpty
func (m *MapStringString) Empty() bool {
	return len(*m.Map) == 0
}
