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

package legacyflag

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/spf13/pflag"
)

// MapStringBoolValue is a reference to a registered map[string]bool flag value.
type MapStringBoolValue struct {
	name  string
	value map[string]bool
	fs    *pflag.FlagSet
}

// MapStringBoolVar registers a flag for map[string]bool against the FlagSet,
// and returns a MapStringBoolValue reference to the registered flag value.
// Format: `--flag "string=bool"`.
// Multiple comma-separated key-value pairs in a single invocation are supported.
// Example usage: `--flag "a=true,b=false"`.
// Multiple flag invocations are supported.
// Example usage: `--flag "a=true" --flag "b=false"`.
func (fs *FlagSet) MapStringBoolVar(name string, def map[string]bool, usage string, options *MapOptions) *MapStringBoolValue {
	val := &MapStringBoolValue{
		name:  name,
		value: make(map[string]bool),
		fs:    fs.fs,
	}
	for k, v := range def {
		val.value[k] = v
	}
	fs.fs.Var(newMapStringBool(&val.value, options), name, usage)
	return val
}

// Set copies the map over the target if the flag was set.
// It completely overwrites any existing target.
func (v *MapStringBoolValue) Set(target *map[string]bool) {
	if v.fs.Changed(v.name) {
		*target = make(map[string]bool)
		for k, v := range v.value {
			(*target)[k] = v
		}
	}
}

// Merge copies the map keys/values piecewise into the target if the flag
// was set. Values in the flag's map override values for corresponding
// keys in the target map.
func (v *MapStringBoolValue) Merge(target *map[string]bool) {
	if v.fs.Changed(v.name) {
		if *target == nil {
			*target = make(map[string]bool)
		}
		for k, v := range v.value {
			(*target)[k] = v
		}
	}
}

// Apply calls the user-provided apply function with the map if the flag was set.
func (v *MapStringBoolValue) Apply(apply func(value map[string]bool)) {
	if v.fs.Changed(v.name) {
		apply(v.value)
	}
}

// mapStringBool implements pflag.Value for map[string]bool
type mapStringBool struct {
	m           *map[string]bool
	initialized bool
	options     *MapOptions
}

// newMapStringBool takes a pointer to a map[string]string and returns the
// mapStringBool flag parsing shim for that map
func newMapStringBool(m *map[string]bool, o *MapOptions) *mapStringBool {
	o.Default()
	return &mapStringBool{m: m, options: o}
}

// String implements github.com/spf13/pflag.Value
func (m *mapStringBool) String() string {
	if m == nil || m.m == nil {
		return ""
	}
	pairs := []string{}
	for k, v := range *m.m {
		pairs = append(pairs, fmt.Sprintf("%s%s%t", k, m.options.KeyValueSep, v))
	}
	sort.Strings(pairs)
	return strings.Join(pairs, m.options.PairSep)
}

// Set implements github.com/spf13/pflag.Value
func (m *mapStringBool) Set(value string) error {
	if m.m == nil {
		return fmt.Errorf("no target (nil pointer to map[string]bool)")
	}
	if !m.initialized || *m.m == nil {
		// clear default values, or allocate if no existing map
		*m.m = make(map[string]bool)
		m.initialized = true
	}
	if !m.options.DisableCommaSeparatedPairs {
		for _, s := range strings.Split(value, m.options.PairSep) {
			if len(s) == 0 {
				continue
			}
			arr := strings.SplitN(s, m.options.KeyValueSep, 2)
			if len(arr) != 2 {
				return fmt.Errorf("malformed pair, expect string%sbool", m.options.KeyValueSep)
			}
			k := strings.TrimSpace(arr[0])
			v := strings.TrimSpace(arr[1])
			boolValue, err := strconv.ParseBool(v)
			if err != nil {
				return fmt.Errorf("invalid value of %s: %s, err: %v", k, v, err)
			}
			(*m.m)[k] = boolValue
		}
		return nil
	}

	// account for only one key-value pair in a single invocation
	arr := strings.SplitN(value, m.options.KeyValueSep, 2)
	if len(arr) != 2 {
		return fmt.Errorf("malformed pair, expect string%sbool", m.options.KeyValueSep)
	}
	k := strings.TrimSpace(arr[0])
	v := strings.TrimSpace(arr[1])
	boolValue, err := strconv.ParseBool(v)
	if err != nil {
		return fmt.Errorf("invalid value of %s: %s, err: %v", k, v, err)
	}
	(*m.m)[k] = boolValue

	return nil
}

// Type implements github.com/spf13/pflag.Value
func (*mapStringBool) Type() string {
	return "mapStringBool"
}

// Empty implements OmitEmpty
func (m *mapStringBool) Empty() bool {
	return len(*m.m) == 0
}
