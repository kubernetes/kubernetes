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
	"strings"

	"github.com/spf13/pflag"
)

// MapStringStringValue is a reference to a registered map[string]string flag value.
type MapStringStringValue struct {
	name  string
	value map[string]string
	fs    *pflag.FlagSet
}

// MapStringStringVar registers a flag for map[string]string against the FlagSet,
// and returns a MapStringStringValue reference to the registered flag value.
// Format: `--flag "string=string"`.
// Multiple comma-separated key-value pairs in a single invocation are supported,
// when MapOptions.DisableBatch=false.
// For example: `--flag "a=foo,b=bar"`.
// Multiple flag invocations are supported.
// For example: `--flag "a=foo" --flag "b=bar"`.
func (fs *FlagSet) MapStringStringVar(name string, def map[string]string, usage string, options *MapOptions) *MapStringStringValue {
	val := &MapStringStringValue{
		name:  name,
		value: make(map[string]string),
		fs:    fs.fs,
	}
	for k, v := range def {
		val.value[k] = v
	}
	fs.fs.Var(newMapStringString(&val.value, options), name, usage)
	return val
}

// Set copies the map over the target if the flag was set.
// It completely overwrites any existing target.
func (v *MapStringStringValue) Set(target *map[string]string) {
	if v.fs.Changed(v.name) {
		*target = make(map[string]string)
		for k, v := range v.value {
			(*target)[k] = v
		}
	}
}

// Merge copies the map keys/values piecewise into the target if the flag
// was set. Values in the flag's map override values for corresponding
// keys in the target map.
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

// Apply calls the user-provided apply function with the map if the flag was set.
func (v *MapStringStringValue) Apply(apply func(value map[string]string)) {
	if v.fs.Changed(v.name) {
		apply(v.value)
	}
}

// mapStringString implements plfag.Value for map[string]string
type mapStringString struct {
	m           *map[string]string
	initialized bool
	options     *MapOptions
}

// newMapStringString takes a pointer to a map[string]string and returns the
// mapStringString flag parsing shim for that map.
func newMapStringString(m *map[string]string, o *MapOptions) *mapStringString {
	o.Default()
	return &mapStringString{m: m, options: o}
}

// String implements github.com/spf13/pflag.Value
func (m *mapStringString) String() string {
	if m == nil || m.m == nil {
		return ""
	}
	pairs := []string{}
	for k, v := range *m.m {
		pairs = append(pairs, fmt.Sprintf("%s%s%s", k, m.options.KeyValueSep, v))
	}
	sort.Strings(pairs)
	return strings.Join(pairs, m.options.PairSep)
}

// Set implements github.com/spf13/pflag.Value
func (m *mapStringString) Set(value string) error {
	if m.m == nil {
		return fmt.Errorf("no target (nil pointer to map[string]string)")
	}
	if !m.initialized || *m.m == nil {
		// clear default values, or allocate if no existing map
		*m.m = make(map[string]string)
		m.initialized = true
	}

	// account for multiple key-value pairs in a single invocation
	if !m.options.DisableCommaSeparatedPairs {
		for _, s := range strings.Split(value, m.options.PairSep) {
			if len(s) == 0 {
				continue
			}
			arr := strings.SplitN(s, m.options.KeyValueSep, 2)
			if len(arr) != 2 {
				return fmt.Errorf("malformed pair, expect string%sstring", m.options.KeyValueSep)
			}
			k := strings.TrimSpace(arr[0])
			v := strings.TrimSpace(arr[1])
			(*m.m)[k] = v
		}
		return nil
	}

	// account for only one key-value pair in a single invocation
	arr := strings.SplitN(value, m.options.KeyValueSep, 2)
	if len(arr) != 2 {
		return fmt.Errorf("malformed pair, expect string%sstring", m.options.KeyValueSep)
	}
	k := strings.TrimSpace(arr[0])
	v := strings.TrimSpace(arr[1])
	(*m.m)[k] = v
	return nil

}

// Type implements github.com/spf13/pflag.Value
func (*mapStringString) Type() string {
	return "mapStringString"
}

// Empty implements OmitEmpty
func (m *mapStringString) Empty() bool {
	return len(*m.m) == 0
}
