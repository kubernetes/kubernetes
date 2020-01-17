// Copyright 2018 The etcd Authors
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

package flags

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

// SelectiveStringValue implements the flag.Value interface.
type SelectiveStringValue struct {
	v      string
	valids map[string]struct{}
}

// Set verifies the argument to be a valid member of the allowed values
// before setting the underlying flag value.
func (ss *SelectiveStringValue) Set(s string) error {
	if _, ok := ss.valids[s]; ok {
		ss.v = s
		return nil
	}
	return errors.New("invalid value")
}

// String returns the set value (if any) of the SelectiveStringValue
func (ss *SelectiveStringValue) String() string {
	return ss.v
}

// Valids returns the list of valid strings.
func (ss *SelectiveStringValue) Valids() []string {
	s := make([]string, 0, len(ss.valids))
	for k := range ss.valids {
		s = append(s, k)
	}
	sort.Strings(s)
	return s
}

// NewSelectiveStringValue creates a new string flag
// for which any one of the given strings is a valid value,
// and any other value is an error.
//
// valids[0] will be default value. Caller must be sure
// len(valids) != 0 or it will panic.
func NewSelectiveStringValue(valids ...string) *SelectiveStringValue {
	vm := make(map[string]struct{})
	for _, v := range valids {
		vm[v] = struct{}{}
	}
	return &SelectiveStringValue{valids: vm, v: valids[0]}
}

// SelectiveStringsValue implements the flag.Value interface.
type SelectiveStringsValue struct {
	vs     []string
	valids map[string]struct{}
}

// Set verifies the argument to be a valid member of the allowed values
// before setting the underlying flag value.
func (ss *SelectiveStringsValue) Set(s string) error {
	vs := strings.Split(s, ",")
	for i := range vs {
		if _, ok := ss.valids[vs[i]]; ok {
			ss.vs = append(ss.vs, vs[i])
		} else {
			return fmt.Errorf("invalid value %q", vs[i])
		}
	}
	sort.Strings(ss.vs)
	return nil
}

// String returns the set value (if any) of the SelectiveStringsValue.
func (ss *SelectiveStringsValue) String() string {
	return strings.Join(ss.vs, ",")
}

// Valids returns the list of valid strings.
func (ss *SelectiveStringsValue) Valids() []string {
	s := make([]string, 0, len(ss.valids))
	for k := range ss.valids {
		s = append(s, k)
	}
	sort.Strings(s)
	return s
}

// NewSelectiveStringsValue creates a new string slice flag
// for which any one of the given strings is a valid value,
// and any other value is an error.
func NewSelectiveStringsValue(valids ...string) *SelectiveStringsValue {
	vm := make(map[string]struct{})
	for _, v := range valids {
		vm[v] = struct{}{}
	}
	return &SelectiveStringsValue{valids: vm, vs: []string{}}
}
