// Copyright 2016 The rkt Authors
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

package flag

import (
	"fmt"
	"sort"
	"strings"
)

// PairList is a flag value supporting a list of key=value pairs
// It will optionally validate the supplied keys and / or values
// It would parse something like "--arg foo=bar --arg baz=bar"
type PairList struct {
	Pairs       map[string]string
	permissible map[string](map[string]struct{}) // map of allowed keys and, optionally, allowed values
	typeName    string
}

// NewPairList initializes a new pair list
// If permissiblePairs is not empty, then it will validate keys and values. For
// every key in permissiblePairs, if the value is an empty list, then any value
// is allowed. If there is a list, then only those values are allowed for that.
// for example, { "a": {"1", "two", "cheese"}, "b" : {}} would accept
// a=1 and any b, but no other keys
// defaults is a list of default parameters
func NewPairList(permissiblePairs map[string][]string, defaults map[string]string) (*PairList, error) {
	pl := &PairList{
		Pairs:       make(map[string]string),
		permissible: make(map[string]map[string]struct{}),
		typeName:    "PairList",
	}

	// invert the list of allowed vals for fast lookup
	for key, vals := range permissiblePairs {
		pl.permissible[key] = make(map[string]struct{})

		for _, val := range vals {
			pl.permissible[key][val] = struct{}{}
		}
	}

	for key, val := range defaults {
		err := pl.SetOne(key, val)
		if err != nil {
			return nil, err
		}
	}

	return pl, nil
}

// MustNewPairList is the same as NewPairList, but panics instead of returning error
// So check that your defaults are correct :-)
func MustNewPairList(permissiblePairs map[string][]string, defaults map[string]string) *PairList {
	pl, err := NewPairList(permissiblePairs, defaults)
	if err != nil {
		panic(err)
	}
	return pl
}

// Set parses a "k=v[,kk=vv...]" string
func (pl *PairList) Set(s string) error {
	entries := strings.Split(s, ",")

	for _, e := range entries {
		p := strings.SplitN(e, "=", 2)
		if len(p) != 2 {
			return fmt.Errorf("could not parse key=value pair %v", s)
		}
		key := p[0]
		val := p[1]
		err := pl.SetOne(key, val)
		if err != nil {
			return err
		}

	}
	return nil
}

// SetOne validates and sets an individual key-value pair
// It will overwrite an existing value
func (pl *PairList) SetOne(key, val string) error {
	// Check that key is allowed
	if len(pl.permissible) > 0 {
		permVals, ok := pl.permissible[key]
		if !ok {
			return fmt.Errorf("key %v is not allowed", key)
		}

		// Check that value is allowed
		if len(permVals) > 0 {
			_, ok = permVals[val]
			if !ok {
				return fmt.Errorf("key %v does not allow value %v", key, val)
			}
		}
	}

	pl.Pairs[key] = val
	return nil
}

// Keys returns a sorted list of all present keys
func (pl *PairList) Keys() []string {
	keys := make([]string, 0, len(pl.Pairs))
	for k := range pl.Pairs {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// String presents a k=v list in key=sorted order
func (pl *PairList) String() string {
	ps := make([]string, 0, len(pl.Pairs))
	for _, k := range pl.Keys() {
		ps = append(ps, fmt.Sprintf("%v=%v", k, pl.Pairs[k]))
	}
	return strings.Join(ps, " ")
}

func (pl *PairList) Type() string {
	return pl.typeName
}

// PermissibleString generates a posix-cli-doc style string.
// For example, a=[b|c|d] w=[x|y|z] l=*
func (pl *PairList) PermissibleString() string {
	ps := make([]string, 0, len(pl.permissible))

	keys := make([]string, 0, len(pl.permissible))
	for k := range pl.permissible {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		vss := pl.permissible[k]
		//re-invert the index
		vs := make([]string, 0, len(vss))
		for v := range vss {
			vs = append(vs, v)
		}
		sort.Strings(vs)

		vstr := "*"
		if len(vs) > 0 {
			vstr = "[" + strings.Join(vs, "|") + "]"
		}
		ps = append(ps, fmt.Sprintf("%v=%v", k, vstr))
	}

	return strings.Join(ps, " ")
}

// Serialize takes a map and generates a string that can be parsed by Set
func SerializePairs(pairs map[string]string) string {
	tmp := make([]string, 0, len(pairs))

	for k, v := range pairs {
		tmp = append(tmp, fmt.Sprintf("%s=%s", k, v))
	}
	return strings.Join(tmp, ",")
}
