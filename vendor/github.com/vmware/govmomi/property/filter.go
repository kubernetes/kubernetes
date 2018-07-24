/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package property

import (
	"fmt"
	"path"
	"reflect"
	"strconv"
	"strings"

	"github.com/vmware/govmomi/vim25/types"
)

// Filter provides methods for matching against types.DynamicProperty
type Filter map[string]types.AnyType

// Keys returns the Filter map keys as a []string
func (f Filter) Keys() []string {
	keys := make([]string, 0, len(f))

	for key := range f {
		keys = append(keys, key)
	}

	return keys
}

// MatchProperty returns true if a Filter entry matches the given prop.
func (f Filter) MatchProperty(prop types.DynamicProperty) bool {
	match, ok := f[prop.Name]
	if !ok {
		return false
	}

	if match == prop.Val {
		return true
	}

	ptype := reflect.TypeOf(prop.Val)

	if strings.HasPrefix(ptype.Name(), "ArrayOf") {
		pval := reflect.ValueOf(prop.Val).Field(0)

		for i := 0; i < pval.Len(); i++ {
			prop.Val = pval.Index(i).Interface()

			if f.MatchProperty(prop) {
				return true
			}
		}

		return false
	}

	if reflect.TypeOf(match) != ptype {
		s, ok := match.(string)
		if !ok {
			return false
		}

		// convert if we can
		switch prop.Val.(type) {
		case bool:
			match, _ = strconv.ParseBool(s)
		case int16:
			x, _ := strconv.ParseInt(s, 10, 16)
			match = int16(x)
		case int32:
			x, _ := strconv.ParseInt(s, 10, 32)
			match = int32(x)
		case int64:
			match, _ = strconv.ParseInt(s, 10, 64)
		case float32:
			x, _ := strconv.ParseFloat(s, 32)
			match = float32(x)
		case float64:
			match, _ = strconv.ParseFloat(s, 64)
		case fmt.Stringer:
			prop.Val = prop.Val.(fmt.Stringer).String()
		default:
			if ptype.Kind() != reflect.String {
				return false
			}
			// An enum type we can convert to a string type
			prop.Val = reflect.ValueOf(prop.Val).String()
		}
	}

	switch pval := prop.Val.(type) {
	case string:
		s := match.(string)
		if s == "*" {
			return true // TODO: path.Match fails if s contains a '/'
		}
		m, _ := path.Match(s, pval)
		return m
	default:
		return reflect.DeepEqual(match, pval)
	}
}

// MatchPropertyList returns true if all given props match the Filter.
func (f Filter) MatchPropertyList(props []types.DynamicProperty) bool {
	for _, p := range props {
		if !f.MatchProperty(p) {
			return false
		}
	}

	return len(f) == len(props) // false if a property such as VM "guest" is unset
}

// MatchObjectContent returns a list of ObjectContent.Obj where the ObjectContent.PropSet matches the Filter.
func (f Filter) MatchObjectContent(objects []types.ObjectContent) []types.ManagedObjectReference {
	var refs []types.ManagedObjectReference

	for _, o := range objects {
		if f.MatchPropertyList(o.PropSet) {
			refs = append(refs, o.Obj)
		}
	}

	return refs
}
