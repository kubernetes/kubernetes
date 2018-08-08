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

package simulator

import (
	"reflect"
	"strings"

	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type PropertyFilter struct {
	mo.PropertyFilter

	pc   *PropertyCollector
	refs map[types.ManagedObjectReference]struct{}
}

func (f *PropertyFilter) DestroyPropertyFilter(ctx *Context, c *types.DestroyPropertyFilter) soap.HasFault {
	body := &methods.DestroyPropertyFilterBody{}

	RemoveReference(&f.pc.Filter, c.This)

	ctx.Session.Remove(c.This)

	body.Res = &types.DestroyPropertyFilterResponse{}

	return body
}

// matches returns true if the change matches one of the filter Spec.PropSet
func (f *PropertyFilter) matches(ctx *Context, ref types.ManagedObjectReference, change *types.PropertyChange) bool {
	for _, p := range f.Spec.PropSet {
		if p.Type != ref.Type {
			continue
		}

		if isTrue(p.All) {
			return true
		}

		for _, name := range p.PathSet {
			if name == change.Name {
				return true
			}

			// strings.HasPrefix("runtime.powerState", "runtime") == parent field matches
			if strings.HasPrefix(change.Name, name) {
				if obj := ctx.Map.Get(ref); obj != nil { // object may have since been deleted
					change.Name = name
					change.Val, _ = fieldValue(reflect.ValueOf(obj), name)
				}

				return true
			}
		}
	}

	return false
}

// apply the PropertyFilter.Spec to the given ObjectUpdate
func (f *PropertyFilter) apply(ctx *Context, change types.ObjectUpdate) types.ObjectUpdate {
	parents := make(map[string]bool)
	set := change.ChangeSet
	change.ChangeSet = nil

	for i, p := range set {
		if f.matches(ctx, change.Obj, &p) {
			if p.Name != set[i].Name {
				// update matches a parent field from the spec.
				if parents[p.Name] {
					continue // only return 1 instance of the parent
				}
				parents[p.Name] = true
			}
			change.ChangeSet = append(change.ChangeSet, p)
		}
	}

	return change
}
