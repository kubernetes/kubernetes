/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package mo

import (
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"
)

// Ancestors returns the entire ancestry tree of a specified managed object.
// The return value includes the root node and the specified object itself.
func Ancestors(ctx context.Context, rt soap.RoundTripper, pc, obj types.ManagedObjectReference) ([]ManagedEntity, error) {
	ospec := types.ObjectSpec{
		Obj: obj,
		SelectSet: []types.BaseSelectionSpec{
			&types.TraversalSpec{
				SelectionSpec: types.SelectionSpec{Name: "traverseParent"},
				Type:          "ManagedEntity",
				Path:          "parent",
				Skip:          types.NewBool(false),
				SelectSet: []types.BaseSelectionSpec{
					&types.SelectionSpec{Name: "traverseParent"},
				},
			},
		},
		Skip: types.NewBool(false),
	}

	pspec := types.PropertySpec{
		Type:    "ManagedEntity",
		PathSet: []string{"name", "parent"},
	}

	req := types.RetrieveProperties{
		This: pc,
		SpecSet: []types.PropertyFilterSpec{
			{
				ObjectSet: []types.ObjectSpec{ospec},
				PropSet:   []types.PropertySpec{pspec},
			},
		},
	}

	var ifaces []interface{}

	err := RetrievePropertiesForRequest(ctx, rt, req, &ifaces)
	if err != nil {
		return nil, err
	}

	var out []ManagedEntity

	// Build ancestry tree by iteratively finding a new child.
	for len(out) < len(ifaces) {
		var find types.ManagedObjectReference

		if len(out) > 0 {
			find = out[len(out)-1].Self
		}

		// Find entity we're looking for given the last entity in the current tree.
		for _, iface := range ifaces {
			me := iface.(IsManagedEntity).GetManagedEntity()
			if me.Parent == nil {
				out = append(out, me)
				break
			}

			if *me.Parent == find {
				out = append(out, me)
				break
			}
		}
	}

	return out, nil
}
