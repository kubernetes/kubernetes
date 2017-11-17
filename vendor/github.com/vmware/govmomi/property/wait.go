/*
Copyright (c) 2015-2017 VMware, Inc. All Rights Reserved.

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
	"context"

	"github.com/vmware/govmomi/vim25/types"
)

// WaitFilter provides helpers to construct a types.CreateFilter for use with property.Wait
type WaitFilter struct {
	types.CreateFilter
}

// Add a new ObjectSpec and PropertySpec to the WaitFilter
func (f *WaitFilter) Add(obj types.ManagedObjectReference, kind string, ps []string, set ...types.BaseSelectionSpec) *WaitFilter {
	spec := types.ObjectSpec{
		Obj:       obj,
		SelectSet: set,
	}

	pset := types.PropertySpec{
		Type:    kind,
		PathSet: ps,
	}

	if len(ps) == 0 {
		pset.All = types.NewBool(true)
	}

	f.Spec.ObjectSet = append(f.Spec.ObjectSet, spec)

	f.Spec.PropSet = append(f.Spec.PropSet, pset)

	return f
}

// Wait creates a new WaitFilter and calls the specified function for each ObjectUpdate via WaitForUpdates
func Wait(ctx context.Context, c *Collector, obj types.ManagedObjectReference, ps []string, f func([]types.PropertyChange) bool) error {
	filter := new(WaitFilter).Add(obj, obj.Type, ps)

	return WaitForUpdates(ctx, c, filter, func(updates []types.ObjectUpdate) bool {
		for _, update := range updates {
			if f(update.ChangeSet) {
				return true
			}
		}

		return false
	})
}

// WaitForUpdates waits for any of the specified properties of the specified managed
// object to change. It calls the specified function for every update it
// receives. If this function returns false, it continues waiting for
// subsequent updates. If this function returns true, it stops waiting and
// returns.
//
// To only receive updates for the specified managed object, the function
// creates a new property collector and calls CreateFilter. A new property
// collector is required because filters can only be added, not removed.
//
// The newly created collector is destroyed before this function returns (both
// in case of success or error).
//
func WaitForUpdates(ctx context.Context, c *Collector, filter *WaitFilter, f func([]types.ObjectUpdate) bool) error {
	p, err := c.Create(ctx)
	if err != nil {
		return err
	}

	// Attempt to destroy the collector using the background context, as the
	// specified context may have timed out or have been cancelled.
	defer p.Destroy(context.Background())

	err = p.CreateFilter(ctx, filter.CreateFilter)
	if err != nil {
		return err
	}

	for version := ""; ; {
		res, err := p.WaitForUpdates(ctx, version)
		if err != nil {
			return err
		}

		// Retry if the result came back empty
		if res == nil {
			continue
		}

		version = res.Version

		for _, fs := range res.FilterSet {
			if f(fs.ObjectSet) {
				return nil
			}
		}
	}
}
