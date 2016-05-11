/*
Copyright (c) 2016 VMware, Inc. All Rights Reserved.

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

package event

import (
	"fmt"

	"golang.org/x/net/context"

	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/view"
	"github.com/vmware/govmomi/vim25/types"
)

func multipleObjectEvents(ctx context.Context, m Manager, objects []types.ManagedObjectReference, pageSize int32, tail bool, force bool, prop []string, f func([]types.BaseEvent) error) error {
	// create an EventHistoryCollector for each object
	var collectors []types.ManagedObjectReference
	for _, o := range objects {
		filter := types.EventFilterSpec{
			Entity: &types.EventFilterSpecByEntity{
				Entity:    o,
				Recursion: types.EventFilterSpecRecursionOptionAll,
			},
		}

		collector, err := m.CreateCollectorForEvents(ctx, filter)
		if err != nil {
			return fmt.Errorf("[%#v] %s", o, err)
		}
		defer collector.Destroy(ctx)

		err = collector.SetPageSize(ctx, pageSize)
		if err != nil {
			return err
		}

		collectors = append(collectors, collector.Reference())
	}

	// create and populate a ListView
	viewMgr := view.NewManager(m.Client())
	listView, err := viewMgr.CreateListView(ctx, collectors)
	if err != nil {
		return err
	}
	count := 0
	// Retrieve the property from the objects in the ListView
	return property.WaitForView(ctx, property.DefaultCollector(m.Client()), listView.Reference(), collectors[0], prop, func(pc []types.PropertyChange) bool {
		for _, u := range pc {
			if u.Name != prop[0] {
				continue
			}
			if u.Val == nil {
				continue
			}
			f(u.Val.(types.ArrayOfEvent).Event)
		}
		count++
		if count == len(collectors) && !tail {
			return true
		}
		return false
	})

}

func singleObjectEvents(ctx context.Context, m Manager, object types.ManagedObjectReference, pageSize int32, tail bool, force bool, prop []string, f func([]types.BaseEvent) error) error {
	filter := types.EventFilterSpec{
		Entity: &types.EventFilterSpecByEntity{
			Entity:    object,
			Recursion: types.EventFilterSpecRecursionOptionAll,
		},
	}

	collector, err := m.CreateCollectorForEvents(ctx, filter)
	if err != nil {
		return fmt.Errorf("[%#v] %s", object, err)
	}
	defer collector.Destroy(ctx)

	err = collector.SetPageSize(ctx, pageSize)
	if err != nil {
		return err
	}

	return property.Wait(ctx, property.DefaultCollector(m.Client()), collector.Reference(), prop, func(pc []types.PropertyChange) bool {
		for _, u := range pc {
			if u.Name != prop[0] {
				continue
			}
			if u.Val == nil {
				continue
			}
			f(u.Val.(types.ArrayOfEvent).Event)

		}
		if !tail {
			return true
		}
		return false
	})
}
