/*
Copyright (c) 2016-2017 VMware, Inc. All Rights Reserved.

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
	"context"
	"fmt"

	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/view"
	"github.com/vmware/govmomi/vim25/types"
)

type tailInfo struct {
	t         *eventTailer
	obj       types.ManagedObjectReference
	collector *HistoryCollector
}

type eventProcessor struct {
	mgr      Manager
	pageSize int32
	kind     []string
	tailers  map[types.ManagedObjectReference]*tailInfo // tailers by collector ref
	callback func(types.ManagedObjectReference, []types.BaseEvent) error
}

func newEventProcessor(mgr Manager, pageSize int32, callback func(types.ManagedObjectReference, []types.BaseEvent) error, kind []string) *eventProcessor {
	return &eventProcessor{
		mgr:      mgr,
		tailers:  make(map[types.ManagedObjectReference]*tailInfo),
		callback: callback,
		pageSize: pageSize,
		kind:     kind,
	}
}

func (p *eventProcessor) addObject(ctx context.Context, obj types.ManagedObjectReference) error {
	filter := types.EventFilterSpec{
		Entity: &types.EventFilterSpecByEntity{
			Entity:    obj,
			Recursion: types.EventFilterSpecRecursionOptionAll,
		},
		EventTypeId: p.kind,
	}

	collector, err := p.mgr.CreateCollectorForEvents(ctx, filter)
	if err != nil {
		return fmt.Errorf("[%#v] %s", obj, err)
	}

	err = collector.SetPageSize(ctx, p.pageSize)
	if err != nil {
		return err
	}

	p.tailers[collector.Reference()] = &tailInfo{
		t:         newEventTailer(),
		obj:       obj,
		collector: collector,
	}

	return nil
}

func (p *eventProcessor) destroy() {
	for _, info := range p.tailers {
		_ = info.collector.Destroy(context.Background())
	}
}

func (p *eventProcessor) run(ctx context.Context, tail bool) error {
	if len(p.tailers) == 0 {
		return nil
	}

	var collectors []types.ManagedObjectReference
	for ref := range p.tailers {
		collectors = append(collectors, ref)
	}

	c := property.DefaultCollector(p.mgr.Client())
	props := []string{"latestPage"}

	if len(collectors) == 1 {
		// only one object to follow, don't bother creating a view
		return property.Wait(ctx, c, collectors[0], props, func(pc []types.PropertyChange) bool {
			if err := p.process(collectors[0], pc); err != nil {
				return false
			}

			return !tail
		})
	}

	// create and populate a ListView
	m := view.NewManager(p.mgr.Client())

	list, err := m.CreateListView(ctx, collectors)
	if err != nil {
		return err
	}

	defer list.Destroy(context.Background())

	ref := list.Reference()
	filter := new(property.WaitFilter).Add(ref, collectors[0].Type, props, list.TraversalSpec())

	return property.WaitForUpdates(ctx, c, filter, func(updates []types.ObjectUpdate) bool {
		for _, update := range updates {
			if err := p.process(update.Obj, update.ChangeSet); err != nil {
				return false
			}
		}

		return !tail
	})
}

func (p *eventProcessor) process(c types.ManagedObjectReference, pc []types.PropertyChange) error {
	t := p.tailers[c]
	if t == nil {
		return fmt.Errorf("unknown collector %s", c.String())
	}

	for _, u := range pc {
		evs := t.t.newEvents(u.Val.(types.ArrayOfEvent).Event)
		if len(evs) == 0 {
			continue
		}

		if err := p.callback(t.obj, evs); err != nil {
			return err
		}
	}

	return nil
}

const invalidKey = int32(-1)

type eventTailer struct {
	lastKey int32
}

func newEventTailer() *eventTailer {
	return &eventTailer{
		lastKey: invalidKey,
	}
}

func (t *eventTailer) newEvents(evs []types.BaseEvent) []types.BaseEvent {
	var ret []types.BaseEvent
	if t.lastKey == invalidKey {
		ret = evs
	} else {
		found := false
		for i := range evs {
			if evs[i].GetEvent().Key != t.lastKey {
				continue
			}

			found = true
			ret = evs[:i]
			break
		}

		if !found {
			ret = evs
		}
	}

	if len(ret) > 0 {
		t.lastKey = ret[0].GetEvent().Key
	}

	return ret
}
