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

package event

import (
	"context"
	"fmt"
	"reflect"
	"sync"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type Manager struct {
	object.Common

	eventCategory   map[string]string
	eventCategoryMu *sync.Mutex
	maxObjects      int
}

func NewManager(c *vim25.Client) *Manager {
	m := Manager{
		Common: object.NewCommon(c, *c.ServiceContent.EventManager),

		eventCategory:   make(map[string]string),
		eventCategoryMu: new(sync.Mutex),
		maxObjects:      10,
	}

	return &m
}

func (m Manager) CreateCollectorForEvents(ctx context.Context, filter types.EventFilterSpec) (*HistoryCollector, error) {
	req := types.CreateCollectorForEvents{
		This:   m.Common.Reference(),
		Filter: filter,
	}

	res, err := methods.CreateCollectorForEvents(ctx, m.Client(), &req)
	if err != nil {
		return nil, err
	}

	return NewHistoryCollector(m.Client(), res.Returnval), nil
}

func (m Manager) LogUserEvent(ctx context.Context, entity types.ManagedObjectReference, msg string) error {
	req := types.LogUserEvent{
		This:   m.Common.Reference(),
		Entity: entity,
		Msg:    msg,
	}

	_, err := methods.LogUserEvent(ctx, m.Client(), &req)
	if err != nil {
		return err
	}

	return nil
}

func (m Manager) PostEvent(ctx context.Context, eventToPost types.BaseEvent, taskInfo types.TaskInfo) error {
	req := types.PostEvent{
		This:        m.Common.Reference(),
		EventToPost: eventToPost,
		TaskInfo:    &taskInfo,
	}

	_, err := methods.PostEvent(ctx, m.Client(), &req)
	if err != nil {
		return err
	}

	return nil
}

func (m Manager) QueryEvents(ctx context.Context, filter types.EventFilterSpec) ([]types.BaseEvent, error) {
	req := types.QueryEvents{
		This:   m.Common.Reference(),
		Filter: filter,
	}

	res, err := methods.QueryEvents(ctx, m.Client(), &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

func (m Manager) RetrieveArgumentDescription(ctx context.Context, eventTypeID string) ([]types.EventArgDesc, error) {
	req := types.RetrieveArgumentDescription{
		This:        m.Common.Reference(),
		EventTypeId: eventTypeID,
	}

	res, err := methods.RetrieveArgumentDescription(ctx, m.Client(), &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

func (m Manager) eventCategoryMap(ctx context.Context) (map[string]string, error) {
	m.eventCategoryMu.Lock()
	defer m.eventCategoryMu.Unlock()

	if len(m.eventCategory) != 0 {
		return m.eventCategory, nil
	}

	var o mo.EventManager

	ps := []string{"description.eventInfo"}
	err := property.DefaultCollector(m.Client()).RetrieveOne(ctx, m.Common.Reference(), ps, &o)
	if err != nil {
		return nil, err
	}

	for _, info := range o.Description.EventInfo {
		m.eventCategory[info.Key] = info.Category
	}

	return m.eventCategory, nil
}

// EventCategory returns the category for an event, such as "info" or "error" for example.
func (m Manager) EventCategory(ctx context.Context, event types.BaseEvent) (string, error) {
	// Most of the event details are included in the Event.FullFormattedMessage, but the category
	// is only available via the EventManager description.eventInfo property.  The value of this
	// property is static, so we fetch and once and cache.
	eventCategory, err := m.eventCategoryMap(ctx)
	if err != nil {
		return "", err
	}

	class := reflect.TypeOf(event).Elem().Name()

	return eventCategory[class], nil
}

// Get the events from the specified object(s) and optionanlly tail the event stream
func (m Manager) Events(ctx context.Context, objects []types.ManagedObjectReference, pageSize int32, tail bool, force bool, f func(types.ManagedObjectReference, []types.BaseEvent) error) error {

	if len(objects) >= m.maxObjects && !force {
		return fmt.Errorf("Maximum number of objects to monitor (%d) exceeded, refine search", m.maxObjects)
	}

	proc := newEventProcessor(m, pageSize, f)
	for _, o := range objects {
		proc.addObject(ctx, o)
	}

	return proc.run(ctx, tail)
}
