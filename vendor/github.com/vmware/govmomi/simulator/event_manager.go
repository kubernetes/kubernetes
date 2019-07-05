/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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
	"bytes"
	"container/ring"
	"log"
	"reflect"
	"text/template"
	"time"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/simulator/esx"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

var (
	maxPageSize = 1000
	logEvents   = false
)

type EventManager struct {
	mo.EventManager

	root       types.ManagedObjectReference
	page       *ring.Ring
	key        int32
	collectors map[types.ManagedObjectReference]*EventHistoryCollector
	templates  map[string]*template.Template
}

func NewEventManager(ref types.ManagedObjectReference) object.Reference {
	return &EventManager{
		EventManager: mo.EventManager{
			Self: ref,
			Description: types.EventDescription{
				EventInfo: esx.EventInfo,
			},
			MaxCollector: 1000,
		},
		root:       Map.content().RootFolder,
		page:       ring.New(maxPageSize),
		collectors: make(map[types.ManagedObjectReference]*EventHistoryCollector),
		templates:  make(map[string]*template.Template),
	}
}

func (m *EventManager) createCollector(ctx *Context, req *types.CreateCollectorForEvents) (*EventHistoryCollector, *soap.Fault) {
	size, err := validatePageSize(req.Filter.MaxCount)
	if err != nil {
		return nil, err
	}

	if len(m.collectors) >= int(m.MaxCollector) {
		return nil, Fault("Too many event collectors to create", new(types.InvalidState))
	}

	collector := &EventHistoryCollector{
		m:    m,
		page: ring.New(size),
	}
	collector.Filter = req.Filter
	collector.fillPage(size)

	return collector, nil
}

func (m *EventManager) CreateCollectorForEvents(ctx *Context, req *types.CreateCollectorForEvents) soap.HasFault {
	body := new(methods.CreateCollectorForEventsBody)
	collector, err := m.createCollector(ctx, req)
	if err != nil {
		body.Fault_ = err
		return body
	}

	ref := ctx.Session.Put(collector).Reference()
	m.collectors[ref] = collector

	body.Res = &types.CreateCollectorForEventsResponse{
		Returnval: ref,
	}

	return body
}

func (m *EventManager) QueryEvents(ctx *Context, req *types.QueryEvents) soap.HasFault {
	if Map.IsESX() {
		return &methods.QueryEventsBody{
			Fault_: Fault("", new(types.NotImplemented)),
		}
	}

	body := new(methods.QueryEventsBody)
	collector, err := m.createCollector(ctx, &types.CreateCollectorForEvents{Filter: req.Filter})
	if err != nil {
		body.Fault_ = err
		return body
	}

	body.Res = &types.QueryEventsResponse{
		Returnval: collector.GetLatestPage(),
	}

	return body
}

// formatMessage applies the EventDescriptionEventDetail.FullFormat template to the given event's FullFormattedMessage field.
func (m *EventManager) formatMessage(event types.BaseEvent) {
	id := reflect.ValueOf(event).Elem().Type().Name()
	e := event.GetEvent()

	t, ok := m.templates[id]
	if !ok {
		for _, info := range m.Description.EventInfo {
			if info.Key == id {
				t = template.Must(template.New(id).Parse(info.FullFormat))
				m.templates[id] = t
				break
			}
		}
	}

	if t != nil {
		var buf bytes.Buffer
		if err := t.Execute(&buf, event); err != nil {
			log.Print(err)
		}
		e.FullFormattedMessage = buf.String()
	}

	if logEvents {
		log.Printf("[%s] %s", id, e.FullFormattedMessage)
	}
}

func (m *EventManager) PostEvent(ctx *Context, req *types.PostEvent) soap.HasFault {
	m.key++
	event := req.EventToPost.GetEvent()
	event.Key = m.key
	event.ChainId = event.Key
	event.CreatedTime = time.Now()
	event.UserName = ctx.Session.UserName

	m.page = m.page.Prev()
	m.page.Value = req.EventToPost
	m.formatMessage(req.EventToPost)

	for _, c := range m.collectors {
		ctx.WithLock(c, func() {
			if c.eventMatches(req.EventToPost) {
				c.page = c.page.Prev()
				c.page.Value = req.EventToPost
				Map.Update(c, []types.PropertyChange{{Name: "latestPage", Val: c.GetLatestPage()}})
			}
		})
	}

	return &methods.PostEventBody{
		Res: new(types.PostEventResponse),
	}
}

type EventHistoryCollector struct {
	mo.EventHistoryCollector

	m    *EventManager
	page *ring.Ring
	pos  int
}

// doEntityEventArgument calls f for each entity argument in the event.
// If f returns true, the iteration stops.
func doEntityEventArgument(event types.BaseEvent, f func(types.ManagedObjectReference, *types.EntityEventArgument) bool) bool {
	e := event.GetEvent()

	if arg := e.Vm; arg != nil {
		if f(arg.Vm, &arg.EntityEventArgument) {
			return true
		}
	}

	if arg := e.Host; arg != nil {
		if f(arg.Host, &arg.EntityEventArgument) {
			return true
		}
	}

	if arg := e.ComputeResource; arg != nil {
		if f(arg.ComputeResource, &arg.EntityEventArgument) {
			return true
		}
	}

	if arg := e.Ds; arg != nil {
		if f(arg.Datastore, &arg.EntityEventArgument) {
			return true
		}
	}

	if arg := e.Net; arg != nil {
		if f(arg.Network, &arg.EntityEventArgument) {
			return true
		}
	}

	if arg := e.Dvs; arg != nil {
		if f(arg.Dvs, &arg.EntityEventArgument) {
			return true
		}
	}

	if arg := e.Datacenter; arg != nil {
		if f(arg.Datacenter, &arg.EntityEventArgument) {
			return true
		}
	}

	return false
}

// eventFilterSelf returns true if self is one of the entity arguments in the event.
func eventFilterSelf(event types.BaseEvent, self types.ManagedObjectReference) bool {
	return doEntityEventArgument(event, func(ref types.ManagedObjectReference, _ *types.EntityEventArgument) bool {
		return self == ref
	})
}

// eventFilterChildren returns true if a child of self is one of the entity arguments in the event.
func eventFilterChildren(event types.BaseEvent, self types.ManagedObjectReference) bool {
	return doEntityEventArgument(event, func(ref types.ManagedObjectReference, _ *types.EntityEventArgument) bool {
		seen := false

		var match func(types.ManagedObjectReference)

		match = func(child types.ManagedObjectReference) {
			if child == self {
				seen = true
				return
			}

			walk(child, match)
		}

		walk(ref, match)

		return seen
	})
}

// entityMatches returns true if the spec Entity filter matches the event.
func (c *EventHistoryCollector) entityMatches(event types.BaseEvent, spec *types.EventFilterSpec) bool {
	e := spec.Entity
	if e == nil {
		return true
	}

	isRootFolder := c.m.root == e.Entity

	switch e.Recursion {
	case types.EventFilterSpecRecursionOptionSelf:
		return isRootFolder || eventFilterSelf(event, e.Entity)
	case types.EventFilterSpecRecursionOptionChildren:
		return eventFilterChildren(event, e.Entity)
	case types.EventFilterSpecRecursionOptionAll:
		if isRootFolder || eventFilterSelf(event, e.Entity) {
			return true
		}
		return eventFilterChildren(event, e.Entity)
	}

	return false
}

// typeMatches returns true if one of the spec EventTypeId types matches the event.
func (c *EventHistoryCollector) typeMatches(event types.BaseEvent, spec *types.EventFilterSpec) bool {
	if len(spec.EventTypeId) == 0 {
		return true
	}

	matches := func(name string) bool {
		for _, id := range spec.EventTypeId {
			if id == name {
				return true
			}
		}
		return false
	}
	kind := reflect.ValueOf(event).Elem().Type()

	if matches(kind.Name()) {
		return true // concrete type
	}

	field, ok := kind.FieldByNameFunc(matches)
	if ok {
		return field.Anonymous // base type (embedded field)
	}
	return false
}

// eventMatches returns true one of the filters matches the event.
func (c *EventHistoryCollector) eventMatches(event types.BaseEvent) bool {
	spec := c.Filter.(types.EventFilterSpec)

	if !c.typeMatches(event, &spec) {
		return false
	}

	// TODO: spec.Time, spec.UserName, etc

	return c.entityMatches(event, &spec)
}

// filePage copies the manager's latest events into the collector's page with Filter applied.
func (c *EventHistoryCollector) fillPage(size int) {
	c.pos = 0
	l := c.page.Len()
	delta := size - l

	if delta < 0 {
		// Shrink ring size
		c.page = c.page.Unlink(-delta)
		return
	}

	matches := 0
	mpage := c.m.page
	page := c.page

	if delta != 0 {
		// Grow ring size
		c.page = c.page.Link(ring.New(delta))
	}

	for i := 0; i < maxPageSize; i++ {
		event, ok := mpage.Value.(types.BaseEvent)
		mpage = mpage.Prev()
		if !ok {
			continue
		}

		if c.eventMatches(event) {
			page.Value = event
			page = page.Prev()
			matches++
			if matches == size {
				break
			}
		}
	}
}

func validatePageSize(count int32) (int, *soap.Fault) {
	size := int(count)

	if size == 0 {
		size = 10 // defaultPageSize
	} else if size < 0 || size > maxPageSize {
		return -1, Fault("", &types.InvalidArgument{InvalidProperty: "maxCount"})
	}

	return size, nil
}

func (c *EventHistoryCollector) SetCollectorPageSize(ctx *Context, req *types.SetCollectorPageSize) soap.HasFault {
	body := new(methods.SetCollectorPageSizeBody)
	size, err := validatePageSize(req.MaxCount)
	if err != nil {
		body.Fault_ = err
		return body
	}

	ctx.WithLock(c.m, func() {
		c.fillPage(size)
	})

	body.Res = new(types.SetCollectorPageSizeResponse)
	return body
}

func (c *EventHistoryCollector) RewindCollector(ctx *Context, req *types.RewindCollector) soap.HasFault {
	c.pos = 0
	return &methods.RewindCollectorBody{
		Res: new(types.RewindCollectorResponse),
	}
}

func (c *EventHistoryCollector) ReadNextEvents(ctx *Context, req *types.ReadNextEvents) soap.HasFault {
	body := &methods.ReadNextEventsBody{}
	if req.MaxCount <= 0 {
		body.Fault_ = Fault("", &types.InvalidArgument{InvalidProperty: "maxCount"})
		return body
	}
	body.Res = new(types.ReadNextEventsResponse)

	events := c.GetLatestPage()
	nevents := len(events)
	if c.pos == nevents {
		return body // already read to EOF
	}

	start := c.pos
	end := start + int(req.MaxCount)
	c.pos += int(req.MaxCount)
	if end > nevents {
		end = nevents
		c.pos = nevents
	}

	body.Res.Returnval = events[start:end]

	return body
}

func (c *EventHistoryCollector) ReadPreviousEvents(ctx *Context, req *types.ReadPreviousEvents) soap.HasFault {
	body := &methods.ReadPreviousEventsBody{}
	if req.MaxCount <= 0 {
		body.Fault_ = Fault("", &types.InvalidArgument{InvalidProperty: "maxCount"})
		return body
	}
	body.Res = new(types.ReadPreviousEventsResponse)

	events := c.GetLatestPage()
	if c.pos == 0 {
		return body // already read to EOF
	}

	start := c.pos - int(req.MaxCount)
	end := c.pos
	c.pos -= int(req.MaxCount)
	if start < 0 {
		start = 0
		c.pos = 0
	}

	body.Res.Returnval = events[start:end]

	return body
}

func (c *EventHistoryCollector) DestroyCollector(ctx *Context, req *types.DestroyCollector) soap.HasFault {
	ctx.Session.Remove(req.This)

	ctx.WithLock(c.m, func() {
		delete(c.m.collectors, req.This)
	})

	return &methods.DestroyCollectorBody{
		Res: new(types.DestroyCollectorResponse),
	}
}

func (c *EventHistoryCollector) GetLatestPage() []types.BaseEvent {
	var latestPage []types.BaseEvent

	c.page.Do(func(val interface{}) {
		if val == nil {
			return
		}
		latestPage = append(latestPage, val.(types.BaseEvent))
	})

	return latestPage
}

func (c *EventHistoryCollector) Get() mo.Reference {
	clone := *c

	clone.LatestPage = clone.GetLatestPage()

	return &clone
}
