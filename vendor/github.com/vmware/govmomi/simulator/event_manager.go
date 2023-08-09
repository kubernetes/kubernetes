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
	"container/list"
	"log"
	"reflect"
	"text/template"
	"time"

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

	root types.ManagedObjectReference

	history *list.List
	key     int32

	collectors map[types.ManagedObjectReference]*EventHistoryCollector
	templates  map[string]*template.Template
}

func (m *EventManager) init(r *Registry) {
	if len(m.Description.EventInfo) == 0 {
		m.Description.EventInfo = esx.EventInfo
	}
	if m.MaxCollector == 0 {
		m.MaxCollector = 1000
	}
	m.root = r.content().RootFolder
	m.history = list.New()
	m.collectors = make(map[types.ManagedObjectReference]*EventHistoryCollector)
	m.templates = make(map[string]*template.Template)
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
		page: list.New(),
		size: size,
	}
	collector.Filter = req.Filter
	collector.fillPage()

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

func pushEvent(l *list.List, event types.BaseEvent) {
	if l.Len() > maxPageSize*5 {
		l.Remove(l.Front()) // Prune history
	}
	l.PushBack(event)
}

func (m *EventManager) PostEvent(ctx *Context, req *types.PostEvent) soap.HasFault {
	m.key++
	event := req.EventToPost.GetEvent()
	event.Key = m.key
	event.ChainId = event.Key
	event.CreatedTime = time.Now()
	event.UserName = ctx.Session.UserName

	m.formatMessage(req.EventToPost)

	pushEvent(m.history, req.EventToPost)

	for _, c := range m.collectors {
		ctx.WithLock(c, func() {
			if c.eventMatches(req.EventToPost) {
				pushEvent(c.page, req.EventToPost)
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
	size int
	page *list.List
	pos  *list.Element
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

func (c *EventHistoryCollector) timeMatches(event types.BaseEvent, spec *types.EventFilterSpec) bool {
	if spec.Time == nil {
		return true
	}

	created := event.GetEvent().CreatedTime

	if begin := spec.Time.BeginTime; begin != nil {
		if created.Before(*begin) {
			return false
		}
	}

	if end := spec.Time.EndTime; end != nil {
		if created.After(*end) {
			return false
		}
	}

	return true
}

// eventMatches returns true one of the filters matches the event.
func (c *EventHistoryCollector) eventMatches(event types.BaseEvent) bool {
	spec := c.Filter.(types.EventFilterSpec)

	matchers := []func(types.BaseEvent, *types.EventFilterSpec) bool{
		c.typeMatches,
		c.timeMatches,
		c.entityMatches,
		// TODO: spec.UserName, etc
	}

	for _, match := range matchers {
		if !match(event, &spec) {
			return false
		}
	}

	return true
}

// fillPage copies the manager's latest events into the collector's page with Filter applied.
func (c *EventHistoryCollector) fillPage() {
	for e := c.m.history.Front(); e != nil; e = e.Next() {
		event := e.Value.(types.BaseEvent)

		if c.eventMatches(event) {
			c.page.PushBack(event)
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

	c.size = size
	c.page = list.New()
	ctx.WithLock(c.m, c.fillPage)

	body.Res = new(types.SetCollectorPageSizeResponse)
	return body
}

func (c *EventHistoryCollector) ResetCollector(ctx *Context, req *types.ResetCollector) soap.HasFault {
	c.pos = c.page.Back()

	return &methods.ResetCollectorBody{
		Res: new(types.ResetCollectorResponse),
	}
}

func (c *EventHistoryCollector) RewindCollector(ctx *Context, req *types.RewindCollector) soap.HasFault {
	c.pos = c.page.Front()

	return &methods.RewindCollectorBody{
		Res: new(types.RewindCollectorResponse),
	}
}

// readEvents returns the next max Events from the EventManager's history
func (c *EventHistoryCollector) readEvents(ctx *Context, max int32, next func() *list.Element) []types.BaseEvent {
	var events []types.BaseEvent

	for i := 0; i < int(max); i++ {
		e := next()
		if e == nil {
			break
		}

		events = append(events, e.Value.(types.BaseEvent))
		c.pos = e
	}

	return events
}

func (c *EventHistoryCollector) ReadNextEvents(ctx *Context, req *types.ReadNextEvents) soap.HasFault {
	body := &methods.ReadNextEventsBody{}
	if req.MaxCount <= 0 {
		body.Fault_ = Fault("", &types.InvalidArgument{InvalidProperty: "maxCount"})
		return body
	}
	body.Res = new(types.ReadNextEventsResponse)

	next := func() *list.Element {
		if c.pos != nil {
			return c.pos.Next()
		}
		return c.page.Front()
	}

	body.Res.Returnval = c.readEvents(ctx, req.MaxCount, next)

	return body
}

func (c *EventHistoryCollector) ReadPreviousEvents(ctx *Context, req *types.ReadPreviousEvents) soap.HasFault {
	body := &methods.ReadPreviousEventsBody{}
	if req.MaxCount <= 0 {
		body.Fault_ = Fault("", &types.InvalidArgument{InvalidProperty: "maxCount"})
		return body
	}
	body.Res = new(types.ReadPreviousEventsResponse)

	next := func() *list.Element {
		if c.pos != nil {
			return c.pos.Prev()
		}
		return c.page.Back()
	}

	body.Res.Returnval = c.readEvents(ctx, req.MaxCount, next)

	return body
}

func (c *EventHistoryCollector) DestroyCollector(ctx *Context, req *types.DestroyCollector) soap.HasFault {
	ctx.Session.Remove(ctx, req.This)

	ctx.WithLock(c.m, func() {
		delete(c.m.collectors, req.This)
	})

	return &methods.DestroyCollectorBody{
		Res: new(types.DestroyCollectorResponse),
	}
}

func (c *EventHistoryCollector) GetLatestPage() []types.BaseEvent {
	var latestPage []types.BaseEvent

	e := c.page.Back()
	for i := 0; i < c.size; i++ {
		if e == nil {
			break
		}
		latestPage = append(latestPage, e.Value.(types.BaseEvent))
		e = e.Prev()
	}

	return latestPage
}

func (c *EventHistoryCollector) Get() mo.Reference {
	clone := *c

	clone.LatestPage = clone.GetLatestPage()

	return &clone
}
