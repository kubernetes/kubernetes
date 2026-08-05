// Copyright 2018 Google LLC. All Rights Reserved.
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

package nftables

import (
	"math"
	"strings"
	"sync"

	"github.com/mdlayher/netlink"
	"golang.org/x/sys/unix"
)

type MonitorAction uint8

// Possible MonitorAction values.
const (
	MonitorActionNew MonitorAction = 1 << iota
	MonitorActionDel
	MonitorActionMask MonitorAction = (1 << iota) - 1
	MonitorActionAny  MonitorAction = MonitorActionMask
)

type MonitorObject uint32

// Possible MonitorObject values.
const (
	MonitorObjectTables MonitorObject = 1 << iota
	MonitorObjectChains
	MonitorObjectSets
	MonitorObjectRules
	MonitorObjectElements
	MonitorObjectRuleset
	MonitorObjectMask MonitorObject = (1 << iota) - 1
	MonitorObjectAny  MonitorObject = MonitorObjectMask
)

var (
	monitorFlags = map[MonitorAction]map[MonitorObject]uint32{
		MonitorActionAny: {
			MonitorObjectAny:      0xffffffff,
			MonitorObjectTables:   1<<unix.NFT_MSG_NEWTABLE | 1<<unix.NFT_MSG_DELTABLE,
			MonitorObjectChains:   1<<unix.NFT_MSG_NEWCHAIN | 1<<unix.NFT_MSG_DELCHAIN,
			MonitorObjectRules:    1<<unix.NFT_MSG_NEWRULE | 1<<unix.NFT_MSG_DELRULE,
			MonitorObjectSets:     1<<unix.NFT_MSG_NEWSET | 1<<unix.NFT_MSG_DELSET,
			MonitorObjectElements: 1<<unix.NFT_MSG_NEWSETELEM | 1<<unix.NFT_MSG_DELSETELEM,
			MonitorObjectRuleset: 1<<unix.NFT_MSG_NEWTABLE | 1<<unix.NFT_MSG_DELTABLE |
				1<<unix.NFT_MSG_NEWCHAIN | 1<<unix.NFT_MSG_DELCHAIN |
				1<<unix.NFT_MSG_NEWRULE | 1<<unix.NFT_MSG_DELRULE |
				1<<unix.NFT_MSG_NEWSET | 1<<unix.NFT_MSG_DELSET |
				1<<unix.NFT_MSG_NEWSETELEM | 1<<unix.NFT_MSG_DELSETELEM |
				1<<unix.NFT_MSG_NEWOBJ | 1<<unix.NFT_MSG_DELOBJ,
		},
		MonitorActionNew: {
			MonitorObjectAny: 1<<unix.NFT_MSG_NEWTABLE |
				1<<unix.NFT_MSG_NEWCHAIN |
				1<<unix.NFT_MSG_NEWRULE |
				1<<unix.NFT_MSG_NEWSET |
				1<<unix.NFT_MSG_NEWSETELEM,
			MonitorObjectTables: 1 << unix.NFT_MSG_NEWTABLE,
			MonitorObjectChains: 1 << unix.NFT_MSG_NEWCHAIN,
			MonitorObjectRules:  1 << unix.NFT_MSG_NEWRULE,
			MonitorObjectSets:   1 << unix.NFT_MSG_NEWSET,
			MonitorObjectRuleset: 1<<unix.NFT_MSG_NEWTABLE |
				1<<unix.NFT_MSG_NEWCHAIN |
				1<<unix.NFT_MSG_NEWRULE |
				1<<unix.NFT_MSG_NEWSET |
				1<<unix.NFT_MSG_NEWSETELEM |
				1<<unix.NFT_MSG_NEWOBJ,
		},
		MonitorActionDel: {
			MonitorObjectAny: 1<<unix.NFT_MSG_DELTABLE |
				1<<unix.NFT_MSG_DELCHAIN |
				1<<unix.NFT_MSG_DELRULE |
				1<<unix.NFT_MSG_DELSET |
				1<<unix.NFT_MSG_DELSETELEM |
				1<<unix.NFT_MSG_DELOBJ,
		},
	}
)

type MonitorEventType int

const (
	MonitorEventTypeNewTable   MonitorEventType = unix.NFT_MSG_NEWTABLE
	MonitorEventTypeDelTable   MonitorEventType = unix.NFT_MSG_DELTABLE
	MonitorEventTypeNewChain   MonitorEventType = unix.NFT_MSG_NEWCHAIN
	MonitorEventTypeDelChain   MonitorEventType = unix.NFT_MSG_DELCHAIN
	MonitorEventTypeNewRule    MonitorEventType = unix.NFT_MSG_NEWRULE
	MonitorEventTypeDelRule    MonitorEventType = unix.NFT_MSG_DELRULE
	MonitorEventTypeNewSet     MonitorEventType = unix.NFT_MSG_NEWSET
	MonitorEventTypeDelSet     MonitorEventType = unix.NFT_MSG_DELSET
	MonitorEventTypeNewSetElem MonitorEventType = unix.NFT_MSG_NEWSETELEM
	MonitorEventTypeDelSetElem MonitorEventType = unix.NFT_MSG_DELSETELEM
	MonitorEventTypeNewObj     MonitorEventType = unix.NFT_MSG_NEWOBJ
	MonitorEventTypeDelObj     MonitorEventType = unix.NFT_MSG_DELOBJ
	MonitorEventTypeOOB        MonitorEventType = math.MaxInt // out of band event
)

// A MonitorEvent represents a single change received via a [Monitor].
//
// Depending on the Type, the Data field can be type-asserted to the specific
// data type for this event, e.g. when Type is
// nftables.MonitorEventTypeNewTable, you can access the corresponding table
// details via Data.(*nftables.Table).
type MonitorEvent struct {
	Header netlink.Header
	Type   MonitorEventType
	Data   any
	Error  error
}

type MonitorEvents struct {
	GeneratedBy *MonitorEvent
	Changes     []*MonitorEvent
}

const (
	monitorOK = iota
	monitorClosed
)

// A Monitor is an event-based nftables monitor that will receive one event per
// new (or deleted) table, chain, rule, set, etc., depending on the monitor
// configuration.
type Monitor struct {
	action       MonitorAction
	object       MonitorObject
	monitorFlags uint32

	conn   *netlink.Conn
	closer netlinkCloser

	// mu covers eventCh and status
	mu      sync.Mutex
	eventCh chan *MonitorEvents
	status  int
}

type MonitorOption func(*Monitor)

func WithMonitorEventBuffer(size int) MonitorOption {
	return func(monitor *Monitor) {
		monitor.eventCh = make(chan *MonitorEvents, size)
	}
}

// WithMonitorAction to set monitor actions like new, del or any.
func WithMonitorAction(action MonitorAction) MonitorOption {
	return func(monitor *Monitor) {
		monitor.action = action
	}
}

// WithMonitorObject to set monitor objects.
func WithMonitorObject(object MonitorObject) MonitorOption {
	return func(monitor *Monitor) {
		monitor.object = object
	}
}

// NewMonitor returns a Monitor with options to be started.
//
// Note that NewMonitor only prepares a Monitor. To install the monitor, call
// [Conn.AddMonitor].
func NewMonitor(opts ...MonitorOption) *Monitor {
	monitor := &Monitor{
		status: monitorOK,
	}
	for _, opt := range opts {
		opt(monitor)
	}
	if monitor.eventCh == nil {
		monitor.eventCh = make(chan *MonitorEvents)
	}
	objects, ok := monitorFlags[monitor.action]
	if !ok {
		objects = monitorFlags[MonitorActionAny]
	}
	flags, ok := objects[monitor.object]
	if !ok {
		flags = objects[MonitorObjectAny]
	}
	monitor.monitorFlags = flags
	return monitor
}

func (monitor *Monitor) monitor() {
	var changesEvents []*MonitorEvent

	for {
		msgs, err := monitor.conn.Receive()
		if err != nil {
			if strings.Contains(err.Error(), "use of closed file") {
				// ignore the error that be closed
				break
			} else {
				// any other errors will be sent to user, and then to close eventCh
				event := &MonitorEvent{
					Type:  MonitorEventTypeOOB,
					Data:  nil,
					Error: err,
				}

				changesEvents = append(changesEvents, event)

				monitor.eventCh <- &MonitorEvents{
					GeneratedBy: event,
					Changes:     changesEvents,
				}
				changesEvents = nil

				break
			}
		}
		for _, msg := range msgs {
			if msg.Header.Type&0xff00>>8 != netlink.HeaderType(unix.NFNL_SUBSYS_NFTABLES) {
				continue
			}
			msgType := msg.Header.Type & 0x00ff
			if monitor.monitorFlags&1<<msgType == 0 {
				continue
			}
			switch msgType {
			case unix.NFT_MSG_NEWTABLE, unix.NFT_MSG_DELTABLE:
				table, err := tableFromMsg(msg)
				event := &MonitorEvent{
					Type:   MonitorEventType(msgType),
					Data:   table,
					Error:  err,
					Header: msg.Header,
				}
				changesEvents = append(changesEvents, event)
			case unix.NFT_MSG_NEWCHAIN, unix.NFT_MSG_DELCHAIN:
				chain, err := chainFromMsg(msg)
				event := &MonitorEvent{
					Type:   MonitorEventType(msgType),
					Data:   chain,
					Error:  err,
					Header: msg.Header,
				}
				changesEvents = append(changesEvents, event)
			case unix.NFT_MSG_NEWRULE, unix.NFT_MSG_DELRULE:
				rule, err := parseRuleFromMsg(msg)
				event := &MonitorEvent{
					Type:   MonitorEventType(msgType),
					Data:   rule,
					Error:  err,
					Header: msg.Header,
				}
				changesEvents = append(changesEvents, event)
			case unix.NFT_MSG_NEWSET, unix.NFT_MSG_DELSET:
				set, err := setsFromMsg(msg)
				event := &MonitorEvent{
					Type:   MonitorEventType(msgType),
					Data:   set,
					Error:  err,
					Header: msg.Header,
				}
				changesEvents = append(changesEvents, event)
			case unix.NFT_MSG_NEWSETELEM, unix.NFT_MSG_DELSETELEM:
				elems, err := elementsFromMsg(uint8(TableFamilyUnspecified), msg)
				event := &MonitorEvent{
					Type:   MonitorEventType(msgType),
					Data:   elems,
					Error:  err,
					Header: msg.Header,
				}
				changesEvents = append(changesEvents, event)
			case unix.NFT_MSG_NEWOBJ, unix.NFT_MSG_DELOBJ:
				obj, err := objFromMsg(msg, true)
				event := &MonitorEvent{
					Type:   MonitorEventType(msgType),
					Data:   obj,
					Error:  err,
					Header: msg.Header,
				}
				changesEvents = append(changesEvents, event)
			case unix.NFT_MSG_NEWGEN:
				gen, err := genFromMsg(msg)
				event := &MonitorEvent{
					Type:   MonitorEventType(msgType),
					Data:   gen,
					Error:  err,
					Header: msg.Header,
				}

				monitor.eventCh <- &MonitorEvents{
					GeneratedBy: event,
					Changes:     changesEvents,
				}

				changesEvents = nil
			}
		}
	}

	monitor.mu.Lock()
	defer monitor.mu.Unlock()

	if monitor.status != monitorClosed {
		monitor.status = monitorClosed
	}
	close(monitor.eventCh)
}

func (monitor *Monitor) Close() error {
	monitor.mu.Lock()
	defer monitor.mu.Unlock()

	if monitor.status != monitorClosed {
		monitor.status = monitorClosed
		return monitor.closer()
	}
	return nil
}

// AddMonitor to perform the monitor immediately. The channel will be closed after
// calling Close on Monitor or encountering a netlink conn error while Receive.
// Caller may receive a MonitorEventTypeOOB event which contains an error we didn't
// handle, for now.
func (cc *Conn) AddMonitor(monitor *Monitor) (chan *MonitorEvent, error) {
	generationalEventCh, err := cc.AddGenerationalMonitor(monitor)
	if err != nil {
		return nil, err
	}

	eventCh := make(chan *MonitorEvent)

	go func() {
		defer close(eventCh)
		for monitorEvents := range generationalEventCh {
			for _, event := range monitorEvents.Changes {
				eventCh <- event
			}
		}
	}()

	return eventCh, nil
}

func (cc *Conn) AddGenerationalMonitor(monitor *Monitor) (chan *MonitorEvents, error) {
	conn, closer, err := cc.netlinkConn()
	if err != nil {
		return nil, err
	}
	monitor.conn = conn
	monitor.closer = closer

	if monitor.monitorFlags != 0 {
		if err = conn.JoinGroup(uint32(unix.NFNLGRP_NFTABLES)); err != nil {
			monitor.closer()
			return nil, err
		}
	}

	go monitor.monitor()
	return monitor.eventCh, nil
}

func parseRuleFromMsg(msg netlink.Message) (*Rule, error) {
	genmsg := &NFGenMsg{}
	genmsg.Decode(msg.Data[:4])
	return ruleFromMsg(TableFamily(genmsg.NFGenFamily), msg)
}
