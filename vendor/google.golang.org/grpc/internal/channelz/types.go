/*
 *
 * Copyright 2018 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package channelz

import (
	"net"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/grpclog"
)

// entry represents a node in the channelz database.
type entry interface {
	// addChild adds a child e, whose channelz id is id to child list
	addChild(id int64, e entry)
	// deleteChild deletes a child with channelz id to be id from child list
	deleteChild(id int64)
	// triggerDelete tries to delete self from channelz database. However, if child
	// list is not empty, then deletion from the database is on hold until the last
	// child is deleted from database.
	triggerDelete()
	// deleteSelfIfReady check whether triggerDelete() has been called before, and whether child
	// list is now empty. If both conditions are met, then delete self from database.
	deleteSelfIfReady()
	// getParentID returns parent ID of the entry. 0 value parent ID means no parent.
	getParentID() int64
}

// dummyEntry is a fake entry to handle entry not found case.
type dummyEntry struct {
	idNotFound int64
}

func (d *dummyEntry) addChild(id int64, e entry) {
	// Note: It is possible for a normal program to reach here under race condition.
	// For example, there could be a race between ClientConn.Close() info being propagated
	// to addrConn and http2Client. ClientConn.Close() cancel the context and result
	// in http2Client to error. The error info is then caught by transport monitor
	// and before addrConn.tearDown() is called in side ClientConn.Close(). Therefore,
	// the addrConn will create a new transport. And when registering the new transport in
	// channelz, its parent addrConn could have already been torn down and deleted
	// from channelz tracking, and thus reach the code here.
	grpclog.Infof("attempt to add child of type %T with id %d to a parent (id=%d) that doesn't currently exist", e, id, d.idNotFound)
}

func (d *dummyEntry) deleteChild(id int64) {
	// It is possible for a normal program to reach here under race condition.
	// Refer to the example described in addChild().
	grpclog.Infof("attempt to delete child with id %d from a parent (id=%d) that doesn't currently exist", id, d.idNotFound)
}

func (d *dummyEntry) triggerDelete() {
	grpclog.Warningf("attempt to delete an entry (id=%d) that doesn't currently exist", d.idNotFound)
}

func (*dummyEntry) deleteSelfIfReady() {
	// code should not reach here. deleteSelfIfReady is always called on an existing entry.
}

func (*dummyEntry) getParentID() int64 {
	return 0
}

// ChannelMetric defines the info channelz provides for a specific Channel, which
// includes ChannelInternalMetric and channelz-specific data, such as channelz id,
// child list, etc.
type ChannelMetric struct {
	// ID is the channelz id of this channel.
	ID int64
	// RefName is the human readable reference string of this channel.
	RefName string
	// ChannelData contains channel internal metric reported by the channel through
	// ChannelzMetric().
	ChannelData *ChannelInternalMetric
	// NestedChans tracks the nested channel type children of this channel in the format of
	// a map from nested channel channelz id to corresponding reference string.
	NestedChans map[int64]string
	// SubChans tracks the subchannel type children of this channel in the format of a
	// map from subchannel channelz id to corresponding reference string.
	SubChans map[int64]string
	// Sockets tracks the socket type children of this channel in the format of a map
	// from socket channelz id to corresponding reference string.
	// Note current grpc implementation doesn't allow channel having sockets directly,
	// therefore, this is field is unused.
	Sockets map[int64]string
	// Trace contains the most recent traced events.
	Trace *ChannelTrace
}

// SubChannelMetric defines the info channelz provides for a specific SubChannel,
// which includes ChannelInternalMetric and channelz-specific data, such as
// channelz id, child list, etc.
type SubChannelMetric struct {
	// ID is the channelz id of this subchannel.
	ID int64
	// RefName is the human readable reference string of this subchannel.
	RefName string
	// ChannelData contains subchannel internal metric reported by the subchannel
	// through ChannelzMetric().
	ChannelData *ChannelInternalMetric
	// NestedChans tracks the nested channel type children of this subchannel in the format of
	// a map from nested channel channelz id to corresponding reference string.
	// Note current grpc implementation doesn't allow subchannel to have nested channels
	// as children, therefore, this field is unused.
	NestedChans map[int64]string
	// SubChans tracks the subchannel type children of this subchannel in the format of a
	// map from subchannel channelz id to corresponding reference string.
	// Note current grpc implementation doesn't allow subchannel to have subchannels
	// as children, therefore, this field is unused.
	SubChans map[int64]string
	// Sockets tracks the socket type children of this subchannel in the format of a map
	// from socket channelz id to corresponding reference string.
	Sockets map[int64]string
	// Trace contains the most recent traced events.
	Trace *ChannelTrace
}

// ChannelInternalMetric defines the struct that the implementor of Channel interface
// should return from ChannelzMetric().
type ChannelInternalMetric struct {
	// current connectivity state of the channel.
	State connectivity.State
	// The target this channel originally tried to connect to.  May be absent
	Target string
	// The number of calls started on the channel.
	CallsStarted int64
	// The number of calls that have completed with an OK status.
	CallsSucceeded int64
	// The number of calls that have a completed with a non-OK status.
	CallsFailed int64
	// The last time a call was started on the channel.
	LastCallStartedTimestamp time.Time
}

// ChannelTrace stores traced events on a channel/subchannel and related info.
type ChannelTrace struct {
	// EventNum is the number of events that ever got traced (i.e. including those that have been deleted)
	EventNum int64
	// CreationTime is the creation time of the trace.
	CreationTime time.Time
	// Events stores the most recent trace events (up to $maxTraceEntry, newer event will overwrite the
	// oldest one)
	Events []*TraceEvent
}

// TraceEvent represent a single trace event
type TraceEvent struct {
	// Desc is a simple description of the trace event.
	Desc string
	// Severity states the severity of this trace event.
	Severity Severity
	// Timestamp is the event time.
	Timestamp time.Time
	// RefID is the id of the entity that gets referenced in the event. RefID is 0 if no other entity is
	// involved in this event.
	// e.g. SubChannel (id: 4[]) Created. --> RefID = 4, RefName = "" (inside [])
	RefID int64
	// RefName is the reference name for the entity that gets referenced in the event.
	RefName string
	// RefType indicates the referenced entity type, i.e Channel or SubChannel.
	RefType RefChannelType
}

// Channel is the interface that should be satisfied in order to be tracked by
// channelz as Channel or SubChannel.
type Channel interface {
	ChannelzMetric() *ChannelInternalMetric
}

type dummyChannel struct{}

func (d *dummyChannel) ChannelzMetric() *ChannelInternalMetric {
	return &ChannelInternalMetric{}
}

type channel struct {
	refName     string
	c           Channel
	closeCalled bool
	nestedChans map[int64]string
	subChans    map[int64]string
	id          int64
	pid         int64
	cm          *channelMap
	trace       *channelTrace
	// traceRefCount is the number of trace events that reference this channel.
	// Non-zero traceRefCount means the trace of this channel cannot be deleted.
	traceRefCount int32
}

func (c *channel) addChild(id int64, e entry) {
	switch v := e.(type) {
	case *subChannel:
		c.subChans[id] = v.refName
	case *channel:
		c.nestedChans[id] = v.refName
	default:
		grpclog.Errorf("cannot add a child (id = %d) of type %T to a channel", id, e)
	}
}

func (c *channel) deleteChild(id int64) {
	delete(c.subChans, id)
	delete(c.nestedChans, id)
	c.deleteSelfIfReady()
}

func (c *channel) triggerDelete() {
	c.closeCalled = true
	c.deleteSelfIfReady()
}

func (c *channel) getParentID() int64 {
	return c.pid
}

// deleteSelfFromTree tries to delete the channel from the channelz entry relation tree, which means
// deleting the channel reference from its parent's child list.
//
// In order for a channel to be deleted from the tree, it must meet the criteria that, removal of the
// corresponding grpc object has been invoked, and the channel does not have any children left.
//
// The returned boolean value indicates whether the channel has been successfully deleted from tree.
func (c *channel) deleteSelfFromTree() (deleted bool) {
	if !c.closeCalled || len(c.subChans)+len(c.nestedChans) != 0 {
		return false
	}
	// not top channel
	if c.pid != 0 {
		c.cm.findEntry(c.pid).deleteChild(c.id)
	}
	return true
}

// deleteSelfFromMap checks whether it is valid to delete the channel from the map, which means
// deleting the channel from channelz's tracking entirely. Users can no longer use id to query the
// channel, and its memory will be garbage collected.
//
// The trace reference count of the channel must be 0 in order to be deleted from the map. This is
// specified in the channel tracing gRFC that as long as some other trace has reference to an entity,
// the trace of the referenced entity must not be deleted. In order to release the resource allocated
// by grpc, the reference to the grpc object is reset to a dummy object.
//
// deleteSelfFromMap must be called after deleteSelfFromTree returns true.
//
// It returns a bool to indicate whether the channel can be safely deleted from map.
func (c *channel) deleteSelfFromMap() (delete bool) {
	if c.getTraceRefCount() != 0 {
		c.c = &dummyChannel{}
		return false
	}
	return true
}

// deleteSelfIfReady tries to delete the channel itself from the channelz database.
// The delete process includes two steps:
// 1. delete the channel from the entry relation tree, i.e. delete the channel reference from its
//    parent's child list.
// 2. delete the channel from the map, i.e. delete the channel entirely from channelz. Lookup by id
//    will return entry not found error.
func (c *channel) deleteSelfIfReady() {
	if !c.deleteSelfFromTree() {
		return
	}
	if !c.deleteSelfFromMap() {
		return
	}
	c.cm.deleteEntry(c.id)
	c.trace.clear()
}

func (c *channel) getChannelTrace() *channelTrace {
	return c.trace
}

func (c *channel) incrTraceRefCount() {
	atomic.AddInt32(&c.traceRefCount, 1)
}

func (c *channel) decrTraceRefCount() {
	atomic.AddInt32(&c.traceRefCount, -1)
}

func (c *channel) getTraceRefCount() int {
	i := atomic.LoadInt32(&c.traceRefCount)
	return int(i)
}

func (c *channel) getRefName() string {
	return c.refName
}

type subChannel struct {
	refName       string
	c             Channel
	closeCalled   bool
	sockets       map[int64]string
	id            int64
	pid           int64
	cm            *channelMap
	trace         *channelTrace
	traceRefCount int32
}

func (sc *subChannel) addChild(id int64, e entry) {
	if v, ok := e.(*normalSocket); ok {
		sc.sockets[id] = v.refName
	} else {
		grpclog.Errorf("cannot add a child (id = %d) of type %T to a subChannel", id, e)
	}
}

func (sc *subChannel) deleteChild(id int64) {
	delete(sc.sockets, id)
	sc.deleteSelfIfReady()
}

func (sc *subChannel) triggerDelete() {
	sc.closeCalled = true
	sc.deleteSelfIfReady()
}

func (sc *subChannel) getParentID() int64 {
	return sc.pid
}

// deleteSelfFromTree tries to delete the subchannel from the channelz entry relation tree, which
// means deleting the subchannel reference from its parent's child list.
//
// In order for a subchannel to be deleted from the tree, it must meet the criteria that, removal of
// the corresponding grpc object has been invoked, and the subchannel does not have any children left.
//
// The returned boolean value indicates whether the channel has been successfully deleted from tree.
func (sc *subChannel) deleteSelfFromTree() (deleted bool) {
	if !sc.closeCalled || len(sc.sockets) != 0 {
		return false
	}
	sc.cm.findEntry(sc.pid).deleteChild(sc.id)
	return true
}

// deleteSelfFromMap checks whether it is valid to delete the subchannel from the map, which means
// deleting the subchannel from channelz's tracking entirely. Users can no longer use id to query
// the subchannel, and its memory will be garbage collected.
//
// The trace reference count of the subchannel must be 0 in order to be deleted from the map. This is
// specified in the channel tracing gRFC that as long as some other trace has reference to an entity,
// the trace of the referenced entity must not be deleted. In order to release the resource allocated
// by grpc, the reference to the grpc object is reset to a dummy object.
//
// deleteSelfFromMap must be called after deleteSelfFromTree returns true.
//
// It returns a bool to indicate whether the channel can be safely deleted from map.
func (sc *subChannel) deleteSelfFromMap() (delete bool) {
	if sc.getTraceRefCount() != 0 {
		// free the grpc struct (i.e. addrConn)
		sc.c = &dummyChannel{}
		return false
	}
	return true
}

// deleteSelfIfReady tries to delete the subchannel itself from the channelz database.
// The delete process includes two steps:
// 1. delete the subchannel from the entry relation tree, i.e. delete the subchannel reference from
//    its parent's child list.
// 2. delete the subchannel from the map, i.e. delete the subchannel entirely from channelz. Lookup
//    by id will return entry not found error.
func (sc *subChannel) deleteSelfIfReady() {
	if !sc.deleteSelfFromTree() {
		return
	}
	if !sc.deleteSelfFromMap() {
		return
	}
	sc.cm.deleteEntry(sc.id)
	sc.trace.clear()
}

func (sc *subChannel) getChannelTrace() *channelTrace {
	return sc.trace
}

func (sc *subChannel) incrTraceRefCount() {
	atomic.AddInt32(&sc.traceRefCount, 1)
}

func (sc *subChannel) decrTraceRefCount() {
	atomic.AddInt32(&sc.traceRefCount, -1)
}

func (sc *subChannel) getTraceRefCount() int {
	i := atomic.LoadInt32(&sc.traceRefCount)
	return int(i)
}

func (sc *subChannel) getRefName() string {
	return sc.refName
}

// SocketMetric defines the info channelz provides for a specific Socket, which
// includes SocketInternalMetric and channelz-specific data, such as channelz id, etc.
type SocketMetric struct {
	// ID is the channelz id of this socket.
	ID int64
	// RefName is the human readable reference string of this socket.
	RefName string
	// SocketData contains socket internal metric reported by the socket through
	// ChannelzMetric().
	SocketData *SocketInternalMetric
}

// SocketInternalMetric defines the struct that the implementor of Socket interface
// should return from ChannelzMetric().
type SocketInternalMetric struct {
	// The number of streams that have been started.
	StreamsStarted int64
	// The number of streams that have ended successfully:
	// On client side, receiving frame with eos bit set.
	// On server side, sending frame with eos bit set.
	StreamsSucceeded int64
	// The number of streams that have ended unsuccessfully:
	// On client side, termination without receiving frame with eos bit set.
	// On server side, termination without sending frame with eos bit set.
	StreamsFailed int64
	// The number of messages successfully sent on this socket.
	MessagesSent     int64
	MessagesReceived int64
	// The number of keep alives sent.  This is typically implemented with HTTP/2
	// ping messages.
	KeepAlivesSent int64
	// The last time a stream was created by this endpoint.  Usually unset for
	// servers.
	LastLocalStreamCreatedTimestamp time.Time
	// The last time a stream was created by the remote endpoint.  Usually unset
	// for clients.
	LastRemoteStreamCreatedTimestamp time.Time
	// The last time a message was sent by this endpoint.
	LastMessageSentTimestamp time.Time
	// The last time a message was received by this endpoint.
	LastMessageReceivedTimestamp time.Time
	// The amount of window, granted to the local endpoint by the remote endpoint.
	// This may be slightly out of date due to network latency.  This does NOT
	// include stream level or TCP level flow control info.
	LocalFlowControlWindow int64
	// The amount of window, granted to the remote endpoint by the local endpoint.
	// This may be slightly out of date due to network latency.  This does NOT
	// include stream level or TCP level flow control info.
	RemoteFlowControlWindow int64
	// The locally bound address.
	LocalAddr net.Addr
	// The remote bound address.  May be absent.
	RemoteAddr net.Addr
	// Optional, represents the name of the remote endpoint, if different than
	// the original target name.
	RemoteName    string
	SocketOptions *SocketOptionData
	Security      credentials.ChannelzSecurityValue
}

// Socket is the interface that should be satisfied in order to be tracked by
// channelz as Socket.
type Socket interface {
	ChannelzMetric() *SocketInternalMetric
}

type listenSocket struct {
	refName string
	s       Socket
	id      int64
	pid     int64
	cm      *channelMap
}

func (ls *listenSocket) addChild(id int64, e entry) {
	grpclog.Errorf("cannot add a child (id = %d) of type %T to a listen socket", id, e)
}

func (ls *listenSocket) deleteChild(id int64) {
	grpclog.Errorf("cannot delete a child (id = %d) from a listen socket", id)
}

func (ls *listenSocket) triggerDelete() {
	ls.cm.deleteEntry(ls.id)
	ls.cm.findEntry(ls.pid).deleteChild(ls.id)
}

func (ls *listenSocket) deleteSelfIfReady() {
	grpclog.Errorf("cannot call deleteSelfIfReady on a listen socket")
}

func (ls *listenSocket) getParentID() int64 {
	return ls.pid
}

type normalSocket struct {
	refName string
	s       Socket
	id      int64
	pid     int64
	cm      *channelMap
}

func (ns *normalSocket) addChild(id int64, e entry) {
	grpclog.Errorf("cannot add a child (id = %d) of type %T to a normal socket", id, e)
}

func (ns *normalSocket) deleteChild(id int64) {
	grpclog.Errorf("cannot delete a child (id = %d) from a normal socket", id)
}

func (ns *normalSocket) triggerDelete() {
	ns.cm.deleteEntry(ns.id)
	ns.cm.findEntry(ns.pid).deleteChild(ns.id)
}

func (ns *normalSocket) deleteSelfIfReady() {
	grpclog.Errorf("cannot call deleteSelfIfReady on a normal socket")
}

func (ns *normalSocket) getParentID() int64 {
	return ns.pid
}

// ServerMetric defines the info channelz provides for a specific Server, which
// includes ServerInternalMetric and channelz-specific data, such as channelz id,
// child list, etc.
type ServerMetric struct {
	// ID is the channelz id of this server.
	ID int64
	// RefName is the human readable reference string of this server.
	RefName string
	// ServerData contains server internal metric reported by the server through
	// ChannelzMetric().
	ServerData *ServerInternalMetric
	// ListenSockets tracks the listener socket type children of this server in the
	// format of a map from socket channelz id to corresponding reference string.
	ListenSockets map[int64]string
}

// ServerInternalMetric defines the struct that the implementor of Server interface
// should return from ChannelzMetric().
type ServerInternalMetric struct {
	// The number of incoming calls started on the server.
	CallsStarted int64
	// The number of incoming calls that have completed with an OK status.
	CallsSucceeded int64
	// The number of incoming calls that have a completed with a non-OK status.
	CallsFailed int64
	// The last time a call was started on the server.
	LastCallStartedTimestamp time.Time
}

// Server is the interface to be satisfied in order to be tracked by channelz as
// Server.
type Server interface {
	ChannelzMetric() *ServerInternalMetric
}

type server struct {
	refName       string
	s             Server
	closeCalled   bool
	sockets       map[int64]string
	listenSockets map[int64]string
	id            int64
	cm            *channelMap
}

func (s *server) addChild(id int64, e entry) {
	switch v := e.(type) {
	case *normalSocket:
		s.sockets[id] = v.refName
	case *listenSocket:
		s.listenSockets[id] = v.refName
	default:
		grpclog.Errorf("cannot add a child (id = %d) of type %T to a server", id, e)
	}
}

func (s *server) deleteChild(id int64) {
	delete(s.sockets, id)
	delete(s.listenSockets, id)
	s.deleteSelfIfReady()
}

func (s *server) triggerDelete() {
	s.closeCalled = true
	s.deleteSelfIfReady()
}

func (s *server) deleteSelfIfReady() {
	if !s.closeCalled || len(s.sockets)+len(s.listenSockets) != 0 {
		return
	}
	s.cm.deleteEntry(s.id)
}

func (s *server) getParentID() int64 {
	return 0
}

type tracedChannel interface {
	getChannelTrace() *channelTrace
	incrTraceRefCount()
	decrTraceRefCount()
	getRefName() string
}

type channelTrace struct {
	cm          *channelMap
	createdTime time.Time
	eventCount  int64
	mu          sync.Mutex
	events      []*TraceEvent
}

func (c *channelTrace) append(e *TraceEvent) {
	c.mu.Lock()
	if len(c.events) == getMaxTraceEntry() {
		del := c.events[0]
		c.events = c.events[1:]
		if del.RefID != 0 {
			// start recursive cleanup in a goroutine to not block the call originated from grpc.
			go func() {
				// need to acquire c.cm.mu lock to call the unlocked attemptCleanup func.
				c.cm.mu.Lock()
				c.cm.decrTraceRefCount(del.RefID)
				c.cm.mu.Unlock()
			}()
		}
	}
	e.Timestamp = time.Now()
	c.events = append(c.events, e)
	c.eventCount++
	c.mu.Unlock()
}

func (c *channelTrace) clear() {
	c.mu.Lock()
	for _, e := range c.events {
		if e.RefID != 0 {
			// caller should have already held the c.cm.mu lock.
			c.cm.decrTraceRefCount(e.RefID)
		}
	}
	c.mu.Unlock()
}

// Severity is the severity level of a trace event.
// The canonical enumeration of all valid values is here:
// https://github.com/grpc/grpc-proto/blob/9b13d199cc0d4703c7ea26c9c330ba695866eb23/grpc/channelz/v1/channelz.proto#L126.
type Severity int

const (
	// CtUNKNOWN indicates unknown severity of a trace event.
	CtUNKNOWN Severity = iota
	// CtINFO indicates info level severity of a trace event.
	CtINFO
	// CtWarning indicates warning level severity of a trace event.
	CtWarning
	// CtError indicates error level severity of a trace event.
	CtError
)

// RefChannelType is the type of the entity being referenced in a trace event.
type RefChannelType int

const (
	// RefChannel indicates the referenced entity is a Channel.
	RefChannel RefChannelType = iota
	// RefSubChannel indicates the referenced entity is a SubChannel.
	RefSubChannel
)

func (c *channelTrace) dumpData() *ChannelTrace {
	c.mu.Lock()
	ct := &ChannelTrace{EventNum: c.eventCount, CreationTime: c.createdTime}
	ct.Events = c.events[:len(c.events)]
	c.mu.Unlock()
	return ct
}
