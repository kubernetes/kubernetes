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

// Package channelz defines APIs for enabling channelz service, entry
// registration/deletion, and accessing channelz data. It also defines channelz
// metric struct formats.
//
// All APIs in this package are experimental.
package channelz

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc/grpclog"
)

const (
	defaultMaxTraceEntry int32 = 30
)

var (
	db    dbWrapper
	idGen idGenerator
	// EntryPerPage defines the number of channelz entries to be shown on a web page.
	EntryPerPage  = int64(50)
	curState      int32
	maxTraceEntry = defaultMaxTraceEntry
)

// TurnOn turns on channelz data collection.
func TurnOn() {
	if !IsOn() {
		db.set(newChannelMap())
		idGen.reset()
		atomic.StoreInt32(&curState, 1)
	}
}

// IsOn returns whether channelz data collection is on.
func IsOn() bool {
	return atomic.CompareAndSwapInt32(&curState, 1, 1)
}

// SetMaxTraceEntry sets maximum number of trace entry per entity (i.e. channel/subchannel).
// Setting it to 0 will disable channel tracing.
func SetMaxTraceEntry(i int32) {
	atomic.StoreInt32(&maxTraceEntry, i)
}

// ResetMaxTraceEntryToDefault resets the maximum number of trace entry per entity to default.
func ResetMaxTraceEntryToDefault() {
	atomic.StoreInt32(&maxTraceEntry, defaultMaxTraceEntry)
}

func getMaxTraceEntry() int {
	i := atomic.LoadInt32(&maxTraceEntry)
	return int(i)
}

// dbWarpper wraps around a reference to internal channelz data storage, and
// provide synchronized functionality to set and get the reference.
type dbWrapper struct {
	mu sync.RWMutex
	DB *channelMap
}

func (d *dbWrapper) set(db *channelMap) {
	d.mu.Lock()
	d.DB = db
	d.mu.Unlock()
}

func (d *dbWrapper) get() *channelMap {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.DB
}

// NewChannelzStorageForTesting initializes channelz data storage and id
// generator for testing purposes.
//
// Returns a cleanup function to be invoked by the test, which waits for up to
// 10s for all channelz state to be reset by the grpc goroutines when those
// entities get closed. This cleanup function helps with ensuring that tests
// don't mess up each other.
func NewChannelzStorageForTesting() (cleanup func() error) {
	db.set(newChannelMap())
	idGen.reset()

	return func() error {
		cm := db.get()
		if cm == nil {
			return nil
		}

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		ticker := time.NewTicker(10 * time.Millisecond)
		defer ticker.Stop()
		for {
			cm.mu.RLock()
			topLevelChannels, servers, channels, subChannels, listenSockets, normalSockets := len(cm.topLevelChannels), len(cm.servers), len(cm.channels), len(cm.subChannels), len(cm.listenSockets), len(cm.normalSockets)
			cm.mu.RUnlock()

			if err := ctx.Err(); err != nil {
				return fmt.Errorf("after 10s the channelz map has not been cleaned up yet, topchannels: %d, servers: %d, channels: %d, subchannels: %d, listen sockets: %d, normal sockets: %d", topLevelChannels, servers, channels, subChannels, listenSockets, normalSockets)
			}
			if topLevelChannels == 0 && servers == 0 && channels == 0 && subChannels == 0 && listenSockets == 0 && normalSockets == 0 {
				return nil
			}
			<-ticker.C
		}
	}
}

// GetTopChannels returns a slice of top channel's ChannelMetric, along with a
// boolean indicating whether there's more top channels to be queried for.
//
// The arg id specifies that only top channel with id at or above it will be included
// in the result. The returned slice is up to a length of the arg maxResults or
// EntryPerPage if maxResults is zero, and is sorted in ascending id order.
func GetTopChannels(id int64, maxResults int64) ([]*ChannelMetric, bool) {
	return db.get().GetTopChannels(id, maxResults)
}

// GetServers returns a slice of server's ServerMetric, along with a
// boolean indicating whether there's more servers to be queried for.
//
// The arg id specifies that only server with id at or above it will be included
// in the result. The returned slice is up to a length of the arg maxResults or
// EntryPerPage if maxResults is zero, and is sorted in ascending id order.
func GetServers(id int64, maxResults int64) ([]*ServerMetric, bool) {
	return db.get().GetServers(id, maxResults)
}

// GetServerSockets returns a slice of server's (identified by id) normal socket's
// SocketMetric, along with a boolean indicating whether there's more sockets to
// be queried for.
//
// The arg startID specifies that only sockets with id at or above it will be
// included in the result. The returned slice is up to a length of the arg maxResults
// or EntryPerPage if maxResults is zero, and is sorted in ascending id order.
func GetServerSockets(id int64, startID int64, maxResults int64) ([]*SocketMetric, bool) {
	return db.get().GetServerSockets(id, startID, maxResults)
}

// GetChannel returns the ChannelMetric for the channel (identified by id).
func GetChannel(id int64) *ChannelMetric {
	return db.get().GetChannel(id)
}

// GetSubChannel returns the SubChannelMetric for the subchannel (identified by id).
func GetSubChannel(id int64) *SubChannelMetric {
	return db.get().GetSubChannel(id)
}

// GetSocket returns the SocketInternalMetric for the socket (identified by id).
func GetSocket(id int64) *SocketMetric {
	return db.get().GetSocket(id)
}

// GetServer returns the ServerMetric for the server (identified by id).
func GetServer(id int64) *ServerMetric {
	return db.get().GetServer(id)
}

// RegisterChannel registers the given channel c in the channelz database with
// ref as its reference name, and adds it to the child list of its parent
// (identified by pid). pid == nil means no parent.
//
// Returns a unique channelz identifier assigned to this channel.
//
// If channelz is not turned ON, the channelz database is not mutated.
func RegisterChannel(c Channel, pid *Identifier, ref string) *Identifier {
	id := idGen.genID()
	var parent int64
	isTopChannel := true
	if pid != nil {
		isTopChannel = false
		parent = pid.Int()
	}

	if !IsOn() {
		return newIdentifer(RefChannel, id, pid)
	}

	cn := &channel{
		refName:     ref,
		c:           c,
		subChans:    make(map[int64]string),
		nestedChans: make(map[int64]string),
		id:          id,
		pid:         parent,
		trace:       &channelTrace{createdTime: time.Now(), events: make([]*TraceEvent, 0, getMaxTraceEntry())},
	}
	db.get().addChannel(id, cn, isTopChannel, parent)
	return newIdentifer(RefChannel, id, pid)
}

// RegisterSubChannel registers the given subChannel c in the channelz database
// with ref as its reference name, and adds it to the child list of its parent
// (identified by pid).
//
// Returns a unique channelz identifier assigned to this subChannel.
//
// If channelz is not turned ON, the channelz database is not mutated.
func RegisterSubChannel(c Channel, pid *Identifier, ref string) (*Identifier, error) {
	if pid == nil {
		return nil, errors.New("a SubChannel's parent id cannot be nil")
	}
	id := idGen.genID()
	if !IsOn() {
		return newIdentifer(RefSubChannel, id, pid), nil
	}

	sc := &subChannel{
		refName: ref,
		c:       c,
		sockets: make(map[int64]string),
		id:      id,
		pid:     pid.Int(),
		trace:   &channelTrace{createdTime: time.Now(), events: make([]*TraceEvent, 0, getMaxTraceEntry())},
	}
	db.get().addSubChannel(id, sc, pid.Int())
	return newIdentifer(RefSubChannel, id, pid), nil
}

// RegisterServer registers the given server s in channelz database. It returns
// the unique channelz tracking id assigned to this server.
//
// If channelz is not turned ON, the channelz database is not mutated.
func RegisterServer(s Server, ref string) *Identifier {
	id := idGen.genID()
	if !IsOn() {
		return newIdentifer(RefServer, id, nil)
	}

	svr := &server{
		refName:       ref,
		s:             s,
		sockets:       make(map[int64]string),
		listenSockets: make(map[int64]string),
		id:            id,
	}
	db.get().addServer(id, svr)
	return newIdentifer(RefServer, id, nil)
}

// RegisterListenSocket registers the given listen socket s in channelz database
// with ref as its reference name, and add it to the child list of its parent
// (identified by pid). It returns the unique channelz tracking id assigned to
// this listen socket.
//
// If channelz is not turned ON, the channelz database is not mutated.
func RegisterListenSocket(s Socket, pid *Identifier, ref string) (*Identifier, error) {
	if pid == nil {
		return nil, errors.New("a ListenSocket's parent id cannot be 0")
	}
	id := idGen.genID()
	if !IsOn() {
		return newIdentifer(RefListenSocket, id, pid), nil
	}

	ls := &listenSocket{refName: ref, s: s, id: id, pid: pid.Int()}
	db.get().addListenSocket(id, ls, pid.Int())
	return newIdentifer(RefListenSocket, id, pid), nil
}

// RegisterNormalSocket registers the given normal socket s in channelz database
// with ref as its reference name, and adds it to the child list of its parent
// (identified by pid). It returns the unique channelz tracking id assigned to
// this normal socket.
//
// If channelz is not turned ON, the channelz database is not mutated.
func RegisterNormalSocket(s Socket, pid *Identifier, ref string) (*Identifier, error) {
	if pid == nil {
		return nil, errors.New("a NormalSocket's parent id cannot be 0")
	}
	id := idGen.genID()
	if !IsOn() {
		return newIdentifer(RefNormalSocket, id, pid), nil
	}

	ns := &normalSocket{refName: ref, s: s, id: id, pid: pid.Int()}
	db.get().addNormalSocket(id, ns, pid.Int())
	return newIdentifer(RefNormalSocket, id, pid), nil
}

// RemoveEntry removes an entry with unique channelz tracking id to be id from
// channelz database.
//
// If channelz is not turned ON, this function is a no-op.
func RemoveEntry(id *Identifier) {
	if !IsOn() {
		return
	}
	db.get().removeEntry(id.Int())
}

// TraceEventDesc is what the caller of AddTraceEvent should provide to describe
// the event to be added to the channel trace.
//
// The Parent field is optional. It is used for an event that will be recorded
// in the entity's parent trace.
type TraceEventDesc struct {
	Desc     string
	Severity Severity
	Parent   *TraceEventDesc
}

// AddTraceEvent adds trace related to the entity with specified id, using the
// provided TraceEventDesc.
//
// If channelz is not turned ON, this will simply log the event descriptions.
func AddTraceEvent(l grpclog.DepthLoggerV2, id *Identifier, depth int, desc *TraceEventDesc) {
	// Log only the trace description associated with the bottom most entity.
	switch desc.Severity {
	case CtUnknown, CtInfo:
		l.InfoDepth(depth+1, withParens(id)+desc.Desc)
	case CtWarning:
		l.WarningDepth(depth+1, withParens(id)+desc.Desc)
	case CtError:
		l.ErrorDepth(depth+1, withParens(id)+desc.Desc)
	}

	if getMaxTraceEntry() == 0 {
		return
	}
	if IsOn() {
		db.get().traceEvent(id.Int(), desc)
	}
}

// channelMap is the storage data structure for channelz.
// Methods of channelMap can be divided in two two categories with respect to locking.
// 1. Methods acquire the global lock.
// 2. Methods that can only be called when global lock is held.
// A second type of method need always to be called inside a first type of method.
type channelMap struct {
	mu               sync.RWMutex
	topLevelChannels map[int64]struct{}
	servers          map[int64]*server
	channels         map[int64]*channel
	subChannels      map[int64]*subChannel
	listenSockets    map[int64]*listenSocket
	normalSockets    map[int64]*normalSocket
}

func newChannelMap() *channelMap {
	return &channelMap{
		topLevelChannels: make(map[int64]struct{}),
		channels:         make(map[int64]*channel),
		listenSockets:    make(map[int64]*listenSocket),
		normalSockets:    make(map[int64]*normalSocket),
		servers:          make(map[int64]*server),
		subChannels:      make(map[int64]*subChannel),
	}
}

func (c *channelMap) addServer(id int64, s *server) {
	c.mu.Lock()
	s.cm = c
	c.servers[id] = s
	c.mu.Unlock()
}

func (c *channelMap) addChannel(id int64, cn *channel, isTopChannel bool, pid int64) {
	c.mu.Lock()
	cn.cm = c
	cn.trace.cm = c
	c.channels[id] = cn
	if isTopChannel {
		c.topLevelChannels[id] = struct{}{}
	} else {
		c.findEntry(pid).addChild(id, cn)
	}
	c.mu.Unlock()
}

func (c *channelMap) addSubChannel(id int64, sc *subChannel, pid int64) {
	c.mu.Lock()
	sc.cm = c
	sc.trace.cm = c
	c.subChannels[id] = sc
	c.findEntry(pid).addChild(id, sc)
	c.mu.Unlock()
}

func (c *channelMap) addListenSocket(id int64, ls *listenSocket, pid int64) {
	c.mu.Lock()
	ls.cm = c
	c.listenSockets[id] = ls
	c.findEntry(pid).addChild(id, ls)
	c.mu.Unlock()
}

func (c *channelMap) addNormalSocket(id int64, ns *normalSocket, pid int64) {
	c.mu.Lock()
	ns.cm = c
	c.normalSockets[id] = ns
	c.findEntry(pid).addChild(id, ns)
	c.mu.Unlock()
}

// removeEntry triggers the removal of an entry, which may not indeed delete the entry, if it has to
// wait on the deletion of its children and until no other entity's channel trace references it.
// It may lead to a chain of entry deletion. For example, deleting the last socket of a gracefully
// shutting down server will lead to the server being also deleted.
func (c *channelMap) removeEntry(id int64) {
	c.mu.Lock()
	c.findEntry(id).triggerDelete()
	c.mu.Unlock()
}

// c.mu must be held by the caller
func (c *channelMap) decrTraceRefCount(id int64) {
	e := c.findEntry(id)
	if v, ok := e.(tracedChannel); ok {
		v.decrTraceRefCount()
		e.deleteSelfIfReady()
	}
}

// c.mu must be held by the caller.
func (c *channelMap) findEntry(id int64) entry {
	var v entry
	var ok bool
	if v, ok = c.channels[id]; ok {
		return v
	}
	if v, ok = c.subChannels[id]; ok {
		return v
	}
	if v, ok = c.servers[id]; ok {
		return v
	}
	if v, ok = c.listenSockets[id]; ok {
		return v
	}
	if v, ok = c.normalSockets[id]; ok {
		return v
	}
	return &dummyEntry{idNotFound: id}
}

// c.mu must be held by the caller
// deleteEntry simply deletes an entry from the channelMap. Before calling this
// method, caller must check this entry is ready to be deleted, i.e removeEntry()
// has been called on it, and no children still exist.
// Conditionals are ordered by the expected frequency of deletion of each entity
// type, in order to optimize performance.
func (c *channelMap) deleteEntry(id int64) {
	var ok bool
	if _, ok = c.normalSockets[id]; ok {
		delete(c.normalSockets, id)
		return
	}
	if _, ok = c.subChannels[id]; ok {
		delete(c.subChannels, id)
		return
	}
	if _, ok = c.channels[id]; ok {
		delete(c.channels, id)
		delete(c.topLevelChannels, id)
		return
	}
	if _, ok = c.listenSockets[id]; ok {
		delete(c.listenSockets, id)
		return
	}
	if _, ok = c.servers[id]; ok {
		delete(c.servers, id)
		return
	}
}

func (c *channelMap) traceEvent(id int64, desc *TraceEventDesc) {
	c.mu.Lock()
	child := c.findEntry(id)
	childTC, ok := child.(tracedChannel)
	if !ok {
		c.mu.Unlock()
		return
	}
	childTC.getChannelTrace().append(&TraceEvent{Desc: desc.Desc, Severity: desc.Severity, Timestamp: time.Now()})
	if desc.Parent != nil {
		parent := c.findEntry(child.getParentID())
		var chanType RefChannelType
		switch child.(type) {
		case *channel:
			chanType = RefChannel
		case *subChannel:
			chanType = RefSubChannel
		}
		if parentTC, ok := parent.(tracedChannel); ok {
			parentTC.getChannelTrace().append(&TraceEvent{
				Desc:      desc.Parent.Desc,
				Severity:  desc.Parent.Severity,
				Timestamp: time.Now(),
				RefID:     id,
				RefName:   childTC.getRefName(),
				RefType:   chanType,
			})
			childTC.incrTraceRefCount()
		}
	}
	c.mu.Unlock()
}

type int64Slice []int64

func (s int64Slice) Len() int           { return len(s) }
func (s int64Slice) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s int64Slice) Less(i, j int) bool { return s[i] < s[j] }

func copyMap(m map[int64]string) map[int64]string {
	n := make(map[int64]string)
	for k, v := range m {
		n[k] = v
	}
	return n
}

func min(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func (c *channelMap) GetTopChannels(id int64, maxResults int64) ([]*ChannelMetric, bool) {
	if maxResults <= 0 {
		maxResults = EntryPerPage
	}
	c.mu.RLock()
	l := int64(len(c.topLevelChannels))
	ids := make([]int64, 0, l)
	cns := make([]*channel, 0, min(l, maxResults))

	for k := range c.topLevelChannels {
		ids = append(ids, k)
	}
	sort.Sort(int64Slice(ids))
	idx := sort.Search(len(ids), func(i int) bool { return ids[i] >= id })
	count := int64(0)
	var end bool
	var t []*ChannelMetric
	for i, v := range ids[idx:] {
		if count == maxResults {
			break
		}
		if cn, ok := c.channels[v]; ok {
			cns = append(cns, cn)
			t = append(t, &ChannelMetric{
				NestedChans: copyMap(cn.nestedChans),
				SubChans:    copyMap(cn.subChans),
			})
			count++
		}
		if i == len(ids[idx:])-1 {
			end = true
			break
		}
	}
	c.mu.RUnlock()
	if count == 0 {
		end = true
	}

	for i, cn := range cns {
		t[i].ChannelData = cn.c.ChannelzMetric()
		t[i].ID = cn.id
		t[i].RefName = cn.refName
		t[i].Trace = cn.trace.dumpData()
	}
	return t, end
}

func (c *channelMap) GetServers(id, maxResults int64) ([]*ServerMetric, bool) {
	if maxResults <= 0 {
		maxResults = EntryPerPage
	}
	c.mu.RLock()
	l := int64(len(c.servers))
	ids := make([]int64, 0, l)
	ss := make([]*server, 0, min(l, maxResults))
	for k := range c.servers {
		ids = append(ids, k)
	}
	sort.Sort(int64Slice(ids))
	idx := sort.Search(len(ids), func(i int) bool { return ids[i] >= id })
	count := int64(0)
	var end bool
	var s []*ServerMetric
	for i, v := range ids[idx:] {
		if count == maxResults {
			break
		}
		if svr, ok := c.servers[v]; ok {
			ss = append(ss, svr)
			s = append(s, &ServerMetric{
				ListenSockets: copyMap(svr.listenSockets),
			})
			count++
		}
		if i == len(ids[idx:])-1 {
			end = true
			break
		}
	}
	c.mu.RUnlock()
	if count == 0 {
		end = true
	}

	for i, svr := range ss {
		s[i].ServerData = svr.s.ChannelzMetric()
		s[i].ID = svr.id
		s[i].RefName = svr.refName
	}
	return s, end
}

func (c *channelMap) GetServerSockets(id int64, startID int64, maxResults int64) ([]*SocketMetric, bool) {
	if maxResults <= 0 {
		maxResults = EntryPerPage
	}
	var svr *server
	var ok bool
	c.mu.RLock()
	if svr, ok = c.servers[id]; !ok {
		// server with id doesn't exist.
		c.mu.RUnlock()
		return nil, true
	}
	svrskts := svr.sockets
	l := int64(len(svrskts))
	ids := make([]int64, 0, l)
	sks := make([]*normalSocket, 0, min(l, maxResults))
	for k := range svrskts {
		ids = append(ids, k)
	}
	sort.Sort(int64Slice(ids))
	idx := sort.Search(len(ids), func(i int) bool { return ids[i] >= startID })
	count := int64(0)
	var end bool
	for i, v := range ids[idx:] {
		if count == maxResults {
			break
		}
		if ns, ok := c.normalSockets[v]; ok {
			sks = append(sks, ns)
			count++
		}
		if i == len(ids[idx:])-1 {
			end = true
			break
		}
	}
	c.mu.RUnlock()
	if count == 0 {
		end = true
	}
	s := make([]*SocketMetric, 0, len(sks))
	for _, ns := range sks {
		sm := &SocketMetric{}
		sm.SocketData = ns.s.ChannelzMetric()
		sm.ID = ns.id
		sm.RefName = ns.refName
		s = append(s, sm)
	}
	return s, end
}

func (c *channelMap) GetChannel(id int64) *ChannelMetric {
	cm := &ChannelMetric{}
	var cn *channel
	var ok bool
	c.mu.RLock()
	if cn, ok = c.channels[id]; !ok {
		// channel with id doesn't exist.
		c.mu.RUnlock()
		return nil
	}
	cm.NestedChans = copyMap(cn.nestedChans)
	cm.SubChans = copyMap(cn.subChans)
	// cn.c can be set to &dummyChannel{} when deleteSelfFromMap is called. Save a copy of cn.c when
	// holding the lock to prevent potential data race.
	chanCopy := cn.c
	c.mu.RUnlock()
	cm.ChannelData = chanCopy.ChannelzMetric()
	cm.ID = cn.id
	cm.RefName = cn.refName
	cm.Trace = cn.trace.dumpData()
	return cm
}

func (c *channelMap) GetSubChannel(id int64) *SubChannelMetric {
	cm := &SubChannelMetric{}
	var sc *subChannel
	var ok bool
	c.mu.RLock()
	if sc, ok = c.subChannels[id]; !ok {
		// subchannel with id doesn't exist.
		c.mu.RUnlock()
		return nil
	}
	cm.Sockets = copyMap(sc.sockets)
	// sc.c can be set to &dummyChannel{} when deleteSelfFromMap is called. Save a copy of sc.c when
	// holding the lock to prevent potential data race.
	chanCopy := sc.c
	c.mu.RUnlock()
	cm.ChannelData = chanCopy.ChannelzMetric()
	cm.ID = sc.id
	cm.RefName = sc.refName
	cm.Trace = sc.trace.dumpData()
	return cm
}

func (c *channelMap) GetSocket(id int64) *SocketMetric {
	sm := &SocketMetric{}
	c.mu.RLock()
	if ls, ok := c.listenSockets[id]; ok {
		c.mu.RUnlock()
		sm.SocketData = ls.s.ChannelzMetric()
		sm.ID = ls.id
		sm.RefName = ls.refName
		return sm
	}
	if ns, ok := c.normalSockets[id]; ok {
		c.mu.RUnlock()
		sm.SocketData = ns.s.ChannelzMetric()
		sm.ID = ns.id
		sm.RefName = ns.refName
		return sm
	}
	c.mu.RUnlock()
	return nil
}

func (c *channelMap) GetServer(id int64) *ServerMetric {
	sm := &ServerMetric{}
	var svr *server
	var ok bool
	c.mu.RLock()
	if svr, ok = c.servers[id]; !ok {
		c.mu.RUnlock()
		return nil
	}
	sm.ListenSockets = copyMap(svr.listenSockets)
	c.mu.RUnlock()
	sm.ID = svr.id
	sm.RefName = svr.refName
	sm.ServerData = svr.s.ChannelzMetric()
	return sm
}

type idGenerator struct {
	id int64
}

func (i *idGenerator) reset() {
	atomic.StoreInt64(&i.id, 0)
}

func (i *idGenerator) genID() int64 {
	return atomic.AddInt64(&i.id, 1)
}
