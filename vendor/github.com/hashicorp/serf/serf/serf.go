package serf

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"net"
	"strconv"
	"sync"
	"time"

	"github.com/armon/go-metrics"
	"github.com/hashicorp/go-msgpack/codec"
	"github.com/hashicorp/memberlist"
	"github.com/hashicorp/serf/coordinate"
)

// These are the protocol versions that Serf can _understand_. These are
// Serf-level protocol versions that are passed down as the delegate
// version to memberlist below.
const (
	ProtocolVersionMin uint8 = 2
	ProtocolVersionMax       = 4
)

const (
	// Used to detect if the meta data is tags
	// or if it is a raw role
	tagMagicByte uint8 = 255
)

var (
	// FeatureNotSupported is returned if a feature cannot be used
	// due to an older protocol version being used.
	FeatureNotSupported = fmt.Errorf("Feature not supported")
)

func init() {
	// Seed the random number generator
	rand.Seed(time.Now().UnixNano())
}

// Serf is a single node that is part of a single cluster that gets
// events about joins/leaves/failures/etc. It is created with the Create
// method.
//
// All functions on the Serf structure are safe to call concurrently.
type Serf struct {
	// The clocks for different purposes. These MUST be the first things
	// in this struct due to Golang issue #599.
	clock      LamportClock
	eventClock LamportClock
	queryClock LamportClock

	broadcasts    *memberlist.TransmitLimitedQueue
	config        *Config
	failedMembers []*memberState
	leftMembers   []*memberState
	memberlist    *memberlist.Memberlist
	memberLock    sync.RWMutex
	members       map[string]*memberState

	// Circular buffers for recent intents, used
	// in case we get the intent before the relevant event
	recentLeave      []nodeIntent
	recentLeaveIndex int
	recentJoin       []nodeIntent
	recentJoinIndex  int

	eventBroadcasts *memberlist.TransmitLimitedQueue
	eventBuffer     []*userEvents
	eventJoinIgnore bool
	eventMinTime    LamportTime
	eventLock       sync.RWMutex

	queryBroadcasts *memberlist.TransmitLimitedQueue
	queryBuffer     []*queries
	queryMinTime    LamportTime
	queryResponse   map[LamportTime]*QueryResponse
	queryLock       sync.RWMutex

	logger     *log.Logger
	joinLock   sync.Mutex
	stateLock  sync.Mutex
	state      SerfState
	shutdownCh chan struct{}

	snapshotter *Snapshotter
	keyManager  *KeyManager

	coordClient    *coordinate.Client
	coordCache     map[string]*coordinate.Coordinate
	coordCacheLock sync.RWMutex
}

// SerfState is the state of the Serf instance.
type SerfState int

const (
	SerfAlive SerfState = iota
	SerfLeaving
	SerfLeft
	SerfShutdown
)

func (s SerfState) String() string {
	switch s {
	case SerfAlive:
		return "alive"
	case SerfLeaving:
		return "leaving"
	case SerfLeft:
		return "left"
	case SerfShutdown:
		return "shutdown"
	default:
		return "unknown"
	}
}

// Member is a single member of the Serf cluster.
type Member struct {
	Name   string
	Addr   net.IP
	Port   uint16
	Tags   map[string]string
	Status MemberStatus

	// The minimum, maximum, and current values of the protocol versions
	// and delegate (Serf) protocol versions that each member can understand
	// or is speaking.
	ProtocolMin uint8
	ProtocolMax uint8
	ProtocolCur uint8
	DelegateMin uint8
	DelegateMax uint8
	DelegateCur uint8
}

// MemberStatus is the state that a member is in.
type MemberStatus int

const (
	StatusNone MemberStatus = iota
	StatusAlive
	StatusLeaving
	StatusLeft
	StatusFailed
)

func (s MemberStatus) String() string {
	switch s {
	case StatusNone:
		return "none"
	case StatusAlive:
		return "alive"
	case StatusLeaving:
		return "leaving"
	case StatusLeft:
		return "left"
	case StatusFailed:
		return "failed"
	default:
		panic(fmt.Sprintf("unknown MemberStatus: %d", s))
	}
}

// memberState is used to track members that are no longer active due to
// leaving, failing, partitioning, etc. It tracks the member along with
// when that member was marked as leaving.
type memberState struct {
	Member
	statusLTime LamportTime // lamport clock time of last received message
	leaveTime   time.Time   // wall clock time of leave
}

// nodeIntent is used to buffer intents for out-of-order deliveries
type nodeIntent struct {
	LTime LamportTime
	Node  string
}

// userEvent is used to buffer events to prevent re-delivery
type userEvent struct {
	Name    string
	Payload []byte
}

func (ue *userEvent) Equals(other *userEvent) bool {
	if ue.Name != other.Name {
		return false
	}
	if bytes.Compare(ue.Payload, other.Payload) != 0 {
		return false
	}
	return true
}

// userEvents stores all the user events at a specific time
type userEvents struct {
	LTime  LamportTime
	Events []userEvent
}

// queries stores all the query ids at a specific time
type queries struct {
	LTime    LamportTime
	QueryIDs []uint32
}

const (
	UserEventSizeLimit     = 512        // Maximum byte size for event name and payload
	snapshotSizeLimit      = 128 * 1024 // Maximum 128 KB snapshot
)

// Create creates a new Serf instance, starting all the background tasks
// to maintain cluster membership information.
//
// After calling this function, the configuration should no longer be used
// or modified by the caller.
func Create(conf *Config) (*Serf, error) {
	conf.Init()
	if conf.ProtocolVersion < ProtocolVersionMin {
		return nil, fmt.Errorf("Protocol version '%d' too low. Must be in range: [%d, %d]",
			conf.ProtocolVersion, ProtocolVersionMin, ProtocolVersionMax)
	} else if conf.ProtocolVersion > ProtocolVersionMax {
		return nil, fmt.Errorf("Protocol version '%d' too high. Must be in range: [%d, %d]",
			conf.ProtocolVersion, ProtocolVersionMin, ProtocolVersionMax)
	}

	serf := &Serf{
		config:        conf,
		logger:        log.New(conf.LogOutput, "", log.LstdFlags),
		members:       make(map[string]*memberState),
		queryResponse: make(map[LamportTime]*QueryResponse),
		shutdownCh:    make(chan struct{}),
		state:         SerfAlive,
	}

	// Check that the meta data length is okay
	if len(serf.encodeTags(conf.Tags)) > memberlist.MetaMaxSize {
		return nil, fmt.Errorf("Encoded length of tags exceeds limit of %d bytes", memberlist.MetaMaxSize)
	}

	// Check if serf member event coalescing is enabled
	if conf.CoalescePeriod > 0 && conf.QuiescentPeriod > 0 && conf.EventCh != nil {
		c := &memberEventCoalescer{
			lastEvents:   make(map[string]EventType),
			latestEvents: make(map[string]coalesceEvent),
		}

		conf.EventCh = coalescedEventCh(conf.EventCh, serf.shutdownCh,
			conf.CoalescePeriod, conf.QuiescentPeriod, c)
	}

	// Check if user event coalescing is enabled
	if conf.UserCoalescePeriod > 0 && conf.UserQuiescentPeriod > 0 && conf.EventCh != nil {
		c := &userEventCoalescer{
			events: make(map[string]*latestUserEvents),
		}

		conf.EventCh = coalescedEventCh(conf.EventCh, serf.shutdownCh,
			conf.UserCoalescePeriod, conf.UserQuiescentPeriod, c)
	}

	// Listen for internal Serf queries. This is setup before the snapshotter, since
	// we want to capture the query-time, but the internal listener does not passthrough
	// the queries
	outCh, err := newSerfQueries(serf, serf.logger, conf.EventCh, serf.shutdownCh)
	if err != nil {
		return nil, fmt.Errorf("Failed to setup serf query handler: %v", err)
	}
	conf.EventCh = outCh

	// Set up network coordinate client.
	if !conf.DisableCoordinates {
		serf.coordClient, err = coordinate.NewClient(coordinate.DefaultConfig())
		if err != nil {
			return nil, fmt.Errorf("Failed to create coordinate client: %v", err)
		}
	}

	// Try access the snapshot
	var oldClock, oldEventClock, oldQueryClock LamportTime
	var prev []*PreviousNode
	if conf.SnapshotPath != "" {
		eventCh, snap, err := NewSnapshotter(
			conf.SnapshotPath,
			snapshotSizeLimit,
			conf.RejoinAfterLeave,
			serf.logger,
			&serf.clock,
			serf.coordClient,
			conf.EventCh,
			serf.shutdownCh)
		if err != nil {
			return nil, fmt.Errorf("Failed to setup snapshot: %v", err)
		}
		serf.snapshotter = snap
		conf.EventCh = eventCh
		prev = snap.AliveNodes()
		oldClock = snap.LastClock()
		oldEventClock = snap.LastEventClock()
		oldQueryClock = snap.LastQueryClock()
		serf.eventMinTime = oldEventClock + 1
		serf.queryMinTime = oldQueryClock + 1
	}

	// Set up the coordinate cache. We do this after we read the snapshot to
	// make sure we get a good initial value from there, if we got one.
	if !conf.DisableCoordinates {
		serf.coordCache = make(map[string]*coordinate.Coordinate)
		serf.coordCache[conf.NodeName] = serf.coordClient.GetCoordinate()
	}

	// Setup the various broadcast queues, which we use to send our own
	// custom broadcasts along the gossip channel.
	serf.broadcasts = &memberlist.TransmitLimitedQueue{
		NumNodes: func() int {
			return len(serf.members)
		},
		RetransmitMult: conf.MemberlistConfig.RetransmitMult,
	}
	serf.eventBroadcasts = &memberlist.TransmitLimitedQueue{
		NumNodes: func() int {
			return len(serf.members)
		},
		RetransmitMult: conf.MemberlistConfig.RetransmitMult,
	}
	serf.queryBroadcasts = &memberlist.TransmitLimitedQueue{
		NumNodes: func() int {
			return len(serf.members)
		},
		RetransmitMult: conf.MemberlistConfig.RetransmitMult,
	}

	// Create the buffer for recent intents
	serf.recentJoin = make([]nodeIntent, conf.RecentIntentBuffer)
	serf.recentLeave = make([]nodeIntent, conf.RecentIntentBuffer)

	// Create a buffer for events and queries
	serf.eventBuffer = make([]*userEvents, conf.EventBuffer)
	serf.queryBuffer = make([]*queries, conf.QueryBuffer)

	// Ensure our lamport clock is at least 1, so that the default
	// join LTime of 0 does not cause issues
	serf.clock.Increment()
	serf.eventClock.Increment()
	serf.queryClock.Increment()

	// Restore the clock from snap if we have one
	serf.clock.Witness(oldClock)
	serf.eventClock.Witness(oldEventClock)
	serf.queryClock.Witness(oldQueryClock)

	// Modify the memberlist configuration with keys that we set
	conf.MemberlistConfig.Events = &eventDelegate{serf: serf}
	conf.MemberlistConfig.Conflict = &conflictDelegate{serf: serf}
	conf.MemberlistConfig.Delegate = &delegate{serf: serf}
	conf.MemberlistConfig.DelegateProtocolVersion = conf.ProtocolVersion
	conf.MemberlistConfig.DelegateProtocolMin = ProtocolVersionMin
	conf.MemberlistConfig.DelegateProtocolMax = ProtocolVersionMax
	conf.MemberlistConfig.Name = conf.NodeName
	conf.MemberlistConfig.ProtocolVersion = ProtocolVersionMap[conf.ProtocolVersion]
	if !conf.DisableCoordinates {
		conf.MemberlistConfig.Ping = &pingDelegate{serf: serf}
	}

	// Setup a merge delegate if necessary
	if conf.Merge != nil {
		md := &mergeDelegate{serf: serf}
		conf.MemberlistConfig.Merge = md
		conf.MemberlistConfig.Alive = md
	}

	// Create the underlying memberlist that will manage membership
	// and failure detection for the Serf instance.
	memberlist, err := memberlist.Create(conf.MemberlistConfig)
	if err != nil {
		return nil, fmt.Errorf("Failed to create memberlist: %v", err)
	}

	serf.memberlist = memberlist

	// Create a key manager for handling all encryption key changes
	serf.keyManager = &KeyManager{serf: serf}

	// Start the background tasks. See the documentation above each method
	// for more information on their role.
	go serf.handleReap()
	go serf.handleReconnect()
	go serf.checkQueueDepth("Intent", serf.broadcasts)
	go serf.checkQueueDepth("Event", serf.eventBroadcasts)
	go serf.checkQueueDepth("Query", serf.queryBroadcasts)

	// Attempt to re-join the cluster if we have known nodes
	if len(prev) != 0 {
		go serf.handleRejoin(prev)
	}

	return serf, nil
}

// ProtocolVersion returns the current protocol version in use by Serf.
// This is the Serf protocol version, not the memberlist protocol version.
func (s *Serf) ProtocolVersion() uint8 {
	return s.config.ProtocolVersion
}

// EncryptionEnabled is a predicate that determines whether or not encryption
// is enabled, which can be possible in one of 2 cases:
//   - Single encryption key passed at agent start (no persistence)
//   - Keyring file provided at agent start
func (s *Serf) EncryptionEnabled() bool {
	return s.config.MemberlistConfig.Keyring != nil
}

// KeyManager returns the key manager for the current Serf instance.
func (s *Serf) KeyManager() *KeyManager {
	return s.keyManager
}

// UserEvent is used to broadcast a custom user event with a given
// name and payload. The events must be fairly small, and if the
// size limit is exceeded and error will be returned. If coalesce is enabled,
// nodes are allowed to coalesce this event. Coalescing is only available
// starting in v0.2
func (s *Serf) UserEvent(name string, payload []byte, coalesce bool) error {
	// Check the size limit
	if len(name)+len(payload) > UserEventSizeLimit {
		return fmt.Errorf("user event exceeds limit of %d bytes", UserEventSizeLimit)
	}

	// Create a message
	msg := messageUserEvent{
		LTime:   s.eventClock.Time(),
		Name:    name,
		Payload: payload,
		CC:      coalesce,
	}
	s.eventClock.Increment()

	// Process update locally
	s.handleUserEvent(&msg)

	// Start broadcasting the event
	raw, err := encodeMessage(messageUserEventType, &msg)
	if err != nil {
		return err
	}
	s.eventBroadcasts.QueueBroadcast(&broadcast{
		msg: raw,
	})
	return nil
}

// Query is used to broadcast a new query. The query must be fairly small,
// and an error will be returned if the size limit is exceeded. This is only
// available with protocol version 4 and newer. Query parameters are optional,
// and if not provided, a sane set of defaults will be used.
func (s *Serf) Query(name string, payload []byte, params *QueryParam) (*QueryResponse, error) {
	// Check that the latest protocol is in use
	if s.ProtocolVersion() < 4 {
		return nil, FeatureNotSupported
	}

	// Provide default parameters if none given
	if params == nil {
		params = s.DefaultQueryParams()
	} else if params.Timeout == 0 {
		params.Timeout = s.DefaultQueryTimeout()
	}

	// Get the local node
	local := s.memberlist.LocalNode()

	// Encode the filters
	filters, err := params.encodeFilters()
	if err != nil {
		return nil, fmt.Errorf("Failed to format filters: %v", err)
	}

	// Setup the flags
	var flags uint32
	if params.RequestAck {
		flags |= queryFlagAck
	}

	// Create a message
	q := messageQuery{
		LTime:   s.queryClock.Time(),
		ID:      uint32(rand.Int31()),
		Addr:    local.Addr,
		Port:    local.Port,
		Filters: filters,
		Flags:   flags,
		Timeout: params.Timeout,
		Name:    name,
		Payload: payload,
	}

	// Encode the query
	raw, err := encodeMessage(messageQueryType, &q)
	if err != nil {
		return nil, err
	}

	// Check the size
	if len(raw) > s.config.QuerySizeLimit {
		return nil, fmt.Errorf("query exceeds limit of %d bytes", s.config.QuerySizeLimit)
	}

	// Register QueryResponse to track acks and responses
	resp := newQueryResponse(s.memberlist.NumMembers(), &q)
	s.registerQueryResponse(params.Timeout, resp)

	// Process query locally
	s.handleQuery(&q)

	// Start broadcasting the event
	s.queryBroadcasts.QueueBroadcast(&broadcast{
		msg: raw,
	})
	return resp, nil
}

// registerQueryResponse is used to setup the listeners for the query,
// and to schedule closing the query after the timeout.
func (s *Serf) registerQueryResponse(timeout time.Duration, resp *QueryResponse) {
	s.queryLock.Lock()
	defer s.queryLock.Unlock()

	// Map the LTime to the QueryResponse. This is necessarily 1-to-1,
	// since we increment the time for each new query.
	s.queryResponse[resp.lTime] = resp

	// Setup a timer to close the response and deregister after the timeout
	time.AfterFunc(timeout, func() {
		s.queryLock.Lock()
		delete(s.queryResponse, resp.lTime)
		resp.Close()
		s.queryLock.Unlock()
	})
}

// SetTags is used to dynamically update the tags associated with
// the local node. This will propagate the change to the rest of
// the cluster. Blocks until a the message is broadcast out.
func (s *Serf) SetTags(tags map[string]string) error {
	// Check that the meta data length is okay
	if len(s.encodeTags(tags)) > memberlist.MetaMaxSize {
		return fmt.Errorf("Encoded length of tags exceeds limit of %d bytes",
			memberlist.MetaMaxSize)
	}

	// Update the config
	s.config.Tags = tags

	// Trigger a memberlist update
	return s.memberlist.UpdateNode(s.config.BroadcastTimeout)
}

// Join joins an existing Serf cluster. Returns the number of nodes
// successfully contacted. The returned error will be non-nil only in the
// case that no nodes could be contacted. If ignoreOld is true, then any
// user messages sent prior to the join will be ignored.
func (s *Serf) Join(existing []string, ignoreOld bool) (int, error) {
	// Do a quick state check
	if s.State() != SerfAlive {
		return 0, fmt.Errorf("Serf can't Join after Leave or Shutdown")
	}

	// Hold the joinLock, this is to make eventJoinIgnore safe
	s.joinLock.Lock()
	defer s.joinLock.Unlock()

	// Ignore any events from a potential join. This is safe since we hold
	// the joinLock and nobody else can be doing a Join
	if ignoreOld {
		s.eventJoinIgnore = true
		defer func() {
			s.eventJoinIgnore = false
		}()
	}

	// Have memberlist attempt to join
	num, err := s.memberlist.Join(existing)

	// If we joined any nodes, broadcast the join message
	if num > 0 {
		// Start broadcasting the update
		if err := s.broadcastJoin(s.clock.Time()); err != nil {
			return num, err
		}
	}

	return num, err
}

// broadcastJoin broadcasts a new join intent with a
// given clock value. It is used on either join, or if
// we need to refute an older leave intent. Cannot be called
// with the memberLock held.
func (s *Serf) broadcastJoin(ltime LamportTime) error {
	// Construct message to update our lamport clock
	msg := messageJoin{
		LTime: ltime,
		Node:  s.config.NodeName,
	}
	s.clock.Witness(ltime)

	// Process update locally
	s.handleNodeJoinIntent(&msg)

	// Start broadcasting the update
	if err := s.broadcast(messageJoinType, &msg, nil); err != nil {
		s.logger.Printf("[WARN] serf: Failed to broadcast join intent: %v", err)
		return err
	}
	return nil
}

// Leave gracefully exits the cluster. It is safe to call this multiple
// times.
func (s *Serf) Leave() error {
	// Check the current state
	s.stateLock.Lock()
	if s.state == SerfLeft {
		s.stateLock.Unlock()
		return nil
	} else if s.state == SerfLeaving {
		s.stateLock.Unlock()
		return fmt.Errorf("Leave already in progress")
	} else if s.state == SerfShutdown {
		s.stateLock.Unlock()
		return fmt.Errorf("Leave called after Shutdown")
	}
	s.state = SerfLeaving
	s.stateLock.Unlock()

	// If we have a snapshot, mark we are leaving
	if s.snapshotter != nil {
		s.snapshotter.Leave()
	}

	// Construct the message for the graceful leave
	msg := messageLeave{
		LTime: s.clock.Time(),
		Node:  s.config.NodeName,
	}
	s.clock.Increment()

	// Process the leave locally
	s.handleNodeLeaveIntent(&msg)

	// Only broadcast the leave message if there is at least one
	// other node alive.
	if s.hasAliveMembers() {
		notifyCh := make(chan struct{})
		if err := s.broadcast(messageLeaveType, &msg, notifyCh); err != nil {
			return err
		}

		select {
		case <-notifyCh:
		case <-time.After(s.config.BroadcastTimeout):
			return errors.New("timeout while waiting for graceful leave")
		}
	}

	// Attempt the memberlist leave
	err := s.memberlist.Leave(s.config.BroadcastTimeout)
	if err != nil {
		return err
	}

	// Transition to Left only if we not already shutdown
	s.stateLock.Lock()
	if s.state != SerfShutdown {
		s.state = SerfLeft
	}
	s.stateLock.Unlock()
	return nil
}

// hasAliveMembers is called to check for any alive members other than
// ourself.
func (s *Serf) hasAliveMembers() bool {
	s.memberLock.RLock()
	defer s.memberLock.RUnlock()

	hasAlive := false
	for _, m := range s.members {
		// Skip ourself, we want to know if OTHER members are alive
		if m.Name == s.config.NodeName {
			continue
		}

		if m.Status == StatusAlive {
			hasAlive = true
			break
		}
	}
	return hasAlive
}

// LocalMember returns the Member information for the local node
func (s *Serf) LocalMember() Member {
	s.memberLock.RLock()
	defer s.memberLock.RUnlock()
	return s.members[s.config.NodeName].Member
}

// Members returns a point-in-time snapshot of the members of this cluster.
func (s *Serf) Members() []Member {
	s.memberLock.RLock()
	defer s.memberLock.RUnlock()

	members := make([]Member, 0, len(s.members))
	for _, m := range s.members {
		members = append(members, m.Member)
	}

	return members
}

// RemoveFailedNode forcibly removes a failed node from the cluster
// immediately, instead of waiting for the reaper to eventually reclaim it.
// This also has the effect that Serf will no longer attempt to reconnect
// to this node.
func (s *Serf) RemoveFailedNode(node string) error {
	// Construct the message to broadcast
	msg := messageLeave{
		LTime: s.clock.Time(),
		Node:  node,
	}
	s.clock.Increment()

	// Process our own event
	s.handleNodeLeaveIntent(&msg)

	// If we have no members, then we don't need to broadcast
	if !s.hasAliveMembers() {
		return nil
	}

	// Broadcast the remove
	notifyCh := make(chan struct{})
	if err := s.broadcast(messageLeaveType, &msg, notifyCh); err != nil {
		return err
	}

	// Wait for the broadcast
	select {
	case <-notifyCh:
	case <-time.After(s.config.BroadcastTimeout):
		return fmt.Errorf("timed out broadcasting node removal")
	}

	return nil
}

// Shutdown forcefully shuts down the Serf instance, stopping all network
// activity and background maintenance associated with the instance.
//
// This is not a graceful shutdown, and should be preceded by a call
// to Leave. Otherwise, other nodes in the cluster will detect this node's
// exit as a node failure.
//
// It is safe to call this method multiple times.
func (s *Serf) Shutdown() error {
	s.stateLock.Lock()
	defer s.stateLock.Unlock()

	if s.state == SerfShutdown {
		return nil
	}

	if s.state != SerfLeft {
		s.logger.Printf("[WARN] serf: Shutdown without a Leave")
	}

	s.state = SerfShutdown
	close(s.shutdownCh)

	err := s.memberlist.Shutdown()
	if err != nil {
		return err
	}

	// Wait for the snapshoter to finish if we have one
	if s.snapshotter != nil {
		s.snapshotter.Wait()
	}

	return nil
}

// ShutdownCh returns a channel that can be used to wait for
// Serf to shutdown.
func (s *Serf) ShutdownCh() <-chan struct{} {
	return s.shutdownCh
}

// Memberlist is used to get access to the underlying Memberlist instance
func (s *Serf) Memberlist() *memberlist.Memberlist {
	return s.memberlist
}

// State is the current state of this Serf instance.
func (s *Serf) State() SerfState {
	s.stateLock.Lock()
	defer s.stateLock.Unlock()
	return s.state
}

// broadcast takes a Serf message type, encodes it for the wire, and queues
// the broadcast. If a notify channel is given, this channel will be closed
// when the broadcast is sent.
func (s *Serf) broadcast(t messageType, msg interface{}, notify chan<- struct{}) error {
	raw, err := encodeMessage(t, msg)
	if err != nil {
		return err
	}

	s.broadcasts.QueueBroadcast(&broadcast{
		msg:    raw,
		notify: notify,
	})
	return nil
}

// handleNodeJoin is called when a node join event is received
// from memberlist.
func (s *Serf) handleNodeJoin(n *memberlist.Node) {
	s.memberLock.Lock()
	defer s.memberLock.Unlock()

	var oldStatus MemberStatus
	member, ok := s.members[n.Name]
	if !ok {
		oldStatus = StatusNone
		member = &memberState{
			Member: Member{
				Name:   n.Name,
				Addr:   net.IP(n.Addr),
				Port:   n.Port,
				Tags:   s.decodeTags(n.Meta),
				Status: StatusAlive,
			},
		}

		// Check if we have a join intent and use the LTime
		if join := recentIntent(s.recentJoin, n.Name); join != nil {
			member.statusLTime = join.LTime
		}

		// Check if we have a leave intent
		if leave := recentIntent(s.recentLeave, n.Name); leave != nil {
			if leave.LTime > member.statusLTime {
				member.Status = StatusLeaving
				member.statusLTime = leave.LTime
			}
		}

		s.members[n.Name] = member
	} else {
		oldStatus = member.Status
		member.Status = StatusAlive
		member.leaveTime = time.Time{}
		member.Addr = net.IP(n.Addr)
		member.Port = n.Port
		member.Tags = s.decodeTags(n.Meta)
	}

	// Update the protocol versions every time we get an event
	member.ProtocolMin = n.PMin
	member.ProtocolMax = n.PMax
	member.ProtocolCur = n.PCur
	member.DelegateMin = n.DMin
	member.DelegateMax = n.DMax
	member.DelegateCur = n.DCur

	// If node was previously in a failed state, then clean up some
	// internal accounting.
	// TODO(mitchellh): needs tests to verify not reaped
	if oldStatus == StatusFailed || oldStatus == StatusLeft {
		s.failedMembers = removeOldMember(s.failedMembers, member.Name)
		s.leftMembers = removeOldMember(s.leftMembers, member.Name)
	}

	// Update some metrics
	metrics.IncrCounter([]string{"serf", "member", "join"}, 1)

	// Send an event along
	s.logger.Printf("[INFO] serf: EventMemberJoin: %s %s",
		member.Member.Name, member.Member.Addr)
	if s.config.EventCh != nil {
		s.config.EventCh <- MemberEvent{
			Type:    EventMemberJoin,
			Members: []Member{member.Member},
		}
	}
}

// handleNodeLeave is called when a node leave event is received
// from memberlist.
func (s *Serf) handleNodeLeave(n *memberlist.Node) {
	s.memberLock.Lock()
	defer s.memberLock.Unlock()

	member, ok := s.members[n.Name]
	if !ok {
		// We've never even heard of this node that is supposedly
		// leaving. Just ignore it completely.
		return
	}

	switch member.Status {
	case StatusLeaving:
		member.Status = StatusLeft
		member.leaveTime = time.Now()
		s.leftMembers = append(s.leftMembers, member)
	case StatusAlive:
		member.Status = StatusFailed
		member.leaveTime = time.Now()
		s.failedMembers = append(s.failedMembers, member)
	default:
		// Unknown state that it was in? Just don't do anything
		s.logger.Printf("[WARN] serf: Bad state when leave: %d", member.Status)
		return
	}

	// Send an event along
	event := EventMemberLeave
	eventStr := "EventMemberLeave"
	if member.Status != StatusLeft {
		event = EventMemberFailed
		eventStr = "EventMemberFailed"
	}

	// Update some metrics
	metrics.IncrCounter([]string{"serf", "member", member.Status.String()}, 1)

	s.logger.Printf("[INFO] serf: %s: %s %s",
		eventStr, member.Member.Name, member.Member.Addr)
	if s.config.EventCh != nil {
		s.config.EventCh <- MemberEvent{
			Type:    event,
			Members: []Member{member.Member},
		}
	}
}

// handleNodeUpdate is called when a node meta data update
// has taken place
func (s *Serf) handleNodeUpdate(n *memberlist.Node) {
	s.memberLock.Lock()
	defer s.memberLock.Unlock()

	member, ok := s.members[n.Name]
	if !ok {
		// We've never even heard of this node that is updating.
		// Just ignore it completely.
		return
	}

	// Update the member attributes
	member.Addr = net.IP(n.Addr)
	member.Port = n.Port
	member.Tags = s.decodeTags(n.Meta)

	// Snag the latest versions. NOTE - the current memberlist code will NOT
	// fire an update event if the metadata (for Serf, tags) stays the same
	// and only the protocol versions change. If we wake any Serf-level
	// protocol changes where we want to get this event under those
	// circumstances, we will need to update memberlist to do a check of
	// versions as well as the metadata.
	member.ProtocolMin = n.PMin
	member.ProtocolMax = n.PMax
	member.ProtocolCur = n.PCur
	member.DelegateMin = n.DMin
	member.DelegateMax = n.DMax
	member.DelegateCur = n.DCur

	// Update some metrics
	metrics.IncrCounter([]string{"serf", "member", "update"}, 1)

	// Send an event along
	s.logger.Printf("[INFO] serf: EventMemberUpdate: %s", member.Member.Name)
	if s.config.EventCh != nil {
		s.config.EventCh <- MemberEvent{
			Type:    EventMemberUpdate,
			Members: []Member{member.Member},
		}
	}
}

// handleNodeLeaveIntent is called when an intent to leave is received.
func (s *Serf) handleNodeLeaveIntent(leaveMsg *messageLeave) bool {
	// Witness a potentially newer time
	s.clock.Witness(leaveMsg.LTime)

	s.memberLock.Lock()
	defer s.memberLock.Unlock()

	member, ok := s.members[leaveMsg.Node]
	if !ok {
		// If we've already seen this message don't rebroadcast
		if recentIntent(s.recentLeave, leaveMsg.Node) != nil {
			return false
		}

		// We don't know this member so store it in a buffer for now
		s.recentLeave[s.recentLeaveIndex] = nodeIntent{
			LTime: leaveMsg.LTime,
			Node:  leaveMsg.Node,
		}
		s.recentLeaveIndex = (s.recentLeaveIndex + 1) % len(s.recentLeave)
		return true
	}

	// If the message is old, then it is irrelevant and we can skip it
	if leaveMsg.LTime <= member.statusLTime {
		return false
	}

	// Refute us leaving if we are in the alive state
	// Must be done in another goroutine since we have the memberLock
	if leaveMsg.Node == s.config.NodeName && s.state == SerfAlive {
		s.logger.Printf("[DEBUG] serf: Refuting an older leave intent")
		go s.broadcastJoin(s.clock.Time())
		return false
	}

	// State transition depends on current state
	switch member.Status {
	case StatusAlive:
		member.Status = StatusLeaving
		member.statusLTime = leaveMsg.LTime
		return true
	case StatusFailed:
		member.Status = StatusLeft
		member.statusLTime = leaveMsg.LTime

		// Remove from the failed list and add to the left list. We add
		// to the left list so that when we do a sync, other nodes will
		// remove it from their failed list.
		s.failedMembers = removeOldMember(s.failedMembers, member.Name)
		s.leftMembers = append(s.leftMembers, member)

		// We must push a message indicating the node has now
		// left to allow higher-level applications to handle the
		// graceful leave.
		s.logger.Printf("[INFO] serf: EventMemberLeave (forced): %s %s",
			member.Member.Name, member.Member.Addr)
		if s.config.EventCh != nil {
			s.config.EventCh <- MemberEvent{
				Type:    EventMemberLeave,
				Members: []Member{member.Member},
			}
		}
		return true
	default:
		return false
	}
}

// handleNodeJoinIntent is called when a node broadcasts a
// join message to set the lamport time of its join
func (s *Serf) handleNodeJoinIntent(joinMsg *messageJoin) bool {
	// Witness a potentially newer time
	s.clock.Witness(joinMsg.LTime)

	s.memberLock.Lock()
	defer s.memberLock.Unlock()

	member, ok := s.members[joinMsg.Node]
	if !ok {
		// If we've already seen this message don't rebroadcast
		if recentIntent(s.recentJoin, joinMsg.Node) != nil {
			return false
		}

		// We don't know this member so store it in a buffer for now
		s.recentJoin[s.recentJoinIndex] = nodeIntent{LTime: joinMsg.LTime, Node: joinMsg.Node}
		s.recentJoinIndex = (s.recentJoinIndex + 1) % len(s.recentJoin)
		return true
	}

	// Check if this time is newer than what we have
	if joinMsg.LTime <= member.statusLTime {
		return false
	}

	// Update the LTime
	member.statusLTime = joinMsg.LTime

	// If we are in the leaving state, we should go back to alive,
	// since the leaving message must have been for an older time
	if member.Status == StatusLeaving {
		member.Status = StatusAlive
	}
	return true
}

// handleUserEvent is called when a user event broadcast is
// received. Returns if the message should be rebroadcast.
func (s *Serf) handleUserEvent(eventMsg *messageUserEvent) bool {
	// Witness a potentially newer time
	s.eventClock.Witness(eventMsg.LTime)

	s.eventLock.Lock()
	defer s.eventLock.Unlock()

	// Ignore if it is before our minimum event time
	if eventMsg.LTime < s.eventMinTime {
		return false
	}

	// Check if this message is too old
	curTime := s.eventClock.Time()
	if curTime > LamportTime(len(s.eventBuffer)) &&
		eventMsg.LTime < curTime-LamportTime(len(s.eventBuffer)) {
		s.logger.Printf(
			"[WARN] serf: received old event %s from time %d (current: %d)",
			eventMsg.Name,
			eventMsg.LTime,
			s.eventClock.Time())
		return false
	}

	// Check if we've already seen this
	idx := eventMsg.LTime % LamportTime(len(s.eventBuffer))
	seen := s.eventBuffer[idx]
	userEvent := userEvent{Name: eventMsg.Name, Payload: eventMsg.Payload}
	if seen != nil && seen.LTime == eventMsg.LTime {
		for _, previous := range seen.Events {
			if previous.Equals(&userEvent) {
				return false
			}
		}
	} else {
		seen = &userEvents{LTime: eventMsg.LTime}
		s.eventBuffer[idx] = seen
	}

	// Add to recent events
	seen.Events = append(seen.Events, userEvent)

	// Update some metrics
	metrics.IncrCounter([]string{"serf", "events"}, 1)
	metrics.IncrCounter([]string{"serf", "events", eventMsg.Name}, 1)

	if s.config.EventCh != nil {
		s.config.EventCh <- UserEvent{
			LTime:    eventMsg.LTime,
			Name:     eventMsg.Name,
			Payload:  eventMsg.Payload,
			Coalesce: eventMsg.CC,
		}
	}
	return true
}

// handleQuery is called when a query broadcast is
// received. Returns if the message should be rebroadcast.
func (s *Serf) handleQuery(query *messageQuery) bool {
	// Witness a potentially newer time
	s.queryClock.Witness(query.LTime)

	s.queryLock.Lock()
	defer s.queryLock.Unlock()

	// Ignore if it is before our minimum query time
	if query.LTime < s.queryMinTime {
		return false
	}

	// Check if this message is too old
	curTime := s.queryClock.Time()
	if curTime > LamportTime(len(s.queryBuffer)) &&
		query.LTime < curTime-LamportTime(len(s.queryBuffer)) {
		s.logger.Printf(
			"[WARN] serf: received old query %s from time %d (current: %d)",
			query.Name,
			query.LTime,
			s.queryClock.Time())
		return false
	}

	// Check if we've already seen this
	idx := query.LTime % LamportTime(len(s.queryBuffer))
	seen := s.queryBuffer[idx]
	if seen != nil && seen.LTime == query.LTime {
		for _, previous := range seen.QueryIDs {
			if previous == query.ID {
				// Seen this ID already
				return false
			}
		}
	} else {
		seen = &queries{LTime: query.LTime}
		s.queryBuffer[idx] = seen
	}

	// Add to recent queries
	seen.QueryIDs = append(seen.QueryIDs, query.ID)

	// Update some metrics
	metrics.IncrCounter([]string{"serf", "queries"}, 1)
	metrics.IncrCounter([]string{"serf", "queries", query.Name}, 1)

	// Check if we should rebroadcast, this may be disabled by a flag
	rebroadcast := true
	if query.NoBroadcast() {
		rebroadcast = false
	}

	// Filter the query
	if !s.shouldProcessQuery(query.Filters) {
		// Even if we don't process it further, we should rebroadcast,
		// since it is the first time we've seen this.
		return rebroadcast
	}

	// Send ack if requested, without waiting for client to Respond()
	if query.Ack() {
		ack := messageQueryResponse{
			LTime: query.LTime,
			ID:    query.ID,
			From:  s.config.NodeName,
			Flags: queryFlagAck,
		}
		raw, err := encodeMessage(messageQueryResponseType, &ack)
		if err != nil {
			s.logger.Printf("[ERR] serf: failed to format ack: %v", err)
		} else {
			addr := net.UDPAddr{IP: query.Addr, Port: int(query.Port)}
			if err := s.memberlist.SendTo(&addr, raw); err != nil {
				s.logger.Printf("[ERR] serf: failed to send ack: %v", err)
			}
		}
	}

	if s.config.EventCh != nil {
		s.config.EventCh <- &Query{
			LTime:    query.LTime,
			Name:     query.Name,
			Payload:  query.Payload,
			serf:     s,
			id:       query.ID,
			addr:     query.Addr,
			port:     query.Port,
			deadline: time.Now().Add(query.Timeout),
		}
	}
	return rebroadcast
}

// handleResponse is called when a query response is
// received.
func (s *Serf) handleQueryResponse(resp *messageQueryResponse) {
	// Look for a corresponding QueryResponse
	s.queryLock.RLock()
	query, ok := s.queryResponse[resp.LTime]
	s.queryLock.RUnlock()
	if !ok {
		s.logger.Printf("[WARN] serf: reply for non-running query (LTime: %d, ID: %d) From: %s",
			resp.LTime, resp.ID, resp.From)
		return
	}

	// Verify the ID matches
	if query.id != resp.ID {
		s.logger.Printf("[WARN] serf: query reply ID mismatch (Local: %d, Response: %d)",
			query.id, resp.ID)
		return
	}

	// Check if the query is closed
	if query.Finished() {
		return
	}

	// Process each type of response
	if resp.Ack() {
		metrics.IncrCounter([]string{"serf", "query_acks"}, 1)
		select {
		case query.ackCh <- resp.From:
		default:
			s.logger.Printf("[WARN] serf: Failed to delivery query ack, dropping")
		}
	} else {
		metrics.IncrCounter([]string{"serf", "query_responses"}, 1)
		select {
		case query.respCh <- NodeResponse{From: resp.From, Payload: resp.Payload}:
		default:
			s.logger.Printf("[WARN] serf: Failed to delivery query response, dropping")
		}
	}
}

// handleNodeConflict is invoked when a join detects a conflict over a name.
// This means two different nodes (IP/Port) are claiming the same name. Memberlist
// will reject the "new" node mapping, but we can still be notified
func (s *Serf) handleNodeConflict(existing, other *memberlist.Node) {
	// Log a basic warning if the node is not us...
	if existing.Name != s.config.NodeName {
		s.logger.Printf("[WARN] serf: Name conflict for '%s' both %s:%d and %s:%d are claiming",
			existing.Name, existing.Addr, existing.Port, other.Addr, other.Port)
		return
	}

	// The current node is conflicting! This is an error
	s.logger.Printf("[ERR] serf: Node name conflicts with another node at %s:%d. Names must be unique! (Resolution enabled: %v)",
		other.Addr, other.Port, s.config.EnableNameConflictResolution)

	// If automatic resolution is enabled, kick off the resolution
	if s.config.EnableNameConflictResolution {
		go s.resolveNodeConflict()
	}
}

// resolveNodeConflict is used to determine which node should remain during
// a name conflict. This is done by running an internal query.
func (s *Serf) resolveNodeConflict() {
	// Get the local node
	local := s.memberlist.LocalNode()

	// Start a name resolution query
	qName := internalQueryName(conflictQuery)
	payload := []byte(s.config.NodeName)
	resp, err := s.Query(qName, payload, nil)
	if err != nil {
		s.logger.Printf("[ERR] serf: Failed to start name resolution query: %v", err)
		return
	}

	// Counter to determine winner
	var responses, matching int

	// Gather responses
	respCh := resp.ResponseCh()
	for r := range respCh {
		// Decode the response
		if len(r.Payload) < 1 || messageType(r.Payload[0]) != messageConflictResponseType {
			s.logger.Printf("[ERR] serf: Invalid conflict query response type: %v", r.Payload)
			continue
		}
		var member Member
		if err := decodeMessage(r.Payload[1:], &member); err != nil {
			s.logger.Printf("[ERR] serf: Failed to decode conflict query response: %v", err)
			continue
		}

		// Update the counters
		responses++
		if bytes.Equal(member.Addr, local.Addr) && member.Port == local.Port {
			matching++
		}
	}

	// Query over, determine if we should live
	majority := (responses / 2) + 1
	if matching >= majority {
		s.logger.Printf("[INFO] serf: majority in name conflict resolution [%d / %d]",
			matching, responses)
		return
	}

	// Since we lost the vote, we need to exit
	s.logger.Printf("[WARN] serf: minority in name conflict resolution, quiting [%d / %d]",
		matching, responses)
	if err := s.Shutdown(); err != nil {
		s.logger.Printf("[ERR] serf: Failed to shutdown: %v", err)
	}
}

// handleReap periodically reaps the list of failed and left members.
func (s *Serf) handleReap() {
	for {
		select {
		case <-time.After(s.config.ReapInterval):
			s.memberLock.Lock()
			s.failedMembers = s.reap(s.failedMembers, s.config.ReconnectTimeout)
			s.leftMembers = s.reap(s.leftMembers, s.config.TombstoneTimeout)
			s.memberLock.Unlock()
		case <-s.shutdownCh:
			return
		}
	}
}

// handleReconnect attempts to reconnect to recently failed nodes
// on configured intervals.
func (s *Serf) handleReconnect() {
	for {
		select {
		case <-time.After(s.config.ReconnectInterval):
			s.reconnect()
		case <-s.shutdownCh:
			return
		}
	}
}

// reap is called with a list of old members and a timeout, and removes
// members that have exceeded the timeout. The members are removed from
// both the old list and the members itself. Locking is left to the caller.
func (s *Serf) reap(old []*memberState, timeout time.Duration) []*memberState {
	now := time.Now()
	n := len(old)
	for i := 0; i < n; i++ {
		m := old[i]

		// Skip if the timeout is not yet reached
		if now.Sub(m.leaveTime) <= timeout {
			continue
		}

		// Delete from the list
		old[i], old[n-1] = old[n-1], nil
		old = old[:n-1]
		n--
		i--

		// Delete from members
		delete(s.members, m.Name)

		// Tell the coordinate client the node has gone away and delete
		// its cached coordinates.
		if !s.config.DisableCoordinates {
			s.coordClient.ForgetNode(m.Name)

			s.coordCacheLock.Lock()
			delete(s.coordCache, m.Name)
			s.coordCacheLock.Unlock()
		}

		// Send an event along
		s.logger.Printf("[INFO] serf: EventMemberReap: %s", m.Name)
		if s.config.EventCh != nil {
			s.config.EventCh <- MemberEvent{
				Type:    EventMemberReap,
				Members: []Member{m.Member},
			}
		}
	}

	return old
}

// reconnect attempts to reconnect to recently fail nodes.
func (s *Serf) reconnect() {
	s.memberLock.RLock()

	// Nothing to do if there are no failed members
	n := len(s.failedMembers)
	if n == 0 {
		s.memberLock.RUnlock()
		return
	}

	// Probability we should attempt to reconect is given
	// by num failed / (num members - num failed - num left)
	// This means that we probabilistically expect the cluster
	// to attempt to connect to each failed member once per
	// reconnect interval
	numFailed := float32(len(s.failedMembers))
	numAlive := float32(len(s.members) - len(s.failedMembers) - len(s.leftMembers))
	if numAlive == 0 {
		numAlive = 1 // guard against zero divide
	}
	prob := numFailed / numAlive
	if rand.Float32() > prob {
		s.memberLock.RUnlock()
		s.logger.Printf("[DEBUG] serf: forgoing reconnect for random throttling")
		return
	}

	// Select a random member to try and join
	idx := int(rand.Uint32() % uint32(n))
	mem := s.failedMembers[idx]
	s.memberLock.RUnlock()

	// Format the addr
	addr := net.UDPAddr{IP: mem.Addr, Port: int(mem.Port)}
	s.logger.Printf("[INFO] serf: attempting reconnect to %v %s", mem.Name, addr.String())

	// Attempt to join at the memberlist level
	s.memberlist.Join([]string{addr.String()})
}

// checkQueueDepth periodically checks the size of a queue to see if
// it is too large
func (s *Serf) checkQueueDepth(name string, queue *memberlist.TransmitLimitedQueue) {
	for {
		select {
		case <-time.After(time.Second):
			numq := queue.NumQueued()
			metrics.AddSample([]string{"serf", "queue", name}, float32(numq))
			if numq >= s.config.QueueDepthWarning {
				s.logger.Printf("[WARN] serf: %s queue depth: %d", name, numq)
			}
			if numq > s.config.MaxQueueDepth {
				s.logger.Printf("[WARN] serf: %s queue depth (%d) exceeds limit (%d), dropping messages!",
					name, numq, s.config.MaxQueueDepth)
				queue.Prune(s.config.MaxQueueDepth)
			}
		case <-s.shutdownCh:
			return
		}
	}
}

// removeOldMember is used to remove an old member from a list of old
// members.
func removeOldMember(old []*memberState, name string) []*memberState {
	for i, m := range old {
		if m.Name == name {
			n := len(old)
			old[i], old[n-1] = old[n-1], nil
			return old[:n-1]
		}
	}

	return old
}

// recentIntent checks the recent intent buffer for a matching
// entry for a given node, and either returns the message or nil
func recentIntent(recent []nodeIntent, node string) (intent *nodeIntent) {
	for i := 0; i < len(recent); i++ {
		// Break fast if we hit a zero entry
		if recent[i].LTime == 0 {
			break
		}

		// Check for a node match
		if recent[i].Node == node {
			// Take the most recent entry
			if intent == nil || recent[i].LTime > intent.LTime {
				intent = &recent[i]
			}
		}
	}
	return
}

// handleRejoin attempts to reconnect to previously known alive nodes
func (s *Serf) handleRejoin(previous []*PreviousNode) {
	for _, prev := range previous {
		// Do not attempt to join ourself
		if prev.Name == s.config.NodeName {
			continue
		}

		s.logger.Printf("[INFO] serf: Attempting re-join to previously known node: %s", prev)
		_, err := s.memberlist.Join([]string{prev.Addr})
		if err == nil {
			s.logger.Printf("[INFO] serf: Re-joined to previously known node: %s", prev)
			return
		}
	}
	s.logger.Printf("[WARN] serf: Failed to re-join any previously known node")
}

// encodeTags is used to encode a tag map
func (s *Serf) encodeTags(tags map[string]string) []byte {
	// Support role-only backwards compatibility
	if s.ProtocolVersion() < 3 {
		role := tags["role"]
		return []byte(role)
	}

	// Use a magic byte prefix and msgpack encode the tags
	var buf bytes.Buffer
	buf.WriteByte(tagMagicByte)
	enc := codec.NewEncoder(&buf, &codec.MsgpackHandle{})
	if err := enc.Encode(tags); err != nil {
		panic(fmt.Sprintf("Failed to encode tags: %v", err))
	}
	return buf.Bytes()
}

// decodeTags is used to decode a tag map
func (s *Serf) decodeTags(buf []byte) map[string]string {
	tags := make(map[string]string)

	// Backwards compatibility mode
	if len(buf) == 0 || buf[0] != tagMagicByte {
		tags["role"] = string(buf)
		return tags
	}

	// Decode the tags
	r := bytes.NewReader(buf[1:])
	dec := codec.NewDecoder(r, &codec.MsgpackHandle{})
	if err := dec.Decode(&tags); err != nil {
		s.logger.Printf("[ERR] serf: Failed to decode tags: %v", err)
	}
	return tags
}

// Stats is used to provide operator debugging information
func (s *Serf) Stats() map[string]string {
	toString := func(v uint64) string {
		return strconv.FormatUint(v, 10)
	}
	stats := map[string]string{
		"members":      toString(uint64(len(s.members))),
		"failed":       toString(uint64(len(s.failedMembers))),
		"left":         toString(uint64(len(s.leftMembers))),
		"member_time":  toString(uint64(s.clock.Time())),
		"event_time":   toString(uint64(s.eventClock.Time())),
		"query_time":   toString(uint64(s.queryClock.Time())),
		"intent_queue": toString(uint64(s.broadcasts.NumQueued())),
		"event_queue":  toString(uint64(s.eventBroadcasts.NumQueued())),
		"query_queue":  toString(uint64(s.queryBroadcasts.NumQueued())),
		"encrypted":    fmt.Sprintf("%v", s.EncryptionEnabled()),
	}
	return stats
}

// WriteKeyringFile will serialize the current keyring and save it to a file.
func (s *Serf) writeKeyringFile() error {
	if len(s.config.KeyringFile) == 0 {
		return nil
	}

	keyring := s.config.MemberlistConfig.Keyring
	keysRaw := keyring.GetKeys()
	keysEncoded := make([]string, len(keysRaw))

	for i, key := range keysRaw {
		keysEncoded[i] = base64.StdEncoding.EncodeToString(key)
	}

	encodedKeys, err := json.MarshalIndent(keysEncoded, "", "  ")
	if err != nil {
		return fmt.Errorf("Failed to encode keys: %s", err)
	}

	// Use 0600 for permissions because key data is sensitive
	if err = ioutil.WriteFile(s.config.KeyringFile, encodedKeys, 0600); err != nil {
		return fmt.Errorf("Failed to write keyring file: %s", err)
	}

	// Success!
	return nil
}

// GetCoordinate returns the network coordinate of the local node.
func (s *Serf) GetCoordinate() (*coordinate.Coordinate, error) {
	if !s.config.DisableCoordinates {
		return s.coordClient.GetCoordinate(), nil
	}

	return nil, fmt.Errorf("Coordinates are disabled")
}

// GetCachedCoordinate returns the network coordinate for the node with the given
// name. This will only be valid if DisableCoordinates is set to false.
func (s *Serf) GetCachedCoordinate(name string) (coord *coordinate.Coordinate, ok bool) {
	if !s.config.DisableCoordinates {
		s.coordCacheLock.RLock()
		defer s.coordCacheLock.RUnlock()
		if coord, ok = s.coordCache[name]; ok {
			return coord, true
		}

		return nil, false
	}

	return nil, false
}
