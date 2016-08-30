package memberlist

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"net"
	"sync/atomic"
	"time"

	"github.com/armon/go-metrics"
)

type nodeStateType int

const (
	stateAlive nodeStateType = iota
	stateSuspect
	stateDead
)

// Node represents a node in the cluster.
type Node struct {
	Name string
	Addr net.IP
	Port uint16
	Meta []byte // Metadata from the delegate for this node.
	PMin uint8  // Minimum protocol version this understands
	PMax uint8  // Maximum protocol version this understands
	PCur uint8  // Current version node is speaking
	DMin uint8  // Min protocol version for the delegate to understand
	DMax uint8  // Max protocol version for the delegate to understand
	DCur uint8  // Current version delegate is speaking
}

// NodeState is used to manage our state view of another node
type nodeState struct {
	Node
	Incarnation uint32        // Last known incarnation number
	State       nodeStateType // Current state
	StateChange time.Time     // Time last state change happened
}

// ackHandler is used to register handlers for incoming acks
type ackHandler struct {
	handler func([]byte, time.Time)
	timer   *time.Timer
}

// NoPingResponseError is used to indicate a 'ping' packet was
// successfully issued but no response was received
type NoPingResponseError struct {
	node string
}

func (f NoPingResponseError) Error() string {
	return fmt.Sprintf("No response from node %s", f.node)
}

// Schedule is used to ensure the Tick is performed periodically. This
// function is safe to call multiple times. If the memberlist is already
// scheduled, then it won't do anything.
func (m *Memberlist) schedule() {
	m.tickerLock.Lock()
	defer m.tickerLock.Unlock()

	// If we already have tickers, then don't do anything, since we're
	// scheduled
	if len(m.tickers) > 0 {
		return
	}

	// Create the stop tick channel, a blocking channel. We close this
	// when we should stop the tickers.
	stopCh := make(chan struct{})

	// Create a new probeTicker
	if m.config.ProbeInterval > 0 {
		t := time.NewTicker(m.config.ProbeInterval)
		go m.triggerFunc(m.config.ProbeInterval, t.C, stopCh, m.probe)
		m.tickers = append(m.tickers, t)
	}

	// Create a push pull ticker if needed
	if m.config.PushPullInterval > 0 {
		go m.pushPullTrigger(stopCh)
	}

	// Create a gossip ticker if needed
	if m.config.GossipInterval > 0 && m.config.GossipNodes > 0 {
		t := time.NewTicker(m.config.GossipInterval)
		go m.triggerFunc(m.config.GossipInterval, t.C, stopCh, m.gossip)
		m.tickers = append(m.tickers, t)
	}

	// If we made any tickers, then record the stopTick channel for
	// later.
	if len(m.tickers) > 0 {
		m.stopTick = stopCh
	}
}

// triggerFunc is used to trigger a function call each time a
// message is received until a stop tick arrives.
func (m *Memberlist) triggerFunc(stagger time.Duration, C <-chan time.Time, stop <-chan struct{}, f func()) {
	// Use a random stagger to avoid syncronizing
	randStagger := time.Duration(uint64(rand.Int63()) % uint64(stagger))
	select {
	case <-time.After(randStagger):
	case <-stop:
		return
	}
	for {
		select {
		case <-C:
			f()
		case <-stop:
			return
		}
	}
}

// pushPullTrigger is used to periodically trigger a push/pull until
// a stop tick arrives. We don't use triggerFunc since the push/pull
// timer is dynamically scaled based on cluster size to avoid network
// saturation
func (m *Memberlist) pushPullTrigger(stop <-chan struct{}) {
	interval := m.config.PushPullInterval

	// Use a random stagger to avoid syncronizing
	randStagger := time.Duration(uint64(rand.Int63()) % uint64(interval))
	select {
	case <-time.After(randStagger):
	case <-stop:
		return
	}

	// Tick using a dynamic timer
	for {
		tickTime := pushPullScale(interval, m.estNumNodes())
		select {
		case <-time.After(tickTime):
			m.pushPull()
		case <-stop:
			return
		}
	}
}

// Deschedule is used to stop the background maintenence. This is safe
// to call multiple times.
func (m *Memberlist) deschedule() {
	m.tickerLock.Lock()
	defer m.tickerLock.Unlock()

	// If we have no tickers, then we aren't scheduled.
	if len(m.tickers) == 0 {
		return
	}

	// Close the stop channel so all the ticker listeners stop.
	close(m.stopTick)

	// Explicitly stop all the tickers themselves so they don't take
	// up any more resources, and get rid of the list.
	for _, t := range m.tickers {
		t.Stop()
	}
	m.tickers = nil
}

// Tick is used to perform a single round of failure detection and gossip
func (m *Memberlist) probe() {
	// Track the number of indexes we've considered probing
	numCheck := 0
START:
	m.nodeLock.RLock()

	// Make sure we don't wrap around infinitely
	if numCheck >= len(m.nodes) {
		m.nodeLock.RUnlock()
		return
	}

	// Handle the wrap around case
	if m.probeIndex >= len(m.nodes) {
		m.nodeLock.RUnlock()
		m.resetNodes()
		m.probeIndex = 0
		numCheck++
		goto START
	}

	// Determine if we should probe this node
	skip := false
	var node nodeState

	node = *m.nodes[m.probeIndex]
	if node.Name == m.config.Name {
		skip = true
	} else if node.State == stateDead {
		skip = true
	}

	// Potentially skip
	m.nodeLock.RUnlock()
	m.probeIndex++
	if skip {
		numCheck++
		goto START
	}

	// Probe the specific node
	m.probeNode(&node)
}

// probeNode handles a single round of failure checking on a node.
func (m *Memberlist) probeNode(node *nodeState) {
	defer metrics.MeasureSince([]string{"memberlist", "probeNode"}, time.Now())

	// Prepare a ping message and setup an ack handler.
	ping := ping{SeqNo: m.nextSeqNo(), Node: node.Name}
	ackCh := make(chan ackMessage, m.config.IndirectChecks+1)
	m.setAckChannel(ping.SeqNo, ackCh, m.config.ProbeInterval)

	// Send a ping to the node.
	deadline := time.Now().Add(m.config.ProbeInterval)
	destAddr := &net.UDPAddr{IP: node.Addr, Port: int(node.Port)}
	if err := m.encodeAndSendMsg(destAddr, pingMsg, &ping); err != nil {
		m.logger.Printf("[ERR] memberlist: Failed to send ping: %s", err)
		return
	}

	// Mark the sent time here, which should be after any pre-processing and
	// system calls to do the actual send. This probably under-reports a bit,
	// but it's the best we can do.
	sent := time.Now()

	// Wait for response or round-trip-time.
	select {
	case v := <-ackCh:
		if v.Complete == true {
			if m.config.Ping != nil {
				rtt := v.Timestamp.Sub(sent)
				m.config.Ping.NotifyPingComplete(&node.Node, rtt, v.Payload)
			}
			return
		}

		// As an edge case, if we get a timeout, we need to re-enqueue it
		// here to break out of the select below.
		if v.Complete == false {
			ackCh <- v
		}
	case <-time.After(m.config.ProbeTimeout):
		m.logger.Printf("[DEBUG] memberlist: Failed UDP ping: %v (timeout reached)", node.Name)
	}

	// Get some random live nodes.
	m.nodeLock.RLock()
	excludes := []string{m.config.Name, node.Name}
	kNodes := kRandomNodes(m.config.IndirectChecks, excludes, m.nodes)
	m.nodeLock.RUnlock()

	// Attempt an indirect ping.
	ind := indirectPingReq{SeqNo: ping.SeqNo, Target: node.Addr, Port: node.Port, Node: node.Name}
	for _, peer := range kNodes {
		destAddr := &net.UDPAddr{IP: peer.Addr, Port: int(peer.Port)}
		if err := m.encodeAndSendMsg(destAddr, indirectPingMsg, &ind); err != nil {
			m.logger.Printf("[ERR] memberlist: Failed to send indirect ping: %s", err)
		}
	}

	// Also make an attempt to contact the node directly over TCP. This
	// helps prevent confused clients who get isolated from UDP traffic
	// but can still speak TCP (which also means they can possibly report
	// misinformation to other nodes via anti-entropy), avoiding flapping in
	// the cluster.
	//
	// This is a little unusual because we will attempt a TCP ping to any
	// member who understands version 3 of the protocol, regardless of
	// which protocol version we are speaking. That's why we've included a
	// config option to turn this off if desired.
	fallbackCh := make(chan bool, 1)
	if (!m.config.DisableTcpPings) && (node.PMax >= 3) {
		destAddr := &net.TCPAddr{IP: node.Addr, Port: int(node.Port)}
		go func() {
			defer close(fallbackCh)
			didContact, err := m.sendPingAndWaitForAck(destAddr, ping, deadline)
			if err != nil {
				m.logger.Printf("[ERR] memberlist: Failed TCP fallback ping: %s", err)
			} else {
				fallbackCh <- didContact
			}
		}()
	} else {
		close(fallbackCh)
	}

	// Wait for the acks or timeout. Note that we don't check the fallback
	// channel here because we want to issue a warning below if that's the
	// *only* way we hear back from the peer, so we have to let this time
	// out first to allow the normal UDP-based acks to come in.
	select {
	case v := <-ackCh:
		if v.Complete == true {
			return
		}
	}

	// Finally, poll the fallback channel. The timeouts are set such that
	// the channel will have something or be closed without having to wait
	// any additional time here.
	for didContact := range fallbackCh {
		if didContact {
			m.logger.Printf("[WARN] memberlist: Was able to reach %s via TCP but not UDP, network may be misconfigured and not allowing bidirectional UDP", node.Name)
			return
		}
	}

	// No acks received from target, suspect
	m.logger.Printf("[INFO] memberlist: Suspect %s has failed, no acks received", node.Name)
	s := suspect{Incarnation: node.Incarnation, Node: node.Name, From: m.config.Name}
	m.suspectNode(&s)
}

// Ping initiates a ping to the node with the specified name.
func (m *Memberlist) Ping(node string, addr net.Addr) (time.Duration, error) {
	// Prepare a ping message and setup an ack handler.
	ping := ping{SeqNo: m.nextSeqNo(), Node: node}
	ackCh := make(chan ackMessage, m.config.IndirectChecks+1)
	m.setAckChannel(ping.SeqNo, ackCh, m.config.ProbeInterval)

	// Send a ping to the node.
	if err := m.encodeAndSendMsg(addr, pingMsg, &ping); err != nil {
		return 0, err
	}

	// Mark the sent time here, which should be after any pre-processing and
	// system calls to do the actual send. This probably under-reports a bit,
	// but it's the best we can do.
	sent := time.Now()

	// Wait for response or timeout.
	select {
	case v := <-ackCh:
		if v.Complete == true {
			return v.Timestamp.Sub(sent), nil
		}
	case <-time.After(m.config.ProbeTimeout):
		// Timeout, return an error below.
	}

	m.logger.Printf("[DEBUG] memberlist: Failed UDP ping: %v (timeout reached)", node)
	return 0, NoPingResponseError{ping.Node}
}

// resetNodes is used when the tick wraps around. It will reap the
// dead nodes and shuffle the node list.
func (m *Memberlist) resetNodes() {
	m.nodeLock.Lock()
	defer m.nodeLock.Unlock()

	// Move the dead nodes
	deadIdx := moveDeadNodes(m.nodes)

	// Deregister the dead nodes
	for i := deadIdx; i < len(m.nodes); i++ {
		delete(m.nodeMap, m.nodes[i].Name)
		m.nodes[i] = nil
	}

	// Trim the nodes to exclude the dead nodes
	m.nodes = m.nodes[0:deadIdx]

	// Update numNodes after we've trimmed the dead nodes
	atomic.StoreUint32(&m.numNodes, uint32(deadIdx))

	// Shuffle live nodes
	shuffleNodes(m.nodes)
}

// gossip is invoked every GossipInterval period to broadcast our gossip
// messages to a few random nodes.
func (m *Memberlist) gossip() {
	defer metrics.MeasureSince([]string{"memberlist", "gossip"}, time.Now())

	// Get some random live nodes
	m.nodeLock.RLock()
	excludes := []string{m.config.Name}
	kNodes := kRandomNodes(m.config.GossipNodes, excludes, m.nodes)
	m.nodeLock.RUnlock()

	// Compute the bytes available
	bytesAvail := udpSendBuf - compoundHeaderOverhead
	if m.config.EncryptionEnabled() {
		bytesAvail -= encryptOverhead(m.encryptionVersion())
	}

	for _, node := range kNodes {
		// Get any pending broadcasts
		msgs := m.getBroadcasts(compoundOverhead, bytesAvail)
		if len(msgs) == 0 {
			return
		}

		// Create a compound message
		compound := makeCompoundMessage(msgs)

		// Send the compound message
		destAddr := &net.UDPAddr{IP: node.Addr, Port: int(node.Port)}
		if err := m.rawSendMsgUDP(destAddr, compound.Bytes()); err != nil {
			m.logger.Printf("[ERR] memberlist: Failed to send gossip to %s: %s", destAddr, err)
		}
	}
}

// pushPull is invoked periodically to randomly perform a complete state
// exchange. Used to ensure a high level of convergence, but is also
// reasonably expensive as the entire state of this node is exchanged
// with the other node.
func (m *Memberlist) pushPull() {
	// Get a random live node
	m.nodeLock.RLock()
	excludes := []string{m.config.Name}
	nodes := kRandomNodes(1, excludes, m.nodes)
	m.nodeLock.RUnlock()

	// If no nodes, bail
	if len(nodes) == 0 {
		return
	}
	node := nodes[0]

	// Attempt a push pull
	if err := m.pushPullNode(node.Addr, node.Port, false); err != nil {
		m.logger.Printf("[ERR] memberlist: Push/Pull with %s failed: %s", node.Name, err)
	}
}

// pushPullNode does a complete state exchange with a specific node.
func (m *Memberlist) pushPullNode(addr []byte, port uint16, join bool) error {
	defer metrics.MeasureSince([]string{"memberlist", "pushPullNode"}, time.Now())

	// Attempt to send and receive with the node
	remote, userState, err := m.sendAndReceiveState(addr, port, join)
	if err != nil {
		return err
	}

	if err := m.mergeRemoteState(join, remote, userState); err != nil {
		return err
	}
	return nil
}

// verifyProtocol verifies that all the remote nodes can speak with our
// nodes and vice versa on both the core protocol as well as the
// delegate protocol level.
//
// The verification works by finding the maximum minimum and
// minimum maximum understood protocol and delegate versions. In other words,
// it finds the common denominator of protocol and delegate version ranges
// for the entire cluster.
//
// After this, it goes through the entire cluster (local and remote) and
// verifies that everyone's speaking protocol versions satisfy this range.
// If this passes, it means that every node can understand each other.
func (m *Memberlist) verifyProtocol(remote []pushNodeState) error {
	m.nodeLock.RLock()
	defer m.nodeLock.RUnlock()

	// Maximum minimum understood and minimum maximum understood for both
	// the protocol and delegate versions. We use this to verify everyone
	// can be understood.
	var maxpmin, minpmax uint8
	var maxdmin, mindmax uint8
	minpmax = math.MaxUint8
	mindmax = math.MaxUint8

	for _, rn := range remote {
		// If the node isn't alive, then skip it
		if rn.State != stateAlive {
			continue
		}

		// Skip nodes that don't have versions set, it just means
		// their version is zero.
		if len(rn.Vsn) == 0 {
			continue
		}

		if rn.Vsn[0] > maxpmin {
			maxpmin = rn.Vsn[0]
		}

		if rn.Vsn[1] < minpmax {
			minpmax = rn.Vsn[1]
		}

		if rn.Vsn[3] > maxdmin {
			maxdmin = rn.Vsn[3]
		}

		if rn.Vsn[4] < mindmax {
			mindmax = rn.Vsn[4]
		}
	}

	for _, n := range m.nodes {
		// Ignore non-alive nodes
		if n.State != stateAlive {
			continue
		}

		if n.PMin > maxpmin {
			maxpmin = n.PMin
		}

		if n.PMax < minpmax {
			minpmax = n.PMax
		}

		if n.DMin > maxdmin {
			maxdmin = n.DMin
		}

		if n.DMax < mindmax {
			mindmax = n.DMax
		}
	}

	// Now that we definitively know the minimum and maximum understood
	// version that satisfies the whole cluster, we verify that every
	// node in the cluster satisifies this.
	for _, n := range remote {
		var nPCur, nDCur uint8
		if len(n.Vsn) > 0 {
			nPCur = n.Vsn[2]
			nDCur = n.Vsn[5]
		}

		if nPCur < maxpmin || nPCur > minpmax {
			return fmt.Errorf(
				"Node '%s' protocol version (%d) is incompatible: [%d, %d]",
				n.Name, nPCur, maxpmin, minpmax)
		}

		if nDCur < maxdmin || nDCur > mindmax {
			return fmt.Errorf(
				"Node '%s' delegate protocol version (%d) is incompatible: [%d, %d]",
				n.Name, nDCur, maxdmin, mindmax)
		}
	}

	for _, n := range m.nodes {
		nPCur := n.PCur
		nDCur := n.DCur

		if nPCur < maxpmin || nPCur > minpmax {
			return fmt.Errorf(
				"Node '%s' protocol version (%d) is incompatible: [%d, %d]",
				n.Name, nPCur, maxpmin, minpmax)
		}

		if nDCur < maxdmin || nDCur > mindmax {
			return fmt.Errorf(
				"Node '%s' delegate protocol version (%d) is incompatible: [%d, %d]",
				n.Name, nDCur, maxdmin, mindmax)
		}
	}

	return nil
}

// nextSeqNo returns a usable sequence number in a thread safe way
func (m *Memberlist) nextSeqNo() uint32 {
	return atomic.AddUint32(&m.sequenceNum, 1)
}

// nextIncarnation returns the next incarnation number in a thread safe way
func (m *Memberlist) nextIncarnation() uint32 {
	return atomic.AddUint32(&m.incarnation, 1)
}

// estNumNodes is used to get the current estimate of the number of nodes
func (m *Memberlist) estNumNodes() int {
	return int(atomic.LoadUint32(&m.numNodes))
}

type ackMessage struct {
	Complete  bool
	Payload   []byte
	Timestamp time.Time
}

// setAckChannel is used to attach a channel to receive a message when an ack with a given
// sequence number is received. The `complete` field of the message will be false on timeout
func (m *Memberlist) setAckChannel(seqNo uint32, ch chan ackMessage, timeout time.Duration) {
	// Create a handler function
	handler := func(payload []byte, timestamp time.Time) {
		select {
		case ch <- ackMessage{true, payload, timestamp}:
		default:
		}
	}

	// Add the handler
	ah := &ackHandler{handler, nil}
	m.ackLock.Lock()
	m.ackHandlers[seqNo] = ah
	m.ackLock.Unlock()

	// Setup a reaping routing
	ah.timer = time.AfterFunc(timeout, func() {
		m.ackLock.Lock()
		delete(m.ackHandlers, seqNo)
		m.ackLock.Unlock()
		select {
		case ch <- ackMessage{false, nil, time.Now()}:
		default:
		}
	})
}

// setAckHandler is used to attach a handler to be invoked when an
// ack with a given sequence number is received. If a timeout is reached,
// the handler is deleted
func (m *Memberlist) setAckHandler(seqNo uint32, handler func([]byte, time.Time), timeout time.Duration) {
	// Add the handler
	ah := &ackHandler{handler, nil}
	m.ackLock.Lock()
	m.ackHandlers[seqNo] = ah
	m.ackLock.Unlock()

	// Setup a reaping routing
	ah.timer = time.AfterFunc(timeout, func() {
		m.ackLock.Lock()
		delete(m.ackHandlers, seqNo)
		m.ackLock.Unlock()
	})
}

// Invokes an Ack handler if any is associated, and reaps the handler immediately
func (m *Memberlist) invokeAckHandler(ack ackResp, timestamp time.Time) {
	m.ackLock.Lock()
	ah, ok := m.ackHandlers[ack.SeqNo]
	delete(m.ackHandlers, ack.SeqNo)
	m.ackLock.Unlock()
	if !ok {
		return
	}
	ah.timer.Stop()
	ah.handler(ack.Payload, timestamp)
}

// aliveNode is invoked by the network layer when we get a message about a
// live node.
func (m *Memberlist) aliveNode(a *alive, notify chan struct{}, bootstrap bool) {
	m.nodeLock.Lock()
	defer m.nodeLock.Unlock()
	state, ok := m.nodeMap[a.Node]

	// It is possible that during a Leave(), there is already an aliveMsg
	// in-queue to be processed but blocked by the locks above. If we let
	// that aliveMsg process, it'll cause us to re-join the cluster. This
	// ensures that we don't.
	if m.leave && a.Node == m.config.Name {
		return
	}

	// Invoke the Alive delegate if any. This can be used to filter out
	// alive messages based on custom logic. For example, using a cluster name.
	// Using a merge delegate is not enough, as it is possible for passive
	// cluster merging to still occur.
	if m.config.Alive != nil {
		node := &Node{
			Name: a.Node,
			Addr: a.Addr,
			Port: a.Port,
			Meta: a.Meta,
			PMin: a.Vsn[0],
			PMax: a.Vsn[1],
			PCur: a.Vsn[2],
			DMin: a.Vsn[3],
			DMax: a.Vsn[4],
			DCur: a.Vsn[5],
		}
		if err := m.config.Alive.NotifyAlive(node); err != nil {
			m.logger.Printf("[WARN] memberlist: ignoring alive message for '%s': %s",
				a.Node, err)
			return
		}
	}

	// Check if we've never seen this node before, and if not, then
	// store this node in our node map.
	if !ok {
		state = &nodeState{
			Node: Node{
				Name: a.Node,
				Addr: a.Addr,
				Port: a.Port,
				Meta: a.Meta,
			},
			State: stateDead,
		}

		// Add to map
		m.nodeMap[a.Node] = state

		// Get a random offset. This is important to ensure
		// the failure detection bound is low on average. If all
		// nodes did an append, failure detection bound would be
		// very high.
		n := len(m.nodes)
		offset := randomOffset(n)

		// Add at the end and swap with the node at the offset
		m.nodes = append(m.nodes, state)
		m.nodes[offset], m.nodes[n] = m.nodes[n], m.nodes[offset]

		// Update numNodes after we've added a new node
		atomic.AddUint32(&m.numNodes, 1)
	}

	// Check if this address is different than the existing node
	if !bytes.Equal([]byte(state.Addr), a.Addr) || state.Port != a.Port {
		m.logger.Printf("[ERR] memberlist: Conflicting address for %s. Mine: %v:%d Theirs: %v:%d",
			state.Name, state.Addr, state.Port, net.IP(a.Addr), a.Port)

		// Inform the conflict delegate if provided
		if m.config.Conflict != nil {
			other := Node{
				Name: a.Node,
				Addr: a.Addr,
				Port: a.Port,
				Meta: a.Meta,
			}
			m.config.Conflict.NotifyConflict(&state.Node, &other)
		}
		return
	}

	// Bail if the incarnation number is older, and this is not about us
	isLocalNode := state.Name == m.config.Name
	if a.Incarnation <= state.Incarnation && !isLocalNode {
		return
	}

	// Bail if strictly less and this is about us
	if a.Incarnation < state.Incarnation && isLocalNode {
		return
	}

	// Store the old state and meta data
	oldState := state.State
	oldMeta := state.Meta

	// If this is us we need to refute, otherwise re-broadcast
	if !bootstrap && isLocalNode {
		// Compute the version vector
		versions := []uint8{
			state.PMin, state.PMax, state.PCur,
			state.DMin, state.DMax, state.DCur,
		}

		// If the Incarnation is the same, we need special handling, since it
		// possible for the following situation to happen:
		// 1) Start with configuration C, join cluster
		// 2) Hard fail / Kill / Shutdown
		// 3) Restart with configuration C', join cluster
		//
		// In this case, other nodes and the local node see the same incarnation,
		// but the values may not be the same. For this reason, we always
		// need to do an equality check for this Incarnation. In most cases,
		// we just ignore, but we may need to refute.
		//
		if a.Incarnation == state.Incarnation &&
			bytes.Equal(a.Meta, state.Meta) &&
			bytes.Equal(a.Vsn, versions) {
			return
		}

		inc := m.nextIncarnation()
		for a.Incarnation >= inc {
			inc = m.nextIncarnation()
		}
		state.Incarnation = inc

		a := alive{
			Incarnation: inc,
			Node:        state.Name,
			Addr:        state.Addr,
			Port:        state.Port,
			Meta:        state.Meta,
			Vsn:         versions,
		}
		m.encodeBroadcastNotify(a.Node, aliveMsg, a, notify)
		m.logger.Printf("[WARN] memberlist: Refuting an alive message")
	} else {
		m.encodeBroadcastNotify(a.Node, aliveMsg, a, notify)

		// Update protocol versions if it arrived
		if len(a.Vsn) > 0 {
			state.PMin = a.Vsn[0]
			state.PMax = a.Vsn[1]
			state.PCur = a.Vsn[2]
			state.DMin = a.Vsn[3]
			state.DMax = a.Vsn[4]
			state.DCur = a.Vsn[5]
		}

		// Update the state and incarnation number
		state.Incarnation = a.Incarnation
		state.Meta = a.Meta
		if state.State != stateAlive {
			state.State = stateAlive
			state.StateChange = time.Now()
		}
	}

	// Update metrics
	metrics.IncrCounter([]string{"memberlist", "msg", "alive"}, 1)

	// Notify the delegate of any relevant updates
	if m.config.Events != nil {
		if oldState == stateDead {
			// if Dead -> Alive, notify of join
			m.config.Events.NotifyJoin(&state.Node)

		} else if !bytes.Equal(oldMeta, state.Meta) {
			// if Meta changed, trigger an update notification
			m.config.Events.NotifyUpdate(&state.Node)
		}
	}
}

// suspectNode is invoked by the network layer when we get a message
// about a suspect node
func (m *Memberlist) suspectNode(s *suspect) {
	m.nodeLock.Lock()
	defer m.nodeLock.Unlock()
	state, ok := m.nodeMap[s.Node]

	// If we've never heard about this node before, ignore it
	if !ok {
		return
	}

	// Ignore old incarnation numbers
	if s.Incarnation < state.Incarnation {
		return
	}

	// Ignore non-alive nodes
	if state.State != stateAlive {
		return
	}

	// If this is us we need to refute, otherwise re-broadcast
	if state.Name == m.config.Name {
		inc := m.nextIncarnation()
		for s.Incarnation >= inc {
			inc = m.nextIncarnation()
		}
		state.Incarnation = inc

		a := alive{
			Incarnation: inc,
			Node:        state.Name,
			Addr:        state.Addr,
			Port:        state.Port,
			Meta:        state.Meta,
			Vsn: []uint8{
				state.PMin, state.PMax, state.PCur,
				state.DMin, state.DMax, state.DCur,
			},
		}
		m.encodeAndBroadcast(s.Node, aliveMsg, a)
		m.logger.Printf("[WARN] memberlist: Refuting a suspect message (from: %s)", s.From)
		return // Do not mark ourself suspect
	} else {
		m.encodeAndBroadcast(s.Node, suspectMsg, s)
	}

	// Update metrics
	metrics.IncrCounter([]string{"memberlist", "msg", "suspect"}, 1)

	// Update the state
	state.Incarnation = s.Incarnation
	state.State = stateSuspect
	changeTime := time.Now()
	state.StateChange = changeTime

	// Setup a timeout for this
	timeout := suspicionTimeout(m.config.SuspicionMult, m.estNumNodes(), m.config.ProbeInterval)
	time.AfterFunc(timeout, func() {
		m.nodeLock.Lock()
		state, ok := m.nodeMap[s.Node]
		timeout := ok && state.State == stateSuspect && state.StateChange == changeTime
		m.nodeLock.Unlock()

		if timeout {
			m.suspectTimeout(state)
		}
	})
}

// suspectTimeout is invoked when a suspect timeout has occurred
func (m *Memberlist) suspectTimeout(n *nodeState) {
	// Construct a dead message
	m.logger.Printf("[INFO] memberlist: Marking %s as failed, suspect timeout reached", n.Name)
	d := dead{Incarnation: n.Incarnation, Node: n.Name, From: m.config.Name}
	m.deadNode(&d)
}

// deadNode is invoked by the network layer when we get a message
// about a dead node
func (m *Memberlist) deadNode(d *dead) {
	m.nodeLock.Lock()
	defer m.nodeLock.Unlock()
	state, ok := m.nodeMap[d.Node]

	// If we've never heard about this node before, ignore it
	if !ok {
		return
	}

	// Ignore old incarnation numbers
	if d.Incarnation < state.Incarnation {
		return
	}

	// Ignore if node is already dead
	if state.State == stateDead {
		return
	}

	// Check if this is us
	if state.Name == m.config.Name {
		// If we are not leaving we need to refute
		if !m.leave {
			inc := m.nextIncarnation()
			for d.Incarnation >= inc {
				inc = m.nextIncarnation()
			}
			state.Incarnation = inc

			a := alive{
				Incarnation: inc,
				Node:        state.Name,
				Addr:        state.Addr,
				Port:        state.Port,
				Meta:        state.Meta,
				Vsn: []uint8{
					state.PMin, state.PMax, state.PCur,
					state.DMin, state.DMax, state.DCur,
				},
			}
			m.encodeAndBroadcast(d.Node, aliveMsg, a)
			m.logger.Printf("[WARN] memberlist: Refuting a dead message (from: %s)", d.From)
			return // Do not mark ourself dead
		}

		// If we are leaving, we broadcast and wait
		m.encodeBroadcastNotify(d.Node, deadMsg, d, m.leaveBroadcast)
	} else {
		m.encodeAndBroadcast(d.Node, deadMsg, d)
	}

	// Update metrics
	metrics.IncrCounter([]string{"memberlist", "msg", "dead"}, 1)

	// Update the state
	state.Incarnation = d.Incarnation
	state.State = stateDead
	state.StateChange = time.Now()

	// Notify of death
	if m.config.Events != nil {
		m.config.Events.NotifyLeave(&state.Node)
	}
}

// mergeState is invoked by the network layer when we get a Push/Pull
// state transfer
func (m *Memberlist) mergeState(remote []pushNodeState) {
	for _, r := range remote {
		switch r.State {
		case stateAlive:
			a := alive{
				Incarnation: r.Incarnation,
				Node:        r.Name,
				Addr:        r.Addr,
				Port:        r.Port,
				Meta:        r.Meta,
				Vsn:         r.Vsn,
			}
			m.aliveNode(&a, nil, false)

		case stateDead:
			// If the remote node belives a node is dead, we prefer to
			// suspect that node instead of declaring it dead instantly
			fallthrough
		case stateSuspect:
			s := suspect{Incarnation: r.Incarnation, Node: r.Name, From: m.config.Name}
			m.suspectNode(&s)
		}
	}
}
