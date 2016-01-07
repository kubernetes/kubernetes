package raft

import (
	pb "github.com/coreos/etcd/raft/raftpb"
	"golang.org/x/net/context"
)

// MultiNode represents a node that is participating in multiple consensus groups.
// A MultiNode is more efficient than a collection of Nodes.
// The methods of this interface correspond to the methods of Node and are described
// more fully there.
type MultiNode interface {
	// CreateGroup adds a new group to the MultiNode. The application must call CreateGroup
	// on each particpating node with the same group ID; it may create groups on demand as it
	// receives messages. If the given storage contains existing log entries the list of peers
	// may be empty. The Config.ID field will be ignored and replaced by the ID passed
	// to StartMultiNode.
	CreateGroup(group uint64, c *Config, peers []Peer) error
	// RemoveGroup removes a group from the MultiNode.
	RemoveGroup(group uint64) error
	// Tick advances the internal logical clock by a single tick.
	Tick()
	// Campaign causes this MultiNode to transition to candidate state in the given group.
	Campaign(ctx context.Context, group uint64) error
	// Propose proposes that data be appended to the given group's log.
	Propose(ctx context.Context, group uint64, data []byte) error
	// ProposeConfChange proposes a config change.
	ProposeConfChange(ctx context.Context, group uint64, cc pb.ConfChange) error
	// ApplyConfChange applies a config change to the local node.
	ApplyConfChange(group uint64, cc pb.ConfChange) *pb.ConfState
	// Step advances the state machine using the given message.
	Step(ctx context.Context, group uint64, msg pb.Message) error
	// Ready returns a channel that returns the current point-in-time state of any ready
	// groups. Only groups with something to report will appear in the map.
	Ready() <-chan map[uint64]Ready
	// Advance notifies the node that the application has applied and saved progress in the
	// last Ready results. It must be called with the last value returned from the Ready()
	// channel.
	Advance(map[uint64]Ready)
	// Status returns the current status of the given group. Returns nil if no such group
	// exists.
	Status(group uint64) *Status
	// Report reports the given node is not reachable for the last send.
	ReportUnreachable(id, groupID uint64)
	// ReportSnapshot reports the stutus of the sent snapshot.
	ReportSnapshot(id, groupID uint64, status SnapshotStatus)
	// Stop performs any necessary termination of the MultiNode.
	Stop()
}

// StartMultiNode creates a MultiNode and starts its background goroutine.
// The id identifies this node and will be used as its node ID in all groups.
// The election and heartbeat timers are in units of ticks.
func StartMultiNode(id uint64) MultiNode {
	mn := newMultiNode(id)
	go mn.run()
	return &mn
}

// TODO(bdarnell): add group ID to the underlying protos?
type multiMessage struct {
	group uint64
	msg   pb.Message
}

type multiConfChange struct {
	group uint64
	msg   pb.ConfChange
	ch    chan pb.ConfState
}

type multiStatus struct {
	group uint64
	ch    chan *Status
}

type groupCreation struct {
	id     uint64
	config *Config
	peers  []Peer
	// TODO(bdarnell): do we really need the done channel here? It's
	// unlike the rest of this package, but we need the group creation
	// to be complete before any Propose or other calls.
	done chan struct{}
}

type groupRemoval struct {
	id uint64
	// TODO(bdarnell): see comment on groupCreation.done
	done chan struct{}
}

type multiNode struct {
	id       uint64
	groupc   chan groupCreation
	rmgroupc chan groupRemoval
	propc    chan multiMessage
	recvc    chan multiMessage
	confc    chan multiConfChange
	readyc   chan map[uint64]Ready
	advancec chan map[uint64]Ready
	tickc    chan struct{}
	stop     chan struct{}
	done     chan struct{}
	status   chan multiStatus
}

func newMultiNode(id uint64) multiNode {
	return multiNode{
		id:       id,
		groupc:   make(chan groupCreation),
		rmgroupc: make(chan groupRemoval),
		propc:    make(chan multiMessage),
		recvc:    make(chan multiMessage),
		confc:    make(chan multiConfChange),
		readyc:   make(chan map[uint64]Ready),
		advancec: make(chan map[uint64]Ready),
		tickc:    make(chan struct{}),
		stop:     make(chan struct{}),
		done:     make(chan struct{}),
		status:   make(chan multiStatus),
	}
}

type groupState struct {
	id         uint64
	raft       *raft
	prevSoftSt *SoftState
	prevHardSt pb.HardState
	prevSnapi  uint64
}

func (g *groupState) newReady() Ready {
	return newReady(g.raft, g.prevSoftSt, g.prevHardSt)
}

func (g *groupState) commitReady(rd Ready) {
	if rd.SoftState != nil {
		g.prevSoftSt = rd.SoftState
	}
	if !IsEmptyHardState(rd.HardState) {
		g.prevHardSt = rd.HardState
	}
	if g.prevHardSt.Commit != 0 {
		// In most cases, prevHardSt and rd.HardState will be the same
		// because when there are new entries to apply we just sent a
		// HardState with an updated Commit value. However, on initial
		// startup the two are different because we don't send a HardState
		// until something changes, but we do send any un-applied but
		// committed entries (and previously-committed entries may be
		// incorporated into the snapshot, even if rd.CommittedEntries is
		// empty). Therefore we mark all committed entries as applied
		// whether they were included in rd.HardState or not.
		g.raft.raftLog.appliedTo(g.prevHardSt.Commit)
	}
	if len(rd.Entries) > 0 {
		e := rd.Entries[len(rd.Entries)-1]
		g.raft.raftLog.stableTo(e.Index, e.Term)
	}
	if !IsEmptySnap(rd.Snapshot) {
		g.prevSnapi = rd.Snapshot.Metadata.Index
		g.raft.raftLog.stableSnapTo(g.prevSnapi)
	}
}

func (mn *multiNode) run() {
	groups := map[uint64]*groupState{}
	rds := map[uint64]Ready{}
	var advancec chan map[uint64]Ready
	for {
		// Only select readyc if we have something to report and we are not
		// currently waiting for an advance.
		readyc := mn.readyc
		if len(rds) == 0 || advancec != nil {
			readyc = nil
		}

		// group points to the group that was touched on this iteration (if any)
		var group *groupState
		select {
		case gc := <-mn.groupc:
			gc.config.ID = mn.id
			r := newRaft(gc.config)
			group = &groupState{
				id:   gc.id,
				raft: r,
			}
			groups[gc.id] = group
			lastIndex, err := gc.config.Storage.LastIndex()
			if err != nil {
				panic(err) // TODO(bdarnell)
			}
			// If the log is empty, this is a new group (like StartNode); otherwise it's
			// restoring an existing group (like RestartNode).
			// TODO(bdarnell): rethink group initialization and whether the application needs
			// to be able to tell us when it expects the group to exist.
			if lastIndex == 0 {
				r.becomeFollower(1, None)
				ents := make([]pb.Entry, len(gc.peers))
				for i, peer := range gc.peers {
					cc := pb.ConfChange{Type: pb.ConfChangeAddNode, NodeID: peer.ID, Context: peer.Context}
					data, err := cc.Marshal()
					if err != nil {
						panic("unexpected marshal error")
					}
					ents[i] = pb.Entry{Type: pb.EntryConfChange, Term: 1, Index: uint64(i + 1), Data: data}
				}
				r.raftLog.append(ents...)
				r.raftLog.committed = uint64(len(ents))
				for _, peer := range gc.peers {
					r.addNode(peer.ID)
				}
			}
			// Set the initial hard and soft states after performing all initialization.
			group.prevSoftSt = r.softState()
			group.prevHardSt = r.HardState
			close(gc.done)

		case gr := <-mn.rmgroupc:
			delete(groups, gr.id)
			delete(rds, gr.id)
			close(gr.done)

		case mm := <-mn.propc:
			// TODO(bdarnell): single-node impl doesn't read from propc unless the group
			// has a leader; we can't do that since we have one propc for many groups.
			// We'll have to buffer somewhere on a group-by-group basis, or just let
			// raft.Step drop any such proposals on the floor.
			mm.msg.From = mn.id
			var ok bool
			if group, ok = groups[mm.group]; ok {
				group.raft.Step(mm.msg)
			}

		case mm := <-mn.recvc:
			group = groups[mm.group]
			if _, ok := group.raft.prs[mm.msg.From]; ok || !IsResponseMsg(mm.msg) {
				group.raft.Step(mm.msg)
			}

		case mcc := <-mn.confc:
			group = groups[mcc.group]
			if mcc.msg.NodeID == None {
				group.raft.resetPendingConf()
				select {
				case mcc.ch <- pb.ConfState{Nodes: group.raft.nodes()}:
				case <-mn.done:
				}
				break
			}
			switch mcc.msg.Type {
			case pb.ConfChangeAddNode:
				group.raft.addNode(mcc.msg.NodeID)
			case pb.ConfChangeRemoveNode:
				group.raft.removeNode(mcc.msg.NodeID)
			case pb.ConfChangeUpdateNode:
				group.raft.resetPendingConf()
			default:
				panic("unexpected conf type")
			}
			select {
			case mcc.ch <- pb.ConfState{Nodes: group.raft.nodes()}:
			case <-mn.done:
			}

		case <-mn.tickc:
			// TODO(bdarnell): instead of calling every group on every tick,
			// we should have a priority queue of groups based on their next
			// time-based event.
			for _, g := range groups {
				g.raft.tick()
				rd := g.newReady()
				if rd.containsUpdates() {
					rds[g.id] = rd
				}
			}

		case readyc <- rds:
			// Clear outgoing messages as soon as we've passed them to the application.
			for g := range rds {
				groups[g].raft.msgs = nil
			}
			rds = map[uint64]Ready{}
			advancec = mn.advancec

		case advs := <-advancec:
			for groupID, rd := range advs {
				g, ok := groups[groupID]
				if !ok {
					continue
				}
				g.commitReady(rd)

				// We've been accumulating new entries in rds which may now be obsolete.
				// Drop the old Ready object and create a new one if needed.
				delete(rds, groupID)
				newRd := g.newReady()
				if newRd.containsUpdates() {
					rds[groupID] = newRd
				}
			}
			advancec = nil

		case ms := <-mn.status:
			if g, ok := groups[ms.group]; ok {
				s := getStatus(g.raft)
				ms.ch <- &s
			} else {
				ms.ch <- nil
			}

		case <-mn.stop:
			close(mn.done)
			return
		}

		if group != nil {
			rd := group.newReady()
			if rd.containsUpdates() {
				rds[group.id] = rd
			}
		}
	}
}

func (mn *multiNode) CreateGroup(id uint64, config *Config, peers []Peer) error {
	gc := groupCreation{
		id:     id,
		config: config,
		peers:  peers,
		done:   make(chan struct{}),
	}
	mn.groupc <- gc
	select {
	case <-gc.done:
		return nil
	case <-mn.done:
		return ErrStopped
	}
}

func (mn *multiNode) RemoveGroup(id uint64) error {
	gr := groupRemoval{
		id:   id,
		done: make(chan struct{}),
	}
	mn.rmgroupc <- gr
	select {
	case <-gr.done:
		return nil
	case <-mn.done:
		return ErrStopped
	}
}

func (mn *multiNode) Stop() {
	select {
	case mn.stop <- struct{}{}:
	case <-mn.done:
	}
	<-mn.done
}

func (mn *multiNode) Tick() {
	select {
	case mn.tickc <- struct{}{}:
	case <-mn.done:
	}
}

func (mn *multiNode) Campaign(ctx context.Context, group uint64) error {
	return mn.step(ctx, multiMessage{group,
		pb.Message{
			Type: pb.MsgHup,
		},
	})
}

func (mn *multiNode) Propose(ctx context.Context, group uint64, data []byte) error {
	return mn.step(ctx, multiMessage{group,
		pb.Message{
			Type: pb.MsgProp,
			Entries: []pb.Entry{
				{Data: data},
			},
		}})
}

func (mn *multiNode) ProposeConfChange(ctx context.Context, group uint64, cc pb.ConfChange) error {
	data, err := cc.Marshal()
	if err != nil {
		return err
	}
	return mn.Step(ctx, group,
		pb.Message{
			Type: pb.MsgProp,
			Entries: []pb.Entry{
				{Type: pb.EntryConfChange, Data: data},
			},
		})
}

func (mn *multiNode) step(ctx context.Context, m multiMessage) error {
	ch := mn.recvc
	if m.msg.Type == pb.MsgProp {
		ch = mn.propc
	}

	select {
	case ch <- m:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-mn.done:
		return ErrStopped
	}
}

func (mn *multiNode) ApplyConfChange(group uint64, cc pb.ConfChange) *pb.ConfState {
	mcc := multiConfChange{group, cc, make(chan pb.ConfState)}
	select {
	case mn.confc <- mcc:
	case <-mn.done:
	}
	select {
	case cs := <-mcc.ch:
		return &cs
	case <-mn.done:
		// Per comments on Node.ApplyConfChange, this method should never return nil.
		return &pb.ConfState{}
	}
}

func (mn *multiNode) Step(ctx context.Context, group uint64, m pb.Message) error {
	// ignore unexpected local messages receiving over network
	if IsLocalMsg(m) {
		// TODO: return an error?
		return nil
	}
	return mn.step(ctx, multiMessage{group, m})
}

func (mn *multiNode) Ready() <-chan map[uint64]Ready {
	return mn.readyc
}

func (mn *multiNode) Advance(rds map[uint64]Ready) {
	select {
	case mn.advancec <- rds:
	case <-mn.done:
	}
}

func (mn *multiNode) Status(group uint64) *Status {
	ms := multiStatus{
		group: group,
		ch:    make(chan *Status),
	}
	mn.status <- ms
	return <-ms.ch
}

func (mn *multiNode) ReportUnreachable(id, groupID uint64) {
	select {
	case mn.recvc <- multiMessage{
		group: groupID,
		msg:   pb.Message{Type: pb.MsgUnreachable, From: id},
	}:
	case <-mn.done:
	}
}

func (mn *multiNode) ReportSnapshot(id, groupID uint64, status SnapshotStatus) {
	rej := status == SnapshotFailure

	select {
	case mn.recvc <- multiMessage{
		group: groupID,
		msg:   pb.Message{Type: pb.MsgSnapStatus, From: id, Reject: rej},
	}:
	case <-mn.done:
	}
}
