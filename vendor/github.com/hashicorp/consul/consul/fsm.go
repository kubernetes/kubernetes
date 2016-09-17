package consul

import (
	"errors"
	"fmt"
	"io"
	"log"
	"time"

	"github.com/armon/go-metrics"
	"github.com/hashicorp/consul/consul/state"
	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/go-msgpack/codec"
	"github.com/hashicorp/raft"
)

// msgpackHandle is a shared handle for encoding/decoding msgpack payloads
var msgpackHandle = &codec.MsgpackHandle{}

// consulFSM implements a finite state machine that is used
// along with Raft to provide strong consistency. We implement
// this outside the Server to avoid exposing this outside the package.
type consulFSM struct {
	logOutput io.Writer
	logger    *log.Logger
	path      string
	state     *state.StateStore
	gc        *state.TombstoneGC
}

// consulSnapshot is used to provide a snapshot of the current
// state in a way that can be accessed concurrently with operations
// that may modify the live state.
type consulSnapshot struct {
	state *state.StateSnapshot
}

// snapshotHeader is the first entry in our snapshot
type snapshotHeader struct {
	// LastIndex is the last index that affects the data.
	// This is used when we do the restore for watchers.
	LastIndex uint64
}

// NewFSM is used to construct a new FSM with a blank state
func NewFSM(gc *state.TombstoneGC, logOutput io.Writer) (*consulFSM, error) {
	stateNew, err := state.NewStateStore(gc)
	if err != nil {
		return nil, err
	}

	fsm := &consulFSM{
		logOutput: logOutput,
		logger:    log.New(logOutput, "", log.LstdFlags),
		state:     stateNew,
		gc:        gc,
	}
	return fsm, nil
}

// State is used to return a handle to the current state
func (c *consulFSM) State() *state.StateStore {
	return c.state
}

func (c *consulFSM) Apply(log *raft.Log) interface{} {
	buf := log.Data
	msgType := structs.MessageType(buf[0])

	// Check if this message type should be ignored when unknown. This is
	// used so that new commands can be added with developer control if older
	// versions can safely ignore the command, or if they should crash.
	ignoreUnknown := false
	if msgType&structs.IgnoreUnknownTypeFlag == structs.IgnoreUnknownTypeFlag {
		msgType &= ^structs.IgnoreUnknownTypeFlag
		ignoreUnknown = true
	}

	switch msgType {
	case structs.RegisterRequestType:
		return c.applyRegister(buf[1:], log.Index)
	case structs.DeregisterRequestType:
		return c.applyDeregister(buf[1:], log.Index)
	case structs.KVSRequestType:
		return c.applyKVSOperation(buf[1:], log.Index)
	case structs.SessionRequestType:
		return c.applySessionOperation(buf[1:], log.Index)
	case structs.ACLRequestType:
		return c.applyACLOperation(buf[1:], log.Index)
	case structs.TombstoneRequestType:
		return c.applyTombstoneOperation(buf[1:], log.Index)
	case structs.CoordinateBatchUpdateType:
		return c.applyCoordinateBatchUpdate(buf[1:], log.Index)
	case structs.PreparedQueryRequestType:
		return c.applyPreparedQueryOperation(buf[1:], log.Index)
	case structs.TxnRequestType:
		return c.applyTxn(buf[1:], log.Index)
	default:
		if ignoreUnknown {
			c.logger.Printf("[WARN] consul.fsm: ignoring unknown message type (%d), upgrade to newer version", msgType)
			return nil
		} else {
			panic(fmt.Errorf("failed to apply request: %#v", buf))
		}
	}
}

func (c *consulFSM) applyRegister(buf []byte, index uint64) interface{} {
	defer metrics.MeasureSince([]string{"consul", "fsm", "register"}, time.Now())
	var req structs.RegisterRequest
	if err := structs.Decode(buf, &req); err != nil {
		panic(fmt.Errorf("failed to decode request: %v", err))
	}

	// Apply all updates in a single transaction
	if err := c.state.EnsureRegistration(index, &req); err != nil {
		c.logger.Printf("[INFO] consul.fsm: EnsureRegistration failed: %v", err)
		return err
	}
	return nil
}

func (c *consulFSM) applyDeregister(buf []byte, index uint64) interface{} {
	defer metrics.MeasureSince([]string{"consul", "fsm", "deregister"}, time.Now())
	var req structs.DeregisterRequest
	if err := structs.Decode(buf, &req); err != nil {
		panic(fmt.Errorf("failed to decode request: %v", err))
	}

	// Either remove the service entry or the whole node
	if req.ServiceID != "" {
		if err := c.state.DeleteService(index, req.Node, req.ServiceID); err != nil {
			c.logger.Printf("[INFO] consul.fsm: DeleteNodeService failed: %v", err)
			return err
		}
	} else if req.CheckID != "" {
		if err := c.state.DeleteCheck(index, req.Node, req.CheckID); err != nil {
			c.logger.Printf("[INFO] consul.fsm: DeleteNodeCheck failed: %v", err)
			return err
		}
	} else {
		if err := c.state.DeleteNode(index, req.Node); err != nil {
			c.logger.Printf("[INFO] consul.fsm: DeleteNode failed: %v", err)
			return err
		}
	}
	return nil
}

func (c *consulFSM) applyKVSOperation(buf []byte, index uint64) interface{} {
	var req structs.KVSRequest
	if err := structs.Decode(buf, &req); err != nil {
		panic(fmt.Errorf("failed to decode request: %v", err))
	}
	defer metrics.MeasureSince([]string{"consul", "fsm", "kvs", string(req.Op)}, time.Now())
	switch req.Op {
	case structs.KVSSet:
		return c.state.KVSSet(index, &req.DirEnt)
	case structs.KVSDelete:
		return c.state.KVSDelete(index, req.DirEnt.Key)
	case structs.KVSDeleteCAS:
		act, err := c.state.KVSDeleteCAS(index, req.DirEnt.ModifyIndex, req.DirEnt.Key)
		if err != nil {
			return err
		} else {
			return act
		}
	case structs.KVSDeleteTree:
		return c.state.KVSDeleteTree(index, req.DirEnt.Key)
	case structs.KVSCAS:
		act, err := c.state.KVSSetCAS(index, &req.DirEnt)
		if err != nil {
			return err
		} else {
			return act
		}
	case structs.KVSLock:
		act, err := c.state.KVSLock(index, &req.DirEnt)
		if err != nil {
			return err
		} else {
			return act
		}
	case structs.KVSUnlock:
		act, err := c.state.KVSUnlock(index, &req.DirEnt)
		if err != nil {
			return err
		} else {
			return act
		}
	default:
		err := errors.New(fmt.Sprintf("Invalid KVS operation '%s'", req.Op))
		c.logger.Printf("[WARN] consul.fsm: %v", err)
		return err
	}
}

func (c *consulFSM) applySessionOperation(buf []byte, index uint64) interface{} {
	var req structs.SessionRequest
	if err := structs.Decode(buf, &req); err != nil {
		panic(fmt.Errorf("failed to decode request: %v", err))
	}
	defer metrics.MeasureSince([]string{"consul", "fsm", "session", string(req.Op)}, time.Now())
	switch req.Op {
	case structs.SessionCreate:
		if err := c.state.SessionCreate(index, &req.Session); err != nil {
			return err
		} else {
			return req.Session.ID
		}
	case structs.SessionDestroy:
		return c.state.SessionDestroy(index, req.Session.ID)
	default:
		c.logger.Printf("[WARN] consul.fsm: Invalid Session operation '%s'", req.Op)
		return fmt.Errorf("Invalid Session operation '%s'", req.Op)
	}
}

func (c *consulFSM) applyACLOperation(buf []byte, index uint64) interface{} {
	var req structs.ACLRequest
	if err := structs.Decode(buf, &req); err != nil {
		panic(fmt.Errorf("failed to decode request: %v", err))
	}
	defer metrics.MeasureSince([]string{"consul", "fsm", "acl", string(req.Op)}, time.Now())
	switch req.Op {
	case structs.ACLForceSet, structs.ACLSet:
		if err := c.state.ACLSet(index, &req.ACL); err != nil {
			return err
		} else {
			return req.ACL.ID
		}
	case structs.ACLDelete:
		return c.state.ACLDelete(index, req.ACL.ID)
	default:
		c.logger.Printf("[WARN] consul.fsm: Invalid ACL operation '%s'", req.Op)
		return fmt.Errorf("Invalid ACL operation '%s'", req.Op)
	}
}

func (c *consulFSM) applyTombstoneOperation(buf []byte, index uint64) interface{} {
	var req structs.TombstoneRequest
	if err := structs.Decode(buf, &req); err != nil {
		panic(fmt.Errorf("failed to decode request: %v", err))
	}
	defer metrics.MeasureSince([]string{"consul", "fsm", "tombstone", string(req.Op)}, time.Now())
	switch req.Op {
	case structs.TombstoneReap:
		return c.state.ReapTombstones(req.ReapIndex)
	default:
		c.logger.Printf("[WARN] consul.fsm: Invalid Tombstone operation '%s'", req.Op)
		return fmt.Errorf("Invalid Tombstone operation '%s'", req.Op)
	}
}

// applyCoordinateBatchUpdate processes a batch of coordinate updates and applies
// them in a single underlying transaction. This interface isn't 1:1 with the outer
// update interface that the coordinate endpoint exposes, so we made it single
// purpose and avoided the opcode convention.
func (c *consulFSM) applyCoordinateBatchUpdate(buf []byte, index uint64) interface{} {
	var updates structs.Coordinates
	if err := structs.Decode(buf, &updates); err != nil {
		panic(fmt.Errorf("failed to decode batch updates: %v", err))
	}
	defer metrics.MeasureSince([]string{"consul", "fsm", "coordinate", "batch-update"}, time.Now())
	if err := c.state.CoordinateBatchUpdate(index, updates); err != nil {
		return err
	}
	return nil
}

// applyPreparedQueryOperation applies the given prepared query operation to the
// state store.
func (c *consulFSM) applyPreparedQueryOperation(buf []byte, index uint64) interface{} {
	var req structs.PreparedQueryRequest
	if err := structs.Decode(buf, &req); err != nil {
		panic(fmt.Errorf("failed to decode request: %v", err))
	}

	defer metrics.MeasureSince([]string{"consul", "fsm", "prepared-query", string(req.Op)}, time.Now())
	switch req.Op {
	case structs.PreparedQueryCreate, structs.PreparedQueryUpdate:
		return c.state.PreparedQuerySet(index, req.Query)
	case structs.PreparedQueryDelete:
		return c.state.PreparedQueryDelete(index, req.Query.ID)
	default:
		c.logger.Printf("[WARN] consul.fsm: Invalid PreparedQuery operation '%s'", req.Op)
		return fmt.Errorf("Invalid PreparedQuery operation '%s'", req.Op)
	}
}

func (c *consulFSM) applyTxn(buf []byte, index uint64) interface{} {
	var req structs.TxnRequest
	if err := structs.Decode(buf, &req); err != nil {
		panic(fmt.Errorf("failed to decode request: %v", err))
	}
	defer metrics.MeasureSince([]string{"consul", "fsm", "txn"}, time.Now())
	results, errors := c.state.TxnRW(index, req.Ops)
	return structs.TxnResponse{results, errors}
}

func (c *consulFSM) Snapshot() (raft.FSMSnapshot, error) {
	defer func(start time.Time) {
		c.logger.Printf("[INFO] consul.fsm: snapshot created in %v", time.Now().Sub(start))
	}(time.Now())

	return &consulSnapshot{c.state.Snapshot()}, nil
}

func (c *consulFSM) Restore(old io.ReadCloser) error {
	defer old.Close()

	// Create a new state store
	stateNew, err := state.NewStateStore(c.gc)
	if err != nil {
		return err
	}
	c.state = stateNew

	// Set up a new restore transaction
	restore := c.state.Restore()
	defer restore.Abort()

	// Create a decoder
	dec := codec.NewDecoder(old, msgpackHandle)

	// Read in the header
	var header snapshotHeader
	if err := dec.Decode(&header); err != nil {
		return err
	}

	// Populate the new state
	msgType := make([]byte, 1)
	for {
		// Read the message type
		_, err := old.Read(msgType)
		if err == io.EOF {
			break
		} else if err != nil {
			return err
		}

		// Decode
		switch structs.MessageType(msgType[0]) {
		case structs.RegisterRequestType:
			var req structs.RegisterRequest
			if err := dec.Decode(&req); err != nil {
				return err
			}
			if err := restore.Registration(header.LastIndex, &req); err != nil {
				return err
			}

		case structs.KVSRequestType:
			var req structs.DirEntry
			if err := dec.Decode(&req); err != nil {
				return err
			}
			if err := restore.KVS(&req); err != nil {
				return err
			}

		case structs.TombstoneRequestType:
			var req structs.DirEntry
			if err := dec.Decode(&req); err != nil {
				return err
			}

			// For historical reasons, these are serialized in the
			// snapshots as KV entries. We want to keep the snapshot
			// format compatible with pre-0.6 versions for now.
			stone := &state.Tombstone{
				Key:   req.Key,
				Index: req.ModifyIndex,
			}
			if err := restore.Tombstone(stone); err != nil {
				return err
			}

		case structs.SessionRequestType:
			var req structs.Session
			if err := dec.Decode(&req); err != nil {
				return err
			}
			if err := restore.Session(&req); err != nil {
				return err
			}

		case structs.ACLRequestType:
			var req structs.ACL
			if err := dec.Decode(&req); err != nil {
				return err
			}
			if err := restore.ACL(&req); err != nil {
				return err
			}

		case structs.CoordinateBatchUpdateType:
			var req structs.Coordinates
			if err := dec.Decode(&req); err != nil {
				return err

			}
			if err := restore.Coordinates(header.LastIndex, req); err != nil {
				return err
			}

		case structs.PreparedQueryRequestType:
			var req structs.PreparedQuery
			if err := dec.Decode(&req); err != nil {
				return err
			}
			if err := restore.PreparedQuery(&req); err != nil {
				return err
			}

		default:
			return fmt.Errorf("Unrecognized msg type: %v", msgType)
		}
	}

	restore.Commit()
	return nil
}

func (s *consulSnapshot) Persist(sink raft.SnapshotSink) error {
	defer metrics.MeasureSince([]string{"consul", "fsm", "persist"}, time.Now())

	// Register the nodes
	encoder := codec.NewEncoder(sink, msgpackHandle)

	// Write the header
	header := snapshotHeader{
		LastIndex: s.state.LastIndex(),
	}
	if err := encoder.Encode(&header); err != nil {
		sink.Cancel()
		return err
	}

	if err := s.persistNodes(sink, encoder); err != nil {
		sink.Cancel()
		return err
	}

	if err := s.persistSessions(sink, encoder); err != nil {
		sink.Cancel()
		return err
	}

	if err := s.persistACLs(sink, encoder); err != nil {
		sink.Cancel()
		return err
	}

	if err := s.persistKVs(sink, encoder); err != nil {
		sink.Cancel()
		return err
	}

	if err := s.persistTombstones(sink, encoder); err != nil {
		sink.Cancel()
		return err
	}

	if err := s.persistPreparedQueries(sink, encoder); err != nil {
		sink.Cancel()
		return err
	}

	return nil
}

func (s *consulSnapshot) persistNodes(sink raft.SnapshotSink,
	encoder *codec.Encoder) error {

	// Get all the nodes
	nodes, err := s.state.Nodes()
	if err != nil {
		return err
	}

	// Register each node
	for node := nodes.Next(); node != nil; node = nodes.Next() {
		n := node.(*structs.Node)
		req := structs.RegisterRequest{
			Node:            n.Node,
			Address:         n.Address,
			TaggedAddresses: n.TaggedAddresses,
		}

		// Register the node itself
		sink.Write([]byte{byte(structs.RegisterRequestType)})
		if err := encoder.Encode(&req); err != nil {
			return err
		}

		// Register each service this node has
		services, err := s.state.Services(n.Node)
		if err != nil {
			return err
		}
		for service := services.Next(); service != nil; service = services.Next() {
			sink.Write([]byte{byte(structs.RegisterRequestType)})
			req.Service = service.(*structs.ServiceNode).ToNodeService()
			if err := encoder.Encode(&req); err != nil {
				return err
			}
		}

		// Register each check this node has
		req.Service = nil
		checks, err := s.state.Checks(n.Node)
		if err != nil {
			return err
		}
		for check := checks.Next(); check != nil; check = checks.Next() {
			sink.Write([]byte{byte(structs.RegisterRequestType)})
			req.Check = check.(*structs.HealthCheck)
			if err := encoder.Encode(&req); err != nil {
				return err
			}
		}
	}

	// Save the coordinates separately since they are not part of the
	// register request interface. To avoid copying them out, we turn
	// them into batches with a single coordinate each.
	coords, err := s.state.Coordinates()
	if err != nil {
		return err
	}
	for coord := coords.Next(); coord != nil; coord = coords.Next() {
		sink.Write([]byte{byte(structs.CoordinateBatchUpdateType)})
		updates := structs.Coordinates{coord.(*structs.Coordinate)}
		if err := encoder.Encode(&updates); err != nil {
			return err
		}
	}
	return nil
}

func (s *consulSnapshot) persistSessions(sink raft.SnapshotSink,
	encoder *codec.Encoder) error {
	sessions, err := s.state.Sessions()
	if err != nil {
		return err
	}

	for session := sessions.Next(); session != nil; session = sessions.Next() {
		sink.Write([]byte{byte(structs.SessionRequestType)})
		if err := encoder.Encode(session.(*structs.Session)); err != nil {
			return err
		}
	}
	return nil
}

func (s *consulSnapshot) persistACLs(sink raft.SnapshotSink,
	encoder *codec.Encoder) error {
	acls, err := s.state.ACLs()
	if err != nil {
		return err
	}

	for acl := acls.Next(); acl != nil; acl = acls.Next() {
		sink.Write([]byte{byte(structs.ACLRequestType)})
		if err := encoder.Encode(acl.(*structs.ACL)); err != nil {
			return err
		}
	}
	return nil
}

func (s *consulSnapshot) persistKVs(sink raft.SnapshotSink,
	encoder *codec.Encoder) error {
	entries, err := s.state.KVs()
	if err != nil {
		return err
	}

	for entry := entries.Next(); entry != nil; entry = entries.Next() {
		sink.Write([]byte{byte(structs.KVSRequestType)})
		if err := encoder.Encode(entry.(*structs.DirEntry)); err != nil {
			return err
		}
	}
	return nil
}

func (s *consulSnapshot) persistTombstones(sink raft.SnapshotSink,
	encoder *codec.Encoder) error {
	stones, err := s.state.Tombstones()
	if err != nil {
		return err
	}

	for stone := stones.Next(); stone != nil; stone = stones.Next() {
		sink.Write([]byte{byte(structs.TombstoneRequestType)})

		// For historical reasons, these are serialized in the snapshots
		// as KV entries. We want to keep the snapshot format compatible
		// with pre-0.6 versions for now.
		s := stone.(*state.Tombstone)
		fake := &structs.DirEntry{
			Key: s.Key,
			RaftIndex: structs.RaftIndex{
				ModifyIndex: s.Index,
			},
		}
		if err := encoder.Encode(fake); err != nil {
			return err
		}
	}
	return nil
}

func (s *consulSnapshot) persistPreparedQueries(sink raft.SnapshotSink,
	encoder *codec.Encoder) error {
	queries, err := s.state.PreparedQueries()
	if err != nil {
		return err
	}

	for _, query := range queries {
		sink.Write([]byte{byte(structs.PreparedQueryRequestType)})
		if err := encoder.Encode(query); err != nil {
			return err
		}
	}
	return nil
}

func (s *consulSnapshot) Release() {
	s.state.Close()
}
