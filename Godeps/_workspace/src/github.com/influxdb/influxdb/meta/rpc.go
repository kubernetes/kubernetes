package meta

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	"time"

	"github.com/gogo/protobuf/proto"
	"github.com/hashicorp/raft"
	"github.com/influxdb/influxdb/meta/internal"
)

// Max size of a message before we treat the size as invalid
const (
	MaxMessageSize    = 1024 * 1024 * 1024
	leaderDialTimeout = 10 * time.Second
)

// rpc handles request/response style messaging between cluster nodes
type rpc struct {
	logger         *log.Logger
	tracingEnabled bool

	store interface {
		cachedData() *Data
		enableLocalRaft() error
		IsLeader() bool
		Leader() string
		Peers() ([]string, error)
		SetPeers(addrs []string) error
		AddPeer(host string) error
		CreateNode(host string) (*NodeInfo, error)
		NodeByHost(host string) (*NodeInfo, error)
		WaitForDataChanged() error
	}
}

// JoinResult defines the join result structure.
type JoinResult struct {
	RaftEnabled bool
	RaftNodes   []string
	NodeID      uint64
}

// Reply defines the interface for Reply objects.
type Reply interface {
	GetHeader() *internal.ResponseHeader
}

// proxyLeader proxies the connection to the current raft leader
func (r *rpc) proxyLeader(conn *net.TCPConn, buf []byte) {
	if r.store.Leader() == "" {
		r.sendError(conn, "no leader detected during proxyLeader")
		return
	}

	leaderConn, err := net.DialTimeout("tcp", r.store.Leader(), leaderDialTimeout)
	if err != nil {
		r.sendError(conn, fmt.Sprintf("dial leader: %v", err))
		return
	}
	defer leaderConn.Close()

	leaderConn.Write([]byte{MuxRPCHeader})
	// re-write the original message to the leader
	leaderConn.Write(buf)
	if err := proxy(leaderConn.(*net.TCPConn), conn); err != nil {
		r.sendError(conn, fmt.Sprintf("leader proxy error: %v", err))
	}
}

// handleRPCConn reads a command from the connection and executes it.
func (r *rpc) handleRPCConn(conn net.Conn) {
	defer conn.Close()
	// RPC connections should execute on the leader.  If we are not the leader,
	// proxy the connection to the leader so that clients an connect to any node
	// in the cluster.
	r.traceCluster("rpc connection from: %v", conn.RemoteAddr())

	// Read and execute request.
	typ, buf, err := r.readMessage(conn)
	// Handle unexpected RPC errors
	if err != nil {
		r.sendError(conn, err.Error())
		return
	}

	if !r.store.IsLeader() && typ != internal.RPCType_PromoteRaft {
		r.proxyLeader(conn.(*net.TCPConn), pack(typ, buf))
		return
	}

	typ, resp, err := r.executeMessage(conn, typ, buf)

	// Handle unexpected RPC errors
	if err != nil {
		r.sendError(conn, err.Error())
		return
	}

	// Set the status header and error message
	if reply, ok := resp.(Reply); ok {
		reply.GetHeader().OK = proto.Bool(err == nil)
		if err != nil {
			reply.GetHeader().Error = proto.String(err.Error())
		}
	}

	r.sendResponse(conn, typ, resp)
}

func (r *rpc) readMessage(conn net.Conn) (internal.RPCType, []byte, error) {
	// Read request size.
	var sz uint64
	if err := binary.Read(conn, binary.BigEndian, &sz); err != nil {
		return internal.RPCType_Error, nil, fmt.Errorf("read size: %s", err)
	}

	if sz == 0 {
		return internal.RPCType_Error, nil, fmt.Errorf("invalid message size: %d", sz)
	}

	if sz >= MaxMessageSize {
		return internal.RPCType_Error, nil, fmt.Errorf("max message size of %d exceeded: %d", MaxMessageSize, sz)
	}

	// Read request.
	buf := make([]byte, sz)
	if _, err := io.ReadFull(conn, buf); err != nil {
		return internal.RPCType_Error, nil, fmt.Errorf("read request: %s", err)
	}

	// Determine the RPC type
	rpcType := internal.RPCType(btou64(buf[0:8]))
	buf = buf[8:]

	r.traceCluster("recv %v request on: %v", rpcType, conn.RemoteAddr())
	return rpcType, buf, nil
}

func (r *rpc) executeMessage(conn net.Conn, rpcType internal.RPCType, buf []byte) (internal.RPCType, proto.Message, error) {
	switch rpcType {
	case internal.RPCType_FetchData:
		var req internal.FetchDataRequest
		if err := proto.Unmarshal(buf, &req); err != nil {
			return internal.RPCType_Error, nil, fmt.Errorf("fetch request unmarshal: %v", err)
		}
		resp, err := r.handleFetchData(&req)
		return rpcType, resp, err
	case internal.RPCType_Join:
		var req internal.JoinRequest
		if err := proto.Unmarshal(buf, &req); err != nil {
			return internal.RPCType_Error, nil, fmt.Errorf("join request unmarshal: %v", err)
		}
		resp, err := r.handleJoinRequest(&req)
		return rpcType, resp, err
	case internal.RPCType_PromoteRaft:
		var req internal.PromoteRaftRequest
		if err := proto.Unmarshal(buf, &req); err != nil {
			return internal.RPCType_Error, nil, fmt.Errorf("promote to raft request unmarshal: %v", err)
		}
		resp, err := r.handlePromoteRaftRequest(&req)
		return rpcType, resp, err
	default:
		return internal.RPCType_Error, nil, fmt.Errorf("unknown rpc type:%v", rpcType)
	}
}

func (r *rpc) sendResponse(conn net.Conn, typ internal.RPCType, resp proto.Message) {
	// Marshal the response back to a protobuf
	buf, err := proto.Marshal(resp)
	if err != nil {
		r.logger.Printf("unable to marshal response: %v", err)
		return
	}

	// Encode response back to connection.
	if _, err := conn.Write(pack(typ, buf)); err != nil {
		r.logger.Printf("unable to write rpc response: %s", err)
	}
}

func (r *rpc) sendError(conn net.Conn, msg string) {
	r.traceCluster(msg)
	resp := &internal.ErrorResponse{
		Header: &internal.ResponseHeader{
			OK:    proto.Bool(false),
			Error: proto.String(msg),
		},
	}

	r.sendResponse(conn, internal.RPCType_Error, resp)
}

// handleFetchData handles a request for the current nodes meta data
func (r *rpc) handleFetchData(req *internal.FetchDataRequest) (*internal.FetchDataResponse, error) {
	var (
		b    []byte
		data *Data
		err  error
	)

	for {
		data = r.store.cachedData()
		if data.Index != req.GetIndex() {
			b, err = data.MarshalBinary()
			if err != nil {
				return nil, err
			}
			break
		}

		if !req.GetBlocking() {
			break
		}

		if err := r.store.WaitForDataChanged(); err != nil {
			return nil, err
		}
	}

	return &internal.FetchDataResponse{
		Header: &internal.ResponseHeader{
			OK: proto.Bool(true),
		},
		Index: proto.Uint64(data.Index),
		Term:  proto.Uint64(data.Term),
		Data:  b}, nil
}

// handleJoinRequest handles a request to join the cluster
func (r *rpc) handleJoinRequest(req *internal.JoinRequest) (*internal.JoinResponse, error) {
	r.traceCluster("join request from: %v", *req.Addr)

	node, err := func() (*NodeInfo, error) {

		// attempt to create the node
		node, err := r.store.CreateNode(*req.Addr)
		// if it exists, return the existing node
		if err == ErrNodeExists {
			node, err = r.store.NodeByHost(*req.Addr)
			if err != nil {
				return node, err
			}
			r.logger.Printf("existing node re-joined: id=%v addr=%v", node.ID, node.Host)
		} else if err != nil {
			return nil, fmt.Errorf("create node: %v", err)
		}

		peers, err := r.store.Peers()
		if err != nil {
			return nil, fmt.Errorf("list peers: %v", err)
		}

		// If we have less than 3 nodes, add them as raft peers if they are not
		// already a peer
		if len(peers) < MaxRaftNodes && !raft.PeerContained(peers, *req.Addr) {
			r.logger.Printf("adding new raft peer: nodeId=%v addr=%v", node.ID, *req.Addr)
			if err = r.store.AddPeer(*req.Addr); err != nil {
				return node, fmt.Errorf("add peer: %v", err)
			}
		}
		return node, err
	}()

	nodeID := uint64(0)
	if node != nil {
		nodeID = node.ID
	}

	if err != nil {
		return nil, err
	}

	// get the current raft peers
	peers, err := r.store.Peers()
	if err != nil {
		return nil, fmt.Errorf("list peers: %v", err)
	}

	return &internal.JoinResponse{
		Header: &internal.ResponseHeader{
			OK: proto.Bool(true),
		},
		EnableRaft: proto.Bool(raft.PeerContained(peers, *req.Addr)),
		RaftNodes:  peers,
		NodeID:     proto.Uint64(nodeID),
	}, err
}

func (r *rpc) handlePromoteRaftRequest(req *internal.PromoteRaftRequest) (*internal.PromoteRaftResponse, error) {
	r.traceCluster("promote raft request from: %v", *req.Addr)

	// Need to set the local store peers to match what we are about to join
	if err := r.store.SetPeers(req.RaftNodes); err != nil {
		return nil, err
	}

	if err := r.store.enableLocalRaft(); err != nil {
		return nil, err
	}

	if !contains(req.RaftNodes, *req.Addr) {
		req.RaftNodes = append(req.RaftNodes, *req.Addr)
	}

	if err := r.store.SetPeers(req.RaftNodes); err != nil {
		return nil, err
	}

	return &internal.PromoteRaftResponse{
		Header: &internal.ResponseHeader{
			OK: proto.Bool(true),
		},
		Success: proto.Bool(true),
	}, nil
}

// pack returns a TLV style byte slice encoding the size of the payload, the RPC type
// and the RPC data
func pack(typ internal.RPCType, b []byte) []byte {
	buf := u64tob(uint64(len(b)) + 8)
	buf = append(buf, u64tob(uint64(typ))...)
	buf = append(buf, b...)
	return buf
}

// fetchMetaData returns the latest copy of the meta store data from the current
// leader.
func (r *rpc) fetchMetaData(blocking bool) (*Data, error) {
	assert(r.store != nil, "store is nil")

	// Retrieve the current known leader.
	leader := r.store.Leader()
	if leader == "" {
		return nil, errors.New("no leader detected during fetchMetaData")
	}

	var index, term uint64
	data := r.store.cachedData()
	if data != nil {
		index = data.Index
		term = data.Index
	}
	resp, err := r.call(leader, &internal.FetchDataRequest{
		Index:    proto.Uint64(index),
		Term:     proto.Uint64(term),
		Blocking: proto.Bool(blocking),
	})
	if err != nil {
		return nil, err
	}

	switch t := resp.(type) {
	case *internal.FetchDataResponse:
		// If data is nil, then the term and index we sent matches the leader
		if t.GetData() == nil {
			return nil, nil
		}
		ms := &Data{}
		if err := ms.UnmarshalBinary(t.GetData()); err != nil {
			return nil, fmt.Errorf("rpc unmarshal metadata: %v", err)
		}
		return ms, nil
	case *internal.ErrorResponse:
		return nil, fmt.Errorf("rpc failed: %s", t.GetHeader().GetError())
	default:
		return nil, fmt.Errorf("rpc failed: unknown response type: %v", t.String())
	}
}

// join attempts to join a cluster at remoteAddr using localAddr as the current
// node's cluster address
func (r *rpc) join(localAddr, remoteAddr string) (*JoinResult, error) {
	req := &internal.JoinRequest{
		Addr: proto.String(localAddr),
	}

	resp, err := r.call(remoteAddr, req)
	if err != nil {
		return nil, err
	}

	switch t := resp.(type) {
	case *internal.JoinResponse:
		return &JoinResult{
			RaftEnabled: t.GetEnableRaft(),
			RaftNodes:   t.GetRaftNodes(),
			NodeID:      t.GetNodeID(),
		}, nil
	case *internal.ErrorResponse:
		return nil, fmt.Errorf("rpc failed: %s", t.GetHeader().GetError())
	default:
		return nil, fmt.Errorf("rpc failed: unknown response type: %v", t.String())
	}
}

// enableRaft attempts to promote a node at remoteAddr using localAddr as the current
// node's cluster address
func (r *rpc) enableRaft(addr string, peers []string) error {
	req := &internal.PromoteRaftRequest{
		Addr:      proto.String(addr),
		RaftNodes: peers,
	}

	resp, err := r.call(addr, req)
	if err != nil {
		return err
	}

	switch t := resp.(type) {
	case *internal.PromoteRaftResponse:
		return nil
	case *internal.ErrorResponse:
		return fmt.Errorf("rpc failed: %s", t.GetHeader().GetError())
	default:
		return fmt.Errorf("rpc failed: unknown response type: %v", t.String())
	}
}

// call sends an encoded request to the remote leader and returns
// an encoded response value.
func (r *rpc) call(dest string, req proto.Message) (proto.Message, error) {
	// Determine type of request
	var rpcType internal.RPCType
	switch t := req.(type) {
	case *internal.JoinRequest:
		rpcType = internal.RPCType_Join
	case *internal.FetchDataRequest:
		rpcType = internal.RPCType_FetchData
	case *internal.PromoteRaftRequest:
		rpcType = internal.RPCType_PromoteRaft
	default:
		return nil, fmt.Errorf("unknown rpc request type: %v", t)
	}

	// Create a connection to the leader.
	conn, err := net.DialTimeout("tcp", dest, leaderDialTimeout)
	if err != nil {
		return nil, fmt.Errorf("rpc dial: %v", err)
	}
	defer conn.Close()

	// Write a marker byte for rpc messages.
	_, err = conn.Write([]byte{MuxRPCHeader})
	if err != nil {
		return nil, err
	}

	b, err := proto.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("rpc marshal: %v", err)
	}

	// Write request size & bytes.
	if _, err := conn.Write(pack(rpcType, b)); err != nil {
		return nil, fmt.Errorf("write %v rpc: %s", rpcType, err)
	}

	data, err := ioutil.ReadAll(conn)
	if err != nil {
		return nil, fmt.Errorf("read %v rpc: %v", rpcType, err)
	}

	// Should always have a size and type
	if exp := 16; len(data) < exp {
		r.traceCluster("recv: %v", string(data))
		return nil, fmt.Errorf("rpc %v failed: short read: got %v, exp %v", rpcType, len(data), exp)
	}

	sz := btou64(data[0:8])
	if len(data[8:]) != int(sz) {
		r.traceCluster("recv: %v", string(data))
		return nil, fmt.Errorf("rpc %v failed: short read: got %v, exp %v", rpcType, len(data[8:]), sz)
	}

	// See what response type we got back, could get a general error response
	rpcType = internal.RPCType(btou64(data[8:16]))
	data = data[16:]

	var resp proto.Message
	switch rpcType {
	case internal.RPCType_Join:
		resp = &internal.JoinResponse{}
	case internal.RPCType_FetchData:
		resp = &internal.FetchDataResponse{}
	case internal.RPCType_Error:
		resp = &internal.ErrorResponse{}
	case internal.RPCType_PromoteRaft:
		resp = &internal.PromoteRaftResponse{}
	default:
		return nil, fmt.Errorf("unknown rpc response type: %v", rpcType)
	}

	if err := proto.Unmarshal(data, resp); err != nil {
		return nil, fmt.Errorf("rpc unmarshal: %v", err)
	}

	if reply, ok := resp.(Reply); ok {
		if !reply.GetHeader().GetOK() {
			return nil, fmt.Errorf("rpc %v failed: %s", rpcType, reply.GetHeader().GetError())
		}
	}

	return resp, nil
}

func (r *rpc) traceCluster(msg string, args ...interface{}) {
	if r.tracingEnabled {
		r.logger.Printf("rpc: "+msg, args...)
	}
}

func u64tob(v uint64) []byte {
	b := make([]byte, 8)
	binary.BigEndian.PutUint64(b, v)
	return b
}

func btou64(b []byte) uint64 {
	return binary.BigEndian.Uint64(b)
}

func contains(s []string, e string) bool {
	for _, a := range s {
		if a == e {
			return true
		}
	}
	return false
}
