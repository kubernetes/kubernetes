package raft

// RPCHeader is a common sub-structure used to pass along protocol version and
// other information about the cluster. For older Raft implementations before
// versioning was added this will default to a zero-valued structure when read
// by newer Raft versions.
type RPCHeader struct {
	// ProtocolVersion is the version of the protocol the sender is
	// speaking.
	ProtocolVersion ProtocolVersion
}

// WithRPCHeader is an interface that exposes the RPC header.
type WithRPCHeader interface {
	GetRPCHeader() RPCHeader
}

// AppendEntriesRequest is the command used to append entries to the
// replicated log.
type AppendEntriesRequest struct {
	RPCHeader

	// Provide the current term and leader
	Term   uint64
	Leader []byte

	// Provide the previous entries for integrity checking
	PrevLogEntry uint64
	PrevLogTerm  uint64

	// New entries to commit
	Entries []*Log

	// Commit index on the leader
	LeaderCommitIndex uint64
}

// See WithRPCHeader.
func (r *AppendEntriesRequest) GetRPCHeader() RPCHeader {
	return r.RPCHeader
}

// AppendEntriesResponse is the response returned from an
// AppendEntriesRequest.
type AppendEntriesResponse struct {
	RPCHeader

	// Newer term if leader is out of date
	Term uint64

	// Last Log is a hint to help accelerate rebuilding slow nodes
	LastLog uint64

	// We may not succeed if we have a conflicting entry
	Success bool

	// There are scenarios where this request didn't succeed
	// but there's no need to wait/back-off the next attempt.
	NoRetryBackoff bool
}

// See WithRPCHeader.
func (r *AppendEntriesResponse) GetRPCHeader() RPCHeader {
	return r.RPCHeader
}

// RequestVoteRequest is the command used by a candidate to ask a Raft peer
// for a vote in an election.
type RequestVoteRequest struct {
	RPCHeader

	// Provide the term and our id
	Term      uint64
	Candidate []byte

	// Used to ensure safety
	LastLogIndex uint64
	LastLogTerm  uint64
}

// See WithRPCHeader.
func (r *RequestVoteRequest) GetRPCHeader() RPCHeader {
	return r.RPCHeader
}

// RequestVoteResponse is the response returned from a RequestVoteRequest.
type RequestVoteResponse struct {
	RPCHeader

	// Newer term if leader is out of date.
	Term uint64

	// Peers is deprecated, but required by servers that only understand
	// protocol version 0. This is not populated in protocol version 2
	// and later.
	Peers []byte

	// Is the vote granted.
	Granted bool
}

// See WithRPCHeader.
func (r *RequestVoteResponse) GetRPCHeader() RPCHeader {
	return r.RPCHeader
}

// InstallSnapshotRequest is the command sent to a Raft peer to bootstrap its
// log (and state machine) from a snapshot on another peer.
type InstallSnapshotRequest struct {
	RPCHeader
	SnapshotVersion SnapshotVersion

	Term   uint64
	Leader []byte

	// These are the last index/term included in the snapshot
	LastLogIndex uint64
	LastLogTerm  uint64

	// Peer Set in the snapshot. This is deprecated in favor of Configuration
	// but remains here in case we receive an InstallSnapshot from a leader
	// that's running old code.
	Peers []byte

	// Cluster membership.
	Configuration []byte
	// Log index where 'Configuration' entry was originally written.
	ConfigurationIndex uint64

	// Size of the snapshot
	Size int64
}

// See WithRPCHeader.
func (r *InstallSnapshotRequest) GetRPCHeader() RPCHeader {
	return r.RPCHeader
}

// InstallSnapshotResponse is the response returned from an
// InstallSnapshotRequest.
type InstallSnapshotResponse struct {
	RPCHeader

	Term    uint64
	Success bool
}

// See WithRPCHeader.
func (r *InstallSnapshotResponse) GetRPCHeader() RPCHeader {
	return r.RPCHeader
}
