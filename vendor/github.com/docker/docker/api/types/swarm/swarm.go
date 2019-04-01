package swarm // import "github.com/docker/docker/api/types/swarm"

import "time"

// ClusterInfo represents info about the cluster for outputting in "info"
// it contains the same information as "Swarm", but without the JoinTokens
type ClusterInfo struct {
	ID string
	Meta
	Spec                   Spec
	TLSInfo                TLSInfo
	RootRotationInProgress bool
}

// Swarm represents a swarm.
type Swarm struct {
	ClusterInfo
	JoinTokens JoinTokens
}

// JoinTokens contains the tokens workers and managers need to join the swarm.
type JoinTokens struct {
	// Worker is the join token workers may use to join the swarm.
	Worker string
	// Manager is the join token managers may use to join the swarm.
	Manager string
}

// Spec represents the spec of a swarm.
type Spec struct {
	Annotations

	Orchestration    OrchestrationConfig `json:",omitempty"`
	Raft             RaftConfig          `json:",omitempty"`
	Dispatcher       DispatcherConfig    `json:",omitempty"`
	CAConfig         CAConfig            `json:",omitempty"`
	TaskDefaults     TaskDefaults        `json:",omitempty"`
	EncryptionConfig EncryptionConfig    `json:",omitempty"`
}

// OrchestrationConfig represents orchestration configuration.
type OrchestrationConfig struct {
	// TaskHistoryRetentionLimit is the number of historic tasks to keep per instance or
	// node. If negative, never remove completed or failed tasks.
	TaskHistoryRetentionLimit *int64 `json:",omitempty"`
}

// TaskDefaults parameterizes cluster-level task creation with default values.
type TaskDefaults struct {
	// LogDriver selects the log driver to use for tasks created in the
	// orchestrator if unspecified by a service.
	//
	// Updating this value will only have an affect on new tasks. Old tasks
	// will continue use their previously configured log driver until
	// recreated.
	LogDriver *Driver `json:",omitempty"`
}

// EncryptionConfig controls at-rest encryption of data and keys.
type EncryptionConfig struct {
	// AutoLockManagers specifies whether or not managers TLS keys and raft data
	// should be encrypted at rest in such a way that they must be unlocked
	// before the manager node starts up again.
	AutoLockManagers bool
}

// RaftConfig represents raft configuration.
type RaftConfig struct {
	// SnapshotInterval is the number of log entries between snapshots.
	SnapshotInterval uint64 `json:",omitempty"`

	// KeepOldSnapshots is the number of snapshots to keep beyond the
	// current snapshot.
	KeepOldSnapshots *uint64 `json:",omitempty"`

	// LogEntriesForSlowFollowers is the number of log entries to keep
	// around to sync up slow followers after a snapshot is created.
	LogEntriesForSlowFollowers uint64 `json:",omitempty"`

	// ElectionTick is the number of ticks that a follower will wait for a message
	// from the leader before becoming a candidate and starting an election.
	// ElectionTick must be greater than HeartbeatTick.
	//
	// A tick currently defaults to one second, so these translate directly to
	// seconds currently, but this is NOT guaranteed.
	ElectionTick int

	// HeartbeatTick is the number of ticks between heartbeats. Every
	// HeartbeatTick ticks, the leader will send a heartbeat to the
	// followers.
	//
	// A tick currently defaults to one second, so these translate directly to
	// seconds currently, but this is NOT guaranteed.
	HeartbeatTick int
}

// DispatcherConfig represents dispatcher configuration.
type DispatcherConfig struct {
	// HeartbeatPeriod defines how often agent should send heartbeats to
	// dispatcher.
	HeartbeatPeriod time.Duration `json:",omitempty"`
}

// CAConfig represents CA configuration.
type CAConfig struct {
	// NodeCertExpiry is the duration certificates should be issued for
	NodeCertExpiry time.Duration `json:",omitempty"`

	// ExternalCAs is a list of CAs to which a manager node will make
	// certificate signing requests for node certificates.
	ExternalCAs []*ExternalCA `json:",omitempty"`

	// SigningCACert and SigningCAKey specify the desired signing root CA and
	// root CA key for the swarm.  When inspecting the cluster, the key will
	// be redacted.
	SigningCACert string `json:",omitempty"`
	SigningCAKey  string `json:",omitempty"`

	// If this value changes, and there is no specified signing cert and key,
	// then the swarm is forced to generate a new root certificate ane key.
	ForceRotate uint64 `json:",omitempty"`
}

// ExternalCAProtocol represents type of external CA.
type ExternalCAProtocol string

// ExternalCAProtocolCFSSL CFSSL
const ExternalCAProtocolCFSSL ExternalCAProtocol = "cfssl"

// ExternalCA defines external CA to be used by the cluster.
type ExternalCA struct {
	// Protocol is the protocol used by this external CA.
	Protocol ExternalCAProtocol

	// URL is the URL where the external CA can be reached.
	URL string

	// Options is a set of additional key/value pairs whose interpretation
	// depends on the specified CA type.
	Options map[string]string `json:",omitempty"`

	// CACert specifies which root CA is used by this external CA.  This certificate must
	// be in PEM format.
	CACert string
}

// InitRequest is the request used to init a swarm.
type InitRequest struct {
	ListenAddr       string
	AdvertiseAddr    string
	DataPathAddr     string
	ForceNewCluster  bool
	Spec             Spec
	AutoLockManagers bool
	Availability     NodeAvailability
}

// JoinRequest is the request used to join a swarm.
type JoinRequest struct {
	ListenAddr    string
	AdvertiseAddr string
	DataPathAddr  string
	RemoteAddrs   []string
	JoinToken     string // accept by secret
	Availability  NodeAvailability
}

// UnlockRequest is the request used to unlock a swarm.
type UnlockRequest struct {
	// UnlockKey is the unlock key in ASCII-armored format.
	UnlockKey string
}

// LocalNodeState represents the state of the local node.
type LocalNodeState string

const (
	// LocalNodeStateInactive INACTIVE
	LocalNodeStateInactive LocalNodeState = "inactive"
	// LocalNodeStatePending PENDING
	LocalNodeStatePending LocalNodeState = "pending"
	// LocalNodeStateActive ACTIVE
	LocalNodeStateActive LocalNodeState = "active"
	// LocalNodeStateError ERROR
	LocalNodeStateError LocalNodeState = "error"
	// LocalNodeStateLocked LOCKED
	LocalNodeStateLocked LocalNodeState = "locked"
)

// Info represents generic information about swarm.
type Info struct {
	NodeID   string
	NodeAddr string

	LocalNodeState   LocalNodeState
	ControlAvailable bool
	Error            string

	RemoteManagers []Peer
	Nodes          int `json:",omitempty"`
	Managers       int `json:",omitempty"`

	Cluster *ClusterInfo `json:",omitempty"`
}

// Peer represents a peer.
type Peer struct {
	NodeID string
	Addr   string
}

// UpdateFlags contains flags for SwarmUpdate.
type UpdateFlags struct {
	RotateWorkerToken      bool
	RotateManagerToken     bool
	RotateManagerUnlockKey bool
}
