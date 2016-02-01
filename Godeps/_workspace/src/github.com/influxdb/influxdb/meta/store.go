package meta

import (
	"bytes"
	crand "crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math/rand"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gogo/protobuf/proto"
	"github.com/hashicorp/raft"
	"github.com/hashicorp/raft-boltdb"
	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/meta/internal"
	"golang.org/x/crypto/bcrypt"
)

// tcp.Mux header bytes.
const (
	MuxRaftHeader = 0
	MuxExecHeader = 1

	// SaltBytes is the number of bytes used for salts
	SaltBytes = 32
)

// ExecMagic is the first 4 bytes sent to a remote exec connection to verify
// that it is coming from a remote exec client connection.
const ExecMagic = "EXEC"

// Retention policy settings.
const (
	AutoCreateRetentionPolicyName   = "default"
	AutoCreateRetentionPolicyPeriod = 0
	RetentionPolicyMinDuration      = time.Hour
)

// Raft configuration.
const (
	raftLogCacheSize      = 512
	raftSnapshotsRetained = 2
	raftTransportMaxPool  = 3
	raftTransportTimeout  = 10 * time.Second
)

// Store represents a raft-backed metastore.
type Store struct {
	mu     sync.RWMutex
	path   string
	opened bool

	id uint64 // local node id

	// All peers in cluster. Used during bootstrapping.
	peers []string

	data *Data

	remoteAddr net.Addr
	raft       *raft.Raft
	raftLayer  *raftLayer
	peerStore  raft.PeerStore
	transport  *raft.NetworkTransport
	store      *raftboltdb.BoltStore

	ready   chan struct{}
	err     chan error
	closing chan struct{}
	wg      sync.WaitGroup

	retentionAutoCreate bool

	// The listeners to accept raft and remote exec connections from.
	RaftListener net.Listener
	ExecListener net.Listener

	// The advertised hostname of the store.
	Addr net.Addr

	// The amount of time before a follower starts a new election.
	HeartbeatTimeout time.Duration

	// The amount of time before a candidate starts a new election.
	ElectionTimeout time.Duration

	// The amount of time without communication to the cluster before a
	// leader steps down to a follower state.
	LeaderLeaseTimeout time.Duration

	// The amount of time without an apply before sending a heartbeat.
	CommitTimeout time.Duration

	// Authentication cache.
	authCache map[string]authUser

	// hashPassword generates a cryptographically secure hash for password.
	// Returns an error if the password is invalid or a hash cannot be generated.
	hashPassword HashPasswordFn

	Logger *log.Logger
}

type authUser struct {
	salt []byte
	hash []byte
}

// NewStore returns a new instance of Store.
func NewStore(c Config) *Store {
	return &Store{
		path:  c.Dir,
		peers: c.Peers,
		data:  &Data{},

		ready:   make(chan struct{}),
		err:     make(chan error),
		closing: make(chan struct{}),

		retentionAutoCreate: c.RetentionAutoCreate,

		HeartbeatTimeout:   time.Duration(c.HeartbeatTimeout),
		ElectionTimeout:    time.Duration(c.ElectionTimeout),
		LeaderLeaseTimeout: time.Duration(c.LeaderLeaseTimeout),
		CommitTimeout:      time.Duration(c.CommitTimeout),
		authCache:          make(map[string]authUser, 0),
		hashPassword: func(password string) ([]byte, error) {
			return bcrypt.GenerateFromPassword([]byte(password), BcryptCost)
		},
		Logger: log.New(os.Stderr, "[metastore] ", log.LstdFlags),
	}
}

// Path returns the root path when open.
// Returns an empty string when the store is closed.
func (s *Store) Path() string { return s.path }

// IDPath returns the path to the local node ID file.
func (s *Store) IDPath() string { return filepath.Join(s.path, "id") }

// Open opens and initializes the raft store.
func (s *Store) Open() error {
	// Verify that no more than 3 peers.
	// https://github.com/influxdb/influxdb/issues/2750
	if len(s.peers) > 3 {
		return ErrTooManyPeers
	}

	// Verify listeners are set.
	if s.RaftListener == nil {
		panic("Store.RaftListener not set")
	} else if s.ExecListener == nil {
		panic("Store.ExecListener not set")
	}

	if err := func() error {
		s.mu.Lock()
		defer s.mu.Unlock()

		// Check if store has already been opened.
		if s.opened {
			return ErrStoreOpen
		}
		s.opened = true

		// Create the root directory if it doesn't already exist.
		if err := os.MkdirAll(s.path, 0777); err != nil {
			return fmt.Errorf("mkdir all: %s", err)
		}

		// Open the raft store.
		if err := s.openRaft(); err != nil {
			return fmt.Errorf("raft: %s", err)
		}

		// Initialize the store, if necessary.
		if err := s.initialize(); err != nil {
			return fmt.Errorf("initialize raft: %s", err)
		}

		// Load existing ID, if exists.
		if err := s.readID(); err != nil {
			return fmt.Errorf("read id: %s", err)
		}

		return nil
	}(); err != nil {
		s.close()
		return err
	}

	// Begin serving listener.
	s.wg.Add(1)
	go s.serveExecListener()

	// If the ID doesn't exist then create a new node.
	if s.id == 0 {
		go s.init()
	} else {
		close(s.ready)
	}

	return nil
}

// openRaft initializes the raft store.
func (s *Store) openRaft() error {
	// Setup raft configuration.
	config := raft.DefaultConfig()
	config.Logger = s.Logger
	config.HeartbeatTimeout = s.HeartbeatTimeout
	config.ElectionTimeout = s.ElectionTimeout
	config.LeaderLeaseTimeout = s.LeaderLeaseTimeout
	config.CommitTimeout = s.CommitTimeout

	// If no peers are set in the config then start as a single server.
	config.EnableSingleNode = (len(s.peers) == 0)

	// Build raft layer to multiplex listener.
	s.raftLayer = newRaftLayer(s.RaftListener, s.Addr)

	// Create a transport layer
	s.transport = raft.NewNetworkTransport(s.raftLayer, 3, 10*time.Second, os.Stderr)

	// Create peer storage.
	s.peerStore = raft.NewJSONPeers(s.path, s.transport)

	// Create the log store and stable store.
	store, err := raftboltdb.NewBoltStore(filepath.Join(s.path, "raft.db"))
	if err != nil {
		return fmt.Errorf("new bolt store: %s", err)
	}
	s.store = store

	// Create the snapshot store.
	snapshots, err := raft.NewFileSnapshotStore(s.path, raftSnapshotsRetained, os.Stderr)
	if err != nil {
		return fmt.Errorf("file snapshot store: %s", err)
	}

	// Create raft log.
	r, err := raft.NewRaft(config, (*storeFSM)(s), store, store, snapshots, s.peerStore, s.transport)
	if err != nil {
		return fmt.Errorf("new raft: %s", err)
	}
	s.raft = r

	return nil
}

// initialize attempts to bootstrap the raft store if there are no committed entries.
func (s *Store) initialize() error {
	// If we have committed entries then the store is already in the cluster.
	/*
		if index, err := s.store.LastIndex(); err != nil {
			return fmt.Errorf("last index: %s", err)
		} else if index > 0 {
			return nil
		}
	*/

	// Force set peers.
	if err := s.SetPeers(s.peers); err != nil {
		return fmt.Errorf("set raft peers: %s", err)
	}

	return nil
}

// Close closes the store and shuts down the node in the cluster.
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.close()
}

func (s *Store) close() error {
	// Check if store has already been closed.
	if !s.opened {
		return ErrStoreClosed
	}
	s.opened = false

	// Notify goroutines of close.
	close(s.closing)
	// FIXME(benbjohnson): s.wg.Wait()

	// Shutdown raft.
	if s.raft != nil {
		s.raft.Shutdown()
		s.raft = nil
	}
	if s.transport != nil {
		s.transport.Close()
		s.transport = nil
	}
	if s.store != nil {
		s.store.Close()
		s.store = nil
	}

	return nil
}

// readID reads the local node ID from the ID file.
func (s *Store) readID() error {
	b, err := ioutil.ReadFile(s.IDPath())
	if os.IsNotExist(err) {
		s.id = 0
		return nil
	} else if err != nil {
		return fmt.Errorf("read file: %s", err)
	}

	id, err := strconv.ParseUint(string(b), 10, 64)
	if err != nil {
		return fmt.Errorf("parse id: %s", err)
	}
	s.id = id

	s.Logger.Printf("read local node id: %d", s.id)

	return nil
}

// init initializes the store in a separate goroutine.
// This occurs when the store first creates or joins a cluster.
// The ready channel is closed once the store is initialized.
func (s *Store) init() {
	// Create a node for this store.
	if err := s.createLocalNode(); err != nil {
		s.err <- fmt.Errorf("create local node: %s", err)
		return
	}

	// Notify the ready channel.
	close(s.ready)
}

// createLocalNode creates the node for this local instance.
// Writes the id of the node to file on success.
func (s *Store) createLocalNode() error {
	// Wait for leader.
	if err := s.WaitForLeader(0); err != nil {
		return fmt.Errorf("wait for leader: %s", err)
	}

	// Create new node.
	ni, err := s.CreateNode(s.Addr.String())
	if err != nil {
		return fmt.Errorf("create node: %s", err)
	}

	// Write node id to file.
	if err := ioutil.WriteFile(s.IDPath(), []byte(strconv.FormatUint(ni.ID, 10)), 0666); err != nil {
		return fmt.Errorf("write file: %s", err)
	}

	// Set ID locally.
	s.id = ni.ID

	s.Logger.Printf("created local node: id=%d, host=%s", s.id, s.Addr.String())

	return nil
}

// Snapshot saves a snapshot of the current state.
func (s *Store) Snapshot() error {
	future := s.raft.Snapshot()
	return future.Error()
}

// WaitForLeader sleeps until a leader is found or a timeout occurs.
// timeout == 0 means to wait forever.
func (s *Store) WaitForLeader(timeout time.Duration) error {
	if s.raft.Leader() != "" {
		return nil
	}

	// Begin timeout timer.
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	// Continually check for leader until timeout.
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-s.closing:
			return errors.New("closing")
		case <-timer.C:
			if timeout != 0 {
				return errors.New("timeout")
			}
		case <-ticker.C:
			if s.raft.Leader() != "" {
				return nil
			}
		}
	}
}

// Ready returns a channel that is closed once the store is initialized.
func (s *Store) Ready() <-chan struct{} { return s.ready }

// Err returns a channel for all out-of-band errors.
func (s *Store) Err() <-chan error { return s.err }

// IsLeader returns true if the store is currently the leader.
func (s *Store) IsLeader() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.raft == nil {
		return false
	}
	return s.raft.State() == raft.Leader
}

// Leader returns what the store thinks is the current leader. An empty
// string indicates no leader exists.
func (s *Store) Leader() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.raft == nil {
		return ""
	}
	return s.raft.Leader()
}

// LeaderCh returns a channel that notifies on leadership change.
// Panics when the store has not been opened yet.
func (s *Store) LeaderCh() <-chan bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	assert(s.raft != nil, "cannot retrieve leadership channel when closed")
	return s.raft.LeaderCh()
}

// SetPeers sets a list of peers in the cluster.
func (s *Store) SetPeers(addrs []string) error {
	a := make([]string, len(addrs))
	for i, s := range addrs {
		addr, err := net.ResolveTCPAddr("tcp", s)
		if err != nil {
			return fmt.Errorf("cannot resolve addr: %s, err=%s", s, err)
		}
		a[i] = addr.String()
	}
	return s.raft.SetPeers(a).Error()
}

// serveExecListener processes remote exec connections.
// This function runs in a separate goroutine.
func (s *Store) serveExecListener() {
	defer s.wg.Done()

	for {
		// Accept next TCP connection.
		conn, err := s.ExecListener.Accept()
		if err != nil {
			if strings.Contains(err.Error(), "connection closed") {
				return
			} else {
				s.Logger.Printf("temporary accept error: %s", err)
				continue
			}
		}

		// Handle connection in a separate goroutine.
		s.wg.Add(1)
		go s.handleExecConn(conn)
	}
}

// handleExecConn reads a command from the connection and executes it.
func (s *Store) handleExecConn(conn net.Conn) {
	defer s.wg.Done()

	// Read and execute command.
	err := func() error {
		// Read marker message.
		b := make([]byte, 4)
		if _, err := io.ReadFull(conn, b); err != nil {
			return fmt.Errorf("read magic: %s", err)
		} else if string(b) != ExecMagic {
			return fmt.Errorf("invalid exec magic: %q", string(b))
		}

		// Read command size.
		var sz uint64
		if err := binary.Read(conn, binary.BigEndian, &sz); err != nil {
			return fmt.Errorf("read size: %s", err)
		}

		// Read command.
		buf := make([]byte, sz)
		if _, err := io.ReadFull(conn, buf); err != nil {
			return fmt.Errorf("read command: %s", err)
		}

		// Ensure command can be deserialized before applying.
		if err := proto.Unmarshal(buf, &internal.Command{}); err != nil {
			return fmt.Errorf("unable to unmarshal command: %s", err)
		}

		// Apply against the raft log.
		if err := s.apply(buf); err != nil {
			return fmt.Errorf("apply: %s", err)
		}
		return nil
	}()

	// Build response message.
	var resp internal.Response
	resp.OK = proto.Bool(err == nil)
	resp.Index = proto.Uint64(s.raft.LastIndex())
	if err != nil {
		resp.Error = proto.String(err.Error())
	}

	// Encode response back to connection.
	if b, err := proto.Marshal(&resp); err != nil {
		panic(err)
	} else if err = binary.Write(conn, binary.BigEndian, uint64(len(b))); err != nil {
		s.Logger.Printf("unable to write exec response size: %s", err)
	} else if _, err = conn.Write(b); err != nil {
		s.Logger.Printf("unable to write exec response: %s", err)
	}
	conn.Close()
}

// MarshalBinary encodes the store's data to a binary protobuf format.
func (s *Store) MarshalBinary() ([]byte, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.data.MarshalBinary()
}

// ClusterID returns the unique identifier for the cluster.
// This is generated once a node has been created.
func (s *Store) ClusterID() (id uint64, err error) {
	err = s.read(func(data *Data) error {
		id = data.ClusterID
		return nil
	})
	return
}

// NodeID returns the identifier for the local node.
// Panics if the node has not joined the cluster.
func (s *Store) NodeID() uint64 { return s.id }

// Node returns a node by id.
func (s *Store) Node(id uint64) (ni *NodeInfo, err error) {
	err = s.read(func(data *Data) error {
		ni = data.Node(id)
		if ni == nil {
			return errInvalidate
		}
		return nil
	})
	return
}

// NodeByHost returns a node by hostname.
func (s *Store) NodeByHost(host string) (ni *NodeInfo, err error) {
	err = s.read(func(data *Data) error {
		ni = data.NodeByHost(host)
		if ni == nil {
			return errInvalidate
		}
		return nil
	})
	return
}

// Nodes returns a list of all nodes.
func (s *Store) Nodes() (a []NodeInfo, err error) {
	err = s.read(func(data *Data) error {
		a = data.Nodes
		return nil
	})
	return
}

// CreateNode creates a new node in the store.
func (s *Store) CreateNode(host string) (*NodeInfo, error) {
	if err := s.exec(internal.Command_CreateNodeCommand, internal.E_CreateNodeCommand_Command,
		&internal.CreateNodeCommand{
			Host: proto.String(host),
			Rand: proto.Uint64(uint64(rand.Int63())),
		},
	); err != nil {
		return nil, err
	}
	return s.NodeByHost(host)
}

// DeleteNode removes a node from the metastore by id.
func (s *Store) DeleteNode(id uint64) error {
	return s.exec(internal.Command_DeleteNodeCommand, internal.E_DeleteNodeCommand_Command,
		&internal.DeleteNodeCommand{
			ID: proto.Uint64(id),
		},
	)
}

// Database returns a database by name.
func (s *Store) Database(name string) (di *DatabaseInfo, err error) {
	err = s.read(func(data *Data) error {
		di = data.Database(name)
		if di == nil {
			return errInvalidate
		}
		return nil
	})
	return
}

// Databases returns a list of all databases.
func (s *Store) Databases() (dis []DatabaseInfo, err error) {
	err = s.read(func(data *Data) error {
		dis = data.Databases
		return nil
	})
	return
}

// CreateDatabase creates a new database in the store.
func (s *Store) CreateDatabase(name string) (*DatabaseInfo, error) {
	if err := s.exec(internal.Command_CreateDatabaseCommand, internal.E_CreateDatabaseCommand_Command,
		&internal.CreateDatabaseCommand{
			Name: proto.String(name),
		},
	); err != nil {
		return nil, err
	}

	if s.retentionAutoCreate {
		// Read node count.
		// Retention policies must be fully replicated.
		var nodeN int
		if err := s.read(func(data *Data) error {
			nodeN = len(data.Nodes)
			return nil
		}); err != nil {
			return nil, fmt.Errorf("read: %s", err)
		}

		// Create a retention policy.
		rpi := NewRetentionPolicyInfo(AutoCreateRetentionPolicyName)
		rpi.ReplicaN = nodeN
		rpi.Duration = AutoCreateRetentionPolicyPeriod
		if _, err := s.CreateRetentionPolicy(name, rpi); err != nil {
			return nil, err
		}

		// Set it as the default retention policy.
		if err := s.SetDefaultRetentionPolicy(name, AutoCreateRetentionPolicyName); err != nil {
			return nil, err
		}
	}

	return s.Database(name)
}

// CreateDatabaseIfNotExists creates a new database in the store if it doesn't already exist.
func (s *Store) CreateDatabaseIfNotExists(name string) (*DatabaseInfo, error) {
	// Try to find database locally first.
	if di, err := s.Database(name); err != nil {
		return nil, err
	} else if di != nil {
		return di, nil
	}

	// Attempt to create database.
	if di, err := s.CreateDatabase(name); err == ErrDatabaseExists {
		return s.Database(name)
	} else {
		return di, err
	}
}

// DropDatabase removes a database from the metastore by name.
func (s *Store) DropDatabase(name string) error {
	return s.exec(internal.Command_DropDatabaseCommand, internal.E_DropDatabaseCommand_Command,
		&internal.DropDatabaseCommand{
			Name: proto.String(name),
		},
	)
}

// RetentionPolicy returns a retention policy for a database by name.
func (s *Store) RetentionPolicy(database, name string) (rpi *RetentionPolicyInfo, err error) {
	err = s.read(func(data *Data) error {
		rpi, err = data.RetentionPolicy(database, name)
		if err != nil {
			return err
		} else if rpi == nil {
			return errInvalidate
		}
		return nil
	})
	return
}

// DefaultRetentionPolicy returns the default retention policy for a database.
func (s *Store) DefaultRetentionPolicy(database string) (rpi *RetentionPolicyInfo, err error) {
	err = s.read(func(data *Data) error {
		di := data.Database(database)
		if di == nil {
			return ErrDatabaseNotFound
		}

		for i := range di.RetentionPolicies {
			if di.RetentionPolicies[i].Name == di.DefaultRetentionPolicy {
				rpi = &di.RetentionPolicies[i]
				return nil
			}
		}
		return errInvalidate
	})
	return
}

// RetentionPolicies returns a list of all retention policies for a database.
func (s *Store) RetentionPolicies(database string) (a []RetentionPolicyInfo, err error) {
	err = s.read(func(data *Data) error {
		di := data.Database(database)
		if di != nil {
			return ErrDatabaseNotFound
		}
		a = di.RetentionPolicies
		return nil
	})
	return
}

// CreateRetentionPolicy creates a new retention policy for a database.
func (s *Store) CreateRetentionPolicy(database string, rpi *RetentionPolicyInfo) (*RetentionPolicyInfo, error) {
	if rpi.Duration < RetentionPolicyMinDuration && rpi.Duration != 0 {
		return nil, ErrRetentionPolicyDurationTooLow
	}
	if err := s.exec(internal.Command_CreateRetentionPolicyCommand, internal.E_CreateRetentionPolicyCommand_Command,
		&internal.CreateRetentionPolicyCommand{
			Database:        proto.String(database),
			RetentionPolicy: rpi.marshal(),
		},
	); err != nil {
		return nil, err
	}

	return s.RetentionPolicy(database, rpi.Name)
}

// CreateRetentionPolicyIfNotExists creates a new policy in the store if it doesn't already exist.
func (s *Store) CreateRetentionPolicyIfNotExists(database string, rpi *RetentionPolicyInfo) (*RetentionPolicyInfo, error) {
	// Try to find policy locally first.
	if rpi, err := s.RetentionPolicy(database, rpi.Name); err != nil {
		return nil, err
	} else if rpi != nil {
		return rpi, nil
	}

	// Attempt to create policy.
	if other, err := s.CreateRetentionPolicy(database, rpi); err == ErrRetentionPolicyExists {
		return s.RetentionPolicy(database, rpi.Name)
	} else {
		return other, err
	}
}

// SetDefaultRetentionPolicy sets the default retention policy for a database.
func (s *Store) SetDefaultRetentionPolicy(database, name string) error {
	return s.exec(internal.Command_SetDefaultRetentionPolicyCommand, internal.E_SetDefaultRetentionPolicyCommand_Command,
		&internal.SetDefaultRetentionPolicyCommand{
			Database: proto.String(database),
			Name:     proto.String(name),
		},
	)
}

// UpdateRetentionPolicy updates an existing retention policy.
func (s *Store) UpdateRetentionPolicy(database, name string, rpu *RetentionPolicyUpdate) error {
	var newName *string
	if rpu.Name != nil {
		newName = rpu.Name
	}

	var duration *int64
	if rpu.Duration != nil {
		value := int64(*rpu.Duration)
		duration = &value
	}

	var replicaN *uint32
	if rpu.ReplicaN != nil {
		value := uint32(*rpu.ReplicaN)
		replicaN = &value
	}

	return s.exec(internal.Command_UpdateRetentionPolicyCommand, internal.E_UpdateRetentionPolicyCommand_Command,
		&internal.UpdateRetentionPolicyCommand{
			Database: proto.String(database),
			Name:     proto.String(name),
			NewName:  newName,
			Duration: duration,
			ReplicaN: replicaN,
		},
	)
}

// DropRetentionPolicy removes a policy from a database by name.
func (s *Store) DropRetentionPolicy(database, name string) error {
	return s.exec(internal.Command_DropRetentionPolicyCommand, internal.E_DropRetentionPolicyCommand_Command,
		&internal.DropRetentionPolicyCommand{
			Database: proto.String(database),
			Name:     proto.String(name),
		},
	)
}

// FIX: CreateRetentionPolicyIfNotExists(database string, rp *RetentionPolicyInfo) (*RetentionPolicyInfo, error)

// CreateShardGroup creates a new shard group in a retention policy for a given time.
func (s *Store) CreateShardGroup(database, policy string, timestamp time.Time) (*ShardGroupInfo, error) {
	if err := s.exec(internal.Command_CreateShardGroupCommand, internal.E_CreateShardGroupCommand_Command,
		&internal.CreateShardGroupCommand{
			Database:  proto.String(database),
			Policy:    proto.String(policy),
			Timestamp: proto.Int64(timestamp.UnixNano()),
		},
	); err != nil {
		return nil, err
	}

	return s.ShardGroupByTimestamp(database, policy, timestamp)
}

// CreateShardGroupIfNotExists creates a new shard group if one doesn't already exist.
func (s *Store) CreateShardGroupIfNotExists(database, policy string, timestamp time.Time) (*ShardGroupInfo, error) {
	// Try to find shard group locally first.
	if sgi, err := s.ShardGroupByTimestamp(database, policy, timestamp); err != nil {
		return nil, err
	} else if sgi != nil && !sgi.Deleted() {
		return sgi, nil
	}

	// Attempt to create database.
	if sgi, err := s.CreateShardGroup(database, policy, timestamp); err == ErrShardGroupExists {
		return s.ShardGroupByTimestamp(database, policy, timestamp)
	} else {
		return sgi, err
	}
}

// DeleteShardGroup removes an existing shard group from a policy by ID.
func (s *Store) DeleteShardGroup(database, policy string, id uint64) error {
	return s.exec(internal.Command_DeleteShardGroupCommand, internal.E_DeleteShardGroupCommand_Command,
		&internal.DeleteShardGroupCommand{
			Database:     proto.String(database),
			Policy:       proto.String(policy),
			ShardGroupID: proto.Uint64(id),
		},
	)
}

// ShardGroups returns a list of all shard groups for a policy by timestamp.
func (s *Store) ShardGroups(database, policy string) (a []ShardGroupInfo, err error) {
	err = s.read(func(data *Data) error {
		a, err = data.ShardGroups(database, policy)
		if err != nil {
			return err
		}
		return nil
	})
	return
}

// ShardGroupsByTimeRange returns a slice of ShardGroups that may contain data for the given time range. ShardGroups
// are sorted by start time.
func (s *Store) ShardGroupsByTimeRange(database, policy string, tmin, tmax time.Time) (a []ShardGroupInfo, err error) {
	err = s.read(func(data *Data) error {
		a, err = data.ShardGroupsByTimeRange(database, policy, tmin, tmax)
		if err != nil {
			return err
		} else if a == nil {
			return errInvalidate
		}
		return nil
	})
	return
}

// VisitRetentionPolicies calls the given function with full retention policy details.
func (s *Store) VisitRetentionPolicies(f func(d DatabaseInfo, r RetentionPolicyInfo)) {
	s.read(func(data *Data) error {
		for _, di := range data.Databases {
			for _, rp := range di.RetentionPolicies {
				f(di, rp)
			}
		}
		return nil
	})
	return
}

// ShardGroupByTimestamp returns a shard group for a policy by timestamp.
func (s *Store) ShardGroupByTimestamp(database, policy string, timestamp time.Time) (sgi *ShardGroupInfo, err error) {
	err = s.read(func(data *Data) error {
		sgi, err = data.ShardGroupByTimestamp(database, policy, timestamp)
		if err != nil {
			return err
		} else if sgi == nil {
			return errInvalidate
		}
		return nil
	})
	return
}

func (s *Store) ShardOwner(shardID uint64) (database, policy string, sgi *ShardGroupInfo) {
	s.read(func(data *Data) error {
		for _, dbi := range data.Databases {
			for _, rpi := range dbi.RetentionPolicies {
				for _, g := range rpi.ShardGroups {
					if g.Deleted() {
						continue
					}

					for _, sh := range g.Shards {
						if sh.ID == shardID {
							database = dbi.Name
							policy = rpi.Name
							sgi = &g
							return nil
						}
					}
				}
			}
		}
		return errInvalidate
	})
	return
}

// CreateContinuousQuery creates a new continuous query on the store.
func (s *Store) CreateContinuousQuery(database, name, query string) error {
	return s.exec(internal.Command_CreateContinuousQueryCommand, internal.E_CreateContinuousQueryCommand_Command,
		&internal.CreateContinuousQueryCommand{
			Database: proto.String(database),
			Name:     proto.String(name),
			Query:    proto.String(query),
		},
	)
}

// DropContinuousQuery removes a continuous query from the store.
func (s *Store) DropContinuousQuery(database, name string) error {
	return s.exec(internal.Command_DropContinuousQueryCommand, internal.E_DropContinuousQueryCommand_Command,
		&internal.DropContinuousQueryCommand{
			Database: proto.String(database),
			Name:     proto.String(name),
		},
	)
}

// User returns a user by name.
func (s *Store) User(name string) (ui *UserInfo, err error) {
	err = s.read(func(data *Data) error {
		ui = data.User(name)
		if ui == nil {
			return errInvalidate
		}
		return nil
	})
	return
}

// Users returns a list of all users.
func (s *Store) Users() (a []UserInfo, err error) {
	err = s.read(func(data *Data) error {
		a = data.Users
		return nil
	})
	return
}

// AdminUserExists returns true if an admin user exists on the system.
func (s *Store) AdminUserExists() (exists bool, err error) {
	err = s.read(func(data *Data) error {
		for i := range data.Users {
			if data.Users[i].Admin {
				exists = true
				break
			}
		}
		return nil
	})
	return
}

// ErrAuthenticate is returned when authentication fails.
var ErrAuthenticate = errors.New("authentication failed")

// Authenticate retrieves a user with a matching username and password.
func (s *Store) Authenticate(username, password string) (ui *UserInfo, err error) {
	err = s.read(func(data *Data) error {
		s.mu.Lock()
		defer s.mu.Unlock()

		// Find user.
		u := data.User(username)
		if u == nil {
			return ErrUserNotFound
		}

		// Check the local auth cache first.
		if au, ok := s.authCache[username]; ok {
			// verify the password using the cached salt and hash
			hashed, err := s.hashWithSalt(au.salt, password)
			if err != nil {
				return err
			}

			if bytes.Equal(hashed, au.hash) {
				ui = u
				return nil
			} else {
				return ErrAuthenticate
			}
		}

		// Compare password with user hash.
		if err := bcrypt.CompareHashAndPassword([]byte(u.Hash), []byte(password)); err != nil {
			return ErrAuthenticate
		}

		// generate a salt and hash of the password for the cache
		salt, hashed, err := s.saltedHash(password)
		if err != nil {
			return err
		}
		s.authCache[username] = authUser{salt: salt, hash: hashed}

		ui = u
		return nil
	})
	return
}

// hashWithSalt returns a salted hash of password using salt
func (s *Store) hashWithSalt(salt []byte, password string) ([]byte, error) {
	hasher := sha256.New()
	hasher.Write(append(salt, []byte(password)...))
	return hasher.Sum(nil), nil
}

// saltedHash returns a salt and salted hash of password
func (s *Store) saltedHash(password string) (salt, hash []byte, err error) {
	salt = make([]byte, SaltBytes)
	_, err = io.ReadFull(crand.Reader, salt)
	if err != nil {
		return
	}

	hash, err = s.hashWithSalt(salt, password)
	return
}

// CreateUser creates a new user in the store.
func (s *Store) CreateUser(name, password string, admin bool) (*UserInfo, error) {
	// Hash the password before serializing it.
	hash, err := s.hashPassword(password)
	if err != nil {
		return nil, err
	}

	// Serialize command and send it to the leader.
	if err := s.exec(internal.Command_CreateUserCommand, internal.E_CreateUserCommand_Command,
		&internal.CreateUserCommand{
			Name:  proto.String(name),
			Hash:  proto.String(string(hash)),
			Admin: proto.Bool(admin),
		},
	); err != nil {
		return nil, err
	}
	return s.User(name)
}

// DropUser removes a user from the metastore by name.
func (s *Store) DropUser(name string) error {
	return s.exec(internal.Command_DropUserCommand, internal.E_DropUserCommand_Command,
		&internal.DropUserCommand{
			Name: proto.String(name),
		},
	)
}

// UpdateUser updates an existing user in the store.
func (s *Store) UpdateUser(name, password string) error {
	// Hash the password before serializing it.
	hash, err := s.hashPassword(password)
	if err != nil {
		return err
	}

	// Serialize command and send it to the leader.
	return s.exec(internal.Command_UpdateUserCommand, internal.E_UpdateUserCommand_Command,
		&internal.UpdateUserCommand{
			Name: proto.String(name),
			Hash: proto.String(string(hash)),
		},
	)
}

// SetPrivilege sets a privilege for a user on a database.
func (s *Store) SetPrivilege(username, database string, p influxql.Privilege) error {
	return s.exec(internal.Command_SetPrivilegeCommand, internal.E_SetPrivilegeCommand_Command,
		&internal.SetPrivilegeCommand{
			Username:  proto.String(username),
			Database:  proto.String(database),
			Privilege: proto.Int32(int32(p)),
		},
	)
}

// SetAdminPrivilege sets the admin privilege for a user on a database.
func (s *Store) SetAdminPrivilege(username string, admin bool) error {
	return s.exec(internal.Command_SetAdminPrivilegeCommand, internal.E_SetAdminPrivilegeCommand_Command,
		&internal.SetAdminPrivilegeCommand{
			Username: proto.String(username),
			Admin:    proto.Bool(admin),
		},
	)
}

// UserPrivileges returns a list of all databases.
func (s *Store) UserPrivileges(username string) (p map[string]influxql.Privilege, err error) {
	err = s.read(func(data *Data) error {
		p, err = data.UserPrivileges(username)
		return err
	})
	return
}

// UserPrivilege returns the privilege for a database.
func (s *Store) UserPrivilege(username, database string) (p *influxql.Privilege, err error) {
	err = s.read(func(data *Data) error {
		p, err = data.UserPrivilege(username, database)
		return err
	})
	return
}

// UserCount returns the number of users defined in the cluster.
func (s *Store) UserCount() (count int, err error) {
	err = s.read(func(data *Data) error {
		count = len(data.Users)
		return nil
	})
	return
}

// PrecreateShardGroups creates shard groups whose endtime is before the cutoff time passed in. This
// avoid the need for these shards to be created when data for the corresponding time range arrives.
// Shard creation involves Raft consensus, and precreation avoids taking the hit at write-time.
func (s *Store) PrecreateShardGroups(cutoff time.Time) error {
	s.read(func(data *Data) error {
		for _, di := range data.Databases {
			for _, rp := range di.RetentionPolicies {
				for _, g := range rp.ShardGroups {
					// Check to see if it is not deleted and going to end before our interval
					if !g.Deleted() && g.EndTime.Before(cutoff) {
						nextShardGroupTime := g.EndTime.Add(1 * time.Nanosecond)

						// Check if successive shard group exists.
						if sgi, err := s.ShardGroupByTimestamp(di.Name, rp.Name, nextShardGroupTime); err != nil {
							s.Logger.Printf("failed to check if successive shard group for group exists %d: %s",
								g.ID, err.Error())
							continue
						} else if sgi != nil && !sgi.Deleted() {
							continue
						}

						// It doesn't. Create it.
						if newGroup, err := s.CreateShardGroupIfNotExists(di.Name, rp.Name, nextShardGroupTime); err != nil {
							s.Logger.Printf("failed to create successive shard group for group %d: %s",
								g.ID, err.Error())
						} else {
							s.Logger.Printf("new shard group %d successfully created for database %s, retention policy %s",
								newGroup.ID, di.Name, rp.Name)
						}
					}
				}

			}
		}
		return nil
	})
	return nil
}

// SetData force overwrites the root data.
// This should only be used when restoring a snapshot.
func (s *Store) SetData(data *Data) error {
	return s.exec(internal.Command_SetDataCommand, internal.E_SetDataCommand_Command,
		&internal.SetDataCommand{
			Data: data.marshal(),
		},
	)
}

// read executes a function with the current metadata.
// If an error is returned then the cache is invalidated and retried.
//
// The error returned by the retry is passed through to the original caller
// unless the error is errInvalidate. A nil error is passed through when
// errInvalidate is returned.
func (s *Store) read(fn func(*Data) error) error {
	// First use the cached metadata.
	s.mu.RLock()
	data := s.data
	s.mu.RUnlock()

	// Execute fn against cached data.
	// Return immediately if there was no error.
	if err := fn(data); err == nil {
		return nil
	}

	// If an error occurred then invalidate cache and retry.
	if err := s.invalidate(); err != nil {
		return err
	}

	// Re-read the metadata.
	s.mu.RLock()
	data = s.data
	s.mu.RUnlock()

	// Passthrough error unless it is a cache invalidation.
	if err := fn(data); err != nil && err != errInvalidate {
		return err
	}

	return nil
}

// errInvalidate is returned to read() when the cache should be invalidated
// but an error should not be passed through to the caller.
var errInvalidate = errors.New("invalidate cache")

func (s *Store) invalidate() error {
	time.Sleep(1 * time.Second)
	return nil // FIXME(benbjohnson): Reload cache from the leader.
}

func (s *Store) exec(typ internal.Command_Type, desc *proto.ExtensionDesc, value interface{}) error {
	// Create command.
	cmd := &internal.Command{Type: &typ}
	err := proto.SetExtension(cmd, desc, value)
	assert(err == nil, "proto.SetExtension: %s", err)

	// Marshal to a byte slice.
	b, err := proto.Marshal(cmd)
	assert(err == nil, "proto.Marshal: %s", err)

	// Apply the command if this is the leader.
	// Otherwise remotely execute the command against the current leader.
	if s.raft.State() == raft.Leader {
		return s.apply(b)
	} else {
		return s.remoteExec(b)
	}
}

// apply applies a serialized command to the raft log.
func (s *Store) apply(b []byte) error {
	// Apply to raft log.
	f := s.raft.Apply(b, 0)
	if err := f.Error(); err != nil {
		return err
	}

	// Return response if it's an error.
	// No other non-nil objects should be returned.
	resp := f.Response()
	if err, ok := resp.(error); ok {
		return lookupError(err)
	}
	assert(resp == nil, "unexpected response: %#v", resp)

	return nil
}

// remoteExec sends an encoded command to the remote leader.
func (s *Store) remoteExec(b []byte) error {
	// Retrieve the current known leader.
	leader := s.raft.Leader()
	if leader == "" {
		return errors.New("no leader")
	}

	// Create a connection to the leader.
	conn, err := net.DialTimeout("tcp", leader, 10*time.Second)
	if err != nil {
		return err
	}
	defer conn.Close()

	// Write a marker byte for exec messages.
	_, err = conn.Write([]byte{MuxExecHeader})
	if err != nil {
		return err
	}

	// Write a marker message.
	_, err = conn.Write([]byte(ExecMagic))
	if err != nil {
		return err
	}

	// Write command size & bytes.
	if err := binary.Write(conn, binary.BigEndian, uint64(len(b))); err != nil {
		return fmt.Errorf("write command size: %s", err)
	} else if _, err := conn.Write(b); err != nil {
		return fmt.Errorf("write command: %s", err)
	}

	// Read response bytes.
	var sz uint64
	if err := binary.Read(conn, binary.BigEndian, &sz); err != nil {
		return fmt.Errorf("read response size: %s", err)
	}
	buf := make([]byte, sz)
	if _, err := io.ReadFull(conn, buf); err != nil {
		return fmt.Errorf("read response: %s", err)
	}

	// Unmarshal response.
	var resp internal.Response
	if err := proto.Unmarshal(buf, &resp); err != nil {
		return fmt.Errorf("unmarshal response: %s", err)
	} else if !resp.GetOK() {
		return fmt.Errorf("exec failed: %s", resp.GetError())
	}

	// Wait for local FSM to sync to index.
	if err := s.sync(resp.GetIndex(), 5*time.Second); err != nil {
		return fmt.Errorf("sync: %s", err)
	}

	return nil
}

// sync polls the state machine until it reaches a given index.
func (s *Store) sync(index uint64, timeout time.Duration) error {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		// Wait for next tick or timeout.
		select {
		case <-ticker.C:
		case <-timer.C:
			return errors.New("timeout")
		}

		// Compare index against current metadata.
		s.mu.Lock()
		ok := (s.data.Index >= index)
		s.mu.Unlock()

		// Exit if we are at least at the given index.
		if ok {
			return nil
		}
	}
}

// BcryptCost is the cost associated with generating password with Bcrypt.
// This setting is lowered during testing to improve test suite performance.
var BcryptCost = 10

// HashPasswordFn represnets a password hashing function.
type HashPasswordFn func(password string) ([]byte, error)

// GetHashPasswordFn returns the current password hashing function.
func (s *Store) GetHashPasswordFn() HashPasswordFn {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.hashPassword
}

// SetHashPasswordFn sets the password hashing function.
func (s *Store) SetHashPasswordFn(fn HashPasswordFn) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.hashPassword = fn
}

// storeFSM represents the finite state machine used by Store to interact with Raft.
type storeFSM Store

func (fsm *storeFSM) Apply(l *raft.Log) interface{} {
	var cmd internal.Command
	if err := proto.Unmarshal(l.Data, &cmd); err != nil {
		panic(fmt.Errorf("cannot marshal command: %x", l.Data))
	}

	// Lock the store.
	s := (*Store)(fsm)
	s.mu.Lock()
	defer s.mu.Unlock()

	err := func() interface{} {
		switch cmd.GetType() {
		case internal.Command_CreateNodeCommand:
			return fsm.applyCreateNodeCommand(&cmd)
		case internal.Command_DeleteNodeCommand:
			return fsm.applyDeleteNodeCommand(&cmd)
		case internal.Command_CreateDatabaseCommand:
			return fsm.applyCreateDatabaseCommand(&cmd)
		case internal.Command_DropDatabaseCommand:
			return fsm.applyDropDatabaseCommand(&cmd)
		case internal.Command_CreateRetentionPolicyCommand:
			return fsm.applyCreateRetentionPolicyCommand(&cmd)
		case internal.Command_DropRetentionPolicyCommand:
			return fsm.applyDropRetentionPolicyCommand(&cmd)
		case internal.Command_SetDefaultRetentionPolicyCommand:
			return fsm.applySetDefaultRetentionPolicyCommand(&cmd)
		case internal.Command_UpdateRetentionPolicyCommand:
			return fsm.applyUpdateRetentionPolicyCommand(&cmd)
		case internal.Command_CreateShardGroupCommand:
			return fsm.applyCreateShardGroupCommand(&cmd)
		case internal.Command_DeleteShardGroupCommand:
			return fsm.applyDeleteShardGroupCommand(&cmd)
		case internal.Command_CreateContinuousQueryCommand:
			return fsm.applyCreateContinuousQueryCommand(&cmd)
		case internal.Command_DropContinuousQueryCommand:
			return fsm.applyDropContinuousQueryCommand(&cmd)
		case internal.Command_CreateUserCommand:
			return fsm.applyCreateUserCommand(&cmd)
		case internal.Command_DropUserCommand:
			return fsm.applyDropUserCommand(&cmd)
		case internal.Command_UpdateUserCommand:
			return fsm.applyUpdateUserCommand(&cmd)
		case internal.Command_SetPrivilegeCommand:
			return fsm.applySetPrivilegeCommand(&cmd)
		case internal.Command_SetAdminPrivilegeCommand:
			return fsm.applySetAdminPrivilegeCommand(&cmd)
		case internal.Command_SetDataCommand:
			return fsm.applySetDataCommand(&cmd)
		default:
			panic(fmt.Errorf("cannot apply command: %x", l.Data))
		}
	}()

	// Copy term and index to new metadata.
	fsm.data.Term = l.Term
	fsm.data.Index = l.Index

	return err
}

func (fsm *storeFSM) applyCreateNodeCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_CreateNodeCommand_Command)
	v := ext.(*internal.CreateNodeCommand)

	// Copy data and update.
	other := fsm.data.Clone()
	if err := other.CreateNode(v.GetHost()); err != nil {
		return err
	}

	// If the cluster ID hasn't been set then use the command's random number.
	if other.ClusterID == 0 {
		other.ClusterID = uint64(v.GetRand())
	}

	fsm.data = other
	return nil
}

func (fsm *storeFSM) applyDeleteNodeCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_DeleteNodeCommand_Command)
	v := ext.(*internal.DeleteNodeCommand)

	// Copy data and update.
	other := fsm.data.Clone()
	if err := other.DeleteNode(v.GetID()); err != nil {
		return err
	}
	fsm.data = other

	return nil
}

func (fsm *storeFSM) applyCreateDatabaseCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_CreateDatabaseCommand_Command)
	v := ext.(*internal.CreateDatabaseCommand)

	// Copy data and update.
	other := fsm.data.Clone()
	if err := other.CreateDatabase(v.GetName()); err != nil {
		return err
	}
	fsm.data = other

	return nil
}

func (fsm *storeFSM) applyDropDatabaseCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_DropDatabaseCommand_Command)
	v := ext.(*internal.DropDatabaseCommand)

	// Copy data and update.
	other := fsm.data.Clone()
	if err := other.DropDatabase(v.GetName()); err != nil {
		return err
	}
	fsm.data = other

	return nil
}

func (fsm *storeFSM) applyCreateRetentionPolicyCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_CreateRetentionPolicyCommand_Command)
	v := ext.(*internal.CreateRetentionPolicyCommand)
	pb := v.GetRetentionPolicy()

	// Copy data and update.
	other := fsm.data.Clone()
	if err := other.CreateRetentionPolicy(v.GetDatabase(),
		&RetentionPolicyInfo{
			Name:               pb.GetName(),
			ReplicaN:           int(pb.GetReplicaN()),
			Duration:           time.Duration(pb.GetDuration()),
			ShardGroupDuration: time.Duration(pb.GetShardGroupDuration()),
		}); err != nil {
		return err
	}
	fsm.data = other

	return nil
}

func (fsm *storeFSM) applyDropRetentionPolicyCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_DropRetentionPolicyCommand_Command)
	v := ext.(*internal.DropRetentionPolicyCommand)

	// Copy data and update.
	other := fsm.data.Clone()
	if err := other.DropRetentionPolicy(v.GetDatabase(), v.GetName()); err != nil {
		return err
	}
	fsm.data = other

	return nil
}

func (fsm *storeFSM) applySetDefaultRetentionPolicyCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_SetDefaultRetentionPolicyCommand_Command)
	v := ext.(*internal.SetDefaultRetentionPolicyCommand)

	// Copy data and update.
	other := fsm.data.Clone()
	if err := other.SetDefaultRetentionPolicy(v.GetDatabase(), v.GetName()); err != nil {
		return err
	}
	fsm.data = other

	return nil
}

func (fsm *storeFSM) applyUpdateRetentionPolicyCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_UpdateRetentionPolicyCommand_Command)
	v := ext.(*internal.UpdateRetentionPolicyCommand)

	// Create update object.
	rpu := RetentionPolicyUpdate{Name: v.NewName}
	if v.Duration != nil {
		value := time.Duration(v.GetDuration())
		rpu.Duration = &value
	}
	if v.ReplicaN != nil {
		value := int(v.GetReplicaN())
		rpu.ReplicaN = &value
	}

	// Copy data and update.
	other := fsm.data.Clone()
	if err := other.UpdateRetentionPolicy(v.GetDatabase(), v.GetName(), &rpu); err != nil {
		return err
	}
	fsm.data = other

	return nil
}

func (fsm *storeFSM) applyCreateShardGroupCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_CreateShardGroupCommand_Command)
	v := ext.(*internal.CreateShardGroupCommand)

	// Copy data and update.
	other := fsm.data.Clone()
	if err := other.CreateShardGroup(v.GetDatabase(), v.GetPolicy(), time.Unix(0, v.GetTimestamp())); err != nil {
		return err
	}
	fsm.data = other

	return nil
}

func (fsm *storeFSM) applyDeleteShardGroupCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_DeleteShardGroupCommand_Command)
	v := ext.(*internal.DeleteShardGroupCommand)

	// Copy data and update.
	other := fsm.data.Clone()
	if err := other.DeleteShardGroup(v.GetDatabase(), v.GetPolicy(), v.GetShardGroupID()); err != nil {
		return err
	}
	fsm.data = other

	return nil
}

func (fsm *storeFSM) applyCreateContinuousQueryCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_CreateContinuousQueryCommand_Command)
	v := ext.(*internal.CreateContinuousQueryCommand)

	// Copy data and update.
	other := fsm.data.Clone()
	if err := other.CreateContinuousQuery(v.GetDatabase(), v.GetName(), v.GetQuery()); err != nil {
		return err
	}
	fsm.data = other

	return nil
}

func (fsm *storeFSM) applyDropContinuousQueryCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_DropContinuousQueryCommand_Command)
	v := ext.(*internal.DropContinuousQueryCommand)

	// Copy data and update.
	other := fsm.data.Clone()
	if err := other.DropContinuousQuery(v.GetDatabase(), v.GetName()); err != nil {
		return err
	}
	fsm.data = other

	return nil
}

func (fsm *storeFSM) applyCreateUserCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_CreateUserCommand_Command)
	v := ext.(*internal.CreateUserCommand)

	// Copy data and update.
	other := fsm.data.Clone()
	if err := other.CreateUser(v.GetName(), v.GetHash(), v.GetAdmin()); err != nil {
		return err
	}
	fsm.data = other

	return nil
}

func (fsm *storeFSM) applyDropUserCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_DropUserCommand_Command)
	v := ext.(*internal.DropUserCommand)

	// Copy data and update.
	other := fsm.data.Clone()
	if err := other.DropUser(v.GetName()); err != nil {
		return err
	}
	fsm.data = other
	delete(fsm.authCache, v.GetName())
	return nil
}

func (fsm *storeFSM) applyUpdateUserCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_UpdateUserCommand_Command)
	v := ext.(*internal.UpdateUserCommand)

	// Copy data and update.
	other := fsm.data.Clone()
	if err := other.UpdateUser(v.GetName(), v.GetHash()); err != nil {
		return err
	}
	fsm.data = other
	delete(fsm.authCache, v.GetName())
	return nil
}

func (fsm *storeFSM) applySetPrivilegeCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_SetPrivilegeCommand_Command)
	v := ext.(*internal.SetPrivilegeCommand)

	// Copy data and update.
	other := fsm.data.Clone()
	if err := other.SetPrivilege(v.GetUsername(), v.GetDatabase(), influxql.Privilege(v.GetPrivilege())); err != nil {
		return err
	}
	fsm.data = other
	return nil
}

func (fsm *storeFSM) applySetAdminPrivilegeCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_SetAdminPrivilegeCommand_Command)
	v := ext.(*internal.SetAdminPrivilegeCommand)

	// Copy data and update.
	other := fsm.data.Clone()
	if err := other.SetAdminPrivilege(v.GetUsername(), v.GetAdmin()); err != nil {
		return err
	}
	fsm.data = other
	return nil
}

func (fsm *storeFSM) applySetDataCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_SetDataCommand_Command)
	v := ext.(*internal.SetDataCommand)

	// Overwrite data.
	fsm.data = &Data{}
	fsm.data.unmarshal(v.GetData())

	return nil
}

func (fsm *storeFSM) Snapshot() (raft.FSMSnapshot, error) {
	s := (*Store)(fsm)
	s.mu.Lock()
	defer s.mu.Unlock()

	return &storeFSMSnapshot{Data: (*Store)(fsm).data}, nil
}

func (fsm *storeFSM) Restore(r io.ReadCloser) error {
	// Read all bytes.
	b, err := ioutil.ReadAll(r)
	if err != nil {
		return err
	}

	// Decode metadata.
	data := &Data{}
	if err := data.UnmarshalBinary(b); err != nil {
		return err
	}

	// Set metadata on store.
	// NOTE: No lock because Hashicorp Raft doesn't call Restore concurrently
	// with any other function.
	fsm.data = data

	return nil
}

type storeFSMSnapshot struct {
	Data *Data
}

func (s *storeFSMSnapshot) Persist(sink raft.SnapshotSink) error {
	err := func() error {
		// Encode data.
		p, err := s.Data.MarshalBinary()
		if err != nil {
			return err
		}

		// Write data to sink.
		if _, err := sink.Write(p); err != nil {
			return err
		}

		// Close the sink.
		if err := sink.Close(); err != nil {
			return err
		}

		return nil
	}()

	if err != nil {
		sink.Cancel()
		return err
	}

	return nil
}

// Release is invoked when we are finished with the snapshot
func (s *storeFSMSnapshot) Release() {}

// raftLayer wraps the connection so it can be re-used for forwarding.
type raftLayer struct {
	ln     net.Listener
	addr   net.Addr
	conn   chan net.Conn
	closed chan struct{}
}

// newRaftLayer returns a new instance of raftLayer.
func newRaftLayer(ln net.Listener, addr net.Addr) *raftLayer {
	return &raftLayer{
		ln:     ln,
		addr:   addr,
		conn:   make(chan net.Conn),
		closed: make(chan struct{}),
	}
}

// Addr returns the local address for the layer.
func (l *raftLayer) Addr() net.Addr { return l.addr }

// Dial creates a new network connection.
func (l *raftLayer) Dial(addr string, timeout time.Duration) (net.Conn, error) {
	conn, err := net.DialTimeout("tcp", addr, timeout)
	if err != nil {
		return nil, err
	}

	// Write a marker byte for raft messages.
	_, err = conn.Write([]byte{MuxRaftHeader})
	if err != nil {
		conn.Close()
		return nil, err
	}
	return conn, err
}

// Accept waits for the next connection.
func (l *raftLayer) Accept() (net.Conn, error) { return l.ln.Accept() }

// Close closes the layer.
func (l *raftLayer) Close() error { return l.ln.Close() }

// RetentionPolicyUpdate represents retention policy fields to be updated.
type RetentionPolicyUpdate struct {
	Name     *string
	Duration *time.Duration
	ReplicaN *int
}

func (rpu *RetentionPolicyUpdate) SetName(v string)            { rpu.Name = &v }
func (rpu *RetentionPolicyUpdate) SetDuration(v time.Duration) { rpu.Duration = &v }
func (rpu *RetentionPolicyUpdate) SetReplicaN(v int)           { rpu.ReplicaN = &v }

// assert will panic with a given formatted message if the given condition is false.
func assert(condition bool, msg string, v ...interface{}) {
	if !condition {
		panic(fmt.Sprintf("assert failed: "+msg, v...))
	}
}
