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
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/gogo/protobuf/proto"
	"github.com/hashicorp/raft"
	"github.com/influxdb/influxdb"
	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/meta/internal"
	"github.com/influxdb/influxdb/models"
	"golang.org/x/crypto/bcrypt"
)

// tcp.Mux header bytes.
const (
	MuxRaftHeader = 0
	MuxExecHeader = 1
	MuxRPCHeader  = 5

	// SaltBytes is the number of bytes used for salts
	SaltBytes = 32

	DefaultSyncNodeDelay = time.Second
)

// ExecMagic is the first 4 bytes sent to a remote exec connection to verify
// that it is coming from a remote exec client connection.
const ExecMagic = "EXEC"

// Retention policy settings.
const (
	AutoCreateRetentionPolicyName   = "default"
	AutoCreateRetentionPolicyPeriod = 0

	// MaxAutoCreatedRetentionPolicyReplicaN is the maximum replication factor that will
	// be set for auto-created retention policies.
	MaxAutoCreatedRetentionPolicyReplicaN = 3
)

// Raft configuration.
const (
	raftLogCacheSize      = 512
	raftSnapshotsRetained = 2
	raftTransportMaxPool  = 3
	raftTransportTimeout  = 10 * time.Second
	MaxRaftNodes          = 3
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

	rpc *rpc

	// The address used by other nodes to reach this node.
	RemoteAddr net.Addr

	raftState raftState

	ready   chan struct{}
	err     chan error
	closing chan struct{}
	wg      sync.WaitGroup
	changed chan struct{}

	// clusterTracingEnabled controls whether low-level cluster communication is logged.
	// Useful for troubleshooting
	clusterTracingEnabled bool

	retentionAutoCreate bool

	// The listeners to accept raft and remote exec connections from.
	RaftListener net.Listener
	ExecListener net.Listener

	// The listener for higher-level, cluster operations
	RPCListener net.Listener

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

	// raftPromotionEnabled determines if non-raft nodes should be automatically
	// promoted to a raft node to self-heal a raft cluster
	raftPromotionEnabled bool

	Logger *log.Logger
}

type authUser struct {
	salt []byte
	hash []byte
}

// NewStore returns a new instance of Store.
func NewStore(c *Config) *Store {
	s := &Store{
		path:  c.Dir,
		peers: c.Peers,
		data:  &Data{},

		ready:   make(chan struct{}),
		err:     make(chan error),
		closing: make(chan struct{}),
		changed: make(chan struct{}),

		clusterTracingEnabled: c.ClusterTracing,
		retentionAutoCreate:   c.RetentionAutoCreate,
		raftPromotionEnabled:  c.RaftPromotionEnabled,

		HeartbeatTimeout:   time.Duration(c.HeartbeatTimeout),
		ElectionTimeout:    time.Duration(c.ElectionTimeout),
		LeaderLeaseTimeout: time.Duration(c.LeaderLeaseTimeout),
		CommitTimeout:      time.Duration(c.CommitTimeout),
		authCache:          make(map[string]authUser, 0),
		hashPassword: func(password string) ([]byte, error) {
			return bcrypt.GenerateFromPassword([]byte(password), BcryptCost)
		},
	}

	if c.LoggingEnabled {
		s.Logger = log.New(os.Stderr, "[metastore] ", log.LstdFlags)
	} else {
		s.Logger = log.New(ioutil.Discard, "", 0)
	}

	s.raftState = &localRaft{store: s}
	s.rpc = &rpc{
		store:          s,
		tracingEnabled: c.ClusterTracing,
		logger:         s.Logger,
	}
	return s
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
	if len(s.peers) > MaxRaftNodes {
		return ErrTooManyPeers
	}

	// Verify listeners are set.
	if s.RaftListener == nil {
		panic("Store.RaftListener not set")
	} else if s.ExecListener == nil {
		panic("Store.ExecListener not set")
	} else if s.RPCListener == nil {
		panic("Store.RPCListener not set")
	}

	s.Logger.Printf("Using data dir: %v", s.Path())

	if err := func() error {
		s.mu.Lock()
		defer s.mu.Unlock()

		// Check if store has already been opened.
		if s.opened {
			return ErrStoreOpen
		}
		s.opened = true

		// load our raft state
		if err := s.loadState(); err != nil {
			return err
		}

		// Create the root directory if it doesn't already exist.
		if err := s.createRootDir(); err != nil {
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
		return err
	}

	// Begin serving listener.
	s.wg.Add(1)
	go s.serveExecListener()

	s.wg.Add(1)
	go s.serveRPCListener()

	// Join an existing cluster if we needed
	if err := s.joinCluster(); err != nil {
		return fmt.Errorf("join: %v", err)
	}

	// If the ID doesn't exist then create a new node.
	if s.id == 0 {
		go s.init()
	} else {
		go s.syncNodeInfo()
		close(s.ready)
	}

	// Wait for a leader to be elected so we know the raft log is loaded
	// and up to date
	<-s.ready
	if err := s.WaitForLeader(0); err != nil {
		return err
	}

	if s.raftPromotionEnabled {
		s.wg.Add(1)
		s.Logger.Printf("spun up monitoring for %d", s.NodeID())
		go s.monitorPeerHealth()
	}

	return nil
}

// syncNodeInfo continuously tries to update the current nodes hostname
// in the meta store.  It will retry until successful.
func (s *Store) syncNodeInfo() error {
	<-s.ready

	for {
		if err := func() error {
			if err := s.WaitForLeader(0); err != nil {
				return err
			}

			ni, err := s.Node(s.id)
			if err != nil {
				return err
			}

			if ni == nil {
				return ErrNodeNotFound
			}

			if ni.Host == s.RemoteAddr.String() {
				s.Logger.Printf("Updated node id=%d hostname=%v", s.id, s.RemoteAddr.String())
				return nil
			}

			_, err = s.UpdateNode(s.id, s.RemoteAddr.String())
			if err != nil {
				return err
			}
			return nil
		}(); err != nil {
			// If we get an error, the cluster has not stabilized so just try again
			time.Sleep(DefaultSyncNodeDelay)
			continue
		}
		return nil
	}
}

// loadState sets the appropriate raftState from our persistent storage
func (s *Store) loadState() error {
	peers, err := readPeersJSON(filepath.Join(s.path, "peers.json"))
	if err != nil {
		return err
	}

	// If we have existing peers, use those.  This will override what's in the
	// config.
	if len(peers) > 0 {
		s.peers = peers
	}

	// if no peers on disk, we need to start raft in order to initialize a new
	// cluster or join an existing one.
	if len(peers) == 0 {
		s.raftState = &localRaft{store: s}
		// if we have a raft database, (maybe restored), we should start raft locally
	} else if _, err := os.Stat(filepath.Join(s.path, "raft.db")); err == nil {
		s.raftState = &localRaft{store: s}
		// otherwise, we should use remote raft
	} else {
		s.raftState = &remoteRaft{store: s}
	}
	return nil
}

func (s *Store) joinCluster() error {

	// No join options, so nothing to do
	if len(s.peers) == 0 {
		return nil
	}

	// We already have a node ID so were already part of a cluster,
	// don't join again so we can use our existing state.
	if s.id != 0 {
		s.Logger.Printf("Skipping cluster join: already member of cluster: nodeId=%v raftEnabled=%v peers=%v",
			s.id, raft.PeerContained(s.peers, s.RemoteAddr.String()), s.peers)
		return nil
	}

	s.Logger.Printf("Joining cluster at: %v", s.peers)
	for {
		for _, join := range s.peers {
			res, err := s.rpc.join(s.RemoteAddr.String(), join)
			if err != nil {
				s.Logger.Printf("Join node %v failed: %v: retrying...", join, err)
				continue
			}

			s.Logger.Printf("Joined remote node %v", join)
			s.Logger.Printf("nodeId=%v raftEnabled=%v peers=%v", res.NodeID, res.RaftEnabled, res.RaftNodes)

			s.peers = res.RaftNodes
			s.id = res.NodeID

			if err := s.writeNodeID(res.NodeID); err != nil {
				s.Logger.Printf("Write node id failed: %v", err)
				break
			}

			if !res.RaftEnabled {
				// Shutdown our local raft and transition to a remote raft state
				if err := s.enableRemoteRaft(); err != nil {
					s.Logger.Printf("Enable remote raft failed: %v", err)
					break
				}
			}
			return nil
		}
		time.Sleep(time.Second)
	}
}

func (s *Store) enableLocalRaft() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.raftState.(*localRaft); ok {
		return nil
	}
	s.Logger.Printf("Switching to local raft")

	lr := &localRaft{store: s}
	return s.changeState(lr)
}

func (s *Store) enableRemoteRaft() error {
	if _, ok := s.raftState.(*remoteRaft); ok {
		return nil
	}

	s.Logger.Printf("Switching to remote raft")
	rr := &remoteRaft{store: s}
	return s.changeState(rr)
}

func (s *Store) changeState(state raftState) error {
	if s.raftState != nil {
		if err := s.raftState.close(); err != nil {
			return err
		}

		// Clear out any persistent state
		if err := s.raftState.remove(); err != nil {
			return err
		}
	}
	s.raftState = state

	if err := s.raftState.open(); err != nil {
		return err
	}

	return nil
}

// monitorPeerHealth periodically checks if we have a node that can be promoted to a
// raft peer to fill any missing slots.
// This function runs in a separate goroutine.
func (s *Store) monitorPeerHealth() {
	defer s.wg.Done()

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		// Wait for next tick or timeout.
		select {
		case <-ticker.C:
		case <-s.closing:
			return
		}
		if err := s.promoteNodeToPeer(); err != nil {
			s.Logger.Printf("error promoting node to raft peer: %s", err)
		}
	}
}

func (s *Store) promoteNodeToPeer() error {
	// Only do this if you are the leader

	if !s.IsLeader() {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	peers, err := s.raftState.peers()
	if err != nil {
		return err
	}

	nodes := s.data.Nodes
	var nonraft NodeInfos
	for _, n := range nodes {
		if contains(peers, n.Host) {
			continue
		}
		nonraft = append(nonraft, n)
	}

	// Check to see if any action is required or possible
	if len(peers) >= 3 || len(nonraft) == 0 {
		return nil
	}

	// Sort the nodes
	sort.Sort(nonraft)

	// Get the lowest node for a deterministic outcome
	n := nonraft[0]
	// Set peers on the leader now to the new peers
	if err := s.AddPeer(n.Host); err != nil {
		return fmt.Errorf("unable to add raft peer %s on leader: %s", n.Host, err)
	}

	// add node to peers list
	peers = append(peers, n.Host)
	if err := s.rpc.enableRaft(n.Host, peers); err != nil {
		return fmt.Errorf("error notifying raft peer: %s", err)
	}
	s.Logger.Printf("promoted nodeID %d, host %s to raft peer", n.ID, n.Host)

	return nil
}

// openRaft initializes the raft store.
func (s *Store) openRaft() error {
	return s.raftState.open()
}

// initialize attempts to bootstrap the raft store if there are no committed entries.
func (s *Store) initialize() error {
	return s.raftState.initialize()
}

// Close closes the store and shuts down the node in the cluster.
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.close()
}

// WaitForDataChanged will block the current goroutine until the metastore index has
// be updated.
func (s *Store) WaitForDataChanged() error {
	s.mu.RLock()
	changed := s.changed
	s.mu.RUnlock()

	for {
		select {
		case <-s.closing:
			return errors.New("closing")
		case <-changed:
			return nil
		}
	}
}

func (s *Store) close() error {
	// Check if store has already been closed.
	if !s.opened {
		return ErrStoreClosed
	}
	s.opened = false

	// Close our exec listener
	if err := s.ExecListener.Close(); err != nil {
		s.Logger.Printf("error closing ExecListener %s", err)
	}

	// Close our RPC listener
	if err := s.RPCListener.Close(); err != nil {
		s.Logger.Printf("error closing ExecListener %s", err)
	}

	if s.raftState != nil {
		s.raftState.close()
	}

	// Because a go routine could of already fired in the time we acquired the lock
	// it could then try to acquire another lock, and will deadlock.
	// For that reason, we will release our lock and signal the close so that
	// all go routines can exit cleanly and fullfill their contract to the wait group.
	s.mu.Unlock()
	// Notify goroutines of close.
	close(s.closing)
	s.wg.Wait()

	// Now that all go routines are cleaned up, w lock to do final clean up and exit
	s.mu.Lock()

	s.raftState = nil

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
	ni, err := s.CreateNode(s.RemoteAddr.String())
	if err != nil {
		return fmt.Errorf("create node: %s", err)
	}

	// Write node id to file.
	if err := s.writeNodeID(ni.ID); err != nil {
		return fmt.Errorf("write file: %s", err)
	}

	// Set ID locally.
	s.mu.Lock()
	s.id = ni.ID
	s.mu.Unlock()

	s.Logger.Printf("Created local node: id=%d, host=%s", s.id, s.RemoteAddr)

	return nil
}

func (s *Store) createRootDir() error {
	return os.MkdirAll(s.path, 0777)
}

func (s *Store) writeNodeID(id uint64) error {
	if err := s.createRootDir(); err != nil {
		return err
	}
	return ioutil.WriteFile(s.IDPath(), []byte(strconv.FormatUint(id, 10)), 0666)
}

// Snapshot saves a snapshot of the current state.
func (s *Store) Snapshot() error {
	return s.raftState.snapshot()
}

// WaitForLeader sleeps until a leader is found or a timeout occurs.
// timeout == 0 means to wait forever.
func (s *Store) WaitForLeader(timeout time.Duration) error {
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
			if s.Leader() != "" {
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
	return s.raftState.isLeader()
}

// IsLocal returns true if the store is currently participating in local raft.
func (s *Store) IsLocal() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.raftState.isLocal()
}

// Leader returns what the store thinks is the current leader. An empty
// string indicates no leader exists.
func (s *Store) Leader() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.raftState == nil {
		return ""
	}
	return s.raftState.leader()
}

// SetPeers sets a list of peers in the cluster.
func (s *Store) SetPeers(addrs []string) error {
	return s.raftState.setPeers(addrs)
}

// AddPeer adds addr to the list of peers in the cluster.
func (s *Store) AddPeer(addr string) error {
	return s.raftState.addPeer(addr)
}

// Peers returns the list of peers in the cluster.
func (s *Store) Peers() ([]string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.raftState.peers()
}

// serveExecListener processes remote exec connections.
// This function runs in a separate goroutine.
func (s *Store) serveExecListener() {
	defer s.wg.Done()

	for {
		// Accept next TCP connection.
		var err error
		conn, err := s.ExecListener.Accept()
		if opErr, ok := err.(*net.OpError); ok && opErr.Temporary() {
			s.Logger.Printf("exec listener temporary accept error: %s", err)
			continue
		} else if err != nil {
			s.Logger.Printf("exec listener accept error and closed: %s", err)
			return
		}

		// Handle connection in a separate goroutine.
		s.wg.Add(1)
		go s.handleExecConn(conn)

		select {
		case <-s.closing:
			return
		default:
		}
	}
}

// handleExecConn reads a command from the connection and executes it.
func (s *Store) handleExecConn(conn net.Conn) {
	defer s.wg.Done()

	// Nodes not part of the raft cluster may initiate remote exec commands
	// but may not know who the current leader of the cluster.  If we are not
	// the leader, proxy the request to the current leader.
	if !s.IsLeader() {

		if s.Leader() == s.RemoteAddr.String() {
			s.Logger.Printf("No leader")
			return
		}

		leaderConn, err := net.DialTimeout("tcp", s.Leader(), 10*time.Second)
		if err != nil {
			s.Logger.Printf("Dial leader: %v", err)
			return
		}
		defer leaderConn.Close()
		leaderConn.Write([]byte{MuxExecHeader})

		if err := proxy(leaderConn.(*net.TCPConn), conn.(*net.TCPConn)); err != nil {
			s.Logger.Printf("Leader proxy error: %v", err)
		}
		conn.Close()
		return
	}

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
			return err
		}
		return nil
	}()

	// Build response message.
	var resp internal.Response
	resp.OK = proto.Bool(err == nil)
	resp.Index = proto.Uint64(s.raftState.lastIndex())
	if err != nil {
		resp.Error = proto.String(err.Error())
	}

	// Encode response back to connection.
	if b, err := proto.Marshal(&resp); err != nil {
		panic(err)
	} else if err = binary.Write(conn, binary.BigEndian, uint64(len(b))); err != nil {
		s.Logger.Printf("Unable to write exec response size: %s", err)
	} else if _, err = conn.Write(b); err != nil {
		s.Logger.Printf("Unable to write exec response: %s", err)
	}
	conn.Close()
}

// serveRPCListener processes remote exec connections.
// This function runs in a separate goroutine.
func (s *Store) serveRPCListener() {
	defer s.wg.Done()

	for {
		// Accept next TCP connection.
		conn, err := s.RPCListener.Accept()
		if opErr, ok := err.(*net.OpError); ok && opErr.Temporary() {
			s.Logger.Printf("RPC listener temporary accept error: %s", err)
			continue
		} else if err != nil {
			s.Logger.Printf("RPC listener accept error and closed: %s", err)
			return
		}

		// Handle connection in a separate goroutine.
		s.wg.Add(1)
		go func() {
			defer s.wg.Done()
			s.rpc.handleRPCConn(conn)
		}()

		select {
		case <-s.closing:
			return
		default:
		}
	}
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

// UpdateNode updates an existing node in the store.
func (s *Store) UpdateNode(id uint64, host string) (*NodeInfo, error) {
	if err := s.exec(internal.Command_UpdateNodeCommand, internal.E_UpdateNodeCommand_Command,
		&internal.UpdateNodeCommand{
			ID:   proto.Uint64(id),
			Host: proto.String(host),
		},
	); err != nil {
		return nil, err
	}
	return s.NodeByHost(host)
}

// DeleteNode removes a node from the metastore by id.
func (s *Store) DeleteNode(id uint64, force bool) error {
	ni := s.data.Node(id)
	if ni == nil {
		return ErrNodeNotFound
	}

	err := s.exec(internal.Command_DeleteNodeCommand, internal.E_DeleteNodeCommand_Command,
		&internal.DeleteNodeCommand{
			ID:    proto.Uint64(id),
			Force: proto.Bool(force),
		},
	)
	if err != nil {
		return err
	}

	// Need to send a second message to remove the peer
	return s.exec(internal.Command_RemovePeerCommand, internal.E_RemovePeerCommand_Command,
		&internal.RemovePeerCommand{
			ID:   proto.Uint64(id),
			Addr: proto.String(ni.Host),
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
	s.Logger.Printf("database '%s' created", name)

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

		if nodeN > MaxAutoCreatedRetentionPolicyReplicaN {
			nodeN = MaxAutoCreatedRetentionPolicyReplicaN
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

// CreateDatabaseWithRetentionPolicy creates a new database with an explicit retention policy in the store.
func (s *Store) CreateDatabaseWithRetentionPolicy(name string, rpi *RetentionPolicyInfo) (*DatabaseInfo, error) {
	if err := s.exec(internal.Command_CreateDatabaseCommand, internal.E_CreateDatabaseCommand_Command,
		&internal.CreateDatabaseCommand{
			Name: proto.String(name),
		},
	); err != nil {
		return nil, err
	}
	s.Logger.Printf("database '%s' created", name)

	if _, err := s.CreateRetentionPolicy(name, rpi); err != nil {
		return nil, err
	}

	// Set it as the default retention policy.
	if err := s.SetDefaultRetentionPolicy(name, rpi.Name); err != nil {
		return nil, err
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
	di, err := s.CreateDatabase(name)
	if err == ErrDatabaseExists {
		return s.Database(name)
	}
	return di, err
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
			return influxdb.ErrDatabaseNotFound(database)
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
			return influxdb.ErrDatabaseNotFound(database)
		}
		a = di.RetentionPolicies
		return nil
	})
	return
}

// CreateRetentionPolicy creates a new retention policy for a database.
func (s *Store) CreateRetentionPolicy(database string, rpi *RetentionPolicyInfo) (*RetentionPolicyInfo, error) {
	if rpi.Duration < MinRetentionPolicyDuration && rpi.Duration != 0 {
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

	s.Logger.Printf("retention policy '%s' for database '%s' created", rpi.Name, database)
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
	other, err := s.CreateRetentionPolicy(database, rpi)
	if err == ErrRetentionPolicyExists {
		return s.RetentionPolicy(database, rpi.Name)
	}
	return other, err
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

// CreateShardGroup creates a new shard group in a retention policy for a given time.
func (s *Store) CreateShardGroup(database, policy string, timestamp time.Time) (*ShardGroupInfo, error) {
	// Check the time is valid since we are about to encode it as an int64
	if err := models.CheckTime(timestamp); err != nil {
		return nil, err
	}
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
	sgi, err := s.CreateShardGroup(database, policy, timestamp)
	if err == ErrShardGroupExists {
		sgi, err = s.ShardGroupByTimestamp(database, policy, timestamp)
	}
	// Check that we are returning either an error or a valid shard group.
	if sgi == nil && err == nil {
		return nil, errors.New("failed to create a new shard group, error unknown.")
	}
	return sgi, err
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

// ShardOwner looks up for a specific shard and return the shard group information
// related with the shard.
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

// CreateSubscription creates a new subscription on the store.
func (s *Store) CreateSubscription(database, rp, name, mode string, destinations []string) error {
	return s.exec(internal.Command_CreateSubscriptionCommand, internal.E_CreateSubscriptionCommand_Command,
		&internal.CreateSubscriptionCommand{
			Database:        proto.String(database),
			RetentionPolicy: proto.String(rp),
			Name:            proto.String(name),
			Mode:            proto.String(mode),
			Destinations:    destinations,
		},
	)
}

// DropSubscription removes a subscription from the store.
func (s *Store) DropSubscription(database, rp, name string) error {
	return s.exec(internal.Command_DropSubscriptionCommand, internal.E_DropSubscriptionCommand_Command,
		&internal.DropSubscriptionCommand{
			Database:        proto.String(database),
			RetentionPolicy: proto.String(rp),
			Name:            proto.String(name),
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
			}
			return ErrAuthenticate
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

// PrecreateShardGroups creates shard groups whose endtime is before the 'to' time passed in, but
// is yet to expire before 'from'. This is to avoid the need for these shards to be created when data
// for the corresponding time range arrives. Shard creation involves Raft consensus, and precreation
// avoids taking the hit at write-time.
func (s *Store) PrecreateShardGroups(from, to time.Time) error {
	s.read(func(data *Data) error {
		for _, di := range data.Databases {
			for _, rp := range di.RetentionPolicies {
				if len(rp.ShardGroups) == 0 {
					// No data was ever written to this group, or all groups have been deleted.
					continue
				}
				g := rp.ShardGroups[len(rp.ShardGroups)-1] // Get the last group in time.
				if !g.Deleted() && g.EndTime.Before(to) && g.EndTime.After(from) {
					// Group is not deleted, will end before the future time, but is still yet to expire.
					// This last check is important, so the system doesn't create shards groups wholly
					// in the past.

					// Create successive shard group.
					nextShardGroupTime := g.EndTime.Add(1 * time.Nanosecond)
					if newGroup, err := s.CreateShardGroupIfNotExists(di.Name, rp.Name, nextShardGroupTime); err != nil {
						s.Logger.Printf("failed to precreate successive shard group for group %d: %s",
							g.ID, err.Error())
					} else {
						s.Logger.Printf("new shard group %d successfully precreated for database %s, retention policy %s",
							newGroup.ID, di.Name, rp.Name)
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
	return s.raftState.invalidate()
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
	if s.raftState.isLeader() {
		return s.apply(b)
	}
	return s.remoteExec(b)
}

// apply applies a serialized command to the raft log.
func (s *Store) apply(b []byte) error {
	return s.raftState.apply(b)
}

// remoteExec sends an encoded command to the remote leader.
func (s *Store) remoteExec(b []byte) error {
	// Retrieve the current known leader.
	leader := s.raftState.leader()
	if leader == "" {
		return errors.New("no leader detected during remoteExec")
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
		return lookupError(fmt.Errorf(resp.GetError()))
	}

	// Wait for local FSM to sync to index.
	if err := s.sync(resp.GetIndex(), 5*time.Second); err != nil {
		return fmt.Errorf("sync: %s", err)
	}

	return nil
}

// sync polls the state machine until it reaches a given index.
func (s *Store) sync(index uint64, timeout time.Duration) error {
	return s.raftState.sync(index, timeout)
}

func (s *Store) cachedData() *Data {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.data.Clone()
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

// notifiyChanged will close a changed channel which brooadcasts to all waiting
// goroutines that the meta store has been updated.  Callers are responsible for locking
// the meta store before calling this.
func (s *Store) notifyChanged() {
	close(s.changed)
	s.changed = make(chan struct{})
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
		case internal.Command_RemovePeerCommand:
			return fsm.applyRemovePeerCommand(&cmd)
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
		case internal.Command_CreateSubscriptionCommand:
			return fsm.applyCreateSubscriptionCommand(&cmd)
		case internal.Command_DropSubscriptionCommand:
			return fsm.applyDropSubscriptionCommand(&cmd)
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
		case internal.Command_UpdateNodeCommand:
			return fsm.applyUpdateNodeCommand(&cmd)
		default:
			panic(fmt.Errorf("cannot apply command: %x", l.Data))
		}
	}()

	// Copy term and index to new metadata.
	fsm.data.Term = l.Term
	fsm.data.Index = l.Index
	s.notifyChanged()

	return err
}

func (fsm *storeFSM) applyRemovePeerCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_RemovePeerCommand_Command)
	v := ext.(*internal.RemovePeerCommand)

	id := v.GetID()
	addr := v.GetAddr()

	// Only do this if you are the leader
	if fsm.raftState.isLeader() {
		//Remove that node from the peer
		fsm.Logger.Printf("removing peer for node id %d, %s", id, addr)
		if err := fsm.raftState.removePeer(addr); err != nil {
			fsm.Logger.Printf("error removing peer: %s", err)
		}
	}

	// If this is the node being shutdown, close raft
	if fsm.id == id {
		fsm.Logger.Printf("shutting down raft for %s", addr)
		if err := fsm.raftState.close(); err != nil {
			fsm.Logger.Printf("failed to shut down raft: %s", err)
		}
	}

	return nil
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

func (fsm *storeFSM) applyUpdateNodeCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_UpdateNodeCommand_Command)
	v := ext.(*internal.UpdateNodeCommand)

	// Copy data and update.
	other := fsm.data.Clone()
	ni := other.Node(v.GetID())
	if ni == nil {
		return ErrNodeNotFound
	}

	ni.Host = v.GetHost()

	fsm.data = other
	return nil
}

func (fsm *storeFSM) applyDeleteNodeCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_DeleteNodeCommand_Command)
	v := ext.(*internal.DeleteNodeCommand)

	// Copy data and update.
	other := fsm.data.Clone()
	if err := other.DeleteNode(v.GetID(), v.GetForce()); err != nil {
		return err
	}
	fsm.data = other

	id := v.GetID()
	fsm.Logger.Printf("node '%d' removed", id)

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

func (fsm *storeFSM) applyCreateSubscriptionCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_CreateSubscriptionCommand_Command)
	v := ext.(*internal.CreateSubscriptionCommand)

	// Copy data and update.
	other := fsm.data.Clone()
	if err := other.CreateSubscription(v.GetDatabase(), v.GetRetentionPolicy(), v.GetName(), v.GetMode(), v.GetDestinations()); err != nil {
		return err
	}
	fsm.data = other

	return nil
}

func (fsm *storeFSM) applyDropSubscriptionCommand(cmd *internal.Command) interface{} {
	ext, _ := proto.GetExtension(cmd, internal.E_DropSubscriptionCommand_Command)
	v := ext.(*internal.DropSubscriptionCommand)

	// Copy data and update.
	other := fsm.data.Clone()
	if err := other.DropSubscription(v.GetDatabase(), v.GetRetentionPolicy(), v.GetName()); err != nil {
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

// SetName sets the RetentionPolicyUpdate.Name
func (rpu *RetentionPolicyUpdate) SetName(v string) { rpu.Name = &v }

// SetDuration sets the RetentionPolicyUpdate.Duration
func (rpu *RetentionPolicyUpdate) SetDuration(v time.Duration) { rpu.Duration = &v }

// SetReplicaN sets the RetentionPolicyUpdate.ReplicaN
func (rpu *RetentionPolicyUpdate) SetReplicaN(v int) { rpu.ReplicaN = &v }

// assert will panic with a given formatted message if the given condition is false.
func assert(condition bool, msg string, v ...interface{}) {
	if !condition {
		panic(fmt.Sprintf("assert failed: "+msg, v...))
	}
}
