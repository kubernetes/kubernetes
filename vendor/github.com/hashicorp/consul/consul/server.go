package consul

import (
	"crypto/tls"
	"errors"
	"fmt"
	"log"
	"net"
	"net/rpc"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"sync"
	"time"

	"github.com/hashicorp/consul/acl"
	"github.com/hashicorp/consul/consul/state"
	"github.com/hashicorp/consul/tlsutil"
	"github.com/hashicorp/raft"
	"github.com/hashicorp/raft-boltdb"
	"github.com/hashicorp/serf/coordinate"
	"github.com/hashicorp/serf/serf"
)

// These are the protocol versions that Consul can _understand_. These are
// Consul-level protocol versions, that are used to configure the Serf
// protocol versions.
const (
	ProtocolVersionMin uint8 = 1

	// Version 3 added support for network coordinates but we kept the
	// default protocol version at 2 to ease the transition to this new
	// feature. A Consul agent speaking version 2 of the protocol will
	// attempt to send its coordinates to a server who understands version
	// 3 or greater.
	ProtocolVersion2Compatible = 2

	ProtocolVersionMax = 3
)

const (
	serfLANSnapshot   = "serf/local.snapshot"
	serfWANSnapshot   = "serf/remote.snapshot"
	raftState         = "raft/"
	snapshotsRetained = 2

	// serverRPCCache controls how long we keep an idle connection
	// open to a server
	serverRPCCache = 2 * time.Minute

	// serverMaxStreams controls how many idle streams we keep
	// open to a server
	serverMaxStreams = 64

	// raftLogCacheSize is the maximum number of logs to cache in-memory.
	// This is used to reduce disk I/O for the recently committed entries.
	raftLogCacheSize = 512

	// raftRemoveGracePeriod is how long we wait to allow a RemovePeer
	// to replicate to gracefully leave the cluster.
	raftRemoveGracePeriod = 5 * time.Second
)

// Server is Consul server which manages the service discovery,
// health checking, DC forwarding, Raft, and multiple Serf pools.
type Server struct {
	// aclAuthCache is the authoritative ACL cache
	aclAuthCache *acl.Cache

	// aclCache is the non-authoritative ACL cache.
	aclCache *aclCache

	// Consul configuration
	config *Config

	// Connection pool to other consul servers
	connPool *ConnPool

	// Endpoints holds our RPC endpoints
	endpoints endpoints

	// eventChLAN is used to receive events from the
	// serf cluster in the datacenter
	eventChLAN chan serf.Event

	// eventChWAN is used to receive events from the
	// serf cluster that spans datacenters
	eventChWAN chan serf.Event

	// fsm is the state machine used with Raft to provide
	// strong consistency.
	fsm *consulFSM

	// Have we attempted to leave the cluster
	left bool

	// localConsuls is used to track the known consuls
	// in the local datacenter. Used to do leader forwarding.
	localConsuls map[string]*serverParts
	localLock    sync.RWMutex

	// Logger uses the provided LogOutput
	logger *log.Logger

	// The raft instance is used among Consul nodes within the
	// DC to protect operations that require strong consistency
	raft          *raft.Raft
	raftLayer     *RaftLayer
	raftPeers     raft.PeerStore
	raftStore     *raftboltdb.BoltStore
	raftTransport *raft.NetworkTransport
	raftInmem     *raft.InmemStore

	// reconcileCh is used to pass events from the serf handler
	// into the leader manager, so that the strong state can be
	// updated
	reconcileCh chan serf.Member

	// remoteConsuls is used to track the known consuls in
	// remote datacenters. Used to do DC forwarding.
	remoteConsuls map[string][]*serverParts
	remoteLock    sync.RWMutex

	// rpcListener is used to listen for incoming connections
	rpcListener net.Listener
	rpcServer   *rpc.Server

	// rpcTLS is the TLS config for incoming TLS requests
	rpcTLS *tls.Config

	// serfLAN is the Serf cluster maintained inside the DC
	// which contains all the DC nodes
	serfLAN *serf.Serf

	// serfWAN is the Serf cluster maintained between DC's
	// which SHOULD only consist of Consul servers
	serfWAN *serf.Serf

	// sessionTimers track the expiration time of each Session that has
	// a TTL. On expiration, a SessionDestroy event will occur, and
	// destroy the session via standard session destroy processing
	sessionTimers     map[string]*time.Timer
	sessionTimersLock sync.Mutex

	// tombstoneGC is used to track the pending GC invocations
	// for the KV tombstones
	tombstoneGC *state.TombstoneGC

	shutdown     bool
	shutdownCh   chan struct{}
	shutdownLock sync.Mutex
}

// Holds the RPC endpoints
type endpoints struct {
	Catalog       *Catalog
	Health        *Health
	Status        *Status
	KVS           *KVS
	Session       *Session
	Internal      *Internal
	ACL           *ACL
	Coordinate    *Coordinate
	PreparedQuery *PreparedQuery
}

// NewServer is used to construct a new Consul server from the
// configuration, potentially returning an error
func NewServer(config *Config) (*Server, error) {
	// Check the protocol version
	if err := config.CheckVersion(); err != nil {
		return nil, err
	}

	// Check for a data directory!
	if config.DataDir == "" && !config.DevMode {
		return nil, fmt.Errorf("Config must provide a DataDir")
	}

	// Sanity check the ACLs
	if err := config.CheckACL(); err != nil {
		return nil, err
	}

	// Ensure we have a log output
	if config.LogOutput == nil {
		config.LogOutput = os.Stderr
	}

	// Create the tls wrapper for outgoing connections
	tlsConf := config.tlsConfig()
	tlsWrap, err := tlsConf.OutgoingTLSWrapper()
	if err != nil {
		return nil, err
	}

	// Get the incoming tls config
	incomingTLS, err := tlsConf.IncomingTLSConfig()
	if err != nil {
		return nil, err
	}

	// Create a logger
	logger := log.New(config.LogOutput, "", log.LstdFlags)

	// Create the tombstone GC
	gc, err := state.NewTombstoneGC(config.TombstoneTTL, config.TombstoneTTLGranularity)
	if err != nil {
		return nil, err
	}

	// Create server
	s := &Server{
		config:        config,
		connPool:      NewPool(config.LogOutput, serverRPCCache, serverMaxStreams, tlsWrap),
		eventChLAN:    make(chan serf.Event, 256),
		eventChWAN:    make(chan serf.Event, 256),
		localConsuls:  make(map[string]*serverParts),
		logger:        logger,
		reconcileCh:   make(chan serf.Member, 32),
		remoteConsuls: make(map[string][]*serverParts),
		rpcServer:     rpc.NewServer(),
		rpcTLS:        incomingTLS,
		tombstoneGC:   gc,
		shutdownCh:    make(chan struct{}),
	}

	// Initialize the authoritative ACL cache
	s.aclAuthCache, err = acl.NewCache(aclCacheSize, s.aclFault)
	if err != nil {
		s.Shutdown()
		return nil, fmt.Errorf("Failed to create ACL cache: %v", err)
	}

	// Set up the non-authoritative ACL cache
	if s.aclCache, err = newAclCache(config, logger, s.RPC); err != nil {
		s.Shutdown()
		return nil, err
	}

	// Initialize the RPC layer
	if err := s.setupRPC(tlsWrap); err != nil {
		s.Shutdown()
		return nil, fmt.Errorf("Failed to start RPC layer: %v", err)
	}

	// Initialize the Raft server
	if err := s.setupRaft(); err != nil {
		s.Shutdown()
		return nil, fmt.Errorf("Failed to start Raft: %v", err)
	}

	// Initialize the lan Serf
	s.serfLAN, err = s.setupSerf(config.SerfLANConfig,
		s.eventChLAN, serfLANSnapshot, false)
	if err != nil {
		s.Shutdown()
		return nil, fmt.Errorf("Failed to start lan serf: %v", err)
	}
	go s.lanEventHandler()

	// Initialize the wan Serf
	s.serfWAN, err = s.setupSerf(config.SerfWANConfig,
		s.eventChWAN, serfWANSnapshot, true)
	if err != nil {
		s.Shutdown()
		return nil, fmt.Errorf("Failed to start wan serf: %v", err)
	}
	go s.wanEventHandler()

	// Start listening for RPC requests
	go s.listen()

	// Start the metrics handlers
	go s.sessionStats()
	return s, nil
}

// setupSerf is used to setup and initialize a Serf
func (s *Server) setupSerf(conf *serf.Config, ch chan serf.Event, path string, wan bool) (*serf.Serf, error) {
	addr := s.rpcListener.Addr().(*net.TCPAddr)
	conf.Init()
	if wan {
		conf.NodeName = fmt.Sprintf("%s.%s", s.config.NodeName, s.config.Datacenter)
	} else {
		conf.NodeName = s.config.NodeName
	}
	conf.Tags["role"] = "consul"
	conf.Tags["dc"] = s.config.Datacenter
	conf.Tags["vsn"] = fmt.Sprintf("%d", s.config.ProtocolVersion)
	conf.Tags["vsn_min"] = fmt.Sprintf("%d", ProtocolVersionMin)
	conf.Tags["vsn_max"] = fmt.Sprintf("%d", ProtocolVersionMax)
	conf.Tags["build"] = s.config.Build
	conf.Tags["port"] = fmt.Sprintf("%d", addr.Port)
	if s.config.Bootstrap {
		conf.Tags["bootstrap"] = "1"
	}
	if s.config.BootstrapExpect != 0 {
		conf.Tags["expect"] = fmt.Sprintf("%d", s.config.BootstrapExpect)
	}
	conf.MemberlistConfig.LogOutput = s.config.LogOutput
	conf.LogOutput = s.config.LogOutput
	conf.EventCh = ch
	if !s.config.DevMode {
		conf.SnapshotPath = filepath.Join(s.config.DataDir, path)
	}
	conf.ProtocolVersion = protocolVersionMap[s.config.ProtocolVersion]
	conf.RejoinAfterLeave = s.config.RejoinAfterLeave
	if wan {
		conf.Merge = &wanMergeDelegate{}
	} else {
		conf.Merge = &lanMergeDelegate{dc: s.config.Datacenter}
	}

	// Until Consul supports this fully, we disable automatic resolution.
	// When enabled, the Serf gossip may just turn off if we are the minority
	// node which is rather unexpected.
	conf.EnableNameConflictResolution = false
	if err := ensurePath(conf.SnapshotPath, false); err != nil {
		return nil, err
	}

	// Plumb down the enable coordinates flag.
	conf.DisableCoordinates = s.config.DisableCoordinates

	return serf.Create(conf)
}

// setupRaft is used to setup and initialize Raft
func (s *Server) setupRaft() error {
	// If we are in bootstrap mode, enable a single node cluster
	if s.config.Bootstrap || s.config.DevMode {
		s.config.RaftConfig.EnableSingleNode = true
	}

	// Create the FSM
	var err error
	s.fsm, err = NewFSM(s.tombstoneGC, s.config.LogOutput)
	if err != nil {
		return err
	}

	// Create a transport layer
	trans := raft.NewNetworkTransport(s.raftLayer, 3, 10*time.Second, s.config.LogOutput)
	s.raftTransport = trans

	var log raft.LogStore
	var stable raft.StableStore
	var snap raft.SnapshotStore
	var peers raft.PeerStore

	if s.config.DevMode {
		store := raft.NewInmemStore()
		s.raftInmem = store
		stable = store
		log = store
		snap = raft.NewDiscardSnapshotStore()
		peers = &raft.StaticPeers{}
		s.raftPeers = peers
	} else {
		// Create the base raft path
		path := filepath.Join(s.config.DataDir, raftState)
		if err := ensurePath(path, true); err != nil {
			return err
		}

		// Create the backend raft store for logs and stable storage
		store, err := raftboltdb.NewBoltStore(filepath.Join(path, "raft.db"))
		if err != nil {
			return err
		}
		s.raftStore = store
		stable = store

		// Wrap the store in a LogCache to improve performance
		cacheStore, err := raft.NewLogCache(raftLogCacheSize, store)
		if err != nil {
			store.Close()
			return err
		}
		log = cacheStore

		// Create the snapshot store
		snapshots, err := raft.NewFileSnapshotStore(path, snapshotsRetained, s.config.LogOutput)
		if err != nil {
			store.Close()
			return err
		}
		snap = snapshots

		// Setup the peer store
		s.raftPeers = raft.NewJSONPeers(path, trans)
		peers = s.raftPeers
	}

	// Ensure local host is always included if we are in bootstrap mode
	if s.config.Bootstrap {
		peers, err := s.raftPeers.Peers()
		if err != nil {
			if s.raftStore != nil {
				s.raftStore.Close()
			}
			return err
		}
		if !raft.PeerContained(peers, trans.LocalAddr()) {
			s.raftPeers.SetPeers(raft.AddUniquePeer(peers, trans.LocalAddr()))
		}
	}

	// Make sure we set the LogOutput
	s.config.RaftConfig.LogOutput = s.config.LogOutput

	// Setup the Raft store
	s.raft, err = raft.NewRaft(s.config.RaftConfig, s.fsm, log, stable,
		snap, s.raftPeers, trans)
	if err != nil {
		if s.raftStore != nil {
			s.raftStore.Close()
		}
		trans.Close()
		return err
	}

	// Start monitoring leadership
	go s.monitorLeadership()
	return nil
}

// setupRPC is used to setup the RPC listener
func (s *Server) setupRPC(tlsWrap tlsutil.DCWrapper) error {
	// Create endpoints
	s.endpoints.Status = &Status{s}
	s.endpoints.Catalog = &Catalog{s}
	s.endpoints.Health = &Health{s}
	s.endpoints.KVS = &KVS{s}
	s.endpoints.Session = &Session{s}
	s.endpoints.Internal = &Internal{s}
	s.endpoints.ACL = &ACL{s}
	s.endpoints.Coordinate = NewCoordinate(s)
	s.endpoints.PreparedQuery = &PreparedQuery{s}

	// Register the handlers
	s.rpcServer.Register(s.endpoints.Status)
	s.rpcServer.Register(s.endpoints.Catalog)
	s.rpcServer.Register(s.endpoints.Health)
	s.rpcServer.Register(s.endpoints.KVS)
	s.rpcServer.Register(s.endpoints.Session)
	s.rpcServer.Register(s.endpoints.Internal)
	s.rpcServer.Register(s.endpoints.ACL)
	s.rpcServer.Register(s.endpoints.Coordinate)
	s.rpcServer.Register(s.endpoints.PreparedQuery)

	list, err := net.ListenTCP("tcp", s.config.RPCAddr)
	if err != nil {
		return err
	}
	s.rpcListener = list

	var advertise net.Addr
	if s.config.RPCAdvertise != nil {
		advertise = s.config.RPCAdvertise
	} else {
		advertise = s.rpcListener.Addr()
	}

	// Verify that we have a usable advertise address
	addr, ok := advertise.(*net.TCPAddr)
	if !ok {
		list.Close()
		return fmt.Errorf("RPC advertise address is not a TCP Address: %v", addr)
	}
	if addr.IP.IsUnspecified() {
		list.Close()
		return fmt.Errorf("RPC advertise address is not advertisable: %v", addr)
	}

	// Provide a DC specific wrapper. Raft replication is only
	// ever done in the same datacenter, so we can provide it as a constant.
	wrapper := tlsutil.SpecificDC(s.config.Datacenter, tlsWrap)
	s.raftLayer = NewRaftLayer(advertise, wrapper)
	return nil
}

// Shutdown is used to shutdown the server
func (s *Server) Shutdown() error {
	s.logger.Printf("[INFO] consul: shutting down server")
	s.shutdownLock.Lock()
	defer s.shutdownLock.Unlock()

	if s.shutdown {
		return nil
	}

	s.shutdown = true
	close(s.shutdownCh)

	if s.serfLAN != nil {
		s.serfLAN.Shutdown()
	}

	if s.serfWAN != nil {
		s.serfWAN.Shutdown()
	}

	if s.raft != nil {
		s.raftTransport.Close()
		s.raftLayer.Close()
		future := s.raft.Shutdown()
		if err := future.Error(); err != nil {
			s.logger.Printf("[WARN] consul: Error shutting down raft: %s", err)
		}
		if s.raftStore != nil {
			s.raftStore.Close()
		}

		// Clear the peer set on a graceful leave to avoid
		// triggering elections on a rejoin.
		if s.left {
			s.raftPeers.SetPeers(nil)
		}
	}

	if s.rpcListener != nil {
		s.rpcListener.Close()
	}

	// Close the connection pool
	s.connPool.Shutdown()

	return nil
}

// Leave is used to prepare for a graceful shutdown of the server
func (s *Server) Leave() error {
	s.logger.Printf("[INFO] consul: server starting leave")
	s.left = true

	// Check the number of known peers
	numPeers, err := s.numOtherPeers()
	if err != nil {
		s.logger.Printf("[ERR] consul: failed to check raft peers: %v", err)
		return err
	}

	// If we are the current leader, and we have any other peers (cluster has multiple
	// servers), we should do a RemovePeer to safely reduce the quorum size. If we are
	// not the leader, then we should issue our leave intention and wait to be removed
	// for some sane period of time.
	isLeader := s.IsLeader()
	if isLeader && numPeers > 0 {
		future := s.raft.RemovePeer(s.raftTransport.LocalAddr())
		if err := future.Error(); err != nil && err != raft.ErrUnknownPeer {
			s.logger.Printf("[ERR] consul: failed to remove ourself as raft peer: %v", err)
		}
	}

	// Leave the WAN pool
	if s.serfWAN != nil {
		if err := s.serfWAN.Leave(); err != nil {
			s.logger.Printf("[ERR] consul: failed to leave WAN Serf cluster: %v", err)
		}
	}

	// Leave the LAN pool
	if s.serfLAN != nil {
		if err := s.serfLAN.Leave(); err != nil {
			s.logger.Printf("[ERR] consul: failed to leave LAN Serf cluster: %v", err)
		}
	}

	// If we were not leader, wait to be safely removed from the cluster.
	// We must wait to allow the raft replication to take place, otherwise
	// an immediate shutdown could cause a loss of quorum.
	if !isLeader {
		limit := time.Now().Add(raftRemoveGracePeriod)
		for numPeers > 0 && time.Now().Before(limit) {
			// Update the number of peers
			numPeers, err = s.numOtherPeers()
			if err != nil {
				s.logger.Printf("[ERR] consul: failed to check raft peers: %v", err)
				break
			}

			// Avoid the sleep if we are done
			if numPeers == 0 {
				break
			}

			// Sleep a while and check again
			time.Sleep(50 * time.Millisecond)
		}
		if numPeers != 0 {
			s.logger.Printf("[WARN] consul: failed to leave raft peer set gracefully, timeout")
		}
	}

	return nil
}

// numOtherPeers is used to check on the number of known peers
// excluding the local node
func (s *Server) numOtherPeers() (int, error) {
	peers, err := s.raftPeers.Peers()
	if err != nil {
		return 0, err
	}
	otherPeers := raft.ExcludePeer(peers, s.raftTransport.LocalAddr())
	return len(otherPeers), nil
}

// JoinLAN is used to have Consul join the inner-DC pool
// The target address should be another node inside the DC
// listening on the Serf LAN address
func (s *Server) JoinLAN(addrs []string) (int, error) {
	return s.serfLAN.Join(addrs, true)
}

// JoinWAN is used to have Consul join the cross-WAN Consul ring
// The target address should be another node listening on the
// Serf WAN address
func (s *Server) JoinWAN(addrs []string) (int, error) {
	return s.serfWAN.Join(addrs, true)
}

// LocalMember is used to return the local node
func (c *Server) LocalMember() serf.Member {
	return c.serfLAN.LocalMember()
}

// LANMembers is used to return the members of the LAN cluster
func (s *Server) LANMembers() []serf.Member {
	return s.serfLAN.Members()
}

// WANMembers is used to return the members of the LAN cluster
func (s *Server) WANMembers() []serf.Member {
	return s.serfWAN.Members()
}

// RemoveFailedNode is used to remove a failed node from the cluster
func (s *Server) RemoveFailedNode(node string) error {
	if err := s.serfLAN.RemoveFailedNode(node); err != nil {
		return err
	}
	if err := s.serfWAN.RemoveFailedNode(node); err != nil {
		return err
	}
	return nil
}

// IsLeader checks if this server is the cluster leader
func (s *Server) IsLeader() bool {
	return s.raft.State() == raft.Leader
}

// KeyManagerLAN returns the LAN Serf keyring manager
func (s *Server) KeyManagerLAN() *serf.KeyManager {
	return s.serfLAN.KeyManager()
}

// KeyManagerWAN returns the WAN Serf keyring manager
func (s *Server) KeyManagerWAN() *serf.KeyManager {
	return s.serfWAN.KeyManager()
}

// Encrypted determines if gossip is encrypted
func (s *Server) Encrypted() bool {
	return s.serfLAN.EncryptionEnabled() && s.serfWAN.EncryptionEnabled()
}

// inmemCodec is used to do an RPC call without going over a network
type inmemCodec struct {
	method string
	args   interface{}
	reply  interface{}
	err    error
}

func (i *inmemCodec) ReadRequestHeader(req *rpc.Request) error {
	req.ServiceMethod = i.method
	return nil
}

func (i *inmemCodec) ReadRequestBody(args interface{}) error {
	sourceValue := reflect.Indirect(reflect.Indirect(reflect.ValueOf(i.args)))
	dst := reflect.Indirect(reflect.Indirect(reflect.ValueOf(args)))
	dst.Set(sourceValue)
	return nil
}

func (i *inmemCodec) WriteResponse(resp *rpc.Response, reply interface{}) error {
	if resp.Error != "" {
		i.err = errors.New(resp.Error)
		return nil
	}
	sourceValue := reflect.Indirect(reflect.Indirect(reflect.ValueOf(reply)))
	dst := reflect.Indirect(reflect.Indirect(reflect.ValueOf(i.reply)))
	dst.Set(sourceValue)
	return nil
}

func (i *inmemCodec) Close() error {
	return nil
}

// RPC is used to make a local RPC call
func (s *Server) RPC(method string, args interface{}, reply interface{}) error {
	codec := &inmemCodec{
		method: method,
		args:   args,
		reply:  reply,
	}
	if err := s.rpcServer.ServeRequest(codec); err != nil {
		return err
	}
	return codec.err
}

// InjectEndpoint is used to substitute an endpoint for testing.
func (s *Server) InjectEndpoint(endpoint interface{}) error {
	s.logger.Printf("[WARN] consul: endpoint injected; this should only be used for testing")
	return s.rpcServer.Register(endpoint)
}

// Stats is used to return statistics for debugging and insight
// for various sub-systems
func (s *Server) Stats() map[string]map[string]string {
	toString := func(v uint64) string {
		return strconv.FormatUint(v, 10)
	}
	stats := map[string]map[string]string{
		"consul": map[string]string{
			"server":            "true",
			"leader":            fmt.Sprintf("%v", s.IsLeader()),
			"bootstrap":         fmt.Sprintf("%v", s.config.Bootstrap),
			"known_datacenters": toString(uint64(len(s.remoteConsuls))),
		},
		"raft":     s.raft.Stats(),
		"serf_lan": s.serfLAN.Stats(),
		"serf_wan": s.serfWAN.Stats(),
		"runtime":  runtimeStats(),
	}
	return stats
}

// GetLANCoordinate returns the coordinate of the server in the LAN gossip pool.
func (s *Server) GetLANCoordinate() (*coordinate.Coordinate, error) {
	return s.serfLAN.GetCoordinate()
}

// GetWANCoordinate returns the coordinate of the server in the WAN gossip pool.
func (s *Server) GetWANCoordinate() (*coordinate.Coordinate, error) {
	return s.serfWAN.GetCoordinate()
}
