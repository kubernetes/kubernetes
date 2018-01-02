package cluster

//
// ## Swarmkit integration
//
// Cluster - static configurable object for accessing everything swarm related.
// Contains methods for connecting and controlling the cluster. Exists always,
// even if swarm mode is not enabled.
//
// NodeRunner - Manager for starting the swarmkit node. Is present only and
// always if swarm mode is enabled. Implements backoff restart loop in case of
// errors.
//
// NodeState - Information about the current node status including access to
// gRPC clients if a manager is active.
//
// ### Locking
//
// `cluster.controlMutex` - taken for the whole lifecycle of the processes that
// can reconfigure cluster(init/join/leave etc). Protects that one
// reconfiguration action has fully completed before another can start.
//
// `cluster.mu` - taken when the actual changes in cluster configurations
// happen. Different from `controlMutex` because in some cases we need to
// access current cluster state even if the long-running reconfiguration is
// going on. For example network stack may ask for the current cluster state in
// the middle of the shutdown. Any time current cluster state is asked you
// should take the read lock of `cluster.mu`. If you are writing an API
// responder that returns synchronously, hold `cluster.mu.RLock()` for the
// duration of the whole handler function. That ensures that node will not be
// shut down until the handler has finished.
//
// NodeRunner implements its internal locks that should not be used outside of
// the struct. Instead, you should just call `nodeRunner.State()` method to get
// the current state of the cluster(still need `cluster.mu.RLock()` to access
// `cluster.nr` reference itself). Most of the changes in NodeRunner happen
// because of an external event(network problem, unexpected swarmkit error) and
// Docker shouldn't take any locks that delay these changes from happening.
//

import (
	"fmt"
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/docker/docker/api/types/network"
	types "github.com/docker/docker/api/types/swarm"
	"github.com/docker/docker/daemon/cluster/controllers/plugin"
	executorpkg "github.com/docker/docker/daemon/cluster/executor"
	"github.com/docker/docker/pkg/signal"
	lncluster "github.com/docker/libnetwork/cluster"
	swarmapi "github.com/docker/swarmkit/api"
	swarmnode "github.com/docker/swarmkit/node"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

const swarmDirName = "swarm"
const controlSocket = "control.sock"
const swarmConnectTimeout = 20 * time.Second
const swarmRequestTimeout = 20 * time.Second
const stateFile = "docker-state.json"
const defaultAddr = "0.0.0.0:2377"

const (
	initialReconnectDelay = 100 * time.Millisecond
	maxReconnectDelay     = 30 * time.Second
	contextPrefix         = "com.docker.swarm"
)

// errNoSwarm is returned on leaving a cluster that was never initialized
var errNoSwarm = errors.New("This node is not part of a swarm")

// errSwarmExists is returned on initialize or join request for a cluster that has already been activated
var errSwarmExists = errors.New("This node is already part of a swarm. Use \"docker swarm leave\" to leave this swarm and join another one.")

// errSwarmJoinTimeoutReached is returned when cluster join could not complete before timeout was reached.
var errSwarmJoinTimeoutReached = errors.New("Timeout was reached before node was joined. The attempt to join the swarm will continue in the background. Use the \"docker info\" command to see the current swarm status of your node.")

// errSwarmLocked is returned if the swarm is encrypted and needs a key to unlock it.
var errSwarmLocked = errors.New("Swarm is encrypted and needs to be unlocked before it can be used. Please use \"docker swarm unlock\" to unlock it.")

// errSwarmCertificatesExpired is returned if docker was not started for the whole validity period and they had no chance to renew automatically.
var errSwarmCertificatesExpired = errors.New("Swarm certificates have expired. To replace them, leave the swarm and join again.")

// NetworkSubnetsProvider exposes functions for retrieving the subnets
// of networks managed by Docker, so they can be filtered.
type NetworkSubnetsProvider interface {
	Subnets() ([]net.IPNet, []net.IPNet)
}

// Config provides values for Cluster.
type Config struct {
	Root                   string
	Name                   string
	Backend                executorpkg.Backend
	PluginBackend          plugin.Backend
	NetworkSubnetsProvider NetworkSubnetsProvider

	// DefaultAdvertiseAddr is the default host/IP or network interface to use
	// if no AdvertiseAddr value is specified.
	DefaultAdvertiseAddr string

	// path to store runtime state, such as the swarm control socket
	RuntimeRoot string

	// WatchStream is a channel to pass watch API notifications to daemon
	WatchStream chan *swarmapi.WatchMessage
}

// Cluster provides capabilities to participate in a cluster as a worker or a
// manager.
type Cluster struct {
	mu           sync.RWMutex
	controlMutex sync.RWMutex // protect init/join/leave user operations
	nr           *nodeRunner
	root         string
	runtimeRoot  string
	config       Config
	configEvent  chan lncluster.ConfigEventType // todo: make this array and goroutine safe
	attachers    map[string]*attacher
	watchStream  chan *swarmapi.WatchMessage
}

// attacher manages the in-memory attachment state of a container
// attachment to a global scope network managed by swarm manager. It
// helps in identifying the attachment ID via the taskID and the
// corresponding attachment configuration obtained from the manager.
type attacher struct {
	taskID           string
	config           *network.NetworkingConfig
	inProgress       bool
	attachWaitCh     chan *network.NetworkingConfig
	attachCompleteCh chan struct{}
	detachWaitCh     chan struct{}
}

// New creates a new Cluster instance using provided config.
func New(config Config) (*Cluster, error) {
	root := filepath.Join(config.Root, swarmDirName)
	if err := os.MkdirAll(root, 0700); err != nil {
		return nil, err
	}
	if config.RuntimeRoot == "" {
		config.RuntimeRoot = root
	}
	if err := os.MkdirAll(config.RuntimeRoot, 0700); err != nil {
		return nil, err
	}
	c := &Cluster{
		root:        root,
		config:      config,
		configEvent: make(chan lncluster.ConfigEventType, 10),
		runtimeRoot: config.RuntimeRoot,
		attachers:   make(map[string]*attacher),
		watchStream: config.WatchStream,
	}
	return c, nil
}

// Start the Cluster instance
// TODO The split between New and Start can be join again when the SendClusterEvent
// method is no longer required
func (c *Cluster) Start() error {
	root := filepath.Join(c.config.Root, swarmDirName)

	nodeConfig, err := loadPersistentState(root)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	nr, err := c.newNodeRunner(*nodeConfig)
	if err != nil {
		return err
	}
	c.nr = nr

	select {
	case <-time.After(swarmConnectTimeout):
		logrus.Error("swarm component could not be started before timeout was reached")
	case err := <-nr.Ready():
		if err != nil {
			logrus.WithError(err).Error("swarm component could not be started")
			return nil
		}
	}
	return nil
}

func (c *Cluster) newNodeRunner(conf nodeStartConfig) (*nodeRunner, error) {
	if err := c.config.Backend.IsSwarmCompatible(); err != nil {
		return nil, err
	}

	actualLocalAddr := conf.LocalAddr
	if actualLocalAddr == "" {
		// If localAddr was not specified, resolve it automatically
		// based on the route to joinAddr. localAddr can only be left
		// empty on "join".
		listenHost, _, err := net.SplitHostPort(conf.ListenAddr)
		if err != nil {
			return nil, fmt.Errorf("could not parse listen address: %v", err)
		}

		listenAddrIP := net.ParseIP(listenHost)
		if listenAddrIP == nil || !listenAddrIP.IsUnspecified() {
			actualLocalAddr = listenHost
		} else {
			if conf.RemoteAddr == "" {
				// Should never happen except using swarms created by
				// old versions that didn't save remoteAddr.
				conf.RemoteAddr = "8.8.8.8:53"
			}
			conn, err := net.Dial("udp", conf.RemoteAddr)
			if err != nil {
				return nil, fmt.Errorf("could not find local IP address: %v", err)
			}
			localHostPort := conn.LocalAddr().String()
			actualLocalAddr, _, _ = net.SplitHostPort(localHostPort)
			conn.Close()
		}
	}

	nr := &nodeRunner{cluster: c}
	nr.actualLocalAddr = actualLocalAddr

	if err := nr.Start(conf); err != nil {
		return nil, err
	}

	c.config.Backend.DaemonJoinsCluster(c)

	return nr, nil
}

func (c *Cluster) getRequestContext() (context.Context, func()) { // TODO: not needed when requests don't block on qourum lost
	return context.WithTimeout(context.Background(), swarmRequestTimeout)
}

// IsManager returns true if Cluster is participating as a manager.
func (c *Cluster) IsManager() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.currentNodeState().IsActiveManager()
}

// IsAgent returns true if Cluster is participating as a worker/agent.
func (c *Cluster) IsAgent() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.currentNodeState().status == types.LocalNodeStateActive
}

// GetLocalAddress returns the local address.
func (c *Cluster) GetLocalAddress() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.currentNodeState().actualLocalAddr
}

// GetListenAddress returns the listen address.
func (c *Cluster) GetListenAddress() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.nr != nil {
		return c.nr.config.ListenAddr
	}
	return ""
}

// GetAdvertiseAddress returns the remotely reachable address of this node.
func (c *Cluster) GetAdvertiseAddress() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.nr != nil && c.nr.config.AdvertiseAddr != "" {
		advertiseHost, _, _ := net.SplitHostPort(c.nr.config.AdvertiseAddr)
		return advertiseHost
	}
	return c.currentNodeState().actualLocalAddr
}

// GetDataPathAddress returns the address to be used for the data path traffic, if specified.
func (c *Cluster) GetDataPathAddress() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.nr != nil {
		return c.nr.config.DataPathAddr
	}
	return ""
}

// GetRemoteAddressList returns the advertise address for each of the remote managers if
// available.
func (c *Cluster) GetRemoteAddressList() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.getRemoteAddressList()
}

func (c *Cluster) getRemoteAddressList() []string {
	state := c.currentNodeState()
	if state.swarmNode == nil {
		return []string{}
	}

	nodeID := state.swarmNode.NodeID()
	remotes := state.swarmNode.Remotes()
	addressList := make([]string, 0, len(remotes))
	for _, r := range remotes {
		if r.NodeID != nodeID {
			addressList = append(addressList, r.Addr)
		}
	}
	return addressList
}

// ListenClusterEvents returns a channel that receives messages on cluster
// participation changes.
// todo: make cancelable and accessible to multiple callers
func (c *Cluster) ListenClusterEvents() <-chan lncluster.ConfigEventType {
	return c.configEvent
}

// currentNodeState should not be called without a read lock
func (c *Cluster) currentNodeState() nodeState {
	return c.nr.State()
}

// errNoManager returns error describing why manager commands can't be used.
// Call with read lock.
func (c *Cluster) errNoManager(st nodeState) error {
	if st.swarmNode == nil {
		if errors.Cause(st.err) == errSwarmLocked {
			return errSwarmLocked
		}
		if st.err == errSwarmCertificatesExpired {
			return errSwarmCertificatesExpired
		}
		return errors.New("This node is not a swarm manager. Use \"docker swarm init\" or \"docker swarm join\" to connect this node to swarm and try again.")
	}
	if st.swarmNode.Manager() != nil {
		return errors.New("This node is not a swarm manager. Manager is being prepared or has trouble connecting to the cluster.")
	}
	return errors.New("This node is not a swarm manager. Worker nodes can't be used to view or modify cluster state. Please run this command on a manager node or promote the current node to a manager.")
}

// Cleanup stops active swarm node. This is run before daemon shutdown.
func (c *Cluster) Cleanup() {
	c.controlMutex.Lock()
	defer c.controlMutex.Unlock()

	c.mu.Lock()
	node := c.nr
	if node == nil {
		c.mu.Unlock()
		return
	}
	state := c.currentNodeState()
	c.mu.Unlock()

	if state.IsActiveManager() {
		active, reachable, unreachable, err := managerStats(state.controlClient, state.NodeID())
		if err == nil {
			singlenode := active && isLastManager(reachable, unreachable)
			if active && !singlenode && removingManagerCausesLossOfQuorum(reachable, unreachable) {
				logrus.Errorf("Leaving cluster with %v managers left out of %v. Raft quorum will be lost.", reachable-1, reachable+unreachable)
			}
		}
	}

	if err := node.Stop(); err != nil {
		logrus.Errorf("failed to shut down cluster node: %v", err)
		signal.DumpStacks("")
	}

	c.mu.Lock()
	c.nr = nil
	c.mu.Unlock()
}

func managerStats(client swarmapi.ControlClient, currentNodeID string) (current bool, reachable int, unreachable int, err error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	nodes, err := client.ListNodes(ctx, &swarmapi.ListNodesRequest{})
	if err != nil {
		return false, 0, 0, err
	}
	for _, n := range nodes.Nodes {
		if n.ManagerStatus != nil {
			if n.ManagerStatus.Reachability == swarmapi.RaftMemberStatus_REACHABLE {
				reachable++
				if n.ID == currentNodeID {
					current = true
				}
			}
			if n.ManagerStatus.Reachability == swarmapi.RaftMemberStatus_UNREACHABLE {
				unreachable++
			}
		}
	}
	return
}

func detectLockedError(err error) error {
	if err == swarmnode.ErrInvalidUnlockKey {
		return errors.WithStack(errSwarmLocked)
	}
	return err
}

func (c *Cluster) lockedManagerAction(fn func(ctx context.Context, state nodeState) error) error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	state := c.currentNodeState()
	if !state.IsActiveManager() {
		return c.errNoManager(state)
	}

	ctx, cancel := c.getRequestContext()
	defer cancel()

	return fn(ctx, state)
}

// SendClusterEvent allows to send cluster events on the configEvent channel
// TODO This method should not be exposed.
// Currently it is used to notify the network controller that the keys are
// available
func (c *Cluster) SendClusterEvent(event lncluster.ConfigEventType) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	c.configEvent <- event
}
