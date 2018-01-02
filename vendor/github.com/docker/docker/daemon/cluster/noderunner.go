package cluster

import (
	"fmt"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	types "github.com/docker/docker/api/types/swarm"
	"github.com/docker/docker/daemon/cluster/executor/container"
	lncluster "github.com/docker/libnetwork/cluster"
	swarmapi "github.com/docker/swarmkit/api"
	swarmnode "github.com/docker/swarmkit/node"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

// nodeRunner implements a manager for continuously running swarmkit node, restarting them with backoff delays if needed.
type nodeRunner struct {
	nodeState
	mu             sync.RWMutex
	done           chan struct{} // closed when swarmNode exits
	ready          chan struct{} // closed when swarmNode becomes active
	reconnectDelay time.Duration
	config         nodeStartConfig

	repeatedRun     bool
	cancelReconnect func()
	stopping        bool
	cluster         *Cluster // only for accessing config helpers, never call any methods. TODO: change to config struct
}

// nodeStartConfig holds configuration needed to start a new node. Exported
// fields of this structure are saved to disk in json. Unexported fields
// contain data that shouldn't be persisted between daemon reloads.
type nodeStartConfig struct {
	// LocalAddr is this machine's local IP or hostname, if specified.
	LocalAddr string
	// RemoteAddr is the address that was given to "swarm join". It is used
	// to find LocalAddr if necessary.
	RemoteAddr string
	// ListenAddr is the address we bind to, including a port.
	ListenAddr string
	// AdvertiseAddr is the address other nodes should connect to,
	// including a port.
	AdvertiseAddr string
	// DataPathAddr is the address that has to be used for the data path
	DataPathAddr string
	// JoinInProgress is set to true if a join operation has started, but
	// not completed yet.
	JoinInProgress bool

	joinAddr        string
	forceNewCluster bool
	joinToken       string
	lockKey         []byte
	autolock        bool
	availability    types.NodeAvailability
}

func (n *nodeRunner) Ready() chan error {
	c := make(chan error, 1)
	n.mu.RLock()
	ready, done := n.ready, n.done
	n.mu.RUnlock()
	go func() {
		select {
		case <-ready:
		case <-done:
		}
		select {
		case <-ready:
		default:
			n.mu.RLock()
			c <- n.err
			n.mu.RUnlock()
		}
		close(c)
	}()
	return c
}

func (n *nodeRunner) Start(conf nodeStartConfig) error {
	n.mu.Lock()
	defer n.mu.Unlock()

	n.reconnectDelay = initialReconnectDelay

	return n.start(conf)
}

func (n *nodeRunner) start(conf nodeStartConfig) error {
	var control string
	if runtime.GOOS == "windows" {
		control = `\\.\pipe\` + controlSocket
	} else {
		control = filepath.Join(n.cluster.runtimeRoot, controlSocket)
	}

	joinAddr := conf.joinAddr
	if joinAddr == "" && conf.JoinInProgress {
		// We must have been restarted while trying to join a cluster.
		// Continue trying to join instead of forming our own cluster.
		joinAddr = conf.RemoteAddr
	}

	// Hostname is not set here. Instead, it is obtained from
	// the node description that is reported periodically
	swarmnodeConfig := swarmnode.Config{
		ForceNewCluster:    conf.forceNewCluster,
		ListenControlAPI:   control,
		ListenRemoteAPI:    conf.ListenAddr,
		AdvertiseRemoteAPI: conf.AdvertiseAddr,
		JoinAddr:           joinAddr,
		StateDir:           n.cluster.root,
		JoinToken:          conf.joinToken,
		Executor:           container.NewExecutor(n.cluster.config.Backend, n.cluster.config.PluginBackend),
		HeartbeatTick:      1,
		ElectionTick:       3,
		UnlockKey:          conf.lockKey,
		AutoLockManagers:   conf.autolock,
		PluginGetter:       n.cluster.config.Backend.PluginGetter(),
	}
	if conf.availability != "" {
		avail, ok := swarmapi.NodeSpec_Availability_value[strings.ToUpper(string(conf.availability))]
		if !ok {
			return fmt.Errorf("invalid Availability: %q", conf.availability)
		}
		swarmnodeConfig.Availability = swarmapi.NodeSpec_Availability(avail)
	}
	node, err := swarmnode.New(&swarmnodeConfig)
	if err != nil {
		return err
	}
	if err := node.Start(context.Background()); err != nil {
		return err
	}

	n.done = make(chan struct{})
	n.ready = make(chan struct{})
	n.swarmNode = node
	if conf.joinAddr != "" {
		conf.JoinInProgress = true
	}
	n.config = conf
	savePersistentState(n.cluster.root, conf)

	ctx, cancel := context.WithCancel(context.Background())

	go func() {
		n.handleNodeExit(node)
		cancel()
	}()

	go n.handleReadyEvent(ctx, node, n.ready)
	go n.handleControlSocketChange(ctx, node)

	return nil
}

func (n *nodeRunner) handleControlSocketChange(ctx context.Context, node *swarmnode.Node) {
	for conn := range node.ListenControlSocket(ctx) {
		n.mu.Lock()
		if n.grpcConn != conn {
			if conn == nil {
				n.controlClient = nil
				n.logsClient = nil
			} else {
				n.controlClient = swarmapi.NewControlClient(conn)
				n.logsClient = swarmapi.NewLogsClient(conn)
				// push store changes to daemon
				go n.watchClusterEvents(ctx, conn)
			}
		}
		n.grpcConn = conn
		n.mu.Unlock()
		n.cluster.SendClusterEvent(lncluster.EventSocketChange)
	}
}

func (n *nodeRunner) watchClusterEvents(ctx context.Context, conn *grpc.ClientConn) {
	client := swarmapi.NewWatchClient(conn)
	watch, err := client.Watch(ctx, &swarmapi.WatchRequest{
		Entries: []*swarmapi.WatchRequest_WatchEntry{
			{
				Kind:   "node",
				Action: swarmapi.WatchActionKindCreate | swarmapi.WatchActionKindUpdate | swarmapi.WatchActionKindRemove,
			},
			{
				Kind:   "service",
				Action: swarmapi.WatchActionKindCreate | swarmapi.WatchActionKindUpdate | swarmapi.WatchActionKindRemove,
			},
			{
				Kind:   "network",
				Action: swarmapi.WatchActionKindCreate | swarmapi.WatchActionKindUpdate | swarmapi.WatchActionKindRemove,
			},
			{
				Kind:   "secret",
				Action: swarmapi.WatchActionKindCreate | swarmapi.WatchActionKindUpdate | swarmapi.WatchActionKindRemove,
			},
			{
				Kind:   "config",
				Action: swarmapi.WatchActionKindCreate | swarmapi.WatchActionKindUpdate | swarmapi.WatchActionKindRemove,
			},
		},
		IncludeOldObject: true,
	})
	if err != nil {
		logrus.WithError(err).Error("failed to watch cluster store")
		return
	}
	for {
		msg, err := watch.Recv()
		if err != nil {
			// store watch is broken
			logrus.WithError(err).Error("failed to receive changes from store watch API")
			return
		}
		select {
		case <-ctx.Done():
			return
		case n.cluster.watchStream <- msg:
		}
	}
}

func (n *nodeRunner) handleReadyEvent(ctx context.Context, node *swarmnode.Node, ready chan struct{}) {
	select {
	case <-node.Ready():
		n.mu.Lock()
		n.err = nil
		if n.config.JoinInProgress {
			n.config.JoinInProgress = false
			savePersistentState(n.cluster.root, n.config)
		}
		n.mu.Unlock()
		close(ready)
	case <-ctx.Done():
	}
	n.cluster.SendClusterEvent(lncluster.EventNodeReady)
}

func (n *nodeRunner) handleNodeExit(node *swarmnode.Node) {
	err := detectLockedError(node.Err(context.Background()))
	if err != nil {
		logrus.Errorf("cluster exited with error: %v", err)
	}
	n.mu.Lock()
	n.swarmNode = nil
	n.err = err
	close(n.done)
	select {
	case <-n.ready:
		n.enableReconnectWatcher()
	default:
		if n.repeatedRun {
			n.enableReconnectWatcher()
		}
	}
	n.repeatedRun = true
	n.mu.Unlock()
}

// Stop stops the current swarm node if it is running.
func (n *nodeRunner) Stop() error {
	n.mu.Lock()
	if n.cancelReconnect != nil { // between restarts
		n.cancelReconnect()
		n.cancelReconnect = nil
	}
	if n.swarmNode == nil {
		n.mu.Unlock()
		return nil
	}
	n.stopping = true
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	n.mu.Unlock()
	if err := n.swarmNode.Stop(ctx); err != nil && !strings.Contains(err.Error(), "context canceled") {
		return err
	}
	n.cluster.SendClusterEvent(lncluster.EventNodeLeave)
	<-n.done
	return nil
}

func (n *nodeRunner) State() nodeState {
	if n == nil {
		return nodeState{status: types.LocalNodeStateInactive}
	}
	n.mu.RLock()
	defer n.mu.RUnlock()

	ns := n.nodeState

	if ns.err != nil || n.cancelReconnect != nil {
		if errors.Cause(ns.err) == errSwarmLocked {
			ns.status = types.LocalNodeStateLocked
		} else {
			ns.status = types.LocalNodeStateError
		}
	} else {
		select {
		case <-n.ready:
			ns.status = types.LocalNodeStateActive
		default:
			ns.status = types.LocalNodeStatePending
		}
	}

	return ns
}

func (n *nodeRunner) enableReconnectWatcher() {
	if n.stopping {
		return
	}
	n.reconnectDelay *= 2
	if n.reconnectDelay > maxReconnectDelay {
		n.reconnectDelay = maxReconnectDelay
	}
	logrus.Warnf("Restarting swarm in %.2f seconds", n.reconnectDelay.Seconds())
	delayCtx, cancel := context.WithTimeout(context.Background(), n.reconnectDelay)
	n.cancelReconnect = cancel

	go func() {
		<-delayCtx.Done()
		if delayCtx.Err() != context.DeadlineExceeded {
			return
		}
		n.mu.Lock()
		defer n.mu.Unlock()
		if n.stopping {
			return
		}

		if err := n.start(n.config); err != nil {
			n.err = err
		}
	}()
}

// nodeState represents information about the current state of the cluster and
// provides access to the grpc clients.
type nodeState struct {
	swarmNode       *swarmnode.Node
	grpcConn        *grpc.ClientConn
	controlClient   swarmapi.ControlClient
	logsClient      swarmapi.LogsClient
	status          types.LocalNodeState
	actualLocalAddr string
	err             error
}

// IsActiveManager returns true if node is a manager ready to accept control requests. It is safe to access the client properties if this returns true.
func (ns nodeState) IsActiveManager() bool {
	return ns.controlClient != nil
}

// IsManager returns true if node is a manager.
func (ns nodeState) IsManager() bool {
	return ns.swarmNode != nil && ns.swarmNode.Manager() != nil
}

// NodeID returns node's ID or empty string if node is inactive.
func (ns nodeState) NodeID() string {
	if ns.swarmNode != nil {
		return ns.swarmNode.NodeID()
	}
	return ""
}
