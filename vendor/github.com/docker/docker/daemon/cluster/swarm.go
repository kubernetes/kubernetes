package cluster

import (
	"fmt"
	"net"
	"strings"
	"time"

	apitypes "github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/filters"
	types "github.com/docker/docker/api/types/swarm"
	"github.com/docker/docker/daemon/cluster/convert"
	"github.com/docker/docker/opts"
	"github.com/docker/docker/pkg/signal"
	swarmapi "github.com/docker/swarmkit/api"
	"github.com/docker/swarmkit/manager/encryption"
	swarmnode "github.com/docker/swarmkit/node"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

// Init initializes new cluster from user provided request.
func (c *Cluster) Init(req types.InitRequest) (string, error) {
	c.controlMutex.Lock()
	defer c.controlMutex.Unlock()
	if c.nr != nil {
		if req.ForceNewCluster {

			// Take c.mu temporarily to wait for presently running
			// API handlers to finish before shutting down the node.
			c.mu.Lock()
			if !c.nr.nodeState.IsManager() {
				return "", errSwarmNotManager
			}
			c.mu.Unlock()

			if err := c.nr.Stop(); err != nil {
				return "", err
			}
		} else {
			return "", errSwarmExists
		}
	}

	if err := validateAndSanitizeInitRequest(&req); err != nil {
		return "", validationError{err}
	}

	listenHost, listenPort, err := resolveListenAddr(req.ListenAddr)
	if err != nil {
		return "", err
	}

	advertiseHost, advertisePort, err := c.resolveAdvertiseAddr(req.AdvertiseAddr, listenPort)
	if err != nil {
		return "", err
	}

	dataPathAddr, err := resolveDataPathAddr(req.DataPathAddr)
	if err != nil {
		return "", err
	}

	localAddr := listenHost

	// If the local address is undetermined, the advertise address
	// will be used as local address, if it belongs to this system.
	// If the advertise address is not local, then we try to find
	// a system address to use as local address. If this fails,
	// we give up and ask the user to pass the listen address.
	if net.ParseIP(localAddr).IsUnspecified() {
		advertiseIP := net.ParseIP(advertiseHost)

		found := false
		for _, systemIP := range listSystemIPs() {
			if systemIP.Equal(advertiseIP) {
				localAddr = advertiseIP.String()
				found = true
				break
			}
		}

		if !found {
			ip, err := c.resolveSystemAddr()
			if err != nil {
				logrus.Warnf("Could not find a local address: %v", err)
				return "", errMustSpecifyListenAddr
			}
			localAddr = ip.String()
		}
	}

	nr, err := c.newNodeRunner(nodeStartConfig{
		forceNewCluster: req.ForceNewCluster,
		autolock:        req.AutoLockManagers,
		LocalAddr:       localAddr,
		ListenAddr:      net.JoinHostPort(listenHost, listenPort),
		AdvertiseAddr:   net.JoinHostPort(advertiseHost, advertisePort),
		DataPathAddr:    dataPathAddr,
		availability:    req.Availability,
	})
	if err != nil {
		return "", err
	}
	c.mu.Lock()
	c.nr = nr
	c.mu.Unlock()

	if err := <-nr.Ready(); err != nil {
		c.mu.Lock()
		c.nr = nil
		c.mu.Unlock()
		if !req.ForceNewCluster { // if failure on first attempt don't keep state
			if err := clearPersistentState(c.root); err != nil {
				return "", err
			}
		}
		return "", err
	}
	state := nr.State()
	if state.swarmNode == nil { // should never happen but protect from panic
		return "", errors.New("invalid cluster state for spec initialization")
	}
	if err := initClusterSpec(state.swarmNode, req.Spec); err != nil {
		return "", err
	}
	return state.NodeID(), nil
}

// Join makes current Cluster part of an existing swarm cluster.
func (c *Cluster) Join(req types.JoinRequest) error {
	c.controlMutex.Lock()
	defer c.controlMutex.Unlock()
	c.mu.Lock()
	if c.nr != nil {
		c.mu.Unlock()
		return errors.WithStack(errSwarmExists)
	}
	c.mu.Unlock()

	if err := validateAndSanitizeJoinRequest(&req); err != nil {
		return validationError{err}
	}

	listenHost, listenPort, err := resolveListenAddr(req.ListenAddr)
	if err != nil {
		return err
	}

	var advertiseAddr string
	if req.AdvertiseAddr != "" {
		advertiseHost, advertisePort, err := c.resolveAdvertiseAddr(req.AdvertiseAddr, listenPort)
		// For joining, we don't need to provide an advertise address,
		// since the remote side can detect it.
		if err == nil {
			advertiseAddr = net.JoinHostPort(advertiseHost, advertisePort)
		}
	}

	dataPathAddr, err := resolveDataPathAddr(req.DataPathAddr)
	if err != nil {
		return err
	}

	nr, err := c.newNodeRunner(nodeStartConfig{
		RemoteAddr:    req.RemoteAddrs[0],
		ListenAddr:    net.JoinHostPort(listenHost, listenPort),
		AdvertiseAddr: advertiseAddr,
		DataPathAddr:  dataPathAddr,
		joinAddr:      req.RemoteAddrs[0],
		joinToken:     req.JoinToken,
		availability:  req.Availability,
	})
	if err != nil {
		return err
	}

	c.mu.Lock()
	c.nr = nr
	c.mu.Unlock()

	select {
	case <-time.After(swarmConnectTimeout):
		return errSwarmJoinTimeoutReached
	case err := <-nr.Ready():
		if err != nil {
			c.mu.Lock()
			c.nr = nil
			c.mu.Unlock()
			if err := clearPersistentState(c.root); err != nil {
				return err
			}
		}
		return err
	}
}

// Inspect retrieves the configuration properties of a managed swarm cluster.
func (c *Cluster) Inspect() (types.Swarm, error) {
	var swarm types.Swarm
	if err := c.lockedManagerAction(func(ctx context.Context, state nodeState) error {
		s, err := c.inspect(ctx, state)
		if err != nil {
			return err
		}
		swarm = s
		return nil
	}); err != nil {
		return types.Swarm{}, err
	}
	return swarm, nil
}

func (c *Cluster) inspect(ctx context.Context, state nodeState) (types.Swarm, error) {
	s, err := getSwarm(ctx, state.controlClient)
	if err != nil {
		return types.Swarm{}, err
	}
	return convert.SwarmFromGRPC(*s), nil
}

// Update updates configuration of a managed swarm cluster.
func (c *Cluster) Update(version uint64, spec types.Spec, flags types.UpdateFlags) error {
	return c.lockedManagerAction(func(ctx context.Context, state nodeState) error {
		swarm, err := getSwarm(ctx, state.controlClient)
		if err != nil {
			return err
		}

		// Validate spec name.
		if spec.Annotations.Name == "" {
			spec.Annotations.Name = "default"
		} else if spec.Annotations.Name != "default" {
			return validationError{errors.New(`swarm spec must be named "default"`)}
		}

		// In update, client should provide the complete spec of the swarm, including
		// Name and Labels. If a field is specified with 0 or nil, then the default value
		// will be used to swarmkit.
		clusterSpec, err := convert.SwarmSpecToGRPC(spec)
		if err != nil {
			return convertError{err}
		}

		_, err = state.controlClient.UpdateCluster(
			ctx,
			&swarmapi.UpdateClusterRequest{
				ClusterID: swarm.ID,
				Spec:      &clusterSpec,
				ClusterVersion: &swarmapi.Version{
					Index: version,
				},
				Rotation: swarmapi.KeyRotation{
					WorkerJoinToken:  flags.RotateWorkerToken,
					ManagerJoinToken: flags.RotateManagerToken,
					ManagerUnlockKey: flags.RotateManagerUnlockKey,
				},
			},
		)
		return err
	})
}

// GetUnlockKey returns the unlock key for the swarm.
func (c *Cluster) GetUnlockKey() (string, error) {
	var resp *swarmapi.GetUnlockKeyResponse
	if err := c.lockedManagerAction(func(ctx context.Context, state nodeState) error {
		client := swarmapi.NewCAClient(state.grpcConn)

		r, err := client.GetUnlockKey(ctx, &swarmapi.GetUnlockKeyRequest{})
		if err != nil {
			return err
		}
		resp = r
		return nil
	}); err != nil {
		return "", err
	}
	if len(resp.UnlockKey) == 0 {
		// no key
		return "", nil
	}
	return encryption.HumanReadableKey(resp.UnlockKey), nil
}

// UnlockSwarm provides a key to decrypt data that is encrypted at rest.
func (c *Cluster) UnlockSwarm(req types.UnlockRequest) error {
	c.controlMutex.Lock()
	defer c.controlMutex.Unlock()

	c.mu.RLock()
	state := c.currentNodeState()

	if !state.IsActiveManager() {
		// when manager is not active,
		// unless it is locked, otherwise return error.
		if err := c.errNoManager(state); err != errSwarmLocked {
			c.mu.RUnlock()
			return err
		}
	} else {
		// when manager is active, return an error of "not locked"
		c.mu.RUnlock()
		return notLockedError{}
	}

	// only when swarm is locked, code running reaches here
	nr := c.nr
	c.mu.RUnlock()

	key, err := encryption.ParseHumanReadableKey(req.UnlockKey)
	if err != nil {
		return validationError{err}
	}

	config := nr.config
	config.lockKey = key
	if err := nr.Stop(); err != nil {
		return err
	}
	nr, err = c.newNodeRunner(config)
	if err != nil {
		return err
	}

	c.mu.Lock()
	c.nr = nr
	c.mu.Unlock()

	if err := <-nr.Ready(); err != nil {
		if errors.Cause(err) == errSwarmLocked {
			return invalidUnlockKey{}
		}
		return errors.Errorf("swarm component could not be started: %v", err)
	}
	return nil
}

// Leave shuts down Cluster and removes current state.
func (c *Cluster) Leave(force bool) error {
	c.controlMutex.Lock()
	defer c.controlMutex.Unlock()

	c.mu.Lock()
	nr := c.nr
	if nr == nil {
		c.mu.Unlock()
		return errors.WithStack(errNoSwarm)
	}

	state := c.currentNodeState()

	c.mu.Unlock()

	if errors.Cause(state.err) == errSwarmLocked && !force {
		// leave a locked swarm without --force is not allowed
		return errors.WithStack(notAvailableError("Swarm is encrypted and locked. Please unlock it first or use `--force` to ignore this message."))
	}

	if state.IsManager() && !force {
		msg := "You are attempting to leave the swarm on a node that is participating as a manager. "
		if state.IsActiveManager() {
			active, reachable, unreachable, err := managerStats(state.controlClient, state.NodeID())
			if err == nil {
				if active && removingManagerCausesLossOfQuorum(reachable, unreachable) {
					if isLastManager(reachable, unreachable) {
						msg += "Removing the last manager erases all current state of the swarm. Use `--force` to ignore this message. "
						return errors.WithStack(notAvailableError(msg))
					}
					msg += fmt.Sprintf("Removing this node leaves %v managers out of %v. Without a Raft quorum your swarm will be inaccessible. ", reachable-1, reachable+unreachable)
				}
			}
		} else {
			msg += "Doing so may lose the consensus of your cluster. "
		}

		msg += "The only way to restore a swarm that has lost consensus is to reinitialize it with `--force-new-cluster`. Use `--force` to suppress this message."
		return errors.WithStack(notAvailableError(msg))
	}
	// release readers in here
	if err := nr.Stop(); err != nil {
		logrus.Errorf("failed to shut down cluster node: %v", err)
		signal.DumpStacks("")
		return err
	}

	c.mu.Lock()
	c.nr = nil
	c.mu.Unlock()

	if nodeID := state.NodeID(); nodeID != "" {
		nodeContainers, err := c.listContainerForNode(nodeID)
		if err != nil {
			return err
		}
		for _, id := range nodeContainers {
			if err := c.config.Backend.ContainerRm(id, &apitypes.ContainerRmConfig{ForceRemove: true}); err != nil {
				logrus.Errorf("error removing %v: %v", id, err)
			}
		}
	}

	// todo: cleanup optional?
	if err := clearPersistentState(c.root); err != nil {
		return err
	}
	c.config.Backend.DaemonLeavesCluster()
	return nil
}

// Info returns information about the current cluster state.
func (c *Cluster) Info() types.Info {
	info := types.Info{
		NodeAddr: c.GetAdvertiseAddress(),
	}
	c.mu.RLock()
	defer c.mu.RUnlock()

	state := c.currentNodeState()
	info.LocalNodeState = state.status
	if state.err != nil {
		info.Error = state.err.Error()
	}

	ctx, cancel := c.getRequestContext()
	defer cancel()

	if state.IsActiveManager() {
		info.ControlAvailable = true
		swarm, err := c.inspect(ctx, state)
		if err != nil {
			info.Error = err.Error()
		}

		info.Cluster = &swarm.ClusterInfo

		if r, err := state.controlClient.ListNodes(ctx, &swarmapi.ListNodesRequest{}); err != nil {
			info.Error = err.Error()
		} else {
			info.Nodes = len(r.Nodes)
			for _, n := range r.Nodes {
				if n.ManagerStatus != nil {
					info.Managers = info.Managers + 1
				}
			}
		}
	}

	if state.swarmNode != nil {
		for _, r := range state.swarmNode.Remotes() {
			info.RemoteManagers = append(info.RemoteManagers, types.Peer{NodeID: r.NodeID, Addr: r.Addr})
		}
		info.NodeID = state.swarmNode.NodeID()
	}

	return info
}

func validateAndSanitizeInitRequest(req *types.InitRequest) error {
	var err error
	req.ListenAddr, err = validateAddr(req.ListenAddr)
	if err != nil {
		return fmt.Errorf("invalid ListenAddr %q: %v", req.ListenAddr, err)
	}

	if req.Spec.Annotations.Name == "" {
		req.Spec.Annotations.Name = "default"
	} else if req.Spec.Annotations.Name != "default" {
		return errors.New(`swarm spec must be named "default"`)
	}

	return nil
}

func validateAndSanitizeJoinRequest(req *types.JoinRequest) error {
	var err error
	req.ListenAddr, err = validateAddr(req.ListenAddr)
	if err != nil {
		return fmt.Errorf("invalid ListenAddr %q: %v", req.ListenAddr, err)
	}
	if len(req.RemoteAddrs) == 0 {
		return errors.New("at least 1 RemoteAddr is required to join")
	}
	for i := range req.RemoteAddrs {
		req.RemoteAddrs[i], err = validateAddr(req.RemoteAddrs[i])
		if err != nil {
			return fmt.Errorf("invalid remoteAddr %q: %v", req.RemoteAddrs[i], err)
		}
	}
	return nil
}

func validateAddr(addr string) (string, error) {
	if addr == "" {
		return addr, errors.New("invalid empty address")
	}
	newaddr, err := opts.ParseTCPAddr(addr, defaultAddr)
	if err != nil {
		return addr, nil
	}
	return strings.TrimPrefix(newaddr, "tcp://"), nil
}

func initClusterSpec(node *swarmnode.Node, spec types.Spec) error {
	ctx, _ := context.WithTimeout(context.Background(), 5*time.Second)
	for conn := range node.ListenControlSocket(ctx) {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if conn != nil {
			client := swarmapi.NewControlClient(conn)
			var cluster *swarmapi.Cluster
			for i := 0; ; i++ {
				lcr, err := client.ListClusters(ctx, &swarmapi.ListClustersRequest{})
				if err != nil {
					return fmt.Errorf("error on listing clusters: %v", err)
				}
				if len(lcr.Clusters) == 0 {
					if i < 10 {
						time.Sleep(200 * time.Millisecond)
						continue
					}
					return errors.New("empty list of clusters was returned")
				}
				cluster = lcr.Clusters[0]
				break
			}
			// In init, we take the initial default values from swarmkit, and merge
			// any non nil or 0 value from spec to GRPC spec. This will leave the
			// default value alone.
			// Note that this is different from Update(), as in Update() we expect
			// user to specify the complete spec of the cluster (as they already know
			// the existing one and knows which field to update)
			clusterSpec, err := convert.MergeSwarmSpecToGRPC(spec, cluster.Spec)
			if err != nil {
				return fmt.Errorf("error updating cluster settings: %v", err)
			}
			_, err = client.UpdateCluster(ctx, &swarmapi.UpdateClusterRequest{
				ClusterID:      cluster.ID,
				ClusterVersion: &cluster.Meta.Version,
				Spec:           &clusterSpec,
			})
			if err != nil {
				return fmt.Errorf("error updating cluster settings: %v", err)
			}
			return nil
		}
	}
	return ctx.Err()
}

func (c *Cluster) listContainerForNode(nodeID string) ([]string, error) {
	var ids []string
	filters := filters.NewArgs()
	filters.Add("label", fmt.Sprintf("com.docker.swarm.node.id=%s", nodeID))
	containers, err := c.config.Backend.Containers(&apitypes.ContainerListOptions{
		Filters: filters,
	})
	if err != nil {
		return []string{}, err
	}
	for _, c := range containers {
		ids = append(ids, c.ID)
	}
	return ids, nil
}
