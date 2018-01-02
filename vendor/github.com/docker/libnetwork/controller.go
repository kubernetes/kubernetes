/*
Package libnetwork provides the basic functionality and extension points to
create network namespaces and allocate interfaces for containers to use.

	networkType := "bridge"

	// Create a new controller instance
	driverOptions := options.Generic{}
	genericOption := make(map[string]interface{})
	genericOption[netlabel.GenericData] = driverOptions
	controller, err := libnetwork.New(config.OptionDriverConfig(networkType, genericOption))
	if err != nil {
		return
	}

	// Create a network for containers to join.
	// NewNetwork accepts Variadic optional arguments that libnetwork and Drivers can make use of
	network, err := controller.NewNetwork(networkType, "network1", "")
	if err != nil {
		return
	}

	// For each new container: allocate IP and interfaces. The returned network
	// settings will be used for container infos (inspect and such), as well as
	// iptables rules for port publishing. This info is contained or accessible
	// from the returned endpoint.
	ep, err := network.CreateEndpoint("Endpoint1")
	if err != nil {
		return
	}

	// Create the sandbox for the container.
	// NewSandbox accepts Variadic optional arguments which libnetwork can use.
	sbx, err := controller.NewSandbox("container1",
		libnetwork.OptionHostname("test"),
		libnetwork.OptionDomainname("docker.io"))

	// A sandbox can join the endpoint via the join api.
	err = ep.Join(sbx)
	if err != nil {
		return
	}
*/
package libnetwork

import (
	"container/heap"
	"fmt"
	"net"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/docker/docker/pkg/discovery"
	"github.com/docker/docker/pkg/locker"
	"github.com/docker/docker/pkg/plugingetter"
	"github.com/docker/docker/pkg/plugins"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/libnetwork/cluster"
	"github.com/docker/libnetwork/config"
	"github.com/docker/libnetwork/datastore"
	"github.com/docker/libnetwork/discoverapi"
	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/drvregistry"
	"github.com/docker/libnetwork/hostdiscovery"
	"github.com/docker/libnetwork/ipamapi"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/osl"
	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
)

// NetworkController provides the interface for controller instance which manages
// networks.
type NetworkController interface {
	// ID provides a unique identity for the controller
	ID() string

	// BuiltinDrivers returns list of builtin drivers
	BuiltinDrivers() []string

	// BuiltinIPAMDrivers returns list of builtin ipam drivers
	BuiltinIPAMDrivers() []string

	// Config method returns the bootup configuration for the controller
	Config() config.Config

	// Create a new network. The options parameter carries network specific options.
	NewNetwork(networkType, name string, id string, options ...NetworkOption) (Network, error)

	// Networks returns the list of Network(s) managed by this controller.
	Networks() []Network

	// WalkNetworks uses the provided function to walk the Network(s) managed by this controller.
	WalkNetworks(walker NetworkWalker)

	// NetworkByName returns the Network which has the passed name. If not found, the error ErrNoSuchNetwork is returned.
	NetworkByName(name string) (Network, error)

	// NetworkByID returns the Network which has the passed id. If not found, the error ErrNoSuchNetwork is returned.
	NetworkByID(id string) (Network, error)

	// NewSandbox creates a new network sandbox for the passed container id
	NewSandbox(containerID string, options ...SandboxOption) (Sandbox, error)

	// Sandboxes returns the list of Sandbox(s) managed by this controller.
	Sandboxes() []Sandbox

	// WalkSandboxes uses the provided function to walk the Sandbox(s) managed by this controller.
	WalkSandboxes(walker SandboxWalker)

	// SandboxByID returns the Sandbox which has the passed id. If not found, a types.NotFoundError is returned.
	SandboxByID(id string) (Sandbox, error)

	// SandboxDestroy destroys a sandbox given a container ID
	SandboxDestroy(id string) error

	// Stop network controller
	Stop()

	// ReloadCondfiguration updates the controller configuration
	ReloadConfiguration(cfgOptions ...config.Option) error

	// SetClusterProvider sets cluster provider
	SetClusterProvider(provider cluster.Provider)

	// Wait for agent initialization complete in libnetwork controller
	AgentInitWait()

	// Wait for agent to stop if running
	AgentStopWait()

	// SetKeys configures the encryption key for gossip and overlay data path
	SetKeys(keys []*types.EncryptionKey) error
}

// NetworkWalker is a client provided function which will be used to walk the Networks.
// When the function returns true, the walk will stop.
type NetworkWalker func(nw Network) bool

// SandboxWalker is a client provided function which will be used to walk the Sandboxes.
// When the function returns true, the walk will stop.
type SandboxWalker func(sb Sandbox) bool

type sandboxTable map[string]*sandbox

type controller struct {
	id                     string
	drvRegistry            *drvregistry.DrvRegistry
	sandboxes              sandboxTable
	cfg                    *config.Config
	stores                 []datastore.DataStore
	discovery              hostdiscovery.HostDiscovery
	extKeyListener         net.Listener
	watchCh                chan *endpoint
	unWatchCh              chan *endpoint
	svcRecords             map[string]svcInfo
	nmap                   map[string]*netWatch
	serviceBindings        map[serviceKey]*service
	defOsSbox              osl.Sandbox
	ingressSandbox         *sandbox
	sboxOnce               sync.Once
	agent                  *agent
	networkLocker          *locker.Locker
	agentInitDone          chan struct{}
	agentStopDone          chan struct{}
	keys                   []*types.EncryptionKey
	clusterConfigAvailable bool
	sync.Mutex
}

type initializer struct {
	fn    drvregistry.InitFunc
	ntype string
}

// New creates a new instance of network controller.
func New(cfgOptions ...config.Option) (NetworkController, error) {
	c := &controller{
		id:              stringid.GenerateRandomID(),
		cfg:             config.ParseConfigOptions(cfgOptions...),
		sandboxes:       sandboxTable{},
		svcRecords:      make(map[string]svcInfo),
		serviceBindings: make(map[serviceKey]*service),
		agentInitDone:   make(chan struct{}),
		networkLocker:   locker.New(),
	}

	if err := c.initStores(); err != nil {
		return nil, err
	}

	drvRegistry, err := drvregistry.New(c.getStore(datastore.LocalScope), c.getStore(datastore.GlobalScope), c.RegisterDriver, nil, c.cfg.PluginGetter)
	if err != nil {
		return nil, err
	}

	for _, i := range getInitializers(c.cfg.Daemon.Experimental) {
		var dcfg map[string]interface{}

		// External plugins don't need config passed through daemon. They can
		// bootstrap themselves
		if i.ntype != "remote" {
			dcfg = c.makeDriverConfig(i.ntype)
		}

		if err := drvRegistry.AddDriver(i.ntype, i.fn, dcfg); err != nil {
			return nil, err
		}
	}

	if err = initIPAMDrivers(drvRegistry, nil, c.getStore(datastore.GlobalScope)); err != nil {
		return nil, err
	}

	c.drvRegistry = drvRegistry

	if c.cfg != nil && c.cfg.Cluster.Watcher != nil {
		if err := c.initDiscovery(c.cfg.Cluster.Watcher); err != nil {
			// Failing to initialize discovery is a bad situation to be in.
			// But it cannot fail creating the Controller
			logrus.Errorf("Failed to Initialize Discovery : %v", err)
		}
	}

	c.WalkNetworks(populateSpecial)

	// Reserve pools first before doing cleanup. Otherwise the
	// cleanups of endpoint/network and sandbox below will
	// generate many unnecessary warnings
	c.reservePools()

	// Cleanup resources
	c.sandboxCleanup(c.cfg.ActiveSandboxes)
	c.cleanupLocalEndpoints()
	c.networkCleanup()

	if err := c.startExternalKeyListener(); err != nil {
		return nil, err
	}

	return c, nil
}

func (c *controller) SetClusterProvider(provider cluster.Provider) {
	var sameProvider bool
	c.Lock()
	// Avoids to spawn multiple goroutine for the same cluster provider
	if c.cfg.Daemon.ClusterProvider == provider {
		// If the cluster provider is already set, there is already a go routine spawned
		// that is listening for events, so nothing to do here
		sameProvider = true
	} else {
		c.cfg.Daemon.ClusterProvider = provider
	}
	c.Unlock()

	if provider == nil || sameProvider {
		return
	}
	// We don't want to spawn a new go routine if the previous one did not exit yet
	c.AgentStopWait()
	go c.clusterAgentInit()
}

func isValidClusteringIP(addr string) bool {
	return addr != "" && !net.ParseIP(addr).IsLoopback() && !net.ParseIP(addr).IsUnspecified()
}

// libnetwork side of agent depends on the keys. On the first receipt of
// keys setup the agent. For subsequent key set handle the key change
func (c *controller) SetKeys(keys []*types.EncryptionKey) error {
	subsysKeys := make(map[string]int)
	for _, key := range keys {
		if key.Subsystem != subsysGossip &&
			key.Subsystem != subsysIPSec {
			return fmt.Errorf("key received for unrecognized subsystem")
		}
		subsysKeys[key.Subsystem]++
	}
	for s, count := range subsysKeys {
		if count != keyringSize {
			return fmt.Errorf("incorrect number of keys for subsystem %v", s)
		}
	}

	agent := c.getAgent()

	if agent == nil {
		c.Lock()
		c.keys = keys
		c.Unlock()
		return nil
	}
	return c.handleKeyChange(keys)
}

func (c *controller) getAgent() *agent {
	c.Lock()
	defer c.Unlock()
	return c.agent
}

func (c *controller) clusterAgentInit() {
	clusterProvider := c.cfg.Daemon.ClusterProvider
	var keysAvailable bool
	for {
		eventType := <-clusterProvider.ListenClusterEvents()
		// The events: EventSocketChange, EventNodeReady and EventNetworkKeysAvailable are not ordered
		// when all the condition for the agent initialization are met then proceed with it
		switch eventType {
		case cluster.EventNetworkKeysAvailable:
			// Validates that the keys are actually available before starting the initialization
			// This will handle old spurious messages left on the channel
			c.Lock()
			keysAvailable = c.keys != nil
			c.Unlock()
			fallthrough
		case cluster.EventSocketChange, cluster.EventNodeReady:
			if keysAvailable && !c.isDistributedControl() {
				c.agentOperationStart()
				if err := c.agentSetup(clusterProvider); err != nil {
					c.agentStopComplete()
				} else {
					c.agentInitComplete()
				}
			}
		case cluster.EventNodeLeave:
			keysAvailable = false
			c.agentOperationStart()
			c.Lock()
			c.keys = nil
			c.Unlock()

			// We are leaving the cluster. Make sure we
			// close the gossip so that we stop all
			// incoming gossip updates before cleaning up
			// any remaining service bindings. But before
			// deleting the networks since the networks
			// should still be present when cleaning up
			// service bindings
			c.agentClose()
			c.cleanupServiceBindings("")

			c.agentStopComplete()

			return
		}
	}
}

// AgentInitWait waits for agent initialization to be completed in the controller.
func (c *controller) AgentInitWait() {
	c.Lock()
	agentInitDone := c.agentInitDone
	c.Unlock()

	if agentInitDone != nil {
		<-agentInitDone
	}
}

// AgentStopWait waits for the Agent stop to be completed in the controller
func (c *controller) AgentStopWait() {
	c.Lock()
	agentStopDone := c.agentStopDone
	c.Unlock()
	if agentStopDone != nil {
		<-agentStopDone
	}
}

// agentOperationStart marks the start of an Agent Init or Agent Stop
func (c *controller) agentOperationStart() {
	c.Lock()
	if c.agentInitDone == nil {
		c.agentInitDone = make(chan struct{})
	}
	if c.agentStopDone == nil {
		c.agentStopDone = make(chan struct{})
	}
	c.Unlock()
}

// agentInitComplete notifies the successful completion of the Agent initialization
func (c *controller) agentInitComplete() {
	c.Lock()
	if c.agentInitDone != nil {
		close(c.agentInitDone)
		c.agentInitDone = nil
	}
	c.Unlock()
}

// agentStopComplete notifies the successful completion of the Agent stop
func (c *controller) agentStopComplete() {
	c.Lock()
	if c.agentStopDone != nil {
		close(c.agentStopDone)
		c.agentStopDone = nil
	}
	c.Unlock()
}

func (c *controller) makeDriverConfig(ntype string) map[string]interface{} {
	if c.cfg == nil {
		return nil
	}

	config := make(map[string]interface{})

	for _, label := range c.cfg.Daemon.Labels {
		if !strings.HasPrefix(netlabel.Key(label), netlabel.DriverPrefix+"."+ntype) {
			continue
		}

		config[netlabel.Key(label)] = netlabel.Value(label)
	}

	drvCfg, ok := c.cfg.Daemon.DriverCfg[ntype]
	if ok {
		for k, v := range drvCfg.(map[string]interface{}) {
			config[k] = v
		}
	}

	for k, v := range c.cfg.Scopes {
		if !v.IsValid() {
			continue
		}
		config[netlabel.MakeKVClient(k)] = discoverapi.DatastoreConfigData{
			Scope:    k,
			Provider: v.Client.Provider,
			Address:  v.Client.Address,
			Config:   v.Client.Config,
		}
	}

	return config
}

var procReloadConfig = make(chan (bool), 1)

func (c *controller) ReloadConfiguration(cfgOptions ...config.Option) error {
	procReloadConfig <- true
	defer func() { <-procReloadConfig }()

	// For now we accept the configuration reload only as a mean to provide a global store config after boot.
	// Refuse the configuration if it alters an existing datastore client configuration.
	update := false
	cfg := config.ParseConfigOptions(cfgOptions...)

	for s := range c.cfg.Scopes {
		if _, ok := cfg.Scopes[s]; !ok {
			return types.ForbiddenErrorf("cannot accept new configuration because it removes an existing datastore client")
		}
	}
	for s, nSCfg := range cfg.Scopes {
		if eSCfg, ok := c.cfg.Scopes[s]; ok {
			if eSCfg.Client.Provider != nSCfg.Client.Provider ||
				eSCfg.Client.Address != nSCfg.Client.Address {
				return types.ForbiddenErrorf("cannot accept new configuration because it modifies an existing datastore client")
			}
		} else {
			if err := c.initScopedStore(s, nSCfg); err != nil {
				return err
			}
			update = true
		}
	}
	if !update {
		return nil
	}

	c.Lock()
	c.cfg = cfg
	c.Unlock()

	var dsConfig *discoverapi.DatastoreConfigData
	for scope, sCfg := range cfg.Scopes {
		if scope == datastore.LocalScope || !sCfg.IsValid() {
			continue
		}
		dsConfig = &discoverapi.DatastoreConfigData{
			Scope:    scope,
			Provider: sCfg.Client.Provider,
			Address:  sCfg.Client.Address,
			Config:   sCfg.Client.Config,
		}
		break
	}
	if dsConfig == nil {
		return nil
	}

	c.drvRegistry.WalkIPAMs(func(name string, driver ipamapi.Ipam, cap *ipamapi.Capability) bool {
		err := driver.DiscoverNew(discoverapi.DatastoreConfig, *dsConfig)
		if err != nil {
			logrus.Errorf("Failed to set datastore in driver %s: %v", name, err)
		}
		return false
	})

	c.drvRegistry.WalkDrivers(func(name string, driver driverapi.Driver, capability driverapi.Capability) bool {
		err := driver.DiscoverNew(discoverapi.DatastoreConfig, *dsConfig)
		if err != nil {
			logrus.Errorf("Failed to set datastore in driver %s: %v", name, err)
		}
		return false
	})

	if c.discovery == nil && c.cfg.Cluster.Watcher != nil {
		if err := c.initDiscovery(c.cfg.Cluster.Watcher); err != nil {
			logrus.Errorf("Failed to Initialize Discovery after configuration update: %v", err)
		}
	}

	return nil
}

func (c *controller) ID() string {
	return c.id
}

func (c *controller) BuiltinDrivers() []string {
	drivers := []string{}
	c.drvRegistry.WalkDrivers(func(name string, driver driverapi.Driver, capability driverapi.Capability) bool {
		if driver.IsBuiltIn() {
			drivers = append(drivers, name)
		}
		return false
	})
	return drivers
}

func (c *controller) BuiltinIPAMDrivers() []string {
	drivers := []string{}
	c.drvRegistry.WalkIPAMs(func(name string, driver ipamapi.Ipam, cap *ipamapi.Capability) bool {
		if driver.IsBuiltIn() {
			drivers = append(drivers, name)
		}
		return false
	})
	return drivers
}

func (c *controller) validateHostDiscoveryConfig() bool {
	if c.cfg == nil || c.cfg.Cluster.Discovery == "" || c.cfg.Cluster.Address == "" {
		return false
	}
	return true
}

func (c *controller) clusterHostID() string {
	c.Lock()
	defer c.Unlock()
	if c.cfg == nil || c.cfg.Cluster.Address == "" {
		return ""
	}
	addr := strings.Split(c.cfg.Cluster.Address, ":")
	return addr[0]
}

func (c *controller) isNodeAlive(node string) bool {
	if c.discovery == nil {
		return false
	}

	nodes := c.discovery.Fetch()
	for _, n := range nodes {
		if n.String() == node {
			return true
		}
	}

	return false
}

func (c *controller) initDiscovery(watcher discovery.Watcher) error {
	if c.cfg == nil {
		return fmt.Errorf("discovery initialization requires a valid configuration")
	}

	c.discovery = hostdiscovery.NewHostDiscovery(watcher)
	return c.discovery.Watch(c.activeCallback, c.hostJoinCallback, c.hostLeaveCallback)
}

func (c *controller) activeCallback() {
	ds := c.getStore(datastore.GlobalScope)
	if ds != nil && !ds.Active() {
		ds.RestartWatch()
	}
}

func (c *controller) hostJoinCallback(nodes []net.IP) {
	c.processNodeDiscovery(nodes, true)
}

func (c *controller) hostLeaveCallback(nodes []net.IP) {
	c.processNodeDiscovery(nodes, false)
}

func (c *controller) processNodeDiscovery(nodes []net.IP, add bool) {
	c.drvRegistry.WalkDrivers(func(name string, driver driverapi.Driver, capability driverapi.Capability) bool {
		c.pushNodeDiscovery(driver, capability, nodes, add)
		return false
	})
}

func (c *controller) pushNodeDiscovery(d driverapi.Driver, cap driverapi.Capability, nodes []net.IP, add bool) {
	var self net.IP
	if c.cfg != nil {
		addr := strings.Split(c.cfg.Cluster.Address, ":")
		self = net.ParseIP(addr[0])
		// if external kvstore is not configured, try swarm-mode config
		if self == nil {
			if agent := c.getAgent(); agent != nil {
				self = net.ParseIP(agent.advertiseAddr)
			}
		}
	}

	if d == nil || cap.ConnectivityScope != datastore.GlobalScope || nodes == nil {
		return
	}

	for _, node := range nodes {
		nodeData := discoverapi.NodeDiscoveryData{Address: node.String(), Self: node.Equal(self)}
		var err error
		if add {
			err = d.DiscoverNew(discoverapi.NodeDiscovery, nodeData)
		} else {
			err = d.DiscoverDelete(discoverapi.NodeDiscovery, nodeData)
		}
		if err != nil {
			logrus.Debugf("discovery notification error: %v", err)
		}
	}
}

func (c *controller) Config() config.Config {
	c.Lock()
	defer c.Unlock()
	if c.cfg == nil {
		return config.Config{}
	}
	return *c.cfg
}

func (c *controller) isManager() bool {
	c.Lock()
	defer c.Unlock()
	if c.cfg == nil || c.cfg.Daemon.ClusterProvider == nil {
		return false
	}
	return c.cfg.Daemon.ClusterProvider.IsManager()
}

func (c *controller) isAgent() bool {
	c.Lock()
	defer c.Unlock()
	if c.cfg == nil || c.cfg.Daemon.ClusterProvider == nil {
		return false
	}
	return c.cfg.Daemon.ClusterProvider.IsAgent()
}

func (c *controller) isDistributedControl() bool {
	return !c.isManager() && !c.isAgent()
}

func (c *controller) GetPluginGetter() plugingetter.PluginGetter {
	return c.drvRegistry.GetPluginGetter()
}

func (c *controller) RegisterDriver(networkType string, driver driverapi.Driver, capability driverapi.Capability) error {
	c.Lock()
	hd := c.discovery
	c.Unlock()

	if hd != nil {
		c.pushNodeDiscovery(driver, capability, hd.Fetch(), true)
	}

	c.agentDriverNotify(driver)
	return nil
}

// NewNetwork creates a new network of the specified network type. The options
// are network specific and modeled in a generic way.
func (c *controller) NewNetwork(networkType, name string, id string, options ...NetworkOption) (Network, error) {
	if id != "" {
		c.networkLocker.Lock(id)
		defer c.networkLocker.Unlock(id)

		if _, err := c.NetworkByID(id); err == nil {
			return nil, NetworkNameError(id)
		}
	}

	if !config.IsValidName(name) {
		return nil, ErrInvalidName(name)
	}

	if id == "" {
		id = stringid.GenerateRandomID()
	}

	defaultIpam := defaultIpamForNetworkType(networkType)
	// Construct the network object
	network := &network{
		name:        name,
		networkType: networkType,
		generic:     map[string]interface{}{netlabel.GenericData: make(map[string]string)},
		ipamType:    defaultIpam,
		id:          id,
		created:     time.Now(),
		ctrlr:       c,
		persist:     true,
		drvOnce:     &sync.Once{},
	}

	network.processOptions(options...)
	if err := network.validateConfiguration(); err != nil {
		return nil, err
	}

	var (
		cap *driverapi.Capability
		err error
	)

	// Reset network types, force local scope and skip allocation and
	// plumbing for configuration networks. Reset of the config-only
	// network drivers is needed so that this special network is not
	// usable by old engine versions.
	if network.configOnly {
		network.scope = datastore.LocalScope
		network.networkType = "null"
		goto addToStore
	}

	_, cap, err = network.resolveDriver(network.networkType, true)
	if err != nil {
		return nil, err
	}

	if network.scope == datastore.LocalScope && cap.DataScope == datastore.GlobalScope {
		return nil, types.ForbiddenErrorf("cannot downgrade network scope for %s networks", networkType)

	}
	if network.ingress && cap.DataScope != datastore.GlobalScope {
		return nil, types.ForbiddenErrorf("Ingress network can only be global scope network")
	}

	// At this point the network scope is still unknown if not set by user
	if (cap.DataScope == datastore.GlobalScope || network.scope == datastore.SwarmScope) &&
		!c.isDistributedControl() && !network.dynamic {
		if c.isManager() {
			// For non-distributed controlled environment, globalscoped non-dynamic networks are redirected to Manager
			return nil, ManagerRedirectError(name)
		}
		return nil, types.ForbiddenErrorf("Cannot create a multi-host network from a worker node. Please create the network from a manager node.")
	}

	if network.scope == datastore.SwarmScope && c.isDistributedControl() {
		return nil, types.ForbiddenErrorf("cannot create a swarm scoped network when swarm is not active")
	}

	// Make sure we have a driver available for this network type
	// before we allocate anything.
	if _, err := network.driver(true); err != nil {
		return nil, err
	}

	// From this point on, we need the network specific configuration,
	// which may come from a configuration-only network
	if network.configFrom != "" {
		t, err := c.getConfigNetwork(network.configFrom)
		if err != nil {
			return nil, types.NotFoundErrorf("configuration network %q does not exist", network.configFrom)
		}
		if err := t.applyConfigurationTo(network); err != nil {
			return nil, types.InternalErrorf("Failed to apply configuration: %v", err)
		}
		defer func() {
			if err == nil {
				if err := t.getEpCnt().IncEndpointCnt(); err != nil {
					logrus.Warnf("Failed to update reference count for configuration network %q on creation of network %q: %v",
						t.Name(), network.Name(), err)
				}
			}
		}()
	}

	err = network.ipamAllocate()
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			network.ipamRelease()
		}
	}()

	err = c.addNetwork(network)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			if e := network.deleteNetwork(); e != nil {
				logrus.Warnf("couldn't roll back driver network on network %s creation failure: %v", network.name, err)
			}
		}
	}()

addToStore:
	// First store the endpoint count, then the network. To avoid to
	// end up with a datastore containing a network and not an epCnt,
	// in case of an ungraceful shutdown during this function call.
	epCnt := &endpointCnt{n: network}
	if err = c.updateToStore(epCnt); err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			if e := c.deleteFromStore(epCnt); e != nil {
				logrus.Warnf("could not rollback from store, epCnt %v on failure (%v): %v", epCnt, err, e)
			}
		}
	}()

	network.epCnt = epCnt
	if err = c.updateToStore(network); err != nil {
		return nil, err
	}
	if network.configOnly {
		return network, nil
	}

	joinCluster(network)
	if !c.isDistributedControl() {
		c.Lock()
		arrangeIngressFilterRule()
		c.Unlock()
	}

	c.Lock()
	arrangeUserFilterRule()
	c.Unlock()

	return network, nil
}

var joinCluster NetworkWalker = func(nw Network) bool {
	n := nw.(*network)
	if n.configOnly {
		return false
	}
	if err := n.joinCluster(); err != nil {
		logrus.Errorf("Failed to join network %s (%s) into agent cluster: %v", n.Name(), n.ID(), err)
	}
	n.addDriverWatches()
	return false
}

func (c *controller) reservePools() {
	networks, err := c.getNetworksForScope(datastore.LocalScope)
	if err != nil {
		logrus.Warnf("Could not retrieve networks from local store during ipam allocation for existing networks: %v", err)
		return
	}

	for _, n := range networks {
		if n.configOnly {
			continue
		}
		if !doReplayPoolReserve(n) {
			continue
		}
		// Construct pseudo configs for the auto IP case
		autoIPv4 := (len(n.ipamV4Config) == 0 || (len(n.ipamV4Config) == 1 && n.ipamV4Config[0].PreferredPool == "")) && len(n.ipamV4Info) > 0
		autoIPv6 := (len(n.ipamV6Config) == 0 || (len(n.ipamV6Config) == 1 && n.ipamV6Config[0].PreferredPool == "")) && len(n.ipamV6Info) > 0
		if autoIPv4 {
			n.ipamV4Config = []*IpamConf{{PreferredPool: n.ipamV4Info[0].Pool.String()}}
		}
		if n.enableIPv6 && autoIPv6 {
			n.ipamV6Config = []*IpamConf{{PreferredPool: n.ipamV6Info[0].Pool.String()}}
		}
		// Account current network gateways
		for i, c := range n.ipamV4Config {
			if c.Gateway == "" && n.ipamV4Info[i].Gateway != nil {
				c.Gateway = n.ipamV4Info[i].Gateway.IP.String()
			}
		}
		if n.enableIPv6 {
			for i, c := range n.ipamV6Config {
				if c.Gateway == "" && n.ipamV6Info[i].Gateway != nil {
					c.Gateway = n.ipamV6Info[i].Gateway.IP.String()
				}
			}
		}
		// Reserve pools
		if err := n.ipamAllocate(); err != nil {
			logrus.Warnf("Failed to allocate ipam pool(s) for network %q (%s): %v", n.Name(), n.ID(), err)
		}
		// Reserve existing endpoints' addresses
		ipam, _, err := n.getController().getIPAMDriver(n.ipamType)
		if err != nil {
			logrus.Warnf("Failed to retrieve ipam driver for network %q (%s) during address reservation", n.Name(), n.ID())
			continue
		}
		epl, err := n.getEndpointsFromStore()
		if err != nil {
			logrus.Warnf("Failed to retrieve list of current endpoints on network %q (%s)", n.Name(), n.ID())
			continue
		}
		for _, ep := range epl {
			if err := ep.assignAddress(ipam, true, ep.Iface().AddressIPv6() != nil); err != nil {
				logrus.Warnf("Failed to reserve current address for endpoint %q (%s) on network %q (%s)",
					ep.Name(), ep.ID(), n.Name(), n.ID())
			}
		}
	}
}

func doReplayPoolReserve(n *network) bool {
	_, caps, err := n.getController().getIPAMDriver(n.ipamType)
	if err != nil {
		logrus.Warnf("Failed to retrieve ipam driver for network %q (%s): %v", n.Name(), n.ID(), err)
		return false
	}
	return caps.RequiresRequestReplay
}

func (c *controller) addNetwork(n *network) error {
	d, err := n.driver(true)
	if err != nil {
		return err
	}

	// Create the network
	if err := d.CreateNetwork(n.id, n.generic, n, n.getIPData(4), n.getIPData(6)); err != nil {
		return err
	}

	n.startResolver()

	return nil
}

func (c *controller) Networks() []Network {
	var list []Network

	networks, err := c.getNetworksFromStore()
	if err != nil {
		logrus.Error(err)
	}

	for _, n := range networks {
		if n.inDelete {
			continue
		}
		list = append(list, n)
	}

	return list
}

func (c *controller) WalkNetworks(walker NetworkWalker) {
	for _, n := range c.Networks() {
		if walker(n) {
			return
		}
	}
}

func (c *controller) NetworkByName(name string) (Network, error) {
	if name == "" {
		return nil, ErrInvalidName(name)
	}
	var n Network

	s := func(current Network) bool {
		if current.Name() == name {
			n = current
			return true
		}
		return false
	}

	c.WalkNetworks(s)

	if n == nil {
		return nil, ErrNoSuchNetwork(name)
	}

	return n, nil
}

func (c *controller) NetworkByID(id string) (Network, error) {
	if id == "" {
		return nil, ErrInvalidID(id)
	}

	n, err := c.getNetworkFromStore(id)
	if err != nil {
		return nil, ErrNoSuchNetwork(id)
	}

	return n, nil
}

// NewSandbox creates a new sandbox for the passed container id
func (c *controller) NewSandbox(containerID string, options ...SandboxOption) (Sandbox, error) {
	if containerID == "" {
		return nil, types.BadRequestErrorf("invalid container ID")
	}

	var sb *sandbox
	c.Lock()
	for _, s := range c.sandboxes {
		if s.containerID == containerID {
			// If not a stub, then we already have a complete sandbox.
			if !s.isStub {
				sbID := s.ID()
				c.Unlock()
				return nil, types.ForbiddenErrorf("container %s is already present in sandbox %s", containerID, sbID)
			}

			// We already have a stub sandbox from the
			// store. Make use of it so that we don't lose
			// the endpoints from store but reset the
			// isStub flag.
			sb = s
			sb.isStub = false
			break
		}
	}
	c.Unlock()

	// Create sandbox and process options first. Key generation depends on an option
	if sb == nil {
		sb = &sandbox{
			id:                 stringid.GenerateRandomID(),
			containerID:        containerID,
			endpoints:          epHeap{},
			epPriority:         map[string]int{},
			populatedEndpoints: map[string]struct{}{},
			config:             containerConfig{},
			controller:         c,
			extDNS:             []extDNSEntry{},
		}
	}

	heap.Init(&sb.endpoints)

	sb.processOptions(options...)

	c.Lock()
	if sb.ingress && c.ingressSandbox != nil {
		c.Unlock()
		return nil, types.ForbiddenErrorf("ingress sandbox already present")
	}

	if sb.ingress {
		c.ingressSandbox = sb
		sb.config.hostsPath = filepath.Join(c.cfg.Daemon.DataDir, "/network/files/hosts")
		sb.config.resolvConfPath = filepath.Join(c.cfg.Daemon.DataDir, "/network/files/resolv.conf")
		sb.id = "ingress_sbox"
	}
	c.Unlock()

	var err error
	defer func() {
		if err != nil {
			c.Lock()
			if sb.ingress {
				c.ingressSandbox = nil
			}
			c.Unlock()
		}
	}()

	if err = sb.setupResolutionFiles(); err != nil {
		return nil, err
	}

	if sb.config.useDefaultSandBox {
		c.sboxOnce.Do(func() {
			c.defOsSbox, err = osl.NewSandbox(sb.Key(), false, false)
		})

		if err != nil {
			c.sboxOnce = sync.Once{}
			return nil, fmt.Errorf("failed to create default sandbox: %v", err)
		}

		sb.osSbox = c.defOsSbox
	}

	if sb.osSbox == nil && !sb.config.useExternalKey {
		if sb.osSbox, err = osl.NewSandbox(sb.Key(), !sb.config.useDefaultSandBox, false); err != nil {
			return nil, fmt.Errorf("failed to create new osl sandbox: %v", err)
		}
	}

	c.Lock()
	c.sandboxes[sb.id] = sb
	c.Unlock()
	defer func() {
		if err != nil {
			c.Lock()
			delete(c.sandboxes, sb.id)
			c.Unlock()
		}
	}()

	err = sb.storeUpdate()
	if err != nil {
		return nil, fmt.Errorf("failed to update the store state of sandbox: %v", err)
	}

	return sb, nil
}

func (c *controller) Sandboxes() []Sandbox {
	c.Lock()
	defer c.Unlock()

	list := make([]Sandbox, 0, len(c.sandboxes))
	for _, s := range c.sandboxes {
		// Hide stub sandboxes from libnetwork users
		if s.isStub {
			continue
		}

		list = append(list, s)
	}

	return list
}

func (c *controller) WalkSandboxes(walker SandboxWalker) {
	for _, sb := range c.Sandboxes() {
		if walker(sb) {
			return
		}
	}
}

func (c *controller) SandboxByID(id string) (Sandbox, error) {
	if id == "" {
		return nil, ErrInvalidID(id)
	}
	c.Lock()
	s, ok := c.sandboxes[id]
	c.Unlock()
	if !ok {
		return nil, types.NotFoundErrorf("sandbox %s not found", id)
	}
	return s, nil
}

// SandboxDestroy destroys a sandbox given a container ID
func (c *controller) SandboxDestroy(id string) error {
	var sb *sandbox
	c.Lock()
	for _, s := range c.sandboxes {
		if s.containerID == id {
			sb = s
			break
		}
	}
	c.Unlock()

	// It is not an error if sandbox is not available
	if sb == nil {
		return nil
	}

	return sb.Delete()
}

// SandboxContainerWalker returns a Sandbox Walker function which looks for an existing Sandbox with the passed containerID
func SandboxContainerWalker(out *Sandbox, containerID string) SandboxWalker {
	return func(sb Sandbox) bool {
		if sb.ContainerID() == containerID {
			*out = sb
			return true
		}
		return false
	}
}

// SandboxKeyWalker returns a Sandbox Walker function which looks for an existing Sandbox with the passed key
func SandboxKeyWalker(out *Sandbox, key string) SandboxWalker {
	return func(sb Sandbox) bool {
		if sb.Key() == key {
			*out = sb
			return true
		}
		return false
	}
}

func (c *controller) loadDriver(networkType string) error {
	var err error

	if pg := c.GetPluginGetter(); pg != nil {
		_, err = pg.Get(networkType, driverapi.NetworkPluginEndpointType, plugingetter.Lookup)
	} else {
		_, err = plugins.Get(networkType, driverapi.NetworkPluginEndpointType)
	}

	if err != nil {
		if err == plugins.ErrNotFound {
			return types.NotFoundErrorf(err.Error())
		}
		return err
	}

	return nil
}

func (c *controller) loadIPAMDriver(name string) error {
	var err error

	if pg := c.GetPluginGetter(); pg != nil {
		_, err = pg.Get(name, ipamapi.PluginEndpointType, plugingetter.Lookup)
	} else {
		_, err = plugins.Get(name, ipamapi.PluginEndpointType)
	}

	if err != nil {
		if err == plugins.ErrNotFound {
			return types.NotFoundErrorf(err.Error())
		}
		return err
	}

	return nil
}

func (c *controller) getIPAMDriver(name string) (ipamapi.Ipam, *ipamapi.Capability, error) {
	id, cap := c.drvRegistry.IPAM(name)
	if id == nil {
		// Might be a plugin name. Try loading it
		if err := c.loadIPAMDriver(name); err != nil {
			return nil, nil, err
		}

		// Now that we resolved the plugin, try again looking up the registry
		id, cap = c.drvRegistry.IPAM(name)
		if id == nil {
			return nil, nil, types.BadRequestErrorf("invalid ipam driver: %q", name)
		}
	}

	return id, cap, nil
}

func (c *controller) Stop() {
	c.closeStores()
	c.stopExternalKeyListener()
	osl.GC()
}
