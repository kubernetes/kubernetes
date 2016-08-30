package agent

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"sync"
	"time"

	"github.com/hashicorp/consul/consul"
	"github.com/hashicorp/consul/consul/state"
	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/lib"
	"github.com/hashicorp/serf/coordinate"
	"github.com/hashicorp/serf/serf"
)

const (
	// Path to save agent service definitions
	servicesDir = "services"

	// Path to save local agent checks
	checksDir     = "checks"
	checkStateDir = "checks/state"

	// The ID of the faux health checks for maintenance mode
	serviceMaintCheckPrefix = "_service_maintenance"
	nodeMaintCheckID        = "_node_maintenance"

	// Default reasons for node/service maintenance mode
	defaultNodeMaintReason = "Maintenance mode is enabled for this node, " +
		"but no reason was provided. This is a default message."
	defaultServiceMaintReason = "Maintenance mode is enabled for this " +
		"service, but no reason was provided. This is a default message."
)

var (
	// dnsNameRe checks if a name or tag is dns-compatible.
	dnsNameRe = regexp.MustCompile(`^[a-zA-Z0-9\-]+$`)
)

/*
 The agent is the long running process that is run on every machine.
 It exposes an RPC interface that is used by the CLI to control the
 agent. The agent runs the query interfaces like HTTP, DNS, and RPC.
 However, it can run in either a client, or server mode. In server
 mode, it runs a full Consul server. In client-only mode, it only forwards
 requests to other Consul servers.
*/
type Agent struct {
	config *Config

	// Used for writing our logs
	logger *log.Logger

	// Output sink for logs
	logOutput io.Writer

	// We have one of a client or a server, depending
	// on our configuration
	server *consul.Server
	client *consul.Client

	// state stores a local representation of the node,
	// services and checks. Used for anti-entropy.
	state localState

	// checkMonitors maps the check ID to an associated monitor
	checkMonitors map[string]*CheckMonitor

	// checkHTTPs maps the check ID to an associated HTTP check
	checkHTTPs map[string]*CheckHTTP

	// checkTCPs maps the check ID to an associated TCP check
	checkTCPs map[string]*CheckTCP

	// checkTTLs maps the check ID to an associated check TTL
	checkTTLs map[string]*CheckTTL

	// checkDockers maps the check ID to an associated Docker Exec based check
	checkDockers map[string]*CheckDocker

	// checkLock protects updates to the check* maps
	checkLock sync.Mutex

	// eventCh is used to receive user events
	eventCh chan serf.UserEvent

	// eventBuf stores the most recent events in a ring buffer
	// using eventIndex as the next index to insert into. This
	// is guarded by eventLock. When an insert happens, the
	// eventNotify group is notified.
	eventBuf    []*UserEvent
	eventIndex  int
	eventLock   sync.RWMutex
	eventNotify state.NotifyGroup

	shutdown     bool
	shutdownCh   chan struct{}
	shutdownLock sync.Mutex

	// endpoints lets you override RPC endpoints for testing. Not all
	// agent methods use this, so use with care and never override
	// outside of a unit test.
	endpoints map[string]string

	// reapLock is used to prevent child process reaping from interfering
	// with normal waiting for subprocesses to complete. Any time you exec
	// and wait, you should take a read lock on this mutex. Only the reaper
	// takes the write lock. This setup prevents us from serializing all the
	// child process management with each other, it just serializes them
	// with the child process reaper.
	reapLock sync.RWMutex
}

// Create is used to create a new Agent. Returns
// the agent or potentially an error.
func Create(config *Config, logOutput io.Writer) (*Agent, error) {
	// Ensure we have a log sink
	if logOutput == nil {
		logOutput = os.Stderr
	}

	// Validate the config
	if config.Datacenter == "" {
		return nil, fmt.Errorf("Must configure a Datacenter")
	}
	if config.DataDir == "" && !config.DevMode {
		return nil, fmt.Errorf("Must configure a DataDir")
	}

	// Try to get an advertise address
	if config.AdvertiseAddr != "" {
		if ip := net.ParseIP(config.AdvertiseAddr); ip == nil {
			return nil, fmt.Errorf("Failed to parse advertise address: %v", config.AdvertiseAddr)
		}
	} else if config.BindAddr != "0.0.0.0" && config.BindAddr != "" {
		config.AdvertiseAddr = config.BindAddr
	} else {
		ip, err := consul.GetPrivateIP()
		if err != nil {
			return nil, fmt.Errorf("Failed to get advertise address: %v", err)
		}
		config.AdvertiseAddr = ip.String()
	}

	// Try to get an advertise address for the wan
	if config.AdvertiseAddrWan != "" {
		if ip := net.ParseIP(config.AdvertiseAddrWan); ip == nil {
			return nil, fmt.Errorf("Failed to parse advertise address for wan: %v", config.AdvertiseAddrWan)
		}
	} else {
		config.AdvertiseAddrWan = config.AdvertiseAddr
	}

	// Create the default set of tagged addresses.
	config.TaggedAddresses = map[string]string{
		"wan": config.AdvertiseAddrWan,
	}

	agent := &Agent{
		config:        config,
		logger:        log.New(logOutput, "", log.LstdFlags),
		logOutput:     logOutput,
		checkMonitors: make(map[string]*CheckMonitor),
		checkTTLs:     make(map[string]*CheckTTL),
		checkHTTPs:    make(map[string]*CheckHTTP),
		checkTCPs:     make(map[string]*CheckTCP),
		checkDockers:  make(map[string]*CheckDocker),
		eventCh:       make(chan serf.UserEvent, 1024),
		eventBuf:      make([]*UserEvent, 256),
		shutdownCh:    make(chan struct{}),
		endpoints:     make(map[string]string),
	}

	// Initialize the local state
	agent.state.Init(config, agent.logger)

	// Setup either the client or the server
	var err error
	if config.Server {
		err = agent.setupServer()
		agent.state.SetIface(agent.server)

		// Automatically register the "consul" service on server nodes
		consulService := structs.NodeService{
			Service: consul.ConsulServiceName,
			ID:      consul.ConsulServiceID,
			Port:    agent.config.Ports.Server,
			Tags:    []string{},
		}
		agent.state.AddService(&consulService, "")
	} else {
		err = agent.setupClient()
		agent.state.SetIface(agent.client)
	}
	if err != nil {
		return nil, err
	}

	// Load checks/services
	if err := agent.loadServices(config); err != nil {
		return nil, err
	}
	if err := agent.loadChecks(config); err != nil {
		return nil, err
	}

	// Start handling events
	go agent.handleEvents()

	// Start sending network coordinate to the server.
	if !config.DisableCoordinates {
		go agent.sendCoordinate()
	}

	// Write out the PID file if necessary
	err = agent.storePid()
	if err != nil {
		return nil, err
	}

	return agent, nil
}

// consulConfig is used to return a consul configuration
func (a *Agent) consulConfig() *consul.Config {
	// Start with the provided config or default config
	var base *consul.Config
	if a.config.ConsulConfig != nil {
		base = a.config.ConsulConfig
	} else {
		base = consul.DefaultConfig()
	}

	// Apply dev mode
	base.DevMode = a.config.DevMode

	// Override with our config
	if a.config.Datacenter != "" {
		base.Datacenter = a.config.Datacenter
	}
	if a.config.DataDir != "" {
		base.DataDir = a.config.DataDir
	}
	if a.config.NodeName != "" {
		base.NodeName = a.config.NodeName
	}
	if a.config.BindAddr != "" {
		base.SerfLANConfig.MemberlistConfig.BindAddr = a.config.BindAddr
		base.SerfWANConfig.MemberlistConfig.BindAddr = a.config.BindAddr
	}
	if a.config.Ports.SerfLan != 0 {
		base.SerfLANConfig.MemberlistConfig.BindPort = a.config.Ports.SerfLan
		base.SerfLANConfig.MemberlistConfig.AdvertisePort = a.config.Ports.SerfLan
	}
	if a.config.Ports.SerfWan != 0 {
		base.SerfWANConfig.MemberlistConfig.BindPort = a.config.Ports.SerfWan
		base.SerfWANConfig.MemberlistConfig.AdvertisePort = a.config.Ports.SerfWan
	}
	if a.config.BindAddr != "" {
		bindAddr := &net.TCPAddr{
			IP:   net.ParseIP(a.config.BindAddr),
			Port: a.config.Ports.Server,
		}
		base.RPCAddr = bindAddr
	}
	if a.config.AdvertiseAddr != "" {
		base.SerfLANConfig.MemberlistConfig.AdvertiseAddr = a.config.AdvertiseAddr
		if a.config.AdvertiseAddrWan != "" {
			base.SerfWANConfig.MemberlistConfig.AdvertiseAddr = a.config.AdvertiseAddrWan
		} else {
			base.SerfWANConfig.MemberlistConfig.AdvertiseAddr = a.config.AdvertiseAddr
		}
		base.RPCAdvertise = &net.TCPAddr{
			IP:   net.ParseIP(a.config.AdvertiseAddr),
			Port: a.config.Ports.Server,
		}
	}
	if a.config.AdvertiseAddrs.SerfLan != nil {
		base.SerfLANConfig.MemberlistConfig.AdvertiseAddr = a.config.AdvertiseAddrs.SerfLan.IP.String()
		base.SerfLANConfig.MemberlistConfig.AdvertisePort = a.config.AdvertiseAddrs.SerfLan.Port
	}
	if a.config.AdvertiseAddrs.SerfWan != nil {
		base.SerfWANConfig.MemberlistConfig.AdvertiseAddr = a.config.AdvertiseAddrs.SerfWan.IP.String()
		base.SerfWANConfig.MemberlistConfig.AdvertisePort = a.config.AdvertiseAddrs.SerfWan.Port
	}
	if a.config.AdvertiseAddrs.RPC != nil {
		base.RPCAdvertise = a.config.AdvertiseAddrs.RPC
	}
	if a.config.Bootstrap {
		base.Bootstrap = true
	}
	if a.config.RejoinAfterLeave {
		base.RejoinAfterLeave = true
	}
	if a.config.BootstrapExpect != 0 {
		base.BootstrapExpect = a.config.BootstrapExpect
	}
	if a.config.Protocol > 0 {
		base.ProtocolVersion = uint8(a.config.Protocol)
	}
	if a.config.ACLToken != "" {
		base.ACLToken = a.config.ACLToken
	}
	if a.config.ACLMasterToken != "" {
		base.ACLMasterToken = a.config.ACLMasterToken
	}
	if a.config.ACLDatacenter != "" {
		base.ACLDatacenter = a.config.ACLDatacenter
	}
	if a.config.ACLTTLRaw != "" {
		base.ACLTTL = a.config.ACLTTL
	}
	if a.config.ACLDefaultPolicy != "" {
		base.ACLDefaultPolicy = a.config.ACLDefaultPolicy
	}
	if a.config.ACLDownPolicy != "" {
		base.ACLDownPolicy = a.config.ACLDownPolicy
	}
	if a.config.SessionTTLMinRaw != "" {
		base.SessionTTLMin = a.config.SessionTTLMin
	}

	// Format the build string
	revision := a.config.Revision
	if len(revision) > 8 {
		revision = revision[:8]
	}
	base.Build = fmt.Sprintf("%s%s:%s",
		a.config.Version, a.config.VersionPrerelease, revision)

	// Copy the TLS configuration
	base.VerifyIncoming = a.config.VerifyIncoming
	base.VerifyOutgoing = a.config.VerifyOutgoing
	base.VerifyServerHostname = a.config.VerifyServerHostname
	base.CAFile = a.config.CAFile
	base.CertFile = a.config.CertFile
	base.KeyFile = a.config.KeyFile
	base.ServerName = a.config.ServerName
	base.Domain = a.config.Domain

	// Setup the ServerUp callback
	base.ServerUp = a.state.ConsulServerUp

	// Setup the user event callback
	base.UserEventHandler = func(e serf.UserEvent) {
		select {
		case a.eventCh <- e:
		case <-a.shutdownCh:
		}
	}

	// Setup the loggers
	base.LogOutput = a.logOutput
	return base
}

// setupServer is used to initialize the Consul server
func (a *Agent) setupServer() error {
	config := a.consulConfig()

	if err := a.setupKeyrings(config); err != nil {
		return fmt.Errorf("Failed to configure keyring: %v", err)
	}

	server, err := consul.NewServer(config)
	if err != nil {
		return fmt.Errorf("Failed to start Consul server: %v", err)
	}
	a.server = server
	return nil
}

// setupClient is used to initialize the Consul client
func (a *Agent) setupClient() error {
	config := a.consulConfig()

	if err := a.setupKeyrings(config); err != nil {
		return fmt.Errorf("Failed to configure keyring: %v", err)
	}

	client, err := consul.NewClient(config)
	if err != nil {
		return fmt.Errorf("Failed to start Consul client: %v", err)
	}
	a.client = client
	return nil
}

// setupKeyrings is used to initialize and load keyrings during agent startup
func (a *Agent) setupKeyrings(config *consul.Config) error {
	fileLAN := filepath.Join(a.config.DataDir, serfLANKeyring)
	fileWAN := filepath.Join(a.config.DataDir, serfWANKeyring)

	if a.config.EncryptKey == "" {
		goto LOAD
	}
	if _, err := os.Stat(fileLAN); err != nil {
		if err := initKeyring(fileLAN, a.config.EncryptKey); err != nil {
			return err
		}
	}
	if a.config.Server {
		if _, err := os.Stat(fileWAN); err != nil {
			if err := initKeyring(fileWAN, a.config.EncryptKey); err != nil {
				return err
			}
		}
	}

LOAD:
	if _, err := os.Stat(fileLAN); err == nil {
		config.SerfLANConfig.KeyringFile = fileLAN
	}
	if err := loadKeyringFile(config.SerfLANConfig); err != nil {
		return err
	}
	if a.config.Server {
		if _, err := os.Stat(fileWAN); err == nil {
			config.SerfWANConfig.KeyringFile = fileWAN
		}
		if err := loadKeyringFile(config.SerfWANConfig); err != nil {
			return err
		}
	}

	// Success!
	return nil
}

// RPC is used to make an RPC call to the Consul servers
// This allows the agent to implement the Consul.Interface
func (a *Agent) RPC(method string, args interface{}, reply interface{}) error {
	if a.server != nil {
		return a.server.RPC(method, args, reply)
	}
	return a.client.RPC(method, args, reply)
}

// Leave is used to prepare the agent for a graceful shutdown
func (a *Agent) Leave() error {
	if a.server != nil {
		return a.server.Leave()
	} else {
		return a.client.Leave()
	}
}

// Shutdown is used to hard stop the agent. Should be
// preceded by a call to Leave to do it gracefully.
func (a *Agent) Shutdown() error {
	a.shutdownLock.Lock()
	defer a.shutdownLock.Unlock()

	if a.shutdown {
		return nil
	}

	// Stop all the checks
	a.checkLock.Lock()
	defer a.checkLock.Unlock()
	for _, chk := range a.checkMonitors {
		chk.Stop()
	}
	for _, chk := range a.checkTTLs {
		chk.Stop()
	}

	for _, chk := range a.checkHTTPs {
		chk.Stop()
	}

	for _, chk := range a.checkTCPs {
		chk.Stop()
	}

	a.logger.Println("[INFO] agent: requesting shutdown")
	var err error
	if a.server != nil {
		err = a.server.Shutdown()
	} else {
		err = a.client.Shutdown()
	}

	pidErr := a.deletePid()
	if pidErr != nil {
		a.logger.Println("[WARN] agent: could not delete pid file ", pidErr)
	}

	a.logger.Println("[INFO] agent: shutdown complete")
	a.shutdown = true
	close(a.shutdownCh)
	return err
}

// ShutdownCh is used to return a channel that can be
// selected to wait for the agent to perform a shutdown.
func (a *Agent) ShutdownCh() <-chan struct{} {
	return a.shutdownCh
}

// JoinLAN is used to have the agent join a LAN cluster
func (a *Agent) JoinLAN(addrs []string) (n int, err error) {
	a.logger.Printf("[INFO] agent: (LAN) joining: %v", addrs)
	if a.server != nil {
		n, err = a.server.JoinLAN(addrs)
	} else {
		n, err = a.client.JoinLAN(addrs)
	}
	a.logger.Printf("[INFO] agent: (LAN) joined: %d Err: %v", n, err)
	return
}

// JoinWAN is used to have the agent join a WAN cluster
func (a *Agent) JoinWAN(addrs []string) (n int, err error) {
	a.logger.Printf("[INFO] agent: (WAN) joining: %v", addrs)
	if a.server != nil {
		n, err = a.server.JoinWAN(addrs)
	} else {
		err = fmt.Errorf("Must be a server to join WAN cluster")
	}
	a.logger.Printf("[INFO] agent: (WAN) joined: %d Err: %v", n, err)
	return
}

// ForceLeave is used to remove a failed node from the cluster
func (a *Agent) ForceLeave(node string) (err error) {
	a.logger.Printf("[INFO] Force leaving node: %v", node)
	if a.server != nil {
		err = a.server.RemoveFailedNode(node)
	} else {
		err = a.client.RemoveFailedNode(node)
	}
	if err != nil {
		a.logger.Printf("[WARN] Failed to remove node: %v", err)
	}
	return err
}

// LocalMember is used to return the local node
func (a *Agent) LocalMember() serf.Member {
	if a.server != nil {
		return a.server.LocalMember()
	} else {
		return a.client.LocalMember()
	}
}

// LANMembers is used to retrieve the LAN members
func (a *Agent) LANMembers() []serf.Member {
	if a.server != nil {
		return a.server.LANMembers()
	} else {
		return a.client.LANMembers()
	}
}

// WANMembers is used to retrieve the WAN members
func (a *Agent) WANMembers() []serf.Member {
	if a.server != nil {
		return a.server.WANMembers()
	} else {
		return nil
	}
}

// StartSync is called once Services and Checks are registered.
// This is called to prevent a race between clients and the anti-entropy routines
func (a *Agent) StartSync() {
	// Start the anti entropy routine
	go a.state.antiEntropy(a.shutdownCh)
}

// PauseSync is used to pause anti-entropy while bulk changes are make
func (a *Agent) PauseSync() {
	a.state.Pause()
}

// ResumeSync is used to unpause anti-entropy after bulk changes are make
func (a *Agent) ResumeSync() {
	a.state.Resume()
}

// Returns the coordinate of this node in the local pool (assumes coordinates
// are enabled, so check that before calling).
func (a *Agent) GetCoordinate() (*coordinate.Coordinate, error) {
	if a.config.Server {
		return a.server.GetLANCoordinate()
	} else {
		return a.client.GetCoordinate()
	}
}

// sendCoordinate is a long-running loop that periodically sends our coordinate
// to the server. Closing the agent's shutdownChannel will cause this to exit.
func (a *Agent) sendCoordinate() {
	for {
		rate := a.config.SyncCoordinateRateTarget
		min := a.config.SyncCoordinateIntervalMin
		intv := lib.RateScaledInterval(rate, min, len(a.LANMembers()))
		intv = intv + lib.RandomStagger(intv)

		select {
		case <-time.After(intv):
			members := a.LANMembers()
			grok, err := consul.CanServersUnderstandProtocol(members, 3)
			if err != nil {
				a.logger.Printf("[ERR] agent: failed to check servers: %s", err)
				continue
			}
			if !grok {
				a.logger.Printf("[DEBUG] agent: skipping coordinate updates until servers are upgraded")
				continue
			}

			c, err := a.GetCoordinate()
			if err != nil {
				a.logger.Printf("[ERR] agent: failed to get coordinate: %s", err)
				continue
			}

			// TODO - Consider adding a distance check so we don't send
			// an update if the position hasn't changed by more than a
			// threshold.
			req := structs.CoordinateUpdateRequest{
				Datacenter:   a.config.Datacenter,
				Node:         a.config.NodeName,
				Coord:        c,
				WriteRequest: structs.WriteRequest{Token: a.config.ACLToken},
			}
			var reply struct{}
			if err := a.RPC("Coordinate.Update", &req, &reply); err != nil {
				a.logger.Printf("[ERR] agent: coordinate update error: %s", err)
				continue
			}
		case <-a.shutdownCh:
			return
		}
	}
}

// persistService saves a service definition to a JSON file in the data dir
func (a *Agent) persistService(service *structs.NodeService) error {
	svcPath := filepath.Join(a.config.DataDir, servicesDir, stringHash(service.ID))
	wrapped := persistedService{
		Token:   a.state.ServiceToken(service.ID),
		Service: service,
	}
	encoded, err := json.Marshal(wrapped)
	if err != nil {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(svcPath), 0700); err != nil {
		return err
	}
	fh, err := os.OpenFile(svcPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0600)
	if err != nil {
		return err
	}
	defer fh.Close()
	if _, err := fh.Write(encoded); err != nil {
		return err
	}
	return nil
}

// purgeService removes a persisted service definition file from the data dir
func (a *Agent) purgeService(serviceID string) error {
	svcPath := filepath.Join(a.config.DataDir, servicesDir, stringHash(serviceID))
	if _, err := os.Stat(svcPath); err == nil {
		return os.Remove(svcPath)
	}
	return nil
}

// persistCheck saves a check definition to the local agent's state directory
func (a *Agent) persistCheck(check *structs.HealthCheck, chkType *CheckType) error {
	checkPath := filepath.Join(a.config.DataDir, checksDir, stringHash(check.CheckID))

	// Create the persisted check
	wrapped := persistedCheck{
		Check:   check,
		ChkType: chkType,
		Token:   a.state.CheckToken(check.CheckID),
	}

	encoded, err := json.Marshal(wrapped)
	if err != nil {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(checkPath), 0700); err != nil {
		return err
	}
	fh, err := os.OpenFile(checkPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0600)
	if err != nil {
		return err
	}
	defer fh.Close()
	if _, err := fh.Write(encoded); err != nil {
		return err
	}
	return nil
}

// purgeCheck removes a persisted check definition file from the data dir
func (a *Agent) purgeCheck(checkID string) error {
	checkPath := filepath.Join(a.config.DataDir, checksDir, stringHash(checkID))
	if _, err := os.Stat(checkPath); err == nil {
		return os.Remove(checkPath)
	}
	return nil
}

// AddService is used to add a service entry.
// This entry is persistent and the agent will make a best effort to
// ensure it is registered
func (a *Agent) AddService(service *structs.NodeService, chkTypes CheckTypes, persist bool, token string) error {
	if service.Service == "" {
		return fmt.Errorf("Service name missing")
	}
	if service.ID == "" && service.Service != "" {
		service.ID = service.Service
	}
	for _, check := range chkTypes {
		if !check.Valid() {
			return fmt.Errorf("Check type is not valid")
		}
	}

	// Warn if the service name is incompatible with DNS
	if !dnsNameRe.MatchString(service.Service) {
		a.logger.Printf("[WARN] Service name %q will not be discoverable "+
			"via DNS due to invalid characters. Valid characters include "+
			"all alpha-numerics and dashes.", service.Service)
	}

	// Warn if any tags are incompatible with DNS
	for _, tag := range service.Tags {
		if !dnsNameRe.MatchString(tag) {
			a.logger.Printf("[WARN] Service tag %q will not be discoverable "+
				"via DNS due to invalid characters. Valid characters include "+
				"all alpha-numerics and dashes.", tag)
		}
	}

	// Pause the service syncs during modification
	a.PauseSync()
	defer a.ResumeSync()

	// Take a snapshot of the current state of checks (if any), and
	// restore them before resuming anti-entropy.
	snap := a.snapshotCheckState()
	defer a.restoreCheckState(snap)

	// Add the service
	a.state.AddService(service, token)

	// Persist the service to a file
	if persist && !a.config.DevMode {
		if err := a.persistService(service); err != nil {
			return err
		}
	}

	// Create an associated health check
	for i, chkType := range chkTypes {
		checkID := fmt.Sprintf("service:%s", service.ID)
		if len(chkTypes) > 1 {
			checkID += fmt.Sprintf(":%d", i+1)
		}
		check := &structs.HealthCheck{
			Node:        a.config.NodeName,
			CheckID:     checkID,
			Name:        fmt.Sprintf("Service '%s' check", service.Service),
			Status:      structs.HealthCritical,
			Notes:       chkType.Notes,
			ServiceID:   service.ID,
			ServiceName: service.Service,
		}
		if chkType.Status != "" {
			check.Status = chkType.Status
		}
		if err := a.AddCheck(check, chkType, persist, token); err != nil {
			return err
		}
	}
	return nil
}

// RemoveService is used to remove a service entry.
// The agent will make a best effort to ensure it is deregistered
func (a *Agent) RemoveService(serviceID string, persist bool) error {
	// Protect "consul" service from deletion by a user
	if a.server != nil && serviceID == consul.ConsulServiceID {
		return fmt.Errorf(
			"Deregistering the %s service is not allowed",
			consul.ConsulServiceID)
	}

	// Validate ServiceID
	if serviceID == "" {
		return fmt.Errorf("ServiceID missing")
	}

	// Remove service immediately
	a.state.RemoveService(serviceID)

	// Remove the service from the data dir
	if persist {
		if err := a.purgeService(serviceID); err != nil {
			return err
		}
	}

	// Deregister any associated health checks
	for checkID, health := range a.state.Checks() {
		if health.ServiceID != serviceID {
			continue
		}
		if err := a.RemoveCheck(checkID, persist); err != nil {
			return err
		}
	}

	log.Printf("[DEBUG] agent: removed service %q", serviceID)
	return nil
}

// AddCheck is used to add a health check to the agent.
// This entry is persistent and the agent will make a best effort to
// ensure it is registered. The Check may include a CheckType which
// is used to automatically update the check status
func (a *Agent) AddCheck(check *structs.HealthCheck, chkType *CheckType, persist bool, token string) error {
	if check.CheckID == "" {
		return fmt.Errorf("CheckID missing")
	}
	if chkType != nil && !chkType.Valid() {
		return fmt.Errorf("Check type is not valid")
	}

	if check.ServiceID != "" {
		svc, ok := a.state.Services()[check.ServiceID]
		if !ok {
			return fmt.Errorf("ServiceID %q does not exist", check.ServiceID)
		}
		check.ServiceName = svc.Service
	}

	a.checkLock.Lock()
	defer a.checkLock.Unlock()

	// Check if already registered
	if chkType != nil {
		if chkType.IsTTL() {
			if existing, ok := a.checkTTLs[check.CheckID]; ok {
				existing.Stop()
			}

			ttl := &CheckTTL{
				Notify:  &a.state,
				CheckID: check.CheckID,
				TTL:     chkType.TTL,
				Logger:  a.logger,
			}

			// Restore persisted state, if any
			if err := a.loadCheckState(check); err != nil {
				a.logger.Printf("[WARN] agent: failed restoring state for check %q: %s",
					check.CheckID, err)
			}

			ttl.Start()
			a.checkTTLs[check.CheckID] = ttl

		} else if chkType.IsHTTP() {
			if existing, ok := a.checkHTTPs[check.CheckID]; ok {
				existing.Stop()
			}
			if chkType.Interval < MinInterval {
				a.logger.Println(fmt.Sprintf("[WARN] agent: check '%s' has interval below minimum of %v",
					check.CheckID, MinInterval))
				chkType.Interval = MinInterval
			}

			http := &CheckHTTP{
				Notify:   &a.state,
				CheckID:  check.CheckID,
				HTTP:     chkType.HTTP,
				Interval: chkType.Interval,
				Timeout:  chkType.Timeout,
				Logger:   a.logger,
			}
			http.Start()
			a.checkHTTPs[check.CheckID] = http

		} else if chkType.IsTCP() {
			if existing, ok := a.checkTCPs[check.CheckID]; ok {
				existing.Stop()
			}
			if chkType.Interval < MinInterval {
				a.logger.Println(fmt.Sprintf("[WARN] agent: check '%s' has interval below minimum of %v",
					check.CheckID, MinInterval))
				chkType.Interval = MinInterval
			}

			tcp := &CheckTCP{
				Notify:   &a.state,
				CheckID:  check.CheckID,
				TCP:      chkType.TCP,
				Interval: chkType.Interval,
				Timeout:  chkType.Timeout,
				Logger:   a.logger,
			}
			tcp.Start()
			a.checkTCPs[check.CheckID] = tcp

		} else if chkType.IsDocker() {
			if existing, ok := a.checkDockers[check.CheckID]; ok {
				existing.Stop()
			}
			if chkType.Interval < MinInterval {
				a.logger.Println(fmt.Sprintf("[WARN] agent: check '%s' has interval below minimum of %v",
					check.CheckID, MinInterval))
				chkType.Interval = MinInterval
			}

			dockerCheck := &CheckDocker{
				Notify:            &a.state,
				CheckID:           check.CheckID,
				DockerContainerID: chkType.DockerContainerID,
				Shell:             chkType.Shell,
				Script:            chkType.Script,
				Interval:          chkType.Interval,
				Logger:            a.logger,
			}
			if err := dockerCheck.Init(); err != nil {
				return err
			}
			dockerCheck.Start()
			a.checkDockers[check.CheckID] = dockerCheck
		} else if chkType.IsMonitor() {
			if existing, ok := a.checkMonitors[check.CheckID]; ok {
				existing.Stop()
			}
			if chkType.Interval < MinInterval {
				a.logger.Println(fmt.Sprintf("[WARN] agent: check '%s' has interval below minimum of %v",
					check.CheckID, MinInterval))
				chkType.Interval = MinInterval
			}

			monitor := &CheckMonitor{
				Notify:   &a.state,
				CheckID:  check.CheckID,
				Script:   chkType.Script,
				Interval: chkType.Interval,
				Logger:   a.logger,
				ReapLock: &a.reapLock,
			}
			monitor.Start()
			a.checkMonitors[check.CheckID] = monitor
		} else {
			return fmt.Errorf("Check type is not valid")
		}
	}

	// Add to the local state for anti-entropy
	a.state.AddCheck(check, token)

	// Persist the check
	if persist && !a.config.DevMode {
		return a.persistCheck(check, chkType)
	}

	return nil
}

// RemoveCheck is used to remove a health check.
// The agent will make a best effort to ensure it is deregistered
func (a *Agent) RemoveCheck(checkID string, persist bool) error {
	// Validate CheckID
	if checkID == "" {
		return fmt.Errorf("CheckID missing")
	}

	// Add to the local state for anti-entropy
	a.state.RemoveCheck(checkID)

	a.checkLock.Lock()
	defer a.checkLock.Unlock()

	// Stop any monitors
	if check, ok := a.checkMonitors[checkID]; ok {
		check.Stop()
		delete(a.checkMonitors, checkID)
	}
	if check, ok := a.checkHTTPs[checkID]; ok {
		check.Stop()
		delete(a.checkHTTPs, checkID)
	}
	if check, ok := a.checkTCPs[checkID]; ok {
		check.Stop()
		delete(a.checkTCPs, checkID)
	}
	if check, ok := a.checkTTLs[checkID]; ok {
		check.Stop()
		delete(a.checkTTLs, checkID)
	}
	if persist {
		if err := a.purgeCheck(checkID); err != nil {
			return err
		}
		if err := a.purgeCheckState(checkID); err != nil {
			return err
		}
	}
	log.Printf("[DEBUG] agent: removed check %q", checkID)
	return nil
}

// UpdateCheck is used to update the status of a check.
// This can only be used with checks of the TTL type.
func (a *Agent) UpdateCheck(checkID, status, output string) error {
	a.checkLock.Lock()
	defer a.checkLock.Unlock()

	check, ok := a.checkTTLs[checkID]
	if !ok {
		return fmt.Errorf("CheckID does not have associated TTL")
	}

	// Set the status through CheckTTL to reset the TTL
	check.SetStatus(status, output)

	if a.config.DevMode {
		return nil
	}

	// Always persist the state for TTL checks
	if err := a.persistCheckState(check, status, output); err != nil {
		return fmt.Errorf("failed persisting state for check %q: %s", checkID, err)
	}

	return nil
}

// persistCheckState is used to record the check status into the data dir.
// This allows the state to be restored on a later agent start. Currently
// only useful for TTL based checks.
func (a *Agent) persistCheckState(check *CheckTTL, status, output string) error {
	// Create the persisted state
	state := persistedCheckState{
		CheckID: check.CheckID,
		Status:  status,
		Output:  output,
		Expires: time.Now().Add(check.TTL).Unix(),
	}

	// Encode the state
	buf, err := json.Marshal(state)
	if err != nil {
		return err
	}

	// Create the state dir if it doesn't exist
	dir := filepath.Join(a.config.DataDir, checkStateDir)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return fmt.Errorf("failed creating check state dir %q: %s", dir, err)
	}

	// Write the state to the file
	file := filepath.Join(dir, stringHash(check.CheckID))
	if err := ioutil.WriteFile(file, buf, 0600); err != nil {
		return fmt.Errorf("failed writing file %q: %s", file, err)
	}

	return nil
}

// loadCheckState is used to restore the persisted state of a check.
func (a *Agent) loadCheckState(check *structs.HealthCheck) error {
	// Try to read the persisted state for this check
	file := filepath.Join(a.config.DataDir, checkStateDir, stringHash(check.CheckID))
	buf, err := ioutil.ReadFile(file)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("failed reading file %q: %s", file, err)
	}

	// Decode the state data
	var p persistedCheckState
	if err := json.Unmarshal(buf, &p); err != nil {
		return fmt.Errorf("failed decoding check state: %s", err)
	}

	// Check if the state has expired
	if time.Now().Unix() >= p.Expires {
		a.logger.Printf("[DEBUG] agent: check state expired for %q, not restoring", check.CheckID)
		return a.purgeCheckState(check.CheckID)
	}

	// Restore the fields from the state
	check.Output = p.Output
	check.Status = p.Status
	return nil
}

// purgeCheckState is used to purge the state of a check from the data dir
func (a *Agent) purgeCheckState(checkID string) error {
	file := filepath.Join(a.config.DataDir, checkStateDir, stringHash(checkID))
	err := os.Remove(file)
	if os.IsNotExist(err) {
		return nil
	}
	return err
}

// Stats is used to get various debugging state from the sub-systems
func (a *Agent) Stats() map[string]map[string]string {
	toString := func(v uint64) string {
		return strconv.FormatUint(v, 10)
	}
	var stats map[string]map[string]string
	if a.server != nil {
		stats = a.server.Stats()
	} else {
		stats = a.client.Stats()
	}
	stats["agent"] = map[string]string{
		"check_monitors": toString(uint64(len(a.checkMonitors))),
		"check_ttls":     toString(uint64(len(a.checkTTLs))),
		"checks":         toString(uint64(len(a.state.checks))),
		"services":       toString(uint64(len(a.state.services))),
	}

	revision := a.config.Revision
	if len(revision) > 8 {
		revision = revision[:8]
	}
	stats["build"] = map[string]string{
		"revision":   revision,
		"version":    a.config.Version,
		"prerelease": a.config.VersionPrerelease,
	}
	return stats
}

// storePid is used to write out our PID to a file if necessary
func (a *Agent) storePid() error {
	// Quit fast if no pidfile
	pidPath := a.config.PidFile
	if pidPath == "" {
		return nil
	}

	// Open the PID file
	pidFile, err := os.OpenFile(pidPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0666)
	if err != nil {
		return fmt.Errorf("Could not open pid file: %v", err)
	}
	defer pidFile.Close()

	// Write out the PID
	pid := os.Getpid()
	_, err = pidFile.WriteString(fmt.Sprintf("%d", pid))
	if err != nil {
		return fmt.Errorf("Could not write to pid file: %s", err)
	}
	return nil
}

// deletePid is used to delete our PID on exit
func (a *Agent) deletePid() error {
	// Quit fast if no pidfile
	pidPath := a.config.PidFile
	if pidPath == "" {
		return nil
	}

	stat, err := os.Stat(pidPath)
	if err != nil {
		return fmt.Errorf("Could not remove pid file: %s", err)
	}

	if stat.IsDir() {
		return fmt.Errorf("Specified pid file path is directory")
	}

	err = os.Remove(pidPath)
	if err != nil {
		return fmt.Errorf("Could not remove pid file: %s", err)
	}
	return nil
}

// loadServices will load service definitions from configuration and persisted
// definitions on disk, and load them into the local agent.
func (a *Agent) loadServices(conf *Config) error {
	// Register the services from config
	for _, service := range conf.Services {
		ns := service.NodeService()
		chkTypes := service.CheckTypes()
		if err := a.AddService(ns, chkTypes, false, service.Token); err != nil {
			return fmt.Errorf("Failed to register service '%s': %v", service.ID, err)
		}
	}

	// Load any persisted services
	svcDir := filepath.Join(a.config.DataDir, servicesDir)
	files, err := ioutil.ReadDir(svcDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("Failed reading services dir %q: %s", svcDir, err)
	}
	for _, fi := range files {
		// Skip all dirs
		if fi.IsDir() {
			continue
		}

		// Open the file for reading
		file := filepath.Join(svcDir, fi.Name())
		fh, err := os.Open(file)
		if err != nil {
			return fmt.Errorf("failed opening service file %q: %s", file, err)
		}

		// Read the contents into a buffer
		buf, err := ioutil.ReadAll(fh)
		fh.Close()
		if err != nil {
			return fmt.Errorf("failed reading service file %q: %s", file, err)
		}

		// Try decoding the service definition
		var p persistedService
		if err := json.Unmarshal(buf, &p); err != nil {
			// Backwards-compatibility for pre-0.5.1 persisted services
			if err := json.Unmarshal(buf, &p.Service); err != nil {
				return fmt.Errorf("failed decoding service file %q: %s", file, err)
			}
		}
		serviceID := p.Service.ID

		if _, ok := a.state.services[serviceID]; ok {
			// Purge previously persisted service. This allows config to be
			// preferred over services persisted from the API.
			a.logger.Printf("[DEBUG] agent: service %q exists, not restoring from %q",
				serviceID, file)
			if err := a.purgeService(serviceID); err != nil {
				return fmt.Errorf("failed purging service %q: %s", serviceID, err)
			}
		} else {
			a.logger.Printf("[DEBUG] agent: restored service definition %q from %q",
				serviceID, file)
			if err := a.AddService(p.Service, nil, false, p.Token); err != nil {
				return fmt.Errorf("failed adding service %q: %s", serviceID, err)
			}
		}
	}

	return nil
}

// unloadServices will deregister all services other than the 'consul' service
// known to the local agent.
func (a *Agent) unloadServices() error {
	for _, service := range a.state.Services() {
		if service.ID == consul.ConsulServiceID {
			continue
		}
		if err := a.RemoveService(service.ID, false); err != nil {
			return fmt.Errorf("Failed deregistering service '%s': %v", service.ID, err)
		}
	}

	return nil
}

// loadChecks loads check definitions and/or persisted check definitions from
// disk and re-registers them with the local agent.
func (a *Agent) loadChecks(conf *Config) error {
	// Register the checks from config
	for _, check := range conf.Checks {
		health := check.HealthCheck(conf.NodeName)
		chkType := &check.CheckType
		if err := a.AddCheck(health, chkType, false, check.Token); err != nil {
			return fmt.Errorf("Failed to register check '%s': %v %v", check.Name, err, check)
		}
	}

	// Load any persisted checks
	checkDir := filepath.Join(a.config.DataDir, checksDir)
	files, err := ioutil.ReadDir(checkDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("Failed reading checks dir %q: %s", checkDir, err)
	}
	for _, fi := range files {
		// Ignore dirs - we only care about the check definition files
		if fi.IsDir() {
			continue
		}

		// Open the file for reading
		file := filepath.Join(checkDir, fi.Name())
		fh, err := os.Open(file)
		if err != nil {
			return fmt.Errorf("Failed opening check file %q: %s", file, err)
		}

		// Read the contents into a buffer
		buf, err := ioutil.ReadAll(fh)
		fh.Close()
		if err != nil {
			return fmt.Errorf("failed reading check file %q: %s", file, err)
		}

		// Decode the check
		var p persistedCheck
		if err := json.Unmarshal(buf, &p); err != nil {
			return fmt.Errorf("Failed decoding check file %q: %s", file, err)
		}
		checkID := p.Check.CheckID

		if _, ok := a.state.checks[checkID]; ok {
			// Purge previously persisted check. This allows config to be
			// preferred over persisted checks from the API.
			a.logger.Printf("[DEBUG] agent: check %q exists, not restoring from %q",
				checkID, file)
			if err := a.purgeCheck(checkID); err != nil {
				return fmt.Errorf("Failed purging check %q: %s", checkID, err)
			}
		} else {
			// Default check to critical to avoid placing potentially unhealthy
			// services into the active pool
			p.Check.Status = structs.HealthCritical

			if err := a.AddCheck(p.Check, p.ChkType, false, p.Token); err != nil {
				// Purge the check if it is unable to be restored.
				a.logger.Printf("[WARN] agent: Failed to restore check %q: %s",
					checkID, err)
				if err := a.purgeCheck(checkID); err != nil {
					return fmt.Errorf("Failed purging check %q: %s", checkID, err)
				}
			}
			a.logger.Printf("[DEBUG] agent: restored health check %q from %q",
				p.Check.CheckID, file)
		}
	}

	return nil
}

// unloadChecks will deregister all checks known to the local agent.
func (a *Agent) unloadChecks() error {
	for _, check := range a.state.Checks() {
		if err := a.RemoveCheck(check.CheckID, false); err != nil {
			return fmt.Errorf("Failed deregistering check '%s': %s", check.CheckID, err)
		}
	}

	return nil
}

// snapshotCheckState is used to snapshot the current state of the health
// checks. This is done before we reload our checks, so that we can properly
// restore into the same state.
func (a *Agent) snapshotCheckState() map[string]*structs.HealthCheck {
	return a.state.Checks()
}

// restoreCheckState is used to reset the health state based on a snapshot.
// This is done after we finish the reload to avoid any unnecessary flaps
// in health state and potential session invalidations.
func (a *Agent) restoreCheckState(snap map[string]*structs.HealthCheck) {
	for id, check := range snap {
		a.state.UpdateCheck(id, check.Status, check.Output)
	}
}

// serviceMaintCheckID returns the ID of a given service's maintenance check
func serviceMaintCheckID(serviceID string) string {
	return fmt.Sprintf("%s:%s", serviceMaintCheckPrefix, serviceID)
}

// EnableServiceMaintenance will register a false health check against the given
// service ID with critical status. This will exclude the service from queries.
func (a *Agent) EnableServiceMaintenance(serviceID, reason, token string) error {
	service, ok := a.state.Services()[serviceID]
	if !ok {
		return fmt.Errorf("No service registered with ID %q", serviceID)
	}

	// Check if maintenance mode is not already enabled
	checkID := serviceMaintCheckID(serviceID)
	if _, ok := a.state.Checks()[checkID]; ok {
		return nil
	}

	// Use default notes if no reason provided
	if reason == "" {
		reason = defaultServiceMaintReason
	}

	// Create and register the critical health check
	check := &structs.HealthCheck{
		Node:        a.config.NodeName,
		CheckID:     checkID,
		Name:        "Service Maintenance Mode",
		Notes:       reason,
		ServiceID:   service.ID,
		ServiceName: service.Service,
		Status:      structs.HealthCritical,
	}
	a.AddCheck(check, nil, true, token)
	a.logger.Printf("[INFO] agent: Service %q entered maintenance mode", serviceID)

	return nil
}

// DisableServiceMaintenance will deregister the fake maintenance mode check
// if the service has been marked as in maintenance.
func (a *Agent) DisableServiceMaintenance(serviceID string) error {
	if _, ok := a.state.Services()[serviceID]; !ok {
		return fmt.Errorf("No service registered with ID %q", serviceID)
	}

	// Check if maintenance mode is enabled
	checkID := serviceMaintCheckID(serviceID)
	if _, ok := a.state.Checks()[checkID]; !ok {
		return nil
	}

	// Deregister the maintenance check
	a.RemoveCheck(checkID, true)
	a.logger.Printf("[INFO] agent: Service %q left maintenance mode", serviceID)

	return nil
}

// EnableNodeMaintenance places a node into maintenance mode.
func (a *Agent) EnableNodeMaintenance(reason, token string) {
	// Ensure node maintenance is not already enabled
	if _, ok := a.state.Checks()[nodeMaintCheckID]; ok {
		return
	}

	// Use a default notes value
	if reason == "" {
		reason = defaultNodeMaintReason
	}

	// Create and register the node maintenance check
	check := &structs.HealthCheck{
		Node:    a.config.NodeName,
		CheckID: nodeMaintCheckID,
		Name:    "Node Maintenance Mode",
		Notes:   reason,
		Status:  structs.HealthCritical,
	}
	a.AddCheck(check, nil, true, token)
	a.logger.Printf("[INFO] agent: Node entered maintenance mode")
}

// DisableNodeMaintenance removes a node from maintenance mode
func (a *Agent) DisableNodeMaintenance() {
	if _, ok := a.state.Checks()[nodeMaintCheckID]; !ok {
		return
	}
	a.RemoveCheck(nodeMaintCheckID, true)
	a.logger.Printf("[INFO] agent: Node left maintenance mode")
}

// InjectEndpoint overrides the given endpoint with a substitute one. Note
// that not all agent methods use this mechanism, and that is should only
// be used for testing.
func (a *Agent) InjectEndpoint(endpoint string, handler interface{}) error {
	if a.server == nil {
		return fmt.Errorf("agent must be a server")
	}

	if err := a.server.InjectEndpoint(handler); err != nil {
		return err
	}
	name := reflect.Indirect(reflect.ValueOf(handler)).Type().Name()
	a.endpoints[endpoint] = name

	a.logger.Printf("[WARN] agent: endpoint injected; this should only be used for testing")
	return nil
}

// getEndpoint returns the endpoint name to use for the given endpoint,
// which may be overridden.
func (a *Agent) getEndpoint(endpoint string) string {
	if override, ok := a.endpoints[endpoint]; ok {
		return override
	}
	return endpoint
}
