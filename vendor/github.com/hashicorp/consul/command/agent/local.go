package agent

import (
	"fmt"
	"log"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hashicorp/consul/consul"
	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/lib"
)

const (
	syncStaggerIntv = 3 * time.Second
	syncRetryIntv   = 15 * time.Second

	// permissionDenied is returned when an ACL based rejection happens
	permissionDenied = "Permission denied"
)

// syncStatus is used to represent the difference between
// the local and remote state, and if action needs to be taken
type syncStatus struct {
	remoteDelete bool // Should this be deleted from the server
	inSync       bool // Is this in sync with the server
}

// localState is used to represent the node's services,
// and checks. We used it to perform anti-entropy with the
// catalog representation
type localState struct {
	// paused is used to check if we are paused. Must be the first
	// element due to a go bug.
	paused int32

	sync.RWMutex
	logger *log.Logger

	// Config is the agent config
	config *Config

	// iface is the consul interface to use for keeping in sync
	iface consul.Interface

	// nodeInfoInSync tracks whether the server has our correct top-level
	// node information in sync (currently only used for tagged addresses)
	nodeInfoInSync bool

	// Services tracks the local services
	services      map[string]*structs.NodeService
	serviceStatus map[string]syncStatus
	serviceTokens map[string]string

	// Checks tracks the local checks
	checks      map[string]*structs.HealthCheck
	checkStatus map[string]syncStatus
	checkTokens map[string]string

	// Used to track checks that are being deferred
	deferCheck map[string]*time.Timer

	// consulCh is used to inform of a change to the known
	// consul nodes. This may be used to retry a sync run
	consulCh chan struct{}

	// triggerCh is used to inform of a change to local state
	// that requires anti-entropy with the server
	triggerCh chan struct{}
}

// Init is used to initialize the local state
func (l *localState) Init(config *Config, logger *log.Logger) {
	l.config = config
	l.logger = logger
	l.services = make(map[string]*structs.NodeService)
	l.serviceStatus = make(map[string]syncStatus)
	l.serviceTokens = make(map[string]string)
	l.checks = make(map[string]*structs.HealthCheck)
	l.checkStatus = make(map[string]syncStatus)
	l.checkTokens = make(map[string]string)
	l.deferCheck = make(map[string]*time.Timer)
	l.consulCh = make(chan struct{}, 1)
	l.triggerCh = make(chan struct{}, 1)
}

// SetIface is used to set the Consul interface. Must be set prior to
// starting anti-entropy
func (l *localState) SetIface(iface consul.Interface) {
	l.iface = iface
}

// changeMade is used to trigger an anti-entropy run
func (l *localState) changeMade() {
	select {
	case l.triggerCh <- struct{}{}:
	default:
	}
}

// ConsulServerUp is used to inform that a new consul server is now
// up. This can be used to speed up the sync process if we are blocking
// waiting to discover a consul server
func (l *localState) ConsulServerUp() {
	select {
	case l.consulCh <- struct{}{}:
	default:
	}
}

// Pause is used to pause state synchronization, this can be
// used to make batch changes
func (l *localState) Pause() {
	atomic.AddInt32(&l.paused, 1)
}

// Resume is used to resume state synchronization
func (l *localState) Resume() {
	paused := atomic.AddInt32(&l.paused, -1)
	if paused < 0 {
		panic("unbalanced localState.Resume() detected")
	}
	l.changeMade()
}

// isPaused is used to check if we are paused
func (l *localState) isPaused() bool {
	return atomic.LoadInt32(&l.paused) > 0
}

// ServiceToken returns the configured ACL token for the given
// service ID. If none is present, the agent's token is returned.
func (l *localState) ServiceToken(id string) string {
	l.RLock()
	defer l.RUnlock()
	return l.serviceToken(id)
}

// serviceToken returns an ACL token associated with a service.
func (l *localState) serviceToken(id string) string {
	token := l.serviceTokens[id]
	if token == "" {
		token = l.config.ACLToken
	}
	return token
}

// AddService is used to add a service entry to the local state.
// This entry is persistent and the agent will make a best effort to
// ensure it is registered
func (l *localState) AddService(service *structs.NodeService, token string) {
	// Assign the ID if none given
	if service.ID == "" && service.Service != "" {
		service.ID = service.Service
	}

	l.Lock()
	defer l.Unlock()

	l.services[service.ID] = service
	l.serviceStatus[service.ID] = syncStatus{}
	l.serviceTokens[service.ID] = token
	l.changeMade()
}

// RemoveService is used to remove a service entry from the local state.
// The agent will make a best effort to ensure it is deregistered
func (l *localState) RemoveService(serviceID string) {
	l.Lock()
	defer l.Unlock()

	delete(l.services, serviceID)
	delete(l.serviceTokens, serviceID)
	l.serviceStatus[serviceID] = syncStatus{remoteDelete: true}
	l.changeMade()
}

// Services returns the locally registered services that the
// agent is aware of and are being kept in sync with the server
func (l *localState) Services() map[string]*structs.NodeService {
	services := make(map[string]*structs.NodeService)
	l.RLock()
	defer l.RUnlock()

	for name, serv := range l.services {
		services[name] = serv
	}
	return services
}

// CheckToken is used to return the configured health check token, or
// if none is configured, the default agent ACL token.
func (l *localState) CheckToken(id string) string {
	l.RLock()
	defer l.RUnlock()
	return l.checkToken(id)
}

// checkToken returns an ACL token associated with a check.
func (l *localState) checkToken(id string) string {
	token := l.checkTokens[id]
	if token == "" {
		token = l.config.ACLToken
	}
	return token
}

// AddCheck is used to add a health check to the local state.
// This entry is persistent and the agent will make a best effort to
// ensure it is registered
func (l *localState) AddCheck(check *structs.HealthCheck, token string) {
	// Set the node name
	check.Node = l.config.NodeName

	l.Lock()
	defer l.Unlock()

	l.checks[check.CheckID] = check
	l.checkStatus[check.CheckID] = syncStatus{}
	l.checkTokens[check.CheckID] = token
	l.changeMade()
}

// RemoveCheck is used to remove a health check from the local state.
// The agent will make a best effort to ensure it is deregistered
func (l *localState) RemoveCheck(checkID string) {
	l.Lock()
	defer l.Unlock()

	delete(l.checks, checkID)
	delete(l.checkTokens, checkID)
	l.checkStatus[checkID] = syncStatus{remoteDelete: true}
	l.changeMade()
}

// UpdateCheck is used to update the status of a check
func (l *localState) UpdateCheck(checkID, status, output string) {
	l.Lock()
	defer l.Unlock()

	check, ok := l.checks[checkID]
	if !ok {
		return
	}

	// Do nothing if update is idempotent
	if check.Status == status && check.Output == output {
		return
	}

	// Defer a sync if the output has changed. This is an optimization around
	// frequent updates of output. Instead, we update the output internally,
	// and periodically do a write-back to the servers. If there is a status
	// change we do the write immediately.
	if l.config.CheckUpdateInterval > 0 && check.Status == status {
		check.Output = output
		if _, ok := l.deferCheck[checkID]; !ok {
			intv := time.Duration(uint64(l.config.CheckUpdateInterval)/2) + lib.RandomStagger(l.config.CheckUpdateInterval)
			deferSync := time.AfterFunc(intv, func() {
				l.Lock()
				if _, ok := l.checkStatus[checkID]; ok {
					l.checkStatus[checkID] = syncStatus{inSync: false}
					l.changeMade()
				}
				delete(l.deferCheck, checkID)
				l.Unlock()
			})
			l.deferCheck[checkID] = deferSync
		}
		return
	}

	// Update status and mark out of sync
	check.Status = status
	check.Output = output
	l.checkStatus[checkID] = syncStatus{inSync: false}
	l.changeMade()
}

// Checks returns the locally registered checks that the
// agent is aware of and are being kept in sync with the server
func (l *localState) Checks() map[string]*structs.HealthCheck {
	checks := make(map[string]*structs.HealthCheck)
	l.RLock()
	defer l.RUnlock()

	for name, check := range l.checks {
		checks[name] = check
	}
	return checks
}

// antiEntropy is a long running method used to perform anti-entropy
// between local and remote state.
func (l *localState) antiEntropy(shutdownCh chan struct{}) {
SYNC:
	// Sync our state with the servers
	for {
		err := l.setSyncState()
		if err == nil {
			break
		}
		l.logger.Printf("[ERR] agent: failed to sync remote state: %v", err)
		select {
		case <-l.consulCh:
			// Stagger the retry on leader election, avoid a thundering heard
			select {
			case <-time.After(lib.RandomStagger(aeScale(syncStaggerIntv, len(l.iface.LANMembers())))):
			case <-shutdownCh:
				return
			}
		case <-time.After(syncRetryIntv + lib.RandomStagger(aeScale(syncRetryIntv, len(l.iface.LANMembers())))):
		case <-shutdownCh:
			return
		}
	}

	// Force-trigger AE to pickup any changes
	l.changeMade()

	// Schedule the next full sync, with a random stagger
	aeIntv := aeScale(l.config.AEInterval, len(l.iface.LANMembers()))
	aeIntv = aeIntv + lib.RandomStagger(aeIntv)
	aeTimer := time.After(aeIntv)

	// Wait for sync events
	for {
		select {
		case <-aeTimer:
			goto SYNC
		case <-l.triggerCh:
			// Skip the sync if we are paused
			if l.isPaused() {
				continue
			}
			if err := l.syncChanges(); err != nil {
				l.logger.Printf("[ERR] agent: failed to sync changes: %v", err)
			}
		case <-shutdownCh:
			return
		}
	}
}

// setSyncState does a read of the server state, and updates
// the local syncStatus as appropriate
func (l *localState) setSyncState() error {
	req := structs.NodeSpecificRequest{
		Datacenter:   l.config.Datacenter,
		Node:         l.config.NodeName,
		QueryOptions: structs.QueryOptions{Token: l.config.ACLToken},
	}
	var out1 structs.IndexedNodeServices
	var out2 structs.IndexedHealthChecks
	if e := l.iface.RPC("Catalog.NodeServices", &req, &out1); e != nil {
		return e
	}
	if err := l.iface.RPC("Health.NodeChecks", &req, &out2); err != nil {
		return err
	}
	checks := out2.HealthChecks

	l.Lock()
	defer l.Unlock()

	// Check the node info (currently limited to tagged addresses since
	// everything else is managed by the Serf layer)
	if out1.NodeServices == nil || out1.NodeServices.Node == nil ||
		!reflect.DeepEqual(out1.NodeServices.Node.TaggedAddresses, l.config.TaggedAddresses) {
		l.nodeInfoInSync = false
	}

	// Check all our services
	services := make(map[string]*structs.NodeService)
	if out1.NodeServices != nil {
		services = out1.NodeServices.Services
	}

	for id, _ := range l.services {
		// If the local service doesn't exist remotely, then sync it
		if _, ok := services[id]; !ok {
			l.serviceStatus[id] = syncStatus{inSync: false}
		}
	}

	for id, service := range services {
		// If we don't have the service locally, deregister it
		existing, ok := l.services[id]
		if !ok {
			l.serviceStatus[id] = syncStatus{remoteDelete: true}
			continue
		}

		// If our definition is different, we need to update it
		if existing.EnableTagOverride {
			existing.Tags = service.Tags
		}
		equal := existing.IsSame(service)
		l.serviceStatus[id] = syncStatus{inSync: equal}
	}

	// Index the remote health checks to improve efficiency
	checkIndex := make(map[string]*structs.HealthCheck, len(checks))
	for _, check := range checks {
		checkIndex[check.CheckID] = check
	}

	// Sync any check which doesn't exist on the remote side
	for id, _ := range l.checks {
		if _, ok := checkIndex[id]; !ok {
			l.checkStatus[id] = syncStatus{inSync: false}
		}
	}

	for _, check := range checks {
		// If we don't have the check locally, deregister it
		id := check.CheckID
		existing, ok := l.checks[id]
		if !ok {
			// The Serf check is created automatically, and does not
			// need to be registered
			if id == consul.SerfCheckID {
				continue
			}
			l.checkStatus[id] = syncStatus{remoteDelete: true}
			continue
		}

		// If our definition is different, we need to update it
		var equal bool
		if l.config.CheckUpdateInterval == 0 {
			equal = existing.IsSame(check)
		} else {
			eCopy := new(structs.HealthCheck)
			*eCopy = *existing
			eCopy.Output = ""
			check.Output = ""
			equal = eCopy.IsSame(check)
		}

		// Update the status
		l.checkStatus[id] = syncStatus{inSync: equal}
	}
	return nil
}

// syncChanges is used to scan the status our local services and checks
// and update any that are out of sync with the server
func (l *localState) syncChanges() error {
	l.Lock()
	defer l.Unlock()

	// We will do node-level info syncing at the end, since it will get
	// updated by a service or check sync anyway, given how the register
	// API works.

	// Sync the services
	for id, status := range l.serviceStatus {
		if status.remoteDelete {
			if err := l.deleteService(id); err != nil {
				return err
			}
		} else if !status.inSync {
			if err := l.syncService(id); err != nil {
				return err
			}
		} else {
			l.logger.Printf("[DEBUG] agent: Service '%s' in sync", id)
		}
	}

	// Sync the checks
	for id, status := range l.checkStatus {
		if status.remoteDelete {
			if err := l.deleteCheck(id); err != nil {
				return err
			}
		} else if !status.inSync {
			// Cancel a deferred sync
			if timer := l.deferCheck[id]; timer != nil {
				timer.Stop()
				delete(l.deferCheck, id)
			}

			if err := l.syncCheck(id); err != nil {
				return err
			}
		} else {
			l.logger.Printf("[DEBUG] agent: Check '%s' in sync", id)
		}
	}

	// Now sync the node level info if we need to, and didn't do any of
	// the other sync operations.
	if !l.nodeInfoInSync {
		if err := l.syncNodeInfo(); err != nil {
			return err
		}
	}

	return nil
}

// deleteService is used to delete a service from the server
func (l *localState) deleteService(id string) error {
	if id == "" {
		return fmt.Errorf("ServiceID missing")
	}

	req := structs.DeregisterRequest{
		Datacenter:   l.config.Datacenter,
		Node:         l.config.NodeName,
		ServiceID:    id,
		WriteRequest: structs.WriteRequest{Token: l.serviceToken(id)},
	}
	var out struct{}
	err := l.iface.RPC("Catalog.Deregister", &req, &out)
	if err == nil {
		delete(l.serviceStatus, id)
		l.logger.Printf("[INFO] agent: Deregistered service '%s'", id)
	}
	return err
}

// deleteCheck is used to delete a service from the server
func (l *localState) deleteCheck(id string) error {
	if id == "" {
		return fmt.Errorf("CheckID missing")
	}

	req := structs.DeregisterRequest{
		Datacenter:   l.config.Datacenter,
		Node:         l.config.NodeName,
		CheckID:      id,
		WriteRequest: structs.WriteRequest{Token: l.checkToken(id)},
	}
	var out struct{}
	err := l.iface.RPC("Catalog.Deregister", &req, &out)
	if err == nil {
		delete(l.checkStatus, id)
		l.logger.Printf("[INFO] agent: Deregistered check '%s'", id)
	}
	return err
}

// syncService is used to sync a service to the server
func (l *localState) syncService(id string) error {
	req := structs.RegisterRequest{
		Datacenter:      l.config.Datacenter,
		Node:            l.config.NodeName,
		Address:         l.config.AdvertiseAddr,
		TaggedAddresses: l.config.TaggedAddresses,
		Service:         l.services[id],
		WriteRequest:    structs.WriteRequest{Token: l.serviceToken(id)},
	}

	// If the service has associated checks that are out of sync,
	// piggyback them on the service sync so they are part of the
	// same transaction and are registered atomically.
	var checks structs.HealthChecks
	for _, check := range l.checks {
		if check.ServiceID == id {
			if stat, ok := l.checkStatus[check.CheckID]; !ok || !stat.inSync {
				checks = append(checks, check)
			}
		}
	}

	// Backwards-compatibility for Consul < 0.5
	if len(checks) == 1 {
		req.Check = checks[0]
	} else {
		req.Checks = checks
	}

	var out struct{}
	err := l.iface.RPC("Catalog.Register", &req, &out)
	if err == nil {
		l.serviceStatus[id] = syncStatus{inSync: true}
		// Given how the register API works, this info is also updated
		// every time we sync a service.
		l.nodeInfoInSync = true
		l.logger.Printf("[INFO] agent: Synced service '%s'", id)
		for _, check := range checks {
			l.checkStatus[check.CheckID] = syncStatus{inSync: true}
		}
	} else if strings.Contains(err.Error(), permissionDenied) {
		l.serviceStatus[id] = syncStatus{inSync: true}
		l.logger.Printf("[WARN] agent: Service '%s' registration blocked by ACLs", id)
		for _, check := range checks {
			l.checkStatus[check.CheckID] = syncStatus{inSync: true}
		}
		return nil
	}
	return err
}

// syncCheck is used to sync a check to the server
func (l *localState) syncCheck(id string) error {
	// Pull in the associated service if any
	check := l.checks[id]
	var service *structs.NodeService
	if check.ServiceID != "" {
		if serv, ok := l.services[check.ServiceID]; ok {
			service = serv
		}
	}

	req := structs.RegisterRequest{
		Datacenter:      l.config.Datacenter,
		Node:            l.config.NodeName,
		Address:         l.config.AdvertiseAddr,
		TaggedAddresses: l.config.TaggedAddresses,
		Service:         service,
		Check:           l.checks[id],
		WriteRequest:    structs.WriteRequest{Token: l.checkToken(id)},
	}
	var out struct{}
	err := l.iface.RPC("Catalog.Register", &req, &out)
	if err == nil {
		l.checkStatus[id] = syncStatus{inSync: true}
		// Given how the register API works, this info is also updated
		// every time we sync a service.
		l.nodeInfoInSync = true
		l.logger.Printf("[INFO] agent: Synced check '%s'", id)
	} else if strings.Contains(err.Error(), permissionDenied) {
		l.checkStatus[id] = syncStatus{inSync: true}
		l.logger.Printf("[WARN] agent: Check '%s' registration blocked by ACLs", id)
		return nil
	}
	return err
}

func (l *localState) syncNodeInfo() error {
	req := structs.RegisterRequest{
		Datacenter:      l.config.Datacenter,
		Node:            l.config.NodeName,
		Address:         l.config.AdvertiseAddr,
		TaggedAddresses: l.config.TaggedAddresses,
		WriteRequest:    structs.WriteRequest{Token: l.config.ACLToken},
	}
	var out struct{}
	err := l.iface.RPC("Catalog.Register", &req, &out)
	if err == nil {
		l.nodeInfoInSync = true
		l.logger.Printf("[INFO] agent: Synced node info")
	} else if strings.Contains(err.Error(), permissionDenied) {
		l.nodeInfoInSync = true
		l.logger.Printf("[WARN] agent: Node info update blocked by ACLs")
		return nil
	}
	return err
}
