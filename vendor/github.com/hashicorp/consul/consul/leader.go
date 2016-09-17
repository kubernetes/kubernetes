package consul

import (
	"fmt"
	"net"
	"strconv"
	"strings"
	"time"

	"github.com/armon/go-metrics"
	"github.com/hashicorp/consul/consul/agent"
	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/types"
	"github.com/hashicorp/raft"
	"github.com/hashicorp/serf/serf"
)

const (
	SerfCheckID           types.CheckID = "serfHealth"
	SerfCheckName                       = "Serf Health Status"
	SerfCheckAliveOutput                = "Agent alive and reachable"
	SerfCheckFailedOutput               = "Agent not live or unreachable"
	ConsulServiceID                     = "consul"
	ConsulServiceName                   = "consul"
	newLeaderEvent                      = "consul:new-leader"
)

// monitorLeadership is used to monitor if we acquire or lose our role
// as the leader in the Raft cluster. There is some work the leader is
// expected to do, so we must react to changes
func (s *Server) monitorLeadership() {
	leaderCh := s.raft.LeaderCh()
	var stopCh chan struct{}
	for {
		select {
		case isLeader := <-leaderCh:
			if isLeader {
				stopCh = make(chan struct{})
				go s.leaderLoop(stopCh)
				s.logger.Printf("[INFO] consul: cluster leadership acquired")
			} else if stopCh != nil {
				close(stopCh)
				stopCh = nil
				s.logger.Printf("[INFO] consul: cluster leadership lost")
			}
		case <-s.shutdownCh:
			return
		}
	}
}

// leaderLoop runs as long as we are the leader to run various
// maintenance activities
func (s *Server) leaderLoop(stopCh chan struct{}) {
	// Ensure we revoke leadership on stepdown
	defer s.revokeLeadership()

	// Fire a user event indicating a new leader
	payload := []byte(s.config.NodeName)
	if err := s.serfLAN.UserEvent(newLeaderEvent, payload, false); err != nil {
		s.logger.Printf("[WARN] consul: failed to broadcast new leader event: %v", err)
	}

	// Reconcile channel is only used once initial reconcile
	// has succeeded
	var reconcileCh chan serf.Member
	establishedLeader := false

RECONCILE:
	// Setup a reconciliation timer
	reconcileCh = nil
	interval := time.After(s.config.ReconcileInterval)

	// Apply a raft barrier to ensure our FSM is caught up
	start := time.Now()
	barrier := s.raft.Barrier(0)
	if err := barrier.Error(); err != nil {
		s.logger.Printf("[ERR] consul: failed to wait for barrier: %v", err)
		goto WAIT
	}
	metrics.MeasureSince([]string{"consul", "leader", "barrier"}, start)

	// Check if we need to handle initial leadership actions
	if !establishedLeader {
		if err := s.establishLeadership(); err != nil {
			s.logger.Printf("[ERR] consul: failed to establish leadership: %v",
				err)
			goto WAIT
		}
		establishedLeader = true
	}

	// Reconcile any missing data
	if err := s.reconcile(); err != nil {
		s.logger.Printf("[ERR] consul: failed to reconcile: %v", err)
		goto WAIT
	}

	// Initial reconcile worked, now we can process the channel
	// updates
	reconcileCh = s.reconcileCh

WAIT:
	// Periodically reconcile as long as we are the leader,
	// or when Serf events arrive
	for {
		select {
		case <-stopCh:
			return
		case <-s.shutdownCh:
			return
		case <-interval:
			goto RECONCILE
		case member := <-reconcileCh:
			s.reconcileMember(member)
		case index := <-s.tombstoneGC.ExpireCh():
			go s.reapTombstones(index)
		}
	}
}

// establishLeadership is invoked once we become leader and are able
// to invoke an initial barrier. The barrier is used to ensure any
// previously inflight transactions have been committed and that our
// state is up-to-date.
func (s *Server) establishLeadership() error {
	// Hint the tombstone expiration timer. When we freshly establish leadership
	// we become the authoritative timer, and so we need to start the clock
	// on any pending GC events.
	s.tombstoneGC.SetEnabled(true)
	lastIndex := s.raft.LastIndex()
	s.tombstoneGC.Hint(lastIndex)
	s.logger.Printf("[DEBUG] consul: reset tombstone GC to index %d", lastIndex)

	// Setup ACLs if we are the leader and need to
	if err := s.initializeACL(); err != nil {
		s.logger.Printf("[ERR] consul: ACL initialization failed: %v", err)
		return err
	}

	// Setup the session timers. This is done both when starting up or when
	// a leader fail over happens. Since the timers are maintained by the leader
	// node along, effectively this means all the timers are renewed at the
	// time of failover. The TTL contract is that the session will not be expired
	// before the TTL, so expiring it later is allowable.
	//
	// This MUST be done after the initial barrier to ensure the latest Sessions
	// are available to be initialized. Otherwise initialization may use stale
	// data.
	if err := s.initializeSessionTimers(); err != nil {
		s.logger.Printf("[ERR] consul: Session Timers initialization failed: %v",
			err)
		return err
	}
	return nil
}

// revokeLeadership is invoked once we step down as leader.
// This is used to cleanup any state that may be specific to a leader.
func (s *Server) revokeLeadership() error {
	// Disable the tombstone GC, since it is only useful as a leader
	s.tombstoneGC.SetEnabled(false)

	// Clear the session timers on either shutdown or step down, since we
	// are no longer responsible for session expirations.
	if err := s.clearAllSessionTimers(); err != nil {
		s.logger.Printf("[ERR] consul: Clearing session timers failed: %v", err)
		return err
	}
	return nil
}

// initializeACL is used to setup the ACLs if we are the leader
// and need to do this.
func (s *Server) initializeACL() error {
	// Bail if not configured or we are not authoritative
	authDC := s.config.ACLDatacenter
	if len(authDC) == 0 || authDC != s.config.Datacenter {
		return nil
	}

	// Purge the cache, since it could've changed while we
	// were not the leader
	s.aclAuthCache.Purge()

	// Look for the anonymous token
	state := s.fsm.State()
	_, acl, err := state.ACLGet(anonymousToken)
	if err != nil {
		return fmt.Errorf("failed to get anonymous token: %v", err)
	}

	// Create anonymous token if missing
	if acl == nil {
		req := structs.ACLRequest{
			Datacenter: authDC,
			Op:         structs.ACLSet,
			ACL: structs.ACL{
				ID:   anonymousToken,
				Name: "Anonymous Token",
				Type: structs.ACLTypeClient,
			},
		}
		_, err := s.raftApply(structs.ACLRequestType, &req)
		if err != nil {
			return fmt.Errorf("failed to create anonymous token: %v", err)
		}
	}

	// Check for configured master token
	master := s.config.ACLMasterToken
	if len(master) == 0 {
		return nil
	}

	// Look for the master token
	_, acl, err = state.ACLGet(master)
	if err != nil {
		return fmt.Errorf("failed to get master token: %v", err)
	}
	if acl == nil {
		req := structs.ACLRequest{
			Datacenter: authDC,
			Op:         structs.ACLSet,
			ACL: structs.ACL{
				ID:   master,
				Name: "Master Token",
				Type: structs.ACLTypeManagement,
			},
		}
		_, err := s.raftApply(structs.ACLRequestType, &req)
		if err != nil {
			return fmt.Errorf("failed to create master token: %v", err)
		}

	}
	return nil
}

// reconcile is used to reconcile the differences between Serf
// membership and what is reflected in our strongly consistent store.
// Mainly we need to ensure all live nodes are registered, all failed
// nodes are marked as such, and all left nodes are de-registered.
func (s *Server) reconcile() (err error) {
	defer metrics.MeasureSince([]string{"consul", "leader", "reconcile"}, time.Now())
	members := s.serfLAN.Members()
	knownMembers := make(map[string]struct{})
	for _, member := range members {
		if err := s.reconcileMember(member); err != nil {
			return err
		}
		knownMembers[member.Name] = struct{}{}
	}

	// Reconcile any members that have been reaped while we were not the leader
	return s.reconcileReaped(knownMembers)
}

// reconcileReaped is used to reconcile nodes that have failed and been reaped
// from Serf but remain in the catalog. This is done by looking for SerfCheckID
// in a critical state that does not correspond to a known Serf member. We generate
// a "reap" event to cause the node to be cleaned up.
func (s *Server) reconcileReaped(known map[string]struct{}) error {
	state := s.fsm.State()
	_, checks, err := state.ChecksInState(structs.HealthAny)
	if err != nil {
		return err
	}
	for _, check := range checks {
		// Ignore any non serf checks
		if check.CheckID != SerfCheckID {
			continue
		}

		// Check if this node is "known" by serf
		if _, ok := known[check.Node]; ok {
			continue
		}

		// Create a fake member
		member := serf.Member{
			Name: check.Node,
			Tags: map[string]string{
				"dc":   s.config.Datacenter,
				"role": "node",
			},
		}

		// Get the node services, look for ConsulServiceID
		_, services, err := state.NodeServices(check.Node)
		if err != nil {
			return err
		}
		serverPort := 0
		for _, service := range services.Services {
			if service.ID == ConsulServiceID {
				serverPort = service.Port
				break
			}
		}

		// Create the appropriate tags if this was a server node
		if serverPort > 0 {
			member.Tags["role"] = "consul"
			member.Tags["port"] = strconv.FormatUint(uint64(serverPort), 10)
		}

		// Attempt to reap this member
		if err := s.handleReapMember(member); err != nil {
			return err
		}
	}
	return nil
}

// reconcileMember is used to do an async reconcile of a single
// serf member
func (s *Server) reconcileMember(member serf.Member) error {
	// Check if this is a member we should handle
	if !s.shouldHandleMember(member) {
		s.logger.Printf("[WARN] consul: skipping reconcile of node %v", member)
		return nil
	}
	defer metrics.MeasureSince([]string{"consul", "leader", "reconcileMember"}, time.Now())
	var err error
	switch member.Status {
	case serf.StatusAlive:
		err = s.handleAliveMember(member)
	case serf.StatusFailed:
		err = s.handleFailedMember(member)
	case serf.StatusLeft:
		err = s.handleLeftMember(member)
	case StatusReap:
		err = s.handleReapMember(member)
	}
	if err != nil {
		s.logger.Printf("[ERR] consul: failed to reconcile member: %v: %v",
			member, err)

		// Permission denied should not bubble up
		if strings.Contains(err.Error(), permissionDenied) {
			return nil
		}
		return err
	}
	return nil
}

// shouldHandleMember checks if this is a Consul pool member
func (s *Server) shouldHandleMember(member serf.Member) bool {
	if valid, dc := isConsulNode(member); valid && dc == s.config.Datacenter {
		return true
	}
	if valid, parts := agent.IsConsulServer(member); valid && parts.Datacenter == s.config.Datacenter {
		return true
	}
	return false
}

// handleAliveMember is used to ensure the node
// is registered, with a passing health check.
func (s *Server) handleAliveMember(member serf.Member) error {
	// Register consul service if a server
	var service *structs.NodeService
	if valid, parts := agent.IsConsulServer(member); valid {
		service = &structs.NodeService{
			ID:      ConsulServiceID,
			Service: ConsulServiceName,
			Port:    parts.Port,
		}

		// Attempt to join the consul server
		if err := s.joinConsulServer(member, parts); err != nil {
			return err
		}
	}

	// Check if the node exists
	state := s.fsm.State()
	_, node, err := state.GetNode(member.Name)
	if err != nil {
		return err
	}
	if node != nil && node.Address == member.Addr.String() {
		// Check if the associated service is available
		if service != nil {
			match := false
			_, services, err := state.NodeServices(member.Name)
			if err != nil {
				return err
			}
			if services != nil {
				for id, _ := range services.Services {
					if id == service.ID {
						match = true
					}
				}
			}
			if !match {
				goto AFTER_CHECK
			}
		}

		// Check if the serfCheck is in the passing state
		_, checks, err := state.NodeChecks(member.Name)
		if err != nil {
			return err
		}
		for _, check := range checks {
			if check.CheckID == SerfCheckID && check.Status == structs.HealthPassing {
				return nil
			}
		}
	}
AFTER_CHECK:
	s.logger.Printf("[INFO] consul: member '%s' joined, marking health alive", member.Name)

	// Register with the catalog
	req := structs.RegisterRequest{
		Datacenter: s.config.Datacenter,
		Node:       member.Name,
		Address:    member.Addr.String(),
		Service:    service,
		Check: &structs.HealthCheck{
			Node:    member.Name,
			CheckID: SerfCheckID,
			Name:    SerfCheckName,
			Status:  structs.HealthPassing,
			Output:  SerfCheckAliveOutput,
		},
		WriteRequest: structs.WriteRequest{Token: s.config.ACLToken},
	}
	var out struct{}
	return s.endpoints.Catalog.Register(&req, &out)
}

// handleFailedMember is used to mark the node's status
// as being critical, along with all checks as unknown.
func (s *Server) handleFailedMember(member serf.Member) error {
	// Check if the node exists
	state := s.fsm.State()
	_, node, err := state.GetNode(member.Name)
	if err != nil {
		return err
	}
	if node != nil && node.Address == member.Addr.String() {
		// Check if the serfCheck is in the critical state
		_, checks, err := state.NodeChecks(member.Name)
		if err != nil {
			return err
		}
		for _, check := range checks {
			if check.CheckID == SerfCheckID && check.Status == structs.HealthCritical {
				return nil
			}
		}
	}
	s.logger.Printf("[INFO] consul: member '%s' failed, marking health critical", member.Name)

	// Register with the catalog
	req := structs.RegisterRequest{
		Datacenter: s.config.Datacenter,
		Node:       member.Name,
		Address:    member.Addr.String(),
		Check: &structs.HealthCheck{
			Node:    member.Name,
			CheckID: SerfCheckID,
			Name:    SerfCheckName,
			Status:  structs.HealthCritical,
			Output:  SerfCheckFailedOutput,
		},
		WriteRequest: structs.WriteRequest{Token: s.config.ACLToken},
	}
	var out struct{}
	return s.endpoints.Catalog.Register(&req, &out)
}

// handleLeftMember is used to handle members that gracefully
// left. They are deregistered if necessary.
func (s *Server) handleLeftMember(member serf.Member) error {
	return s.handleDeregisterMember("left", member)
}

// handleReapMember is used to handle members that have been
// reaped after a prolonged failure. They are deregistered.
func (s *Server) handleReapMember(member serf.Member) error {
	return s.handleDeregisterMember("reaped", member)
}

// handleDeregisterMember is used to deregister a member of a given reason
func (s *Server) handleDeregisterMember(reason string, member serf.Member) error {
	// Do not deregister ourself. This can only happen if the current leader
	// is leaving. Instead, we should allow a follower to take-over and
	// deregister us later.
	if member.Name == s.config.NodeName {
		s.logger.Printf("[WARN] consul: deregistering self (%s) should be done by follower", s.config.NodeName)
		return nil
	}

	// Remove from Raft peers if this was a server
	if valid, parts := agent.IsConsulServer(member); valid {
		if err := s.removeConsulServer(member, parts.Port); err != nil {
			return err
		}
	}

	// Check if the node does not exist
	state := s.fsm.State()
	_, node, err := state.GetNode(member.Name)
	if err != nil {
		return err
	}
	if node == nil {
		return nil
	}

	// Deregister the node
	s.logger.Printf("[INFO] consul: member '%s' %s, deregistering", member.Name, reason)
	req := structs.DeregisterRequest{
		Datacenter: s.config.Datacenter,
		Node:       member.Name,
	}
	var out struct{}
	return s.endpoints.Catalog.Deregister(&req, &out)
}

// joinConsulServer is used to try to join another consul server
func (s *Server) joinConsulServer(m serf.Member, parts *agent.Server) error {
	// Do not join ourself
	if m.Name == s.config.NodeName {
		return nil
	}

	// Check for possibility of multiple bootstrap nodes
	if parts.Bootstrap {
		members := s.serfLAN.Members()
		for _, member := range members {
			valid, p := agent.IsConsulServer(member)
			if valid && member.Name != m.Name && p.Bootstrap {
				s.logger.Printf("[ERR] consul: '%v' and '%v' are both in bootstrap mode. Only one node should be in bootstrap mode, not adding Raft peer.", m.Name, member.Name)
				return nil
			}
		}
	}

	// TODO (slackpad) - This will need to be changed once we support node IDs.
	addr := (&net.TCPAddr{IP: m.Addr, Port: parts.Port}).String()

	// See if it's already in the configuration. It's harmless to re-add it
	// but we want to avoid doing that if possible to prevent useless Raft
	// log entries.
	configFuture := s.raft.GetConfiguration()
	if err := configFuture.Error(); err != nil {
		s.logger.Printf("[ERR] consul: failed to get raft configuration: %v", err)
		return err
	}
	for _, server := range configFuture.Configuration().Servers {
		if server.Address == raft.ServerAddress(addr) {
			return nil
		}
	}

	// Attempt to add as a peer
	addFuture := s.raft.AddPeer(raft.ServerAddress(addr))
	if err := addFuture.Error(); err != nil {
		s.logger.Printf("[ERR] consul: failed to add raft peer: %v", err)
		return err
	}
	return nil
}

// removeConsulServer is used to try to remove a consul server that has left
func (s *Server) removeConsulServer(m serf.Member, port int) error {
	// TODO (slackpad) - This will need to be changed once we support node IDs.
	addr := (&net.TCPAddr{IP: m.Addr, Port: port}).String()

	// See if it's already in the configuration. It's harmless to re-remove it
	// but we want to avoid doing that if possible to prevent useless Raft
	// log entries.
	configFuture := s.raft.GetConfiguration()
	if err := configFuture.Error(); err != nil {
		s.logger.Printf("[ERR] consul: failed to get raft configuration: %v", err)
		return err
	}
	for _, server := range configFuture.Configuration().Servers {
		if server.Address == raft.ServerAddress(addr) {
			goto REMOVE
		}
	}
	return nil

REMOVE:
	// Attempt to remove as a peer.
	future := s.raft.RemovePeer(raft.ServerAddress(addr))
	if err := future.Error(); err != nil {
		s.logger.Printf("[ERR] consul: failed to remove raft peer '%v': %v",
			addr, err)
		return err
	}
	return nil
}

// reapTombstones is invoked by the current leader to manage garbage
// collection of tombstones. When a key is deleted, we trigger a tombstone
// GC clock. Once the expiration is reached, this routine is invoked
// to clear all tombstones before this index. This must be replicated
// through Raft to ensure consistency. We do this outside the leader loop
// to avoid blocking.
func (s *Server) reapTombstones(index uint64) {
	defer metrics.MeasureSince([]string{"consul", "leader", "reapTombstones"}, time.Now())
	req := structs.TombstoneRequest{
		Datacenter:   s.config.Datacenter,
		Op:           structs.TombstoneReap,
		ReapIndex:    index,
		WriteRequest: structs.WriteRequest{Token: s.config.ACLToken},
	}
	_, err := s.raftApply(structs.TombstoneRequestType, &req)
	if err != nil {
		s.logger.Printf("[ERR] consul: failed to reap tombstones up to %d: %v",
			index, err)
	}
}
