// Package servers provides a Manager interface for Manager managed
// agent.Server objects.  The servers package manages servers from a Consul
// client's perspective (i.e. a list of servers that a client talks with for
// RPCs).  The servers package does not provide any API guarantees and should
// be called only by `hashicorp/consul`.
package servers

import (
	"log"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hashicorp/consul/consul/agent"
	"github.com/hashicorp/consul/lib"
)

const (
	// clientRPCJitterFraction determines the amount of jitter added to
	// clientRPCMinReuseDuration before a connection is expired and a new
	// connection is established in order to rebalance load across consul
	// servers.  The cluster-wide number of connections per second from
	// rebalancing is applied after this jitter to ensure the CPU impact
	// is always finite.  See newRebalanceConnsPerSecPerServer's comment
	// for additional commentary.
	//
	// For example, in a 10K consul cluster with 5x servers, this default
	// averages out to ~13 new connections from rebalancing per server
	// per second (each connection is reused for 120s to 180s).
	clientRPCJitterFraction = 2

	// clientRPCMinReuseDuration controls the minimum amount of time RPC
	// queries are sent over an established connection to a single server
	clientRPCMinReuseDuration = 120 * time.Second

	// Limit the number of new connections a server receives per second
	// for connection rebalancing.  This limit caps the load caused by
	// continual rebalancing efforts when a cluster is in equilibrium.  A
	// lower value comes at the cost of increased recovery time after a
	// partition.  This parameter begins to take effect when there are
	// more than ~48K clients querying 5x servers or at lower server
	// values when there is a partition.
	//
	// For example, in a 100K consul cluster with 5x servers, it will
	// take ~5min for all servers to rebalance their connections.  If
	// 99,995 agents are in the minority talking to only one server, it
	// will take ~26min for all servers to rebalance.  A 10K cluster in
	// the same scenario will take ~2.6min to rebalance.
	newRebalanceConnsPerSecPerServer = 64
)

// ConsulClusterInfo is an interface wrapper around serf in order to prevent
// a cyclic import dependency.
type ConsulClusterInfo interface {
	NumNodes() int
}

// Pinger is an interface wrapping client.ConnPool to prevent a cyclic import
// dependency.
type Pinger interface {
	PingConsulServer(s *agent.Server) (bool, error)
}

// serverList is a local copy of the struct used to maintain the list of
// Consul servers used by Manager.
//
// NOTE(sean@): We are explicitly relying on the fact that serverList will
// be copied onto the stack.  Please keep this structure light.
type serverList struct {
	// servers tracks the locally known servers.  List membership is
	// maintained by Serf.
	servers []*agent.Server
}

type Manager struct {
	// listValue manages the atomic load/store of a Manager's serverList
	listValue atomic.Value
	listLock  sync.Mutex

	// rebalanceTimer controls the duration of the rebalance interval
	rebalanceTimer *time.Timer

	// shutdownCh is a copy of the channel in consul.Client
	shutdownCh chan struct{}

	logger *log.Logger

	// clusterInfo is used to estimate the approximate number of nodes in
	// a cluster and limit the rate at which it rebalances server
	// connections.  ConsulClusterInfo is an interface that wraps serf.
	clusterInfo ConsulClusterInfo

	// connPoolPinger is used to test the health of a server in the
	// connection pool.  Pinger is an interface that wraps
	// client.ConnPool.
	connPoolPinger Pinger

	// notifyFailedBarrier is acts as a barrier to prevent queuing behind
	// serverListLog and acts as a TryLock().
	notifyFailedBarrier int32
}

// AddServer takes out an internal write lock and adds a new server.  If the
// server is not known, appends the server to the list.  The new server will
// begin seeing use after the rebalance timer fires or enough servers fail
// organically.  If the server is already known, merge the new server
// details.
func (m *Manager) AddServer(s *agent.Server) {
	m.listLock.Lock()
	defer m.listLock.Unlock()
	l := m.getServerList()

	// Check if this server is known
	found := false
	for idx, existing := range l.servers {
		if existing.Name == s.Name {
			newServers := make([]*agent.Server, len(l.servers))
			copy(newServers, l.servers)

			// Overwrite the existing server details in order to
			// possibly update metadata (e.g. server version)
			newServers[idx] = s

			l.servers = newServers
			found = true
			break
		}
	}

	// Add to the list if not known
	if !found {
		newServers := make([]*agent.Server, len(l.servers), len(l.servers)+1)
		copy(newServers, l.servers)
		newServers = append(newServers, s)
		l.servers = newServers
	}

	m.saveServerList(l)
}

// cycleServers returns a new list of servers that has dequeued the first
// server and enqueued it at the end of the list.  cycleServers assumes the
// caller is holding the listLock.  cycleServer does not test or ping
// the next server inline.  cycleServer may be called when the environment
// has just entered an unhealthy situation and blocking on a server test is
// less desirable than just returning the next server in the firing line.  If
// the next server fails, it will fail fast enough and cycleServer will be
// called again.
func (l *serverList) cycleServer() (servers []*agent.Server) {
	numServers := len(l.servers)
	if numServers < 2 {
		return servers // No action required
	}

	newServers := make([]*agent.Server, 0, numServers)
	newServers = append(newServers, l.servers[1:]...)
	newServers = append(newServers, l.servers[0])

	return newServers
}

// removeServerByKey performs an inline removal of the first matching server
func (l *serverList) removeServerByKey(targetKey *agent.Key) {
	for i, s := range l.servers {
		if targetKey.Equal(s.Key()) {
			copy(l.servers[i:], l.servers[i+1:])
			l.servers[len(l.servers)-1] = nil
			l.servers = l.servers[:len(l.servers)-1]
			return
		}
	}
}

// shuffleServers shuffles the server list in place
func (l *serverList) shuffleServers() {
	for i := len(l.servers) - 1; i > 0; i-- {
		j := rand.Int31n(int32(i + 1))
		l.servers[i], l.servers[j] = l.servers[j], l.servers[i]
	}
}

// FindServer takes out an internal "read lock" and searches through the list
// of servers to find a "healthy" server.  If the server is actually
// unhealthy, we rely on Serf to detect this and remove the node from the
// server list.  If the server at the front of the list has failed or fails
// during an RPC call, it is rotated to the end of the list.  If there are no
// servers available, return nil.
func (m *Manager) FindServer() *agent.Server {
	l := m.getServerList()
	numServers := len(l.servers)
	if numServers == 0 {
		m.logger.Printf("[WARN] manager: No servers available")
		return nil
	} else {
		// Return whatever is at the front of the list because it is
		// assumed to be the oldest in the server list (unless -
		// hypothetically - the server list was rotated right after a
		// server was added).
		return l.servers[0]
	}
}

// getServerList is a convenience method which hides the locking semantics
// of atomic.Value from the caller.
func (m *Manager) getServerList() serverList {
	return m.listValue.Load().(serverList)
}

// saveServerList is a convenience method which hides the locking semantics
// of atomic.Value from the caller.
func (m *Manager) saveServerList(l serverList) {
	m.listValue.Store(l)
}

// New is the only way to safely create a new Manager struct.
func New(logger *log.Logger, shutdownCh chan struct{}, clusterInfo ConsulClusterInfo, connPoolPinger Pinger) (m *Manager) {
	m = new(Manager)
	m.logger = logger
	m.clusterInfo = clusterInfo       // can't pass *consul.Client: import cycle
	m.connPoolPinger = connPoolPinger // can't pass *consul.ConnPool: import cycle
	m.rebalanceTimer = time.NewTimer(clientRPCMinReuseDuration)
	m.shutdownCh = shutdownCh

	l := serverList{}
	l.servers = make([]*agent.Server, 0)
	m.saveServerList(l)
	return m
}

// NotifyFailedServer marks the passed in server as "failed" by rotating it
// to the end of the server list.
func (m *Manager) NotifyFailedServer(s *agent.Server) {
	l := m.getServerList()

	// If the server being failed is not the first server on the list,
	// this is a noop.  If, however, the server is failed and first on
	// the list, acquire the lock, retest, and take the penalty of moving
	// the server to the end of the list.

	// Only rotate the server list when there is more than one server
	if len(l.servers) > 1 && l.servers[0] == s &&
		// Use atomic.CAS to emulate a TryLock().
		atomic.CompareAndSwapInt32(&m.notifyFailedBarrier, 0, 1) {
		defer atomic.StoreInt32(&m.notifyFailedBarrier, 0)

		// Grab a lock, retest, and take the hit of cycling the first
		// server to the end.
		m.listLock.Lock()
		defer m.listLock.Unlock()
		l = m.getServerList()

		if len(l.servers) > 1 && l.servers[0] == s {
			l.servers = l.cycleServer()
			m.saveServerList(l)
		}
	}
}

// NumServers takes out an internal "read lock" and returns the number of
// servers.  numServers includes both healthy and unhealthy servers.
func (m *Manager) NumServers() int {
	l := m.getServerList()
	return len(l.servers)
}

// RebalanceServers shuffles the list of servers on this agent.  The server
// at the front of the list is selected for the next RPC.  RPC calls that
// fail for a particular server are rotated to the end of the list.  This
// method reshuffles the list periodically in order to redistribute work
// across all known consul servers (i.e. guarantee that the order of servers
// in the server list is not positively correlated with the age of a server
// in the Consul cluster).  Periodically shuffling the server list prevents
// long-lived clients from fixating on long-lived servers.
//
// Unhealthy servers are removed when serf notices the server has been
// deregistered.  Before the newly shuffled server list is saved, the new
// remote endpoint is tested to ensure its responsive.
func (m *Manager) RebalanceServers() {
	// Obtain a copy of the current serverList
	l := m.getServerList()

	// Early abort if there is nothing to shuffle
	if len(l.servers) < 2 {
		return
	}

	l.shuffleServers()

	// Iterate through the shuffled server list to find an assumed
	// healthy server.  NOTE: Do not iterate on the list directly because
	// this loop mutates the server list in-place.
	var foundHealthyServer bool
	for i := 0; i < len(l.servers); i++ {
		// Always test the first server.  Failed servers are cycled
		// while Serf detects the node has failed.
		selectedServer := l.servers[0]

		ok, err := m.connPoolPinger.PingConsulServer(selectedServer)
		if ok {
			foundHealthyServer = true
			break
		}
		m.logger.Printf(`[DEBUG] manager: pinging server "%s" failed: %s`, selectedServer.String(), err)

		l.cycleServer()
	}

	// If no healthy servers were found, sleep and wait for Serf to make
	// the world a happy place again.
	if !foundHealthyServer {
		m.logger.Printf("[DEBUG] manager: No healthy servers during rebalance, aborting")
		return
	}

	// Verify that all servers are present
	if m.reconcileServerList(&l) {
		m.logger.Printf("[DEBUG] manager: Rebalanced %d servers, next active server is %s", len(l.servers), l.servers[0].String())
	} else {
		// reconcileServerList failed because Serf removed the server
		// that was at the front of the list that had successfully
		// been Ping'ed.  Between the Ping and reconcile, a Serf
		// event had shown up removing the node.
		//
		// Instead of doing any heroics, "freeze in place" and
		// continue to use the existing connection until the next
		// rebalance occurs.
	}

	return
}

// reconcileServerList returns true when the first server in serverList
// exists in the receiver's serverList.  If true, the merged serverList is
// stored as the receiver's serverList.  Returns false if the first server
// does not exist in the list (i.e. was removed by Serf during a
// PingConsulServer() call.  Newly added servers are appended to the list and
// other missing servers are removed from the list.
func (m *Manager) reconcileServerList(l *serverList) bool {
	m.listLock.Lock()
	defer m.listLock.Unlock()

	// newServerCfg is a serverList that has been kept up to date with
	// Serf node join and node leave events.
	newServerCfg := m.getServerList()

	// If Serf has removed all nodes, or there is no selected server
	// (zero nodes in serverList), abort early.
	if len(newServerCfg.servers) == 0 || len(l.servers) == 0 {
		return false
	}

	type targetServer struct {
		server *agent.Server

		//   'b' == both
		//   'o' == original
		//   'n' == new
		state byte
	}
	mergedList := make(map[agent.Key]*targetServer, len(l.servers))
	for _, s := range l.servers {
		mergedList[*s.Key()] = &targetServer{server: s, state: 'o'}
	}
	for _, s := range newServerCfg.servers {
		k := s.Key()
		_, found := mergedList[*k]
		if found {
			mergedList[*k].state = 'b'
		} else {
			mergedList[*k] = &targetServer{server: s, state: 'n'}
		}
	}

	// Ensure the selected server has not been removed by Serf
	selectedServerKey := l.servers[0].Key()
	if v, found := mergedList[*selectedServerKey]; found && v.state == 'o' {
		return false
	}

	// Append any new servers and remove any old servers
	for k, v := range mergedList {
		switch v.state {
		case 'b':
			// Do nothing, server exists in both
		case 'o':
			// Server has been removed
			l.removeServerByKey(&k)
		case 'n':
			// Server added
			l.servers = append(l.servers, v.server)
		default:
			panic("unknown merge list state")
		}
	}

	m.saveServerList(*l)
	return true
}

// RemoveServer takes out an internal write lock and removes a server from
// the server list.
func (m *Manager) RemoveServer(s *agent.Server) {
	m.listLock.Lock()
	defer m.listLock.Unlock()
	l := m.getServerList()

	// Remove the server if known
	for i, _ := range l.servers {
		if l.servers[i].Name == s.Name {
			newServers := make([]*agent.Server, 0, len(l.servers)-1)
			newServers = append(newServers, l.servers[:i]...)
			newServers = append(newServers, l.servers[i+1:]...)
			l.servers = newServers

			m.saveServerList(l)
			return
		}
	}
}

// refreshServerRebalanceTimer is only called once m.rebalanceTimer expires.
func (m *Manager) refreshServerRebalanceTimer() time.Duration {
	l := m.getServerList()
	numServers := len(l.servers)
	// Limit this connection's life based on the size (and health) of the
	// cluster.  Never rebalance a connection more frequently than
	// connReuseLowWatermarkDuration, and make sure we never exceed
	// clusterWideRebalanceConnsPerSec operations/s across numLANMembers.
	clusterWideRebalanceConnsPerSec := float64(numServers * newRebalanceConnsPerSecPerServer)
	connReuseLowWatermarkDuration := clientRPCMinReuseDuration + lib.RandomStagger(clientRPCMinReuseDuration/clientRPCJitterFraction)
	numLANMembers := m.clusterInfo.NumNodes()
	connRebalanceTimeout := lib.RateScaledInterval(clusterWideRebalanceConnsPerSec, connReuseLowWatermarkDuration, numLANMembers)

	m.rebalanceTimer.Reset(connRebalanceTimeout)
	return connRebalanceTimeout
}

// ResetRebalanceTimer resets the rebalance timer.  This method exists for
// testing and should not be used directly.
func (m *Manager) ResetRebalanceTimer() {
	m.listLock.Lock()
	defer m.listLock.Unlock()
	m.rebalanceTimer.Reset(clientRPCMinReuseDuration)
}

// Start is used to start and manage the task of automatically shuffling and
// rebalancing the list of Consul servers.  This maintenance only happens
// periodically based on the expiration of the timer.  Failed servers are
// automatically cycled to the end of the list.  New servers are appended to
// the list.  The order of the server list must be shuffled periodically to
// distribute load across all known and available Consul servers.
func (m *Manager) Start() {
	for {
		select {
		case <-m.rebalanceTimer.C:
			m.RebalanceServers()
			m.refreshServerRebalanceTimer()

		case <-m.shutdownCh:
			m.logger.Printf("[INFO] manager: shutting down")
			return
		}
	}
}
