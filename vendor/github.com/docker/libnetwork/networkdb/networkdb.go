package networkdb

//go:generate protoc -I.:../vendor/github.com/gogo/protobuf --gogo_out=import_path=github.com/docker/libnetwork/networkdb,Mgogoproto/gogo.proto=github.com/gogo/protobuf/gogoproto:. networkdb.proto

import (
	"fmt"
	"net"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/armon/go-radix"
	"github.com/docker/go-events"
	"github.com/docker/libnetwork/types"
	"github.com/hashicorp/memberlist"
	"github.com/hashicorp/serf/serf"
	"github.com/sirupsen/logrus"
)

const (
	byTable int = 1 + iota
	byNetwork
)

// NetworkDB instance drives the networkdb cluster and acts the broker
// for cluster-scoped and network-scoped gossip and watches.
type NetworkDB struct {
	// The clocks MUST be the first things
	// in this struct due to Golang issue #599.

	// Global lamport clock for node network attach events.
	networkClock serf.LamportClock

	// Global lamport clock for table events.
	tableClock serf.LamportClock

	sync.RWMutex

	// NetworkDB configuration.
	config *Config

	// All the tree index (byTable, byNetwork) that we maintain
	// the db.
	indexes map[int]*radix.Tree

	// Memberlist we use to drive the cluster.
	memberlist *memberlist.Memberlist

	// List of all peer nodes in the cluster not-limited to any
	// network.
	nodes map[string]*node

	// List of all peer nodes which have failed
	failedNodes map[string]*node

	// List of all peer nodes which have left
	leftNodes map[string]*node

	// A multi-dimensional map of network/node attachmemts. The
	// first key is a node name and the second key is a network ID
	// for the network that node is participating in.
	networks map[string]map[string]*network

	// A map of nodes which are participating in a given
	// network. The key is a network ID.
	networkNodes map[string][]string

	// A table of ack channels for every node from which we are
	// waiting for an ack.
	bulkSyncAckTbl map[string]chan struct{}

	// Broadcast queue for network event gossip.
	networkBroadcasts *memberlist.TransmitLimitedQueue

	// Broadcast queue for node event gossip.
	nodeBroadcasts *memberlist.TransmitLimitedQueue

	// A central stop channel to stop all go routines running on
	// behalf of the NetworkDB instance.
	stopCh chan struct{}

	// A central broadcaster for all local watchers watching table
	// events.
	broadcaster *events.Broadcaster

	// List of all tickers which needed to be stopped when
	// cleaning up.
	tickers []*time.Ticker

	// Reference to the memberlist's keyring to add & remove keys
	keyring *memberlist.Keyring

	// bootStrapIP is the list of IPs that can be used to bootstrap
	// the gossip.
	bootStrapIP []net.IP

	// lastStatsTimestamp is the last timestamp when the stats got printed
	lastStatsTimestamp time.Time

	// lastHealthTimestamp is the last timestamp when the health score got printed
	lastHealthTimestamp time.Time
}

// PeerInfo represents the peer (gossip cluster) nodes of a network
type PeerInfo struct {
	Name string
	IP   string
}

// PeerClusterInfo represents the peer (gossip cluster) nodes
type PeerClusterInfo struct {
	PeerInfo
}

type node struct {
	memberlist.Node
	ltime serf.LamportTime
	// Number of hours left before the reaper removes the node
	reapTime time.Duration
}

// network describes the node/network attachment.
type network struct {
	// Network ID
	id string

	// Lamport time for the latest state of the entry.
	ltime serf.LamportTime

	// Node leave is in progress.
	leaving bool

	// Number of seconds still left before a deleted network entry gets
	// removed from networkDB
	reapTime time.Duration

	// The broadcast queue for table event gossip. This is only
	// initialized for this node's network attachment entries.
	tableBroadcasts *memberlist.TransmitLimitedQueue

	// Number of gossip messages sent related to this network during the last stats collection period
	qMessagesSent int
}

// Config represents the configuration of the networdb instance and
// can be passed by the caller.
type Config struct {
	// NodeName is the cluster wide unique name for this node.
	NodeName string

	// BindAddr is the IP on which networkdb listens. It can be
	// 0.0.0.0 to listen on all addresses on the host.
	BindAddr string

	// AdvertiseAddr is the node's IP address that we advertise for
	// cluster communication.
	AdvertiseAddr string

	// BindPort is the local node's port to which we bind to for
	// cluster communication.
	BindPort int

	// Keys to be added to the Keyring of the memberlist. Key at index
	// 0 is the primary key
	Keys [][]byte

	// PacketBufferSize is the maximum number of bytes that memberlist will
	// put in a packet (this will be for UDP packets by default with a NetTransport).
	// A safe value for this is typically 1400 bytes (which is the default). However,
	// depending on your network's MTU (Maximum Transmission Unit) you may
	// be able to increase this to get more content into each gossip packet.
	PacketBufferSize int

	// StatsPrintPeriod the period to use to print queue stats
	// Default is 5min
	StatsPrintPeriod time.Duration

	// HealthPrintPeriod the period to use to print the health score
	// Default is 1min
	HealthPrintPeriod time.Duration
}

// entry defines a table entry
type entry struct {
	// node from which this entry was learned.
	node string

	// Lamport time for the most recent update to the entry
	ltime serf.LamportTime

	// Opaque value store in the entry
	value []byte

	// Deleting the entry is in progress. All entries linger in
	// the cluster for certain amount of time after deletion.
	deleting bool

	// Number of seconds still left before a deleted table entry gets
	// removed from networkDB
	reapTime time.Duration
}

// DefaultConfig returns a NetworkDB config with default values
func DefaultConfig() *Config {
	hostname, _ := os.Hostname()
	return &Config{
		NodeName:          hostname,
		BindAddr:          "0.0.0.0",
		PacketBufferSize:  1400,
		StatsPrintPeriod:  5 * time.Minute,
		HealthPrintPeriod: 1 * time.Minute,
	}
}

// New creates a new instance of NetworkDB using the Config passed by
// the caller.
func New(c *Config) (*NetworkDB, error) {
	nDB := &NetworkDB{
		config:         c,
		indexes:        make(map[int]*radix.Tree),
		networks:       make(map[string]map[string]*network),
		nodes:          make(map[string]*node),
		failedNodes:    make(map[string]*node),
		leftNodes:      make(map[string]*node),
		networkNodes:   make(map[string][]string),
		bulkSyncAckTbl: make(map[string]chan struct{}),
		broadcaster:    events.NewBroadcaster(),
	}

	nDB.indexes[byTable] = radix.New()
	nDB.indexes[byNetwork] = radix.New()

	if err := nDB.clusterInit(); err != nil {
		return nil, err
	}

	return nDB, nil
}

// Join joins this NetworkDB instance with a list of peer NetworkDB
// instances passed by the caller in the form of addr:port
func (nDB *NetworkDB) Join(members []string) error {
	nDB.Lock()
	nDB.bootStrapIP = make([]net.IP, 0, len(members))
	for _, m := range members {
		nDB.bootStrapIP = append(nDB.bootStrapIP, net.ParseIP(m))
	}
	nDB.Unlock()
	return nDB.clusterJoin(members)
}

// Close destroys this NetworkDB instance by leave the cluster,
// stopping timers, canceling goroutines etc.
func (nDB *NetworkDB) Close() {
	if err := nDB.clusterLeave(); err != nil {
		logrus.Errorf("Could not close DB %s: %v", nDB.config.NodeName, err)
	}
}

// ClusterPeers returns all the gossip cluster peers.
func (nDB *NetworkDB) ClusterPeers() []PeerInfo {
	nDB.RLock()
	defer nDB.RUnlock()
	peers := make([]PeerInfo, 0, len(nDB.nodes))
	for _, node := range nDB.nodes {
		peers = append(peers, PeerInfo{
			Name: node.Name,
			IP:   node.Node.Addr.String(),
		})
	}
	return peers
}

// Peers returns the gossip peers for a given network.
func (nDB *NetworkDB) Peers(nid string) []PeerInfo {
	nDB.RLock()
	defer nDB.RUnlock()
	peers := make([]PeerInfo, 0, len(nDB.networkNodes[nid]))
	for _, nodeName := range nDB.networkNodes[nid] {
		if node, ok := nDB.nodes[nodeName]; ok {
			peers = append(peers, PeerInfo{
				Name: node.Name,
				IP:   node.Addr.String(),
			})
		}
	}
	return peers
}

// GetEntry retrieves the value of a table entry in a given (network,
// table, key) tuple
func (nDB *NetworkDB) GetEntry(tname, nid, key string) ([]byte, error) {
	entry, err := nDB.getEntry(tname, nid, key)
	if err != nil {
		return nil, err
	}

	return entry.value, nil
}

func (nDB *NetworkDB) getEntry(tname, nid, key string) (*entry, error) {
	nDB.RLock()
	defer nDB.RUnlock()

	e, ok := nDB.indexes[byTable].Get(fmt.Sprintf("/%s/%s/%s", tname, nid, key))
	if !ok {
		return nil, types.NotFoundErrorf("could not get entry in table %s with network id %s and key %s", tname, nid, key)
	}

	return e.(*entry), nil
}

// CreateEntry creates a table entry in NetworkDB for given (network,
// table, key) tuple and if the NetworkDB is part of the cluster
// propagates this event to the cluster. It is an error to create an
// entry for the same tuple for which there is already an existing
// entry unless the current entry is deleting state.
func (nDB *NetworkDB) CreateEntry(tname, nid, key string, value []byte) error {
	oldEntry, err := nDB.getEntry(tname, nid, key)
	if err != nil {
		if _, ok := err.(types.NotFoundError); !ok {
			return fmt.Errorf("cannot create entry in table %s with network id %s and key %s: %v", tname, nid, key, err)
		}
	}
	if oldEntry != nil && !oldEntry.deleting {
		return fmt.Errorf("cannot create entry in table %s with network id %s and key %s, already exists", tname, nid, key)
	}

	entry := &entry{
		ltime: nDB.tableClock.Increment(),
		node:  nDB.config.NodeName,
		value: value,
	}

	if err := nDB.sendTableEvent(TableEventTypeCreate, nid, tname, key, entry); err != nil {
		return fmt.Errorf("cannot send create event for table %s, %v", tname, err)
	}

	nDB.Lock()
	nDB.indexes[byTable].Insert(fmt.Sprintf("/%s/%s/%s", tname, nid, key), entry)
	nDB.indexes[byNetwork].Insert(fmt.Sprintf("/%s/%s/%s", nid, tname, key), entry)
	nDB.Unlock()

	return nil
}

// UpdateEntry updates a table entry in NetworkDB for given (network,
// table, key) tuple and if the NetworkDB is part of the cluster
// propagates this event to the cluster. It is an error to update a
// non-existent entry.
func (nDB *NetworkDB) UpdateEntry(tname, nid, key string, value []byte) error {
	if _, err := nDB.GetEntry(tname, nid, key); err != nil {
		return fmt.Errorf("cannot update entry as the entry in table %s with network id %s and key %s does not exist", tname, nid, key)
	}

	entry := &entry{
		ltime: nDB.tableClock.Increment(),
		node:  nDB.config.NodeName,
		value: value,
	}

	if err := nDB.sendTableEvent(TableEventTypeUpdate, nid, tname, key, entry); err != nil {
		return fmt.Errorf("cannot send table update event: %v", err)
	}

	nDB.Lock()
	nDB.indexes[byTable].Insert(fmt.Sprintf("/%s/%s/%s", tname, nid, key), entry)
	nDB.indexes[byNetwork].Insert(fmt.Sprintf("/%s/%s/%s", nid, tname, key), entry)
	nDB.Unlock()

	return nil
}

// GetTableByNetwork walks the networkdb by the give table and network id and
// returns a map of keys and values
func (nDB *NetworkDB) GetTableByNetwork(tname, nid string) map[string]interface{} {
	entries := make(map[string]interface{})
	nDB.indexes[byTable].WalkPrefix(fmt.Sprintf("/%s/%s", tname, nid), func(k string, v interface{}) bool {
		entry := v.(*entry)
		if entry.deleting {
			return false
		}
		key := k[strings.LastIndex(k, "/")+1:]
		entries[key] = entry.value
		return false
	})
	return entries
}

// DeleteEntry deletes a table entry in NetworkDB for given (network,
// table, key) tuple and if the NetworkDB is part of the cluster
// propagates this event to the cluster.
func (nDB *NetworkDB) DeleteEntry(tname, nid, key string) error {
	value, err := nDB.GetEntry(tname, nid, key)
	if err != nil {
		return fmt.Errorf("cannot delete entry as the entry in table %s with network id %s and key %s does not exist", tname, nid, key)
	}

	entry := &entry{
		ltime:    nDB.tableClock.Increment(),
		node:     nDB.config.NodeName,
		value:    value,
		deleting: true,
		reapTime: reapInterval,
	}

	if err := nDB.sendTableEvent(TableEventTypeDelete, nid, tname, key, entry); err != nil {
		return fmt.Errorf("cannot send table delete event: %v", err)
	}

	nDB.Lock()
	nDB.indexes[byTable].Insert(fmt.Sprintf("/%s/%s/%s", tname, nid, key), entry)
	nDB.indexes[byNetwork].Insert(fmt.Sprintf("/%s/%s/%s", nid, tname, key), entry)
	nDB.Unlock()

	return nil
}

func (nDB *NetworkDB) deleteNetworkEntriesForNode(deletedNode string) {
	for nid, nodes := range nDB.networkNodes {
		updatedNodes := make([]string, 0, len(nodes))
		for _, node := range nodes {
			if node == deletedNode {
				continue
			}

			updatedNodes = append(updatedNodes, node)
		}

		nDB.networkNodes[nid] = updatedNodes
	}

	delete(nDB.networks, deletedNode)
}

// deleteNodeNetworkEntries is called in 2 conditions with 2 different outcomes:
// 1) when a notification is coming of a node leaving the network
//		- Walk all the network entries and mark the leaving node's entries for deletion
//			These will be garbage collected when the reap timer will expire
// 2) when the local node is leaving the network
//		- Walk all the network entries:
//			A) if the entry is owned by the local node
//		  then we will mark it for deletion. This will ensure that if a node did not
//		  yet received the notification that the local node is leaving, will be aware
//		  of the entries to be deleted.
//			B) if the entry is owned by a remote node, then we can safely delete it. This
//			ensures that if we join back this network as we receive the CREATE event for
//		  entries owned by remote nodes, we will accept them and we notify the application
func (nDB *NetworkDB) deleteNodeNetworkEntries(nid, node string) {
	// Indicates if the delete is triggered for the local node
	isNodeLocal := node == nDB.config.NodeName

	nDB.indexes[byNetwork].WalkPrefix(fmt.Sprintf("/%s", nid),
		func(path string, v interface{}) bool {
			oldEntry := v.(*entry)
			params := strings.Split(path[1:], "/")
			nid := params[0]
			tname := params[1]
			key := params[2]

			// If the entry is owned by a remote node and this node is not leaving the network
			if oldEntry.node != node && !isNodeLocal {
				// Don't do anything because the event is triggered for a node that does not own this entry
				return false
			}

			// If this entry is already marked for deletion and this node is not leaving the network
			if oldEntry.deleting && !isNodeLocal {
				// Don't do anything this entry will be already garbage collected using the old reapTime
				return false
			}

			entry := &entry{
				ltime:    oldEntry.ltime,
				node:     node,
				value:    oldEntry.value,
				deleting: true,
				reapTime: reapInterval,
			}

			// we arrived at this point in 2 cases:
			// 1) this entry is owned by the node that is leaving the network
			// 2) the local node is leaving the network
			if oldEntry.node == node {
				if isNodeLocal {
					// TODO fcrisciani: this can be removed if there is no way to leave the network
					// without doing a delete of all the objects
					entry.ltime++
				}
				nDB.indexes[byTable].Insert(fmt.Sprintf("/%s/%s/%s", tname, nid, key), entry)
				nDB.indexes[byNetwork].Insert(fmt.Sprintf("/%s/%s/%s", nid, tname, key), entry)
			} else {
				// the local node is leaving the network, all the entries of remote nodes can be safely removed
				nDB.indexes[byTable].Delete(fmt.Sprintf("/%s/%s/%s", tname, nid, key))
				nDB.indexes[byNetwork].Delete(fmt.Sprintf("/%s/%s/%s", nid, tname, key))
			}

			nDB.broadcaster.Write(makeEvent(opDelete, tname, nid, key, entry.value))
			return false
		})
}

func (nDB *NetworkDB) deleteNodeTableEntries(node string) {
	nDB.indexes[byTable].Walk(func(path string, v interface{}) bool {
		oldEntry := v.(*entry)
		if oldEntry.node != node {
			return false
		}

		params := strings.Split(path[1:], "/")
		tname := params[0]
		nid := params[1]
		key := params[2]

		nDB.indexes[byTable].Delete(fmt.Sprintf("/%s/%s/%s", tname, nid, key))
		nDB.indexes[byNetwork].Delete(fmt.Sprintf("/%s/%s/%s", nid, tname, key))

		nDB.broadcaster.Write(makeEvent(opDelete, tname, nid, key, oldEntry.value))
		return false
	})
}

// WalkTable walks a single table in NetworkDB and invokes the passed
// function for each entry in the table passing the network, key,
// value. The walk stops if the passed function returns a true.
func (nDB *NetworkDB) WalkTable(tname string, fn func(string, string, []byte, bool) bool) error {
	nDB.RLock()
	values := make(map[string]interface{})
	nDB.indexes[byTable].WalkPrefix(fmt.Sprintf("/%s", tname), func(path string, v interface{}) bool {
		values[path] = v
		return false
	})
	nDB.RUnlock()

	for k, v := range values {
		params := strings.Split(k[1:], "/")
		nid := params[1]
		key := params[2]
		if fn(nid, key, v.(*entry).value, v.(*entry).deleting) {
			return nil
		}
	}

	return nil
}

// JoinNetwork joins this node to a given network and propagates this
// event across the cluster. This triggers this node joining the
// sub-cluster of this network and participates in the network-scoped
// gossip and bulk sync for this network.
func (nDB *NetworkDB) JoinNetwork(nid string) error {
	ltime := nDB.networkClock.Increment()

	nDB.Lock()
	nodeNetworks, ok := nDB.networks[nDB.config.NodeName]
	if !ok {
		nodeNetworks = make(map[string]*network)
		nDB.networks[nDB.config.NodeName] = nodeNetworks
	}
	nodeNetworks[nid] = &network{id: nid, ltime: ltime}
	nodeNetworks[nid].tableBroadcasts = &memberlist.TransmitLimitedQueue{
		NumNodes: func() int {
			nDB.RLock()
			defer nDB.RUnlock()
			return len(nDB.networkNodes[nid])
		},
		RetransmitMult: 4,
	}
	nDB.addNetworkNode(nid, nDB.config.NodeName)
	networkNodes := nDB.networkNodes[nid]
	nDB.Unlock()

	if err := nDB.sendNetworkEvent(nid, NetworkEventTypeJoin, ltime); err != nil {
		return fmt.Errorf("failed to send leave network event for %s: %v", nid, err)
	}

	logrus.Debugf("%s: joined network %s", nDB.config.NodeName, nid)
	if _, err := nDB.bulkSync(networkNodes, true); err != nil {
		logrus.Errorf("Error bulk syncing while joining network %s: %v", nid, err)
	}

	return nil
}

// LeaveNetwork leaves this node from a given network and propagates
// this event across the cluster. This triggers this node leaving the
// sub-cluster of this network and as a result will no longer
// participate in the network-scoped gossip and bulk sync for this
// network. Also remove all the table entries for this network from
// networkdb
func (nDB *NetworkDB) LeaveNetwork(nid string) error {
	ltime := nDB.networkClock.Increment()
	if err := nDB.sendNetworkEvent(nid, NetworkEventTypeLeave, ltime); err != nil {
		return fmt.Errorf("failed to send leave network event for %s: %v", nid, err)
	}

	nDB.Lock()
	defer nDB.Unlock()

	// Remove myself from the list of the nodes participating to the network
	nDB.deleteNetworkNode(nid, nDB.config.NodeName)

	// Update all the local entries marking them for deletion and delete all the remote entries
	nDB.deleteNodeNetworkEntries(nid, nDB.config.NodeName)

	nodeNetworks, ok := nDB.networks[nDB.config.NodeName]
	if !ok {
		return fmt.Errorf("could not find self node for network %s while trying to leave", nid)
	}

	n, ok := nodeNetworks[nid]
	if !ok {
		return fmt.Errorf("could not find network %s while trying to leave", nid)
	}

	n.ltime = ltime
	n.reapTime = reapInterval
	n.leaving = true
	return nil
}

// addNetworkNode adds the node to the list of nodes which participate
// in the passed network only if it is not already present. Caller
// should hold the NetworkDB lock while calling this
func (nDB *NetworkDB) addNetworkNode(nid string, nodeName string) {
	nodes := nDB.networkNodes[nid]
	for _, node := range nodes {
		if node == nodeName {
			return
		}
	}

	nDB.networkNodes[nid] = append(nDB.networkNodes[nid], nodeName)
}

// Deletes the node from the list of nodes which participate in the
// passed network. Caller should hold the NetworkDB lock while calling
// this
func (nDB *NetworkDB) deleteNetworkNode(nid string, nodeName string) {
	nodes, ok := nDB.networkNodes[nid]
	if !ok || len(nodes) == 0 {
		return
	}
	newNodes := make([]string, 0, len(nodes)-1)
	for _, name := range nodes {
		if name == nodeName {
			continue
		}
		newNodes = append(newNodes, name)
	}
	nDB.networkNodes[nid] = newNodes
}

// findCommonnetworks find the networks that both this node and the
// passed node have joined.
func (nDB *NetworkDB) findCommonNetworks(nodeName string) []string {
	nDB.RLock()
	defer nDB.RUnlock()

	var networks []string
	for nid := range nDB.networks[nDB.config.NodeName] {
		if n, ok := nDB.networks[nodeName][nid]; ok {
			if !n.leaving {
				networks = append(networks, nid)
			}
		}
	}

	return networks
}

func (nDB *NetworkDB) updateLocalNetworkTime() {
	nDB.Lock()
	defer nDB.Unlock()

	ltime := nDB.networkClock.Increment()
	for _, n := range nDB.networks[nDB.config.NodeName] {
		n.ltime = ltime
	}
}
