package networkdb

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log"
	"math/big"
	rnd "math/rand"
	"net"
	"strings"
	"time"

	"github.com/hashicorp/memberlist"
	"github.com/sirupsen/logrus"
)

const (
	reapInterval     = 30 * time.Minute
	reapPeriod       = 5 * time.Second
	retryInterval    = 1 * time.Second
	nodeReapInterval = 24 * time.Hour
	nodeReapPeriod   = 2 * time.Hour
)

type logWriter struct{}

func (l *logWriter) Write(p []byte) (int, error) {
	str := string(p)
	str = strings.TrimSuffix(str, "\n")

	switch {
	case strings.HasPrefix(str, "[WARN] "):
		str = strings.TrimPrefix(str, "[WARN] ")
		logrus.Warn(str)
	case strings.HasPrefix(str, "[DEBUG] "):
		str = strings.TrimPrefix(str, "[DEBUG] ")
		logrus.Debug(str)
	case strings.HasPrefix(str, "[INFO] "):
		str = strings.TrimPrefix(str, "[INFO] ")
		logrus.Info(str)
	case strings.HasPrefix(str, "[ERR] "):
		str = strings.TrimPrefix(str, "[ERR] ")
		logrus.Warn(str)
	}

	return len(p), nil
}

// SetKey adds a new key to the key ring
func (nDB *NetworkDB) SetKey(key []byte) {
	logrus.Debugf("Adding key %s", hex.EncodeToString(key)[0:5])
	nDB.Lock()
	defer nDB.Unlock()
	for _, dbKey := range nDB.config.Keys {
		if bytes.Equal(key, dbKey) {
			return
		}
	}
	nDB.config.Keys = append(nDB.config.Keys, key)
	if nDB.keyring != nil {
		nDB.keyring.AddKey(key)
	}
}

// SetPrimaryKey sets the given key as the primary key. This should have
// been added apriori through SetKey
func (nDB *NetworkDB) SetPrimaryKey(key []byte) {
	logrus.Debugf("Primary Key %s", hex.EncodeToString(key)[0:5])
	nDB.RLock()
	defer nDB.RUnlock()
	for _, dbKey := range nDB.config.Keys {
		if bytes.Equal(key, dbKey) {
			if nDB.keyring != nil {
				nDB.keyring.UseKey(dbKey)
			}
			break
		}
	}
}

// RemoveKey removes a key from the key ring. The key being removed
// can't be the primary key
func (nDB *NetworkDB) RemoveKey(key []byte) {
	logrus.Debugf("Remove Key %s", hex.EncodeToString(key)[0:5])
	nDB.Lock()
	defer nDB.Unlock()
	for i, dbKey := range nDB.config.Keys {
		if bytes.Equal(key, dbKey) {
			nDB.config.Keys = append(nDB.config.Keys[:i], nDB.config.Keys[i+1:]...)
			if nDB.keyring != nil {
				nDB.keyring.RemoveKey(dbKey)
			}
			break
		}
	}
}

func (nDB *NetworkDB) clusterInit() error {
	nDB.lastStatsTimestamp = time.Now()
	nDB.lastHealthTimestamp = nDB.lastStatsTimestamp

	config := memberlist.DefaultLANConfig()
	config.Name = nDB.config.NodeName
	config.BindAddr = nDB.config.BindAddr
	config.AdvertiseAddr = nDB.config.AdvertiseAddr
	config.UDPBufferSize = nDB.config.PacketBufferSize

	if nDB.config.BindPort != 0 {
		config.BindPort = nDB.config.BindPort
	}

	config.ProtocolVersion = memberlist.ProtocolVersion2Compatible
	config.Delegate = &delegate{nDB: nDB}
	config.Events = &eventDelegate{nDB: nDB}
	// custom logger that does not add time or date, so they are not
	// duplicated by logrus
	config.Logger = log.New(&logWriter{}, "", 0)

	var err error
	if len(nDB.config.Keys) > 0 {
		for i, key := range nDB.config.Keys {
			logrus.Debugf("Encryption key %d: %s", i+1, hex.EncodeToString(key)[0:5])
		}
		nDB.keyring, err = memberlist.NewKeyring(nDB.config.Keys, nDB.config.Keys[0])
		if err != nil {
			return err
		}
		config.Keyring = nDB.keyring
	}

	nDB.networkBroadcasts = &memberlist.TransmitLimitedQueue{
		NumNodes: func() int {
			nDB.RLock()
			num := len(nDB.nodes)
			nDB.RUnlock()
			return num
		},
		RetransmitMult: config.RetransmitMult,
	}

	nDB.nodeBroadcasts = &memberlist.TransmitLimitedQueue{
		NumNodes: func() int {
			nDB.RLock()
			num := len(nDB.nodes)
			nDB.RUnlock()
			return num
		},
		RetransmitMult: config.RetransmitMult,
	}

	mlist, err := memberlist.Create(config)
	if err != nil {
		return fmt.Errorf("failed to create memberlist: %v", err)
	}

	nDB.stopCh = make(chan struct{})
	nDB.memberlist = mlist

	for _, trigger := range []struct {
		interval time.Duration
		fn       func()
	}{
		{reapPeriod, nDB.reapState},
		{config.GossipInterval, nDB.gossip},
		{config.PushPullInterval, nDB.bulkSyncTables},
		{retryInterval, nDB.reconnectNode},
		{nodeReapPeriod, nDB.reapDeadNode},
	} {
		t := time.NewTicker(trigger.interval)
		go nDB.triggerFunc(trigger.interval, t.C, nDB.stopCh, trigger.fn)
		nDB.tickers = append(nDB.tickers, t)
	}

	return nil
}

func (nDB *NetworkDB) retryJoin(members []string, stop <-chan struct{}) {
	t := time.NewTicker(retryInterval)
	defer t.Stop()

	for {
		select {
		case <-t.C:
			if _, err := nDB.memberlist.Join(members); err != nil {
				logrus.Errorf("Failed to join memberlist %s on retry: %v", members, err)
				continue
			}
			if err := nDB.sendNodeEvent(NodeEventTypeJoin); err != nil {
				logrus.Errorf("failed to send node join on retry: %v", err)
				continue
			}
			return
		case <-stop:
			return
		}
	}

}

func (nDB *NetworkDB) clusterJoin(members []string) error {
	mlist := nDB.memberlist

	if _, err := mlist.Join(members); err != nil {
		// In case of failure, keep retrying join until it succeeds or the cluster is shutdown.
		go nDB.retryJoin(members, nDB.stopCh)
		return fmt.Errorf("could not join node to memberlist: %v", err)
	}

	if err := nDB.sendNodeEvent(NodeEventTypeJoin); err != nil {
		return fmt.Errorf("failed to send node join: %v", err)
	}

	return nil
}

func (nDB *NetworkDB) clusterLeave() error {
	mlist := nDB.memberlist

	if err := nDB.sendNodeEvent(NodeEventTypeLeave); err != nil {
		logrus.Errorf("failed to send node leave: %v", err)
	}

	if err := mlist.Leave(time.Second); err != nil {
		return err
	}

	close(nDB.stopCh)

	for _, t := range nDB.tickers {
		t.Stop()
	}

	return mlist.Shutdown()
}

func (nDB *NetworkDB) triggerFunc(stagger time.Duration, C <-chan time.Time, stop <-chan struct{}, f func()) {
	// Use a random stagger to avoid syncronizing
	randStagger := time.Duration(uint64(rnd.Int63()) % uint64(stagger))
	select {
	case <-time.After(randStagger):
	case <-stop:
		return
	}
	for {
		select {
		case <-C:
			f()
		case <-stop:
			return
		}
	}
}

func (nDB *NetworkDB) reapDeadNode() {
	nDB.Lock()
	defer nDB.Unlock()
	for id, n := range nDB.failedNodes {
		if n.reapTime > 0 {
			n.reapTime -= nodeReapPeriod
			continue
		}
		logrus.Debugf("Removing failed node %v from gossip cluster", n.Name)
		delete(nDB.failedNodes, id)
	}
}

func (nDB *NetworkDB) reconnectNode() {
	nDB.RLock()
	if len(nDB.failedNodes) == 0 {
		nDB.RUnlock()
		return
	}

	nodes := make([]*node, 0, len(nDB.failedNodes))
	for _, n := range nDB.failedNodes {
		nodes = append(nodes, n)
	}
	nDB.RUnlock()

	node := nodes[randomOffset(len(nodes))]
	addr := net.UDPAddr{IP: node.Addr, Port: int(node.Port)}

	if _, err := nDB.memberlist.Join([]string{addr.String()}); err != nil {
		return
	}

	if err := nDB.sendNodeEvent(NodeEventTypeJoin); err != nil {
		return
	}

	logrus.Debugf("Initiating bulk sync with node %s after reconnect", node.Name)
	nDB.bulkSync([]string{node.Name}, true)
}

// For timing the entry deletion in the repaer APIs that doesn't use monotonic clock
// source (time.Now, Sub etc.) should be avoided. Hence we use reapTime in every
// entry which is set initially to reapInterval and decremented by reapPeriod every time
// the reaper runs. NOTE nDB.reapTableEntries updates the reapTime with a readlock. This
// is safe as long as no other concurrent path touches the reapTime field.
func (nDB *NetworkDB) reapState() {
	nDB.reapNetworks()
	nDB.reapTableEntries()
}

func (nDB *NetworkDB) reapNetworks() {
	nDB.Lock()
	for _, nn := range nDB.networks {
		for id, n := range nn {
			if n.leaving {
				if n.reapTime <= 0 {
					delete(nn, id)
					continue
				}
				n.reapTime -= reapPeriod
			}
		}
	}
	nDB.Unlock()
}

func (nDB *NetworkDB) reapTableEntries() {
	var paths []string

	nDB.RLock()
	nDB.indexes[byTable].Walk(func(path string, v interface{}) bool {
		entry, ok := v.(*entry)
		if !ok {
			return false
		}

		if !entry.deleting {
			return false
		}
		if entry.reapTime > 0 {
			entry.reapTime -= reapPeriod
			return false
		}
		paths = append(paths, path)
		return false
	})
	nDB.RUnlock()

	nDB.Lock()
	for _, path := range paths {
		params := strings.Split(path[1:], "/")
		tname := params[0]
		nid := params[1]
		key := params[2]

		if _, ok := nDB.indexes[byTable].Delete(fmt.Sprintf("/%s/%s/%s", tname, nid, key)); !ok {
			logrus.Errorf("Could not delete entry in table %s with network id %s and key %s as it does not exist", tname, nid, key)
		}

		if _, ok := nDB.indexes[byNetwork].Delete(fmt.Sprintf("/%s/%s/%s", nid, tname, key)); !ok {
			logrus.Errorf("Could not delete entry in network %s with table name %s and key %s as it does not exist", nid, tname, key)
		}
	}
	nDB.Unlock()
}

func (nDB *NetworkDB) gossip() {
	networkNodes := make(map[string][]string)
	nDB.RLock()
	thisNodeNetworks := nDB.networks[nDB.config.NodeName]
	for nid := range thisNodeNetworks {
		networkNodes[nid] = nDB.networkNodes[nid]

	}
	printStats := time.Since(nDB.lastStatsTimestamp) >= nDB.config.StatsPrintPeriod
	printHealth := time.Since(nDB.lastHealthTimestamp) >= nDB.config.HealthPrintPeriod
	nDB.RUnlock()

	if printHealth {
		healthScore := nDB.memberlist.GetHealthScore()
		if healthScore != 0 {
			logrus.Warnf("NetworkDB stats - healthscore:%d (connectivity issues)", healthScore)
		}
		nDB.lastHealthTimestamp = time.Now()
	}

	for nid, nodes := range networkNodes {
		mNodes := nDB.mRandomNodes(3, nodes)
		bytesAvail := nDB.config.PacketBufferSize - compoundHeaderOverhead

		nDB.RLock()
		network, ok := thisNodeNetworks[nid]
		nDB.RUnlock()
		if !ok || network == nil {
			// It is normal for the network to be removed
			// between the time we collect the network
			// attachments of this node and processing
			// them here.
			continue
		}

		broadcastQ := network.tableBroadcasts

		if broadcastQ == nil {
			logrus.Errorf("Invalid broadcastQ encountered while gossiping for network %s", nid)
			continue
		}

		msgs := broadcastQ.GetBroadcasts(compoundOverhead, bytesAvail)
		// Collect stats and print the queue info, note this code is here also to have a view of the queues empty
		network.qMessagesSent += len(msgs)
		if printStats {
			logrus.Infof("NetworkDB stats - Queue net:%s qLen:%d netPeers:%d netMsg/s:%d",
				nid, broadcastQ.NumQueued(), broadcastQ.NumNodes(), network.qMessagesSent/int((nDB.config.StatsPrintPeriod/time.Second)))
			network.qMessagesSent = 0
		}

		if len(msgs) == 0 {
			continue
		}

		// Create a compound message
		compound := makeCompoundMessage(msgs)

		for _, node := range mNodes {
			nDB.RLock()
			mnode := nDB.nodes[node]
			nDB.RUnlock()

			if mnode == nil {
				break
			}

			// Send the compound message
			if err := nDB.memberlist.SendBestEffort(&mnode.Node, compound); err != nil {
				logrus.Errorf("Failed to send gossip to %s: %s", mnode.Addr, err)
			}
		}
	}
	// Reset the stats
	if printStats {
		nDB.lastStatsTimestamp = time.Now()
	}
}

func (nDB *NetworkDB) bulkSyncTables() {
	var networks []string
	nDB.RLock()
	for nid, network := range nDB.networks[nDB.config.NodeName] {
		if network.leaving {
			continue
		}
		networks = append(networks, nid)
	}
	nDB.RUnlock()

	for {
		if len(networks) == 0 {
			break
		}

		nid := networks[0]
		networks = networks[1:]

		nDB.RLock()
		nodes := nDB.networkNodes[nid]
		nDB.RUnlock()

		// No peer nodes on this network. Move on.
		if len(nodes) == 0 {
			continue
		}

		completed, err := nDB.bulkSync(nodes, false)
		if err != nil {
			logrus.Errorf("periodic bulk sync failure for network %s: %v", nid, err)
			continue
		}

		// Remove all the networks for which we have
		// successfully completed bulk sync in this iteration.
		updatedNetworks := make([]string, 0, len(networks))
		for _, nid := range networks {
			var found bool
			for _, completedNid := range completed {
				if nid == completedNid {
					found = true
					break
				}
			}

			if !found {
				updatedNetworks = append(updatedNetworks, nid)
			}
		}

		networks = updatedNetworks
	}
}

func (nDB *NetworkDB) bulkSync(nodes []string, all bool) ([]string, error) {
	if !all {
		// Get 2 random nodes. 2nd node will be tried if the bulk sync to
		// 1st node fails.
		nodes = nDB.mRandomNodes(2, nodes)
	}

	if len(nodes) == 0 {
		return nil, nil
	}

	var err error
	var networks []string
	for _, node := range nodes {
		if node == nDB.config.NodeName {
			continue
		}
		logrus.Debugf("%s: Initiating bulk sync with node %v", nDB.config.NodeName, node)
		networks = nDB.findCommonNetworks(node)
		err = nDB.bulkSyncNode(networks, node, true)
		// if its periodic bulksync stop after the first successful sync
		if !all && err == nil {
			break
		}
		if err != nil {
			err = fmt.Errorf("bulk sync to node %s failed: %v", node, err)
			logrus.Warn(err.Error())
		}
	}

	if err != nil {
		return nil, err
	}

	return networks, nil
}

// Bulk sync all the table entries belonging to a set of networks to a
// single peer node. It can be unsolicited or can be in response to an
// unsolicited bulk sync
func (nDB *NetworkDB) bulkSyncNode(networks []string, node string, unsolicited bool) error {
	var msgs [][]byte

	var unsolMsg string
	if unsolicited {
		unsolMsg = "unsolicited"
	}

	logrus.Debugf("%s: Initiating %s bulk sync for networks %v with node %s", nDB.config.NodeName, unsolMsg, networks, node)

	nDB.RLock()
	mnode := nDB.nodes[node]
	if mnode == nil {
		nDB.RUnlock()
		return nil
	}

	for _, nid := range networks {
		nDB.indexes[byNetwork].WalkPrefix(fmt.Sprintf("/%s", nid), func(path string, v interface{}) bool {
			entry, ok := v.(*entry)
			if !ok {
				return false
			}

			eType := TableEventTypeCreate
			if entry.deleting {
				eType = TableEventTypeDelete
			}

			params := strings.Split(path[1:], "/")
			tEvent := TableEvent{
				Type:      eType,
				LTime:     entry.ltime,
				NodeName:  entry.node,
				NetworkID: nid,
				TableName: params[1],
				Key:       params[2],
				Value:     entry.value,
			}

			msg, err := encodeMessage(MessageTypeTableEvent, &tEvent)
			if err != nil {
				logrus.Errorf("Encode failure during bulk sync: %#v", tEvent)
				return false
			}

			msgs = append(msgs, msg)
			return false
		})
	}
	nDB.RUnlock()

	// Create a compound message
	compound := makeCompoundMessage(msgs)

	bsm := BulkSyncMessage{
		LTime:       nDB.tableClock.Time(),
		Unsolicited: unsolicited,
		NodeName:    nDB.config.NodeName,
		Networks:    networks,
		Payload:     compound,
	}

	buf, err := encodeMessage(MessageTypeBulkSync, &bsm)
	if err != nil {
		return fmt.Errorf("failed to encode bulk sync message: %v", err)
	}

	nDB.Lock()
	ch := make(chan struct{})
	nDB.bulkSyncAckTbl[node] = ch
	nDB.Unlock()

	err = nDB.memberlist.SendReliable(&mnode.Node, buf)
	if err != nil {
		nDB.Lock()
		delete(nDB.bulkSyncAckTbl, node)
		nDB.Unlock()

		return fmt.Errorf("failed to send a TCP message during bulk sync: %v", err)
	}

	// Wait on a response only if it is unsolicited.
	if unsolicited {
		startTime := time.Now()
		t := time.NewTimer(30 * time.Second)
		select {
		case <-t.C:
			logrus.Errorf("Bulk sync to node %s timed out", node)
		case <-ch:
			logrus.Debugf("%s: Bulk sync to node %s took %s", nDB.config.NodeName, node, time.Since(startTime))
		}
		t.Stop()
	}

	return nil
}

// Returns a random offset between 0 and n
func randomOffset(n int) int {
	if n == 0 {
		return 0
	}

	val, err := rand.Int(rand.Reader, big.NewInt(int64(n)))
	if err != nil {
		logrus.Errorf("Failed to get a random offset: %v", err)
		return 0
	}

	return int(val.Int64())
}

// mRandomNodes is used to select up to m random nodes. It is possible
// that less than m nodes are returned.
func (nDB *NetworkDB) mRandomNodes(m int, nodes []string) []string {
	n := len(nodes)
	mNodes := make([]string, 0, m)
OUTER:
	// Probe up to 3*n times, with large n this is not necessary
	// since k << n, but with small n we want search to be
	// exhaustive
	for i := 0; i < 3*n && len(mNodes) < m; i++ {
		// Get random node
		idx := randomOffset(n)
		node := nodes[idx]

		if node == nDB.config.NodeName {
			continue
		}

		// Check if we have this node already
		for j := 0; j < len(mNodes); j++ {
			if node == mNodes[j] {
				continue OUTER
			}
		}

		// Append the node
		mNodes = append(mNodes, node)
	}

	return mNodes
}
