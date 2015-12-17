package meta

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/hashicorp/raft"
	"github.com/hashicorp/raft-boltdb"
)

// raftState abstracts the interaction of the raft consensus layer
// across local or remote nodes.  It is a form of the state design pattern and allows
// the meta.Store to change its behavior with the raft layer at runtime.
type raftState interface {
	open() error
	remove() error
	initialize() error
	leader() string
	isLeader() bool
	sync(index uint64, timeout time.Duration) error
	setPeers(addrs []string) error
	addPeer(addr string) error
	removePeer(addr string) error
	peers() ([]string, error)
	invalidate() error
	close() error
	lastIndex() uint64
	apply(b []byte) error
	snapshot() error
	isLocal() bool
}

// localRaft is a consensus strategy that uses a local raft implementation for
// consensus operations.
type localRaft struct {
	wg        sync.WaitGroup
	closing   chan struct{}
	store     *Store
	raft      *raft.Raft
	transport *raft.NetworkTransport
	peerStore raft.PeerStore
	raftStore *raftboltdb.BoltStore
	raftLayer *raftLayer
}

func (r *localRaft) remove() error {
	if err := os.RemoveAll(filepath.Join(r.store.path, "raft.db")); err != nil {
		return err
	}
	if err := os.RemoveAll(filepath.Join(r.store.path, "peers.json")); err != nil {
		return err
	}
	if err := os.RemoveAll(filepath.Join(r.store.path, "snapshots")); err != nil {
		return err
	}
	return nil
}

func (r *localRaft) updateMetaData(ms *Data) {
	if ms == nil {
		return
	}

	updated := false
	r.store.mu.RLock()
	if ms.Index > r.store.data.Index {
		updated = true
	}
	r.store.mu.RUnlock()

	if updated {
		r.store.Logger.Printf("Updating metastore to term=%v index=%v", ms.Term, ms.Index)
		r.store.mu.Lock()
		r.store.data = ms
		// Signal any blocked goroutines that the meta store has been updated
		r.store.notifyChanged()
		r.store.mu.Unlock()
	}
}

func (r *localRaft) invalidate() error {
	if r.store.IsLeader() {
		return nil
	}

	ms, err := r.store.rpc.fetchMetaData(false)
	if err != nil {
		return fmt.Errorf("error fetching meta data: %s", err)
	}

	r.updateMetaData(ms)
	return nil
}

func (r *localRaft) open() error {
	r.closing = make(chan struct{})

	s := r.store
	// Setup raft configuration.
	config := raft.DefaultConfig()
	config.LogOutput = ioutil.Discard

	if s.clusterTracingEnabled {
		config.Logger = s.Logger
	}
	config.HeartbeatTimeout = s.HeartbeatTimeout
	config.ElectionTimeout = s.ElectionTimeout
	config.LeaderLeaseTimeout = s.LeaderLeaseTimeout
	config.CommitTimeout = s.CommitTimeout
	// Since we actually never call `removePeer` this is safe.
	// If in the future we decide to call remove peer we have to re-evaluate how to handle this
	config.ShutdownOnRemove = false

	// If no peers are set in the config or there is one and we are it, then start as a single server.
	if len(s.peers) <= 1 {
		config.EnableSingleNode = true
		// Ensure we can always become the leader
		config.DisableBootstrapAfterElect = false
	}

	// Build raft layer to multiplex listener.
	r.raftLayer = newRaftLayer(s.RaftListener, s.RemoteAddr)

	// Create a transport layer
	r.transport = raft.NewNetworkTransport(r.raftLayer, 3, 10*time.Second, config.LogOutput)

	// Create peer storage.
	r.peerStore = raft.NewJSONPeers(s.path, r.transport)

	peers, err := r.peerStore.Peers()
	if err != nil {
		return err
	}

	// For single-node clusters, we can update the raft peers before we start the cluster if the hostname
	// has changed.
	if config.EnableSingleNode {
		if err := r.peerStore.SetPeers([]string{s.RemoteAddr.String()}); err != nil {
			return err
		}
		peers = []string{s.RemoteAddr.String()}
	}

	// If we have multiple nodes in the cluster, make sure our address is in the raft peers or
	// we won't be able to boot into the cluster because the other peers will reject our new hostname.  This
	// is difficult to resolve automatically because we need to have all the raft peers agree on the current members
	// of the cluster before we can change them.
	if len(peers) > 0 && !raft.PeerContained(peers, s.RemoteAddr.String()) {
		s.Logger.Printf("%s is not in the list of raft peers. Please update %v/peers.json on all raft nodes to have the same contents.", s.RemoteAddr.String(), s.Path())
		return fmt.Errorf("peers out of sync: %v not in %v", s.RemoteAddr.String(), peers)
	}

	// Create the log store and stable store.
	store, err := raftboltdb.NewBoltStore(filepath.Join(s.path, "raft.db"))
	if err != nil {
		return fmt.Errorf("new bolt store: %s", err)
	}
	r.raftStore = store

	// Create the snapshot store.
	snapshots, err := raft.NewFileSnapshotStore(s.path, raftSnapshotsRetained, os.Stderr)
	if err != nil {
		return fmt.Errorf("file snapshot store: %s", err)
	}

	// Create raft log.
	ra, err := raft.NewRaft(config, (*storeFSM)(s), store, store, snapshots, r.peerStore, r.transport)
	if err != nil {
		return fmt.Errorf("new raft: %s", err)
	}
	r.raft = ra

	r.wg.Add(1)
	go r.logLeaderChanges()

	return nil
}

func (r *localRaft) logLeaderChanges() {
	defer r.wg.Done()
	// Logs our current state (Node at 1.2.3.4:8088 [Follower])
	r.store.Logger.Printf(r.raft.String())
	for {
		select {
		case <-r.closing:
			return
		case <-r.raft.LeaderCh():
			peers, err := r.peers()
			if err != nil {
				r.store.Logger.Printf("failed to lookup peers: %v", err)
			}
			r.store.Logger.Printf("%v. peers=%v", r.raft.String(), peers)
		}
	}
}

func (r *localRaft) close() error {
	if r.closing != nil {
		close(r.closing)
	}
	r.wg.Wait()

	if r.transport != nil {
		r.transport.Close()
		r.transport = nil
	}

	// Shutdown raft.
	if r.raft != nil {
		if err := r.raft.Shutdown().Error(); err != nil {
			return err
		}
		r.raft = nil
	}

	if r.raftStore != nil {
		r.raftStore.Close()
		r.raftStore = nil
	}

	return nil
}

func (r *localRaft) initialize() error {
	s := r.store
	// If we have committed entries then the store is already in the cluster.
	if index, err := r.raftStore.LastIndex(); err != nil {
		return fmt.Errorf("last index: %s", err)
	} else if index > 0 {
		return nil
	}

	// Force set peers.
	if err := r.setPeers(s.peers); err != nil {
		return fmt.Errorf("set raft peers: %s", err)
	}

	return nil
}

// apply applies a serialized command to the raft log.
func (r *localRaft) apply(b []byte) error {
	// Apply to raft log.
	f := r.raft.Apply(b, 0)
	if err := f.Error(); err != nil {
		return err
	}

	// Return response if it's an error.
	// No other non-nil objects should be returned.
	resp := f.Response()
	if err, ok := resp.(error); ok {
		return lookupError(err)
	}
	assert(resp == nil, "unexpected response: %#v", resp)

	return nil
}

func (r *localRaft) lastIndex() uint64 {
	return r.raft.LastIndex()
}

func (r *localRaft) sync(index uint64, timeout time.Duration) error {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		// Wait for next tick or timeout.
		select {
		case <-ticker.C:
		case <-timer.C:
			return errors.New("timeout")
		}

		// Compare index against current metadata.
		r.store.mu.Lock()
		ok := (r.store.data.Index >= index)
		r.store.mu.Unlock()

		// Exit if we are at least at the given index.
		if ok {
			return nil
		}
	}
}

func (r *localRaft) snapshot() error {
	future := r.raft.Snapshot()
	return future.Error()
}

// addPeer adds addr to the list of peers in the cluster.
func (r *localRaft) addPeer(addr string) error {
	peers, err := r.peerStore.Peers()
	if err != nil {
		return err
	}

	if len(peers) >= 3 {
		return nil
	}

	if fut := r.raft.AddPeer(addr); fut.Error() != nil {
		return fut.Error()
	}
	return nil
}

// removePeer removes addr from the list of peers in the cluster.
func (r *localRaft) removePeer(addr string) error {
	// Only do this on the leader
	if !r.isLeader() {
		return errors.New("not the leader")
	}
	if fut := r.raft.RemovePeer(addr); fut.Error() != nil {
		return fut.Error()
	}
	return nil
}

// setPeers sets a list of peers in the cluster.
func (r *localRaft) setPeers(addrs []string) error {
	return r.raft.SetPeers(addrs).Error()
}

func (r *localRaft) peers() ([]string, error) {
	return r.peerStore.Peers()
}

func (r *localRaft) leader() string {
	if r.raft == nil {
		return ""
	}

	return r.raft.Leader()
}

func (r *localRaft) isLeader() bool {
	if r.raft == nil {
		return false
	}
	return r.raft.State() == raft.Leader
}

func (r *localRaft) isLocal() bool {
	return true
}

// remoteRaft is a consensus strategy that uses a remote raft cluster for
// consensus operations.
type remoteRaft struct {
	store *Store
}

func (r *remoteRaft) remove() error {
	return nil
}

func (r *remoteRaft) updateMetaData(ms *Data) {
	if ms == nil {
		return
	}

	updated := false
	r.store.mu.RLock()
	if ms.Index > r.store.data.Index {
		updated = true
	}
	r.store.mu.RUnlock()

	if updated {
		r.store.Logger.Printf("Updating metastore to term=%v index=%v", ms.Term, ms.Index)
		r.store.mu.Lock()
		r.store.data = ms
		// Signal any blocked goroutines that the meta store has been updated
		r.store.notifyChanged()
		r.store.mu.Unlock()
	}
}

func (r *remoteRaft) invalidate() error {
	ms, err := r.store.rpc.fetchMetaData(false)
	if err != nil {
		return fmt.Errorf("error fetching meta data: %s", err)
	}

	r.updateMetaData(ms)
	return nil
}

func (r *remoteRaft) setPeers(addrs []string) error {
	// Convert to JSON
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	if err := enc.Encode(addrs); err != nil {
		return err
	}

	// Write out as JSON
	return ioutil.WriteFile(filepath.Join(r.store.path, "peers.json"), buf.Bytes(), 0755)
}

// addPeer adds addr to the list of peers in the cluster.
func (r *remoteRaft) addPeer(addr string) error {
	return fmt.Errorf("cannot add peer using remote raft")
}

// removePeer does nothing for remoteRaft.
func (r *remoteRaft) removePeer(addr string) error {
	return nil
}

func (r *remoteRaft) peers() ([]string, error) {
	return readPeersJSON(filepath.Join(r.store.path, "peers.json"))
}

func (r *remoteRaft) open() error {
	if err := r.setPeers(r.store.peers); err != nil {
		return err
	}

	go func() {
		for {
			select {
			case <-r.store.closing:
				return
			default:
			}

			ms, err := r.store.rpc.fetchMetaData(true)
			if err != nil {
				r.store.Logger.Printf("fetch metastore: %v", err)
				time.Sleep(time.Second)
				continue
			}
			r.updateMetaData(ms)
		}
	}()
	return nil
}

func (r *remoteRaft) close() error {
	return nil
}

// apply applies a serialized command to the raft log.
func (r *remoteRaft) apply(b []byte) error {
	return fmt.Errorf("cannot apply log while in remote raft state")
}

func (r *remoteRaft) initialize() error {
	return nil
}

func (r *remoteRaft) leader() string {
	if len(r.store.peers) == 0 {
		return ""
	}

	return r.store.peers[rand.Intn(len(r.store.peers))]
}

func (r *remoteRaft) isLeader() bool {
	return false
}

func (r *remoteRaft) isLocal() bool {
	return false
}

func (r *remoteRaft) lastIndex() uint64 {
	return r.store.cachedData().Index
}

func (r *remoteRaft) sync(index uint64, timeout time.Duration) error {
	//FIXME: jwilder: check index and timeout
	return r.store.invalidate()
}

func (r *remoteRaft) snapshot() error {
	return fmt.Errorf("cannot snapshot while in remote raft state")
}

func readPeersJSON(path string) ([]string, error) {
	// Read the file
	buf, err := ioutil.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}

	// Check for no peers
	if len(buf) == 0 {
		return nil, nil
	}

	// Decode the peers
	var peers []string
	dec := json.NewDecoder(bytes.NewReader(buf))
	if err := dec.Decode(&peers); err != nil {
		return nil, err
	}

	return peers, nil
}
