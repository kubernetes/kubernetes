package raft

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
)

const (
	jsonPeerPath = "peers.json"
)

// PeerStore provides an interface for persistent storage and
// retrieval of peers. We use a separate interface than StableStore
// since the peers may need to be edited by a human operator. For example,
// in a two node cluster, the failure of either node requires human intervention
// since consensus is impossible.
type PeerStore interface {
	// Peers returns the list of known peers.
	Peers() ([]string, error)

	// SetPeers sets the list of known peers. This is invoked when a peer is
	// added or removed.
	SetPeers([]string) error
}

// StaticPeers is used to provide a static list of peers.
type StaticPeers struct {
	StaticPeers []string
	l           sync.Mutex
}

// Peers implements the PeerStore interface.
func (s *StaticPeers) Peers() ([]string, error) {
	s.l.Lock()
	peers := s.StaticPeers
	s.l.Unlock()
	return peers, nil
}

// SetPeers implements the PeerStore interface.
func (s *StaticPeers) SetPeers(p []string) error {
	s.l.Lock()
	s.StaticPeers = p
	s.l.Unlock()
	return nil
}

// JSONPeers is used to provide peer persistence on disk in the form
// of a JSON file. This allows human operators to manipulate the file.
type JSONPeers struct {
	l     sync.Mutex
	path  string
	trans Transport
}

// NewJSONPeers creates a new JSONPeers store. Requires a transport
// to handle the serialization of network addresses.
func NewJSONPeers(base string, trans Transport) *JSONPeers {
	path := filepath.Join(base, jsonPeerPath)
	store := &JSONPeers{
		path:  path,
		trans: trans,
	}
	return store
}

// Peers implements the PeerStore interface.
func (j *JSONPeers) Peers() ([]string, error) {
	j.l.Lock()
	defer j.l.Unlock()

	// Read the file
	buf, err := ioutil.ReadFile(j.path)
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}

	// Check for no peers
	if len(buf) == 0 {
		return nil, nil
	}

	// Decode the peers
	var peerSet []string
	dec := json.NewDecoder(bytes.NewReader(buf))
	if err := dec.Decode(&peerSet); err != nil {
		return nil, err
	}

	// Deserialize each peer
	var peers []string
	for _, p := range peerSet {
		peers = append(peers, j.trans.DecodePeer([]byte(p)))
	}
	return peers, nil
}

// SetPeers implements the PeerStore interface.
func (j *JSONPeers) SetPeers(peers []string) error {
	j.l.Lock()
	defer j.l.Unlock()

	// Encode each peer
	var peerSet []string
	for _, p := range peers {
		peerSet = append(peerSet, string(j.trans.EncodePeer(p)))
	}

	// Convert to JSON
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	if err := enc.Encode(peerSet); err != nil {
		return err
	}

	// Write out as JSON
	return ioutil.WriteFile(j.path, buf.Bytes(), 0755)
}
