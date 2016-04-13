package raft

import (
	"io/ioutil"
	"os"
	"testing"
)

func TestJSONPeers(t *testing.T) {
	// Create a test dir
	dir, err := ioutil.TempDir("", "raft")
	if err != nil {
		t.Fatalf("err: %v ", err)
	}
	defer os.RemoveAll(dir)

	// Create the store
	_, trans := NewInmemTransport()
	store := NewJSONPeers(dir, trans)

	// Try a read, should get nothing
	peers, err := store.Peers()
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(peers) != 0 {
		t.Fatalf("peers: %v", peers)
	}

	// Initialize some peers
	newPeers := []string{NewInmemAddr(), NewInmemAddr(), NewInmemAddr()}
	if err := store.SetPeers(newPeers); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Try a read, should peers
	peers, err = store.Peers()
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(peers) != 3 {
		t.Fatalf("peers: %v", peers)
	}
}
