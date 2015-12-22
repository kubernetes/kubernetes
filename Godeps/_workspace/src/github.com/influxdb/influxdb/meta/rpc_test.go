package meta

import (
	"net"
	"sync"
	"testing"
)

func TestRPCFetchData(t *testing.T) {

	serverRPC := &rpc{
		store: &fakeStore{
			md: &Data{Index: 99},
		},
	}

	srv := newTestServer(t, serverRPC)
	defer srv.Close()
	go srv.Serve()

	// Wait for the RPC server to be ready
	<-srv.Ready

	// create a new RPC with no existing meta.Data cache
	clientRPC := &rpc{
		store: &fakeStore{
			leader: srv.Listener.Addr().String(),
		},
	}

	// fetch the servers meta-data
	md, err := clientRPC.fetchMetaData(false)
	if err != nil {
		t.Fatalf("failed to fetchMetaData: %v", err)
	}

	if md == nil {
		t.Fatalf("meta-data is nil")
	}

	if exp := uint64(99); md.Index != exp {
		t.Fatalf("meta-data mismatch. got %v, exp %v", md.Index, exp)
	}
}

func TestRPCFetchDataMatchesLeader(t *testing.T) {
	serverRPC := &rpc{
		store: &fakeStore{
			md: &Data{Index: 99},
		},
	}

	srv := newTestServer(t, serverRPC)
	defer srv.Close()
	go srv.Serve()

	// Wait for the RPC server to be ready
	<-srv.Ready

	// create a new RPC with a matching index as the server
	clientRPC := &rpc{
		store: &fakeStore{
			leader: srv.Listener.Addr().String(),
			md:     &Data{Index: 99},
		},
	}

	// fetch the servers meta-data
	md, err := clientRPC.fetchMetaData(false)
	if err != nil {
		t.Fatalf("failed to fetchMetaData: %v", err)
	}

	if md != nil {
		t.Fatalf("meta-data is not nil")
	}
}

func TestRPCFetchDataMatchesBlocking(t *testing.T) {
	fs := &fakeStore{
		md:        &Data{Index: 99},
		blockChan: make(chan struct{}),
	}
	serverRPC := &rpc{
		store: fs,
	}

	srv := newTestServer(t, serverRPC)
	defer srv.Close()
	go srv.Serve()

	// Wait for the RPC server to be ready
	<-srv.Ready

	// create a new RPC with a matching index as the server
	clientRPC := &rpc{
		store: &fakeStore{
			leader: srv.Listener.Addr().String(),
			md:     &Data{Index: 99},
		},
	}

	// Kick off the fetching block
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		// fetch the servers meta-data
		md, err := clientRPC.fetchMetaData(true)
		if err != nil {
			t.Fatalf("failed to fetchMetaData: %v", err)
		}

		if md == nil {
			t.Fatalf("meta-data is nil")
		}

		if exp := uint64(100); md.Index != exp {
			t.Fatalf("meta-data mismatch. got %v, exp %v", md.Index, exp)
		}
	}()

	// Simulate the rmote index changing and unblocking
	fs.mu.Lock()
	fs.md = &Data{Index: 100}
	fs.mu.Unlock()
	close(fs.blockChan)
	wg.Wait()
}

func TestRPCJoin(t *testing.T) {
	fs := &fakeStore{
		leader:    "1.2.3.4:1234",
		md:        &Data{Index: 99},
		newNodeID: uint64(100),
		blockChan: make(chan struct{}),
	}
	serverRPC := &rpc{
		store: fs,
	}

	srv := newTestServer(t, serverRPC)
	defer srv.Close()
	go srv.Serve()

	// Wait for the RPC server to be ready
	<-srv.Ready

	// create a new RPC with a matching index as the server
	clientRPC := &rpc{
		store: &fakeStore{
			leader: srv.Listener.Addr().String(),
			md:     &Data{Index: 99},
		},
	}

	res, err := clientRPC.join("1.2.3.4:1234", srv.Listener.Addr().String())
	if err != nil {
		t.Fatalf("failed to join: %v", err)
	}

	if exp := true; res.RaftEnabled != true {
		t.Fatalf("raft enabled mismatch: got %v, exp %v", res.RaftEnabled, exp)
	}

	if exp := 1; len(res.RaftNodes) != exp {
		t.Fatalf("raft peer mismatch: got %v, exp %v", len(res.RaftNodes), exp)
	}

	if exp := "1.2.3.4:1234"; res.RaftNodes[0] != exp {
		t.Fatalf("raft peer mismatch: got %v, exp %v", res.RaftNodes[0], exp)
	}

	if exp := uint64(100); res.NodeID != exp {
		t.Fatalf("node id mismatch. got %v, exp %v", res.NodeID, exp)
	}
}

type fakeStore struct {
	mu        sync.RWMutex
	leader    string
	newNodeID uint64
	md        *Data
	blockChan chan struct{}
}

type testServer struct {
	Listener net.Listener
	Ready    chan struct{}
	rpc      *rpc
	t        *testing.T
}

func newTestServer(t *testing.T, rpc *rpc) *testServer {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	return &testServer{
		Listener: ln,
		Ready:    make(chan struct{}),
		rpc:      rpc,
	}
}

func (s *testServer) Close() {
	s.Listener.Close()
}

func (s *testServer) Serve() {
	close(s.Ready)
	conn, err := s.Listener.Accept()
	if err != nil {
		s.t.Fatalf("failed to accept: %v", err)
	}

	// Demux...
	b := make([]byte, 1)
	if _, err := conn.Read(b); err != nil {
		s.t.Fatalf("failed to demux: %v", err)
	}
	s.rpc.handleRPCConn(conn)
}

func (f *fakeStore) cachedData() *Data {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.md
}

func (f *fakeStore) IsLeader() bool            { return true }
func (f *fakeStore) Leader() string            { return f.leader }
func (f *fakeStore) Peers() ([]string, error)  { return []string{f.leader}, nil }
func (f *fakeStore) AddPeer(host string) error { return nil }
func (f *fakeStore) CreateNode(host string) (*NodeInfo, error) {
	return &NodeInfo{ID: f.newNodeID, Host: host}, nil
}
func (f *fakeStore) NodeByHost(host string) (*NodeInfo, error) { return nil, nil }
func (f *fakeStore) WaitForDataChanged() error {
	<-f.blockChan
	return nil
}
func (f *fakeStore) enableLocalRaft() error {
	return nil
}
func (f *fakeStore) SetPeers(addrs []string) error {
	return nil
}
