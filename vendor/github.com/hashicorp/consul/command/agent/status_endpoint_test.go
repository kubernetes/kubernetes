package agent

import (
	"github.com/hashicorp/consul/testutil"
	"os"
	"testing"
)

func TestStatusLeader(t *testing.T) {
	dir, srv := makeHTTPServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	obj, err := srv.StatusLeader(nil, nil)
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	val := obj.(string)
	if val == "" {
		t.Fatalf("bad addr: %v", obj)
	}
}

func TestStatusPeers(t *testing.T) {
	dir, srv := makeHTTPServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	obj, err := srv.StatusPeers(nil, nil)
	if err != nil {
		t.Fatalf("Err: %v", err)
	}

	peers := obj.([]string)
	if len(peers) != 1 {
		t.Fatalf("bad peers: %v", peers)
	}
}
