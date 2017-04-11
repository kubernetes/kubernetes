// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package integration

import (
	"testing"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
	"github.com/coreos/etcd/integration"
	"github.com/coreos/etcd/pkg/testutil"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

func TestLeaseNotFoundError(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer clus.Terminate(t)

	lapi := clientv3.NewLease(clus.RandClient())
	defer lapi.Close()

	kv := clientv3.NewKV(clus.RandClient())

	_, err := kv.Put(context.TODO(), "foo", "bar", clientv3.WithLease(clientv3.LeaseID(500)))
	if err != rpctypes.ErrLeaseNotFound {
		t.Fatalf("expected %v, got %v", rpctypes.ErrLeaseNotFound, err)
	}
}

func TestLeaseGrant(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	lapi := clientv3.NewLease(clus.RandClient())
	defer lapi.Close()

	kv := clientv3.NewKV(clus.RandClient())

	resp, err := lapi.Grant(context.Background(), 10)
	if err != nil {
		t.Errorf("failed to create lease %v", err)
	}

	_, err = kv.Put(context.TODO(), "foo", "bar", clientv3.WithLease(resp.ID))
	if err != nil {
		t.Fatalf("failed to create key with lease %v", err)
	}
}

func TestLeaseRevoke(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	lapi := clientv3.NewLease(clus.RandClient())
	defer lapi.Close()

	kv := clientv3.NewKV(clus.RandClient())

	resp, err := lapi.Grant(context.Background(), 10)
	if err != nil {
		t.Errorf("failed to create lease %v", err)
	}

	_, err = lapi.Revoke(context.Background(), clientv3.LeaseID(resp.ID))
	if err != nil {
		t.Errorf("failed to revoke lease %v", err)
	}

	_, err = kv.Put(context.TODO(), "foo", "bar", clientv3.WithLease(resp.ID))
	if err != rpctypes.ErrLeaseNotFound {
		t.Fatalf("err = %v, want %v", err, rpctypes.ErrLeaseNotFound)
	}
}

func TestLeaseKeepAliveOnce(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	lapi := clientv3.NewLease(clus.RandClient())
	defer lapi.Close()

	resp, err := lapi.Grant(context.Background(), 10)
	if err != nil {
		t.Errorf("failed to create lease %v", err)
	}

	_, err = lapi.KeepAliveOnce(context.Background(), resp.ID)
	if err != nil {
		t.Errorf("failed to keepalive lease %v", err)
	}

	_, err = lapi.KeepAliveOnce(context.Background(), clientv3.LeaseID(0))
	if err != rpctypes.ErrLeaseNotFound {
		t.Errorf("expected %v, got %v", rpctypes.ErrLeaseNotFound, err)
	}
}

func TestLeaseKeepAlive(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	lapi := clientv3.NewLease(clus.RandClient())

	resp, err := lapi.Grant(context.Background(), 10)
	if err != nil {
		t.Errorf("failed to create lease %v", err)
	}

	rc, kerr := lapi.KeepAlive(context.Background(), resp.ID)
	if kerr != nil {
		t.Errorf("failed to keepalive lease %v", kerr)
	}

	kresp, ok := <-rc
	if !ok {
		t.Errorf("chan is closed, want not closed")
	}

	if kresp.ID != resp.ID {
		t.Errorf("ID = %x, want %x", kresp.ID, resp.ID)
	}

	lapi.Close()

	_, ok = <-rc
	if ok {
		t.Errorf("chan is not closed, want lease Close() closes chan")
	}
}

// TODO: add a client that can connect to all the members of cluster via unix sock.
// TODO: test handle more complicated failures.
func TestLeaseKeepAliveHandleFailure(t *testing.T) {
	t.Skip("test it when we have a cluster client")

	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	// TODO: change this line to get a cluster client
	lapi := clientv3.NewLease(clus.RandClient())

	resp, err := lapi.Grant(context.Background(), 10)
	if err != nil {
		t.Errorf("failed to create lease %v", err)
	}

	rc, kerr := lapi.KeepAlive(context.Background(), resp.ID)
	if kerr != nil {
		t.Errorf("failed to keepalive lease %v", kerr)
	}

	kresp := <-rc
	if kresp.ID != resp.ID {
		t.Errorf("ID = %x, want %x", kresp.ID, resp.ID)
	}

	// restart the connected member.
	clus.Members[0].Stop(t)

	select {
	case <-rc:
		t.Fatalf("unexpected keepalive")
	case <-time.After(10*time.Second/3 + 1):
	}

	// recover the member.
	clus.Members[0].Restart(t)

	kresp = <-rc
	if kresp.ID != resp.ID {
		t.Errorf("ID = %x, want %x", kresp.ID, resp.ID)
	}

	lapi.Close()

	_, ok := <-rc
	if ok {
		t.Errorf("chan is not closed, want lease Close() closes chan")
	}
}

type leaseCh struct {
	lid clientv3.LeaseID
	ch  <-chan *clientv3.LeaseKeepAliveResponse
}

// TestLeaseKeepAliveNotFound ensures a revoked lease won't stop other keep alives
func TestLeaseKeepAliveNotFound(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer clus.Terminate(t)

	cli := clus.RandClient()
	lchs := []leaseCh{}
	for i := 0; i < 3; i++ {
		resp, rerr := cli.Grant(context.TODO(), 5)
		if rerr != nil {
			t.Fatal(rerr)
		}
		kach, kaerr := cli.KeepAlive(context.Background(), resp.ID)
		if kaerr != nil {
			t.Fatal(kaerr)
		}
		lchs = append(lchs, leaseCh{resp.ID, kach})
	}

	if _, err := cli.Revoke(context.TODO(), lchs[1].lid); err != nil {
		t.Fatal(err)
	}

	<-lchs[0].ch
	if _, ok := <-lchs[0].ch; !ok {
		t.Fatalf("closed keepalive on wrong lease")
	}

	timec := time.After(5 * time.Second)
	for range lchs[1].ch {
		select {
		case <-timec:
			t.Fatalf("revoke did not close keep alive")
		default:
		}
	}
}

func TestLeaseGrantErrConnClosed(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer clus.Terminate(t)

	cli := clus.Client(0)
	le := clientv3.NewLease(cli)

	donec := make(chan struct{})
	go func() {
		defer close(donec)
		_, err := le.Grant(context.TODO(), 5)
		if err != nil && err != grpc.ErrClientConnClosing {
			t.Fatalf("expected %v, got %v", grpc.ErrClientConnClosing, err)
		}
	}()

	if err := cli.Close(); err != nil {
		t.Fatal(err)
	}
	clus.TakeClient(0)

	select {
	case <-time.After(3 * time.Second):
		t.Fatal("le.Grant took too long")
	case <-donec:
	}
}

func TestLeaseGrantNewAfterClose(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer clus.Terminate(t)

	cli := clus.Client(0)
	clus.TakeClient(0)
	if err := cli.Close(); err != nil {
		t.Fatal(err)
	}

	donec := make(chan struct{})
	go func() {
		le := clientv3.NewLease(cli)
		if _, err := le.Grant(context.TODO(), 5); err != grpc.ErrClientConnClosing {
			t.Fatalf("expected %v, got %v", grpc.ErrClientConnClosing, err)
		}
		close(donec)
	}()
	select {
	case <-time.After(3 * time.Second):
		t.Fatal("le.Grant took too long")
	case <-donec:
	}
}

func TestLeaseRevokeNewAfterClose(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer clus.Terminate(t)

	cli := clus.Client(0)
	le := clientv3.NewLease(cli)
	resp, err := le.Grant(context.TODO(), 5)
	if err != nil {
		t.Fatal(err)
	}
	leaseID := resp.ID

	clus.TakeClient(0)
	if err := cli.Close(); err != nil {
		t.Fatal(err)
	}

	donec := make(chan struct{})
	go func() {
		if _, err := le.Revoke(context.TODO(), leaseID); err != grpc.ErrClientConnClosing {
			t.Fatalf("expected %v, got %v", grpc.ErrClientConnClosing, err)
		}
		close(donec)
	}()
	select {
	case <-time.After(3 * time.Second):
		t.Fatal("le.Revoke took too long")
	case <-donec:
	}
}

// TestLeaseKeepAliveCloseAfterDisconnectExpire ensures the keep alive channel is closed
// following a disconnection, lease revoke, then reconnect.
func TestLeaseKeepAliveCloseAfterDisconnectRevoke(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	cli := clus.Client(0)

	// setup lease and do a keepalive
	resp, err := cli.Grant(context.Background(), 10)
	if err != nil {
		t.Fatal(err)
	}
	rc, kerr := cli.KeepAlive(context.Background(), resp.ID)
	if kerr != nil {
		t.Fatal(kerr)
	}
	kresp := <-rc
	if kresp.ID != resp.ID {
		t.Fatalf("ID = %x, want %x", kresp.ID, resp.ID)
	}

	// keep client disconnected
	clus.Members[0].Stop(t)
	time.Sleep(time.Second)
	clus.WaitLeader(t)

	if _, err := clus.Client(1).Revoke(context.TODO(), resp.ID); err != nil {
		t.Fatal(err)
	}

	clus.Members[0].Restart(t)

	// some keep-alives may still be buffered; drain until close
	timer := time.After(time.Duration(kresp.TTL) * time.Second)
	for kresp != nil {
		select {
		case kresp = <-rc:
		case <-timer:
			t.Fatalf("keepalive channel did not close")
		}
	}
}

// TestLeaseKeepAliveInitTimeout ensures the keep alive channel closes if
// the initial keep alive request never gets a response.
func TestLeaseKeepAliveInitTimeout(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer clus.Terminate(t)

	cli := clus.Client(0)

	// setup lease and do a keepalive
	resp, err := cli.Grant(context.Background(), 5)
	if err != nil {
		t.Fatal(err)
	}
	rc, kerr := cli.KeepAlive(context.Background(), resp.ID)
	if kerr != nil {
		t.Fatal(kerr)
	}
	// keep client disconnected
	clus.Members[0].Stop(t)
	select {
	case ka, ok := <-rc:
		if ok {
			t.Fatalf("unexpected keepalive %v, expected closed channel", ka)
		}
	case <-time.After(10 * time.Second):
		t.Fatalf("keepalive channel did not close")
	}

	clus.Members[0].Restart(t)
}

// TestLeaseKeepAliveInitTimeout ensures the keep alive channel closes if
// a keep alive request after the first never gets a response.
func TestLeaseKeepAliveTTLTimeout(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer clus.Terminate(t)

	cli := clus.Client(0)

	// setup lease and do a keepalive
	resp, err := cli.Grant(context.Background(), 5)
	if err != nil {
		t.Fatal(err)
	}
	rc, kerr := cli.KeepAlive(context.Background(), resp.ID)
	if kerr != nil {
		t.Fatal(kerr)
	}
	if kresp := <-rc; kresp.ID != resp.ID {
		t.Fatalf("ID = %x, want %x", kresp.ID, resp.ID)
	}

	// keep client disconnected
	clus.Members[0].Stop(t)
	select {
	case ka, ok := <-rc:
		if ok {
			t.Fatalf("unexpected keepalive %v, expected closed channel", ka)
		}
	case <-time.After(10 * time.Second):
		t.Fatalf("keepalive channel did not close")
	}

	clus.Members[0].Restart(t)
}

// TestLeaseRenewLostQuorum ensures keepalives work after losing quorum
// for a while.
func TestLeaseRenewLostQuorum(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	cli := clus.Client(0)
	r, err := cli.Grant(context.TODO(), 4)
	if err != nil {
		t.Fatal(err)
	}

	kctx, kcancel := context.WithCancel(context.Background())
	defer kcancel()
	ka, err := cli.KeepAlive(kctx, r.ID)
	if err != nil {
		t.Fatal(err)
	}
	// consume first keepalive so next message sends when cluster is down
	<-ka

	// force keepalive stream message to timeout
	clus.Members[1].Stop(t)
	clus.Members[2].Stop(t)
	// Use TTL-1 since the client closes the keepalive channel if no
	// keepalive arrives before the lease deadline.
	// The cluster has 1 second to recover and reply to the keepalive.
	time.Sleep(time.Duration(r.TTL-1) * time.Second)
	clus.Members[1].Restart(t)
	clus.Members[2].Restart(t)

	select {
	case _, ok := <-ka:
		if !ok {
			t.Fatalf("keepalive closed")
		}
	case <-time.After(time.Duration(r.TTL) * time.Second):
		t.Fatalf("timed out waiting for keepalive")
	}
}
