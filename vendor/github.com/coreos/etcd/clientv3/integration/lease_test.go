// Copyright 2016 CoreOS, Inc.
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
)

func TestLeaseCreate(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	lapi := clientv3.NewLease(clus.RandClient())
	defer lapi.Close()

	kv := clientv3.NewKV(clus.RandClient())

	resp, err := lapi.Create(context.Background(), 10)
	if err != nil {
		t.Errorf("failed to create lease %v", err)
	}

	_, err = kv.Put(context.TODO(), "foo", "bar", clientv3.WithLease(clientv3.LeaseID(resp.ID)))
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

	resp, err := lapi.Create(context.Background(), 10)
	if err != nil {
		t.Errorf("failed to create lease %v", err)
	}

	_, err = lapi.Revoke(context.Background(), clientv3.LeaseID(resp.ID))
	if err != nil {
		t.Errorf("failed to revoke lease %v", err)
	}

	_, err = kv.Put(context.TODO(), "foo", "bar", clientv3.WithLease(clientv3.LeaseID(resp.ID)))
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

	resp, err := lapi.Create(context.Background(), 10)
	if err != nil {
		t.Errorf("failed to create lease %v", err)
	}

	_, err = lapi.KeepAliveOnce(context.Background(), clientv3.LeaseID(resp.ID))
	if err != nil {
		t.Errorf("failed to keepalive lease %v", err)
	}
}

func TestLeaseKeepAlive(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	lapi := clientv3.NewLease(clus.RandClient())

	resp, err := lapi.Create(context.Background(), 10)
	if err != nil {
		t.Errorf("failed to create lease %v", err)
	}

	rc, kerr := lapi.KeepAlive(context.Background(), clientv3.LeaseID(resp.ID))
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

	resp, err := lapi.Create(context.Background(), 10)
	if err != nil {
		t.Errorf("failed to create lease %v", err)
	}

	rc, kerr := lapi.KeepAlive(context.Background(), clientv3.LeaseID(resp.ID))
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
