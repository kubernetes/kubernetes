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
	"reflect"
	"testing"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/integration"
	"github.com/coreos/etcd/pkg/testutil"
	"github.com/coreos/etcd/pkg/types"
	"golang.org/x/net/context"
)

func TestMemberList(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	capi := clientv3.NewCluster(clus.RandClient())

	resp, err := capi.MemberList(context.Background())
	if err != nil {
		t.Fatalf("failed to list member %v", err)
	}

	if len(resp.Members) != 3 {
		t.Errorf("number of members = %d, want %d", len(resp.Members), 3)
	}
}

func TestMemberAdd(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	capi := clientv3.NewCluster(clus.RandClient())

	urls := []string{"http://127.0.0.1:1234"}
	resp, err := capi.MemberAdd(context.Background(), urls)
	if err != nil {
		t.Fatalf("failed to add member %v", err)
	}

	if !reflect.DeepEqual(resp.Member.PeerURLs, urls) {
		t.Errorf("urls = %v, want %v", urls, resp.Member.PeerURLs)
	}
}

func TestMemberRemove(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	capi := clientv3.NewCluster(clus.Client(1))
	resp, err := capi.MemberList(context.Background())
	if err != nil {
		t.Fatalf("failed to list member %v", err)
	}

	rmvID := resp.Members[0].ID
	// indexes in capi member list don't necessarily match cluster member list;
	// find member that is not the client to remove
	for _, m := range resp.Members {
		mURLs, _ := types.NewURLs(m.PeerURLs)
		if !reflect.DeepEqual(mURLs, clus.Members[1].ServerConfig.PeerURLs) {
			rmvID = m.ID
			break
		}
	}

	_, err = capi.MemberRemove(context.Background(), rmvID)
	if err != nil {
		t.Fatalf("failed to remove member %v", err)
	}

	resp, err = capi.MemberList(context.Background())
	if err != nil {
		t.Fatalf("failed to list member %v", err)
	}

	if len(resp.Members) != 2 {
		t.Errorf("number of members = %d, want %d", len(resp.Members), 2)
	}
}

func TestMemberUpdate(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	capi := clientv3.NewCluster(clus.RandClient())
	resp, err := capi.MemberList(context.Background())
	if err != nil {
		t.Fatalf("failed to list member %v", err)
	}

	urls := []string{"http://127.0.0.1:1234"}
	_, err = capi.MemberUpdate(context.Background(), resp.Members[0].ID, urls)
	if err != nil {
		t.Fatalf("failed to update member %v", err)
	}

	resp, err = capi.MemberList(context.Background())
	if err != nil {
		t.Fatalf("failed to list member %v", err)
	}

	if !reflect.DeepEqual(resp.Members[0].PeerURLs, urls) {
		t.Errorf("urls = %v, want %v", urls, resp.Members[0].PeerURLs)
	}
}
