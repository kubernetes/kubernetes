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
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/clientv3/concurrency"
	"github.com/coreos/etcd/contrib/recipes"
	"golang.org/x/net/context"
)

func TestMutexSingleNode(t *testing.T) {
	clus := NewClusterV3(t, &ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	var clients []*clientv3.Client
	testMutex(t, 5, makeSingleNodeClients(t, clus.cluster, &clients))
	closeClients(t, clients)
}

func TestMutexMultiNode(t *testing.T) {
	clus := NewClusterV3(t, &ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	var clients []*clientv3.Client
	testMutex(t, 5, makeMultiNodeClients(t, clus.cluster, &clients))
	closeClients(t, clients)
}

func testMutex(t *testing.T, waiters int, chooseClient func() *clientv3.Client) {
	// stream lock acquisitions
	lockedC := make(chan *concurrency.Mutex)
	for i := 0; i < waiters; i++ {
		go func() {
			session, err := concurrency.NewSession(chooseClient())
			if err != nil {
				t.Error(err)
			}
			m := concurrency.NewMutex(session, "test-mutex")
			if err := m.Lock(context.TODO()); err != nil {
				t.Fatalf("could not wait on lock (%v)", err)
			}
			lockedC <- m
		}()
	}
	// unlock locked mutexes
	timerC := time.After(time.Duration(waiters) * time.Second)
	for i := 0; i < waiters; i++ {
		select {
		case <-timerC:
			t.Fatalf("timed out waiting for lock %d", i)
		case m := <-lockedC:
			// lock acquired with m
			select {
			case <-lockedC:
				t.Fatalf("lock %d followers did not wait", i)
			default:
			}
			if err := m.Unlock(context.TODO()); err != nil {
				t.Fatalf("could not release lock (%v)", err)
			}
		}
	}
}

// TestMutexSessionRelock ensures that acquiring the same lock with the same
// session will not result in deadlock.
func TestMutexSessionRelock(t *testing.T) {
	clus := NewClusterV3(t, &ClusterConfig{Size: 3})
	defer clus.Terminate(t)
	session, err := concurrency.NewSession(clus.RandClient())
	if err != nil {
		t.Error(err)
	}

	m := concurrency.NewMutex(session, "test-mutex")
	if err := m.Lock(context.TODO()); err != nil {
		t.Fatal(err)
	}

	m2 := concurrency.NewMutex(session, "test-mutex")
	if err := m2.Lock(context.TODO()); err != nil {
		t.Fatal(err)
	}
}

func BenchmarkMutex4Waiters(b *testing.B) {
	// XXX switch tests to use TB interface
	clus := NewClusterV3(nil, &ClusterConfig{Size: 3})
	defer clus.Terminate(nil)
	for i := 0; i < b.N; i++ {
		testMutex(nil, 4, func() *clientv3.Client { return clus.RandClient() })
	}
}

func TestRWMutexSingleNode(t *testing.T) {
	clus := NewClusterV3(t, &ClusterConfig{Size: 3})
	defer clus.Terminate(t)
	testRWMutex(t, 5, func() *clientv3.Client { return clus.clients[0] })
}

func TestRWMutexMultiNode(t *testing.T) {
	clus := NewClusterV3(t, &ClusterConfig{Size: 3})
	defer clus.Terminate(t)
	testRWMutex(t, 5, func() *clientv3.Client { return clus.RandClient() })
}

func testRWMutex(t *testing.T, waiters int, chooseClient func() *clientv3.Client) {
	// stream rwlock acquistions
	rlockedC := make(chan *recipe.RWMutex, 1)
	wlockedC := make(chan *recipe.RWMutex, 1)
	for i := 0; i < waiters; i++ {
		go func() {
			session, err := concurrency.NewSession(chooseClient())
			if err != nil {
				t.Error(err)
			}
			rwm := recipe.NewRWMutex(session, "test-rwmutex")
			if rand.Intn(2) == 0 {
				if err := rwm.RLock(); err != nil {
					t.Fatalf("could not rlock (%v)", err)
				}
				rlockedC <- rwm
			} else {
				if err := rwm.Lock(); err != nil {
					t.Fatalf("could not lock (%v)", err)
				}
				wlockedC <- rwm
			}
		}()
	}
	// unlock locked rwmutexes
	timerC := time.After(time.Duration(waiters) * time.Second)
	for i := 0; i < waiters; i++ {
		select {
		case <-timerC:
			t.Fatalf("timed out waiting for lock %d", i)
		case wl := <-wlockedC:
			select {
			case <-rlockedC:
				t.Fatalf("rlock %d readers did not wait", i)
			default:
			}
			if err := wl.Unlock(); err != nil {
				t.Fatalf("could not release lock (%v)", err)
			}
		case rl := <-rlockedC:
			select {
			case <-wlockedC:
				t.Fatalf("rlock %d writers did not wait", i)
			default:
			}
			if err := rl.RUnlock(); err != nil {
				t.Fatalf("could not release rlock (%v)", err)
			}
		}
	}
}

func makeClients(t *testing.T, clients *[]*clientv3.Client, choose func() *member) func() *clientv3.Client {
	var mu sync.Mutex
	*clients = nil
	return func() *clientv3.Client {
		cli, err := NewClientV3(choose())
		if err != nil {
			t.Fatalf("cannot create client: %v", err)
		}
		mu.Lock()
		*clients = append(*clients, cli)
		mu.Unlock()
		return cli
	}
}

func makeSingleNodeClients(t *testing.T, clus *cluster, clients *[]*clientv3.Client) func() *clientv3.Client {
	return makeClients(t, clients, func() *member {
		return clus.Members[0]
	})
}

func makeMultiNodeClients(t *testing.T, clus *cluster, clients *[]*clientv3.Client) func() *clientv3.Client {
	return makeClients(t, clients, func() *member {
		return clus.Members[rand.Intn(len(clus.Members))]
	})
}

func closeClients(t *testing.T, clients []*clientv3.Client) {
	for _, cli := range clients {
		if err := cli.Close(); err != nil {
			t.Fatal(err)
		}
	}
}
