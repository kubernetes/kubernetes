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

	"github.com/coreos/etcd/clientv3/concurrency"
	"github.com/coreos/etcd/contrib/recipes"
)

func TestDoubleBarrier(t *testing.T) {
	clus := NewClusterV3(t, &ClusterConfig{Size: 3})
	defer clus.Terminate(t)
	defer dropSessionLease(clus)

	waiters := 10

	b := recipe.NewDoubleBarrier(clus.RandClient(), "test-barrier", waiters)
	donec := make(chan struct{})
	for i := 0; i < waiters-1; i++ {
		go func() {
			bb := recipe.NewDoubleBarrier(clus.RandClient(), "test-barrier", waiters)
			if err := bb.Enter(); err != nil {
				t.Fatalf("could not enter on barrier (%v)", err)
			}
			donec <- struct{}{}
			if err := bb.Leave(); err != nil {
				t.Fatalf("could not leave on barrier (%v)", err)
			}
			donec <- struct{}{}
		}()
	}

	time.Sleep(10 * time.Millisecond)
	select {
	case <-donec:
		t.Fatalf("barrier did not enter-wait")
	default:
	}

	if err := b.Enter(); err != nil {
		t.Fatalf("could not enter last barrier (%v)", err)
	}

	timerC := time.After(time.Duration(waiters*100) * time.Millisecond)
	for i := 0; i < waiters-1; i++ {
		select {
		case <-timerC:
			t.Fatalf("barrier enter timed out")
		case <-donec:
		}
	}

	time.Sleep(10 * time.Millisecond)
	select {
	case <-donec:
		t.Fatalf("barrier did not leave-wait")
	default:
	}

	b.Leave()
	timerC = time.After(time.Duration(waiters*100) * time.Millisecond)
	for i := 0; i < waiters-1; i++ {
		select {
		case <-timerC:
			t.Fatalf("barrier leave timed out")
		case <-donec:
		}
	}
}

func TestDoubleBarrierFailover(t *testing.T) {
	clus := NewClusterV3(t, &ClusterConfig{Size: 3})
	defer clus.Terminate(t)
	defer dropSessionLease(clus)

	waiters := 10
	donec := make(chan struct{})

	// sacrificial barrier holder; lease will be revoked
	go func() {
		b := recipe.NewDoubleBarrier(clus.clients[0], "test-barrier", waiters)
		if err := b.Enter(); err != nil {
			t.Fatalf("could not enter on barrier (%v)", err)
		}
		donec <- struct{}{}
	}()

	for i := 0; i < waiters-1; i++ {
		go func() {
			b := recipe.NewDoubleBarrier(clus.clients[1], "test-barrier", waiters)
			if err := b.Enter(); err != nil {
				t.Fatalf("could not enter on barrier (%v)", err)
			}
			donec <- struct{}{}
			b.Leave()
			donec <- struct{}{}
		}()
	}

	// wait for barrier enter to unblock
	for i := 0; i < waiters; i++ {
		select {
		case <-donec:
		case <-time.After(10 * time.Second):
			t.Fatalf("timed out waiting for enter, %d", i)
		}
	}
	// kill lease, expect Leave unblock
	s, err := concurrency.NewSession(clus.clients[0])
	if err != nil {
		t.Fatal(err)
	}
	if err = s.Close(); err != nil {
		t.Fatal(err)
	}
	// join on rest of waiters
	for i := 0; i < waiters-1; i++ {
		select {
		case <-donec:
		case <-time.After(10 * time.Second):
			t.Fatalf("timed out waiting for leave, %d", i)
		}
	}
}

func dropSessionLease(clus *ClusterV3) {
	for _, client := range clus.clients {
		s, _ := concurrency.NewSession(client)
		s.Orphan()
	}
}
