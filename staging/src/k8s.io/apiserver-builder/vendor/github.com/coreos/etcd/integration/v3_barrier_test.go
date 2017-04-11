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
	"github.com/coreos/etcd/contrib/recipes"
	"github.com/coreos/etcd/pkg/testutil"
)

func TestBarrierSingleNode(t *testing.T) {
	defer testutil.AfterTest(t)
	clus := NewClusterV3(t, &ClusterConfig{Size: 1})
	defer clus.Terminate(t)
	testBarrier(t, 5, func() *clientv3.Client { return clus.clients[0] })
}

func TestBarrierMultiNode(t *testing.T) {
	defer testutil.AfterTest(t)
	clus := NewClusterV3(t, &ClusterConfig{Size: 3})
	defer clus.Terminate(t)
	testBarrier(t, 5, func() *clientv3.Client { return clus.RandClient() })
}

func testBarrier(t *testing.T, waiters int, chooseClient func() *clientv3.Client) {
	b := recipe.NewBarrier(chooseClient(), "test-barrier")
	if err := b.Hold(); err != nil {
		t.Fatalf("could not hold barrier (%v)", err)
	}
	if err := b.Hold(); err == nil {
		t.Fatalf("able to double-hold barrier")
	}

	donec := make(chan struct{})
	for i := 0; i < waiters; i++ {
		go func() {
			br := recipe.NewBarrier(chooseClient(), "test-barrier")
			if err := br.Wait(); err != nil {
				t.Fatalf("could not wait on barrier (%v)", err)
			}
			donec <- struct{}{}
		}()
	}

	select {
	case <-donec:
		t.Fatalf("barrier did not wait")
	default:
	}

	if err := b.Release(); err != nil {
		t.Fatalf("could not release barrier (%v)", err)
	}

	timerC := time.After(time.Duration(waiters*100) * time.Millisecond)
	for i := 0; i < waiters; i++ {
		select {
		case <-timerC:
			t.Fatalf("barrier timed out")
		case <-donec:
		}
	}
}
