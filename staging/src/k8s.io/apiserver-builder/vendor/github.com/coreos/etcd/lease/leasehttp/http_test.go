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

package leasehttp

import (
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/coreos/etcd/lease"
	"github.com/coreos/etcd/mvcc/backend"
	"golang.org/x/net/context"
)

func TestRenewHTTP(t *testing.T) {
	be, tmpPath := backend.NewTmpBackend(time.Hour, 10000)
	defer os.Remove(tmpPath)
	defer be.Close()

	le := lease.NewLessor(be, 3)
	le.Promote(time.Second)
	l, err := le.Grant(1, int64(5))
	if err != nil {
		t.Fatalf("failed to create lease: %v", err)
	}

	ts := httptest.NewServer(NewHandler(le))
	defer ts.Close()

	ttl, err := RenewHTTP(context.TODO(), l.ID, ts.URL+"/leases", http.DefaultTransport)
	if err != nil {
		t.Fatal(err)
	}
	if ttl != 5 {
		t.Fatalf("ttl expected 5, got %d", ttl)
	}
}
