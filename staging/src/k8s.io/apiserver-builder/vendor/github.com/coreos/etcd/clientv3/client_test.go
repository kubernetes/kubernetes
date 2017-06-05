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

package clientv3

import (
	"fmt"
	"testing"
	"time"

	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/pkg/testutil"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

func TestDialTimeout(t *testing.T) {
	defer testutil.AfterTest(t)

	donec := make(chan error)
	go func() {
		// without timeout, grpc keeps redialing if connection refused
		cfg := Config{
			Endpoints:   []string{"localhost:12345"},
			DialTimeout: 2 * time.Second}
		c, err := New(cfg)
		if c != nil || err == nil {
			t.Errorf("new client should fail")
		}
		donec <- err
	}()

	time.Sleep(10 * time.Millisecond)

	select {
	case err := <-donec:
		t.Errorf("dial didn't wait (%v)", err)
	default:
	}

	select {
	case <-time.After(5 * time.Second):
		t.Errorf("failed to timeout dial on time")
	case err := <-donec:
		if err != grpc.ErrClientConnTimeout {
			t.Errorf("unexpected error %v, want %v", err, grpc.ErrClientConnTimeout)
		}
	}
}

func TestDialNoTimeout(t *testing.T) {
	cfg := Config{Endpoints: []string{"127.0.0.1:12345"}}
	c, err := New(cfg)
	if c == nil || err != nil {
		t.Fatalf("new client with DialNoWait should succeed, got %v", err)
	}
	c.Close()
}

func TestIsHaltErr(t *testing.T) {
	if !isHaltErr(nil, fmt.Errorf("etcdserver: some etcdserver error")) {
		t.Errorf(`error prefixed with "etcdserver: " should be Halted by default`)
	}
	if isHaltErr(nil, etcdserver.ErrStopped) {
		t.Errorf("error %v should not halt", etcdserver.ErrStopped)
	}
	if isHaltErr(nil, etcdserver.ErrNoLeader) {
		t.Errorf("error %v should not halt", etcdserver.ErrNoLeader)
	}
	ctx, cancel := context.WithCancel(context.TODO())
	if isHaltErr(ctx, nil) {
		t.Errorf("no error and active context should not be Halted")
	}
	cancel()
	if !isHaltErr(ctx, nil) {
		t.Errorf("cancel on context should be Halted")
	}
}
