// Copyright 2015 flannel authors
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

package subnet

import (
	"fmt"
	"testing"

	etcd "github.com/coreos/etcd/client"
	"golang.org/x/net/context"
)

func expectSuccess(t *testing.T, r *etcd.Response, err error, expected *etcd.Response, expectedValue string) {
	if err != nil {
		t.Fatalf("Failed to get etcd keys: %v", err)
	}
	if r == nil {
		t.Fatal("etcd response was nil")
	}
	if r.Action != expected.Action {
		t.Fatalf("Unexpected action %s (expected %s)", r.Action, expected.Action)
	}
	if r.Index < expected.Index {
		t.Fatalf("Unexpected response index %v (expected >= %v)", r.Index, expected.Index)
	}
	if expected.Node != nil {
		if expected.Node.Key != r.Node.Key {
			t.Fatalf("Unexpected response node %s key %s (expected %s)", r.Node.Key, r.Node.Key, expected.Node.Key)
		}
		if expected.Node.Value != r.Node.Value {
			t.Fatalf("Unexpected response node %s value %s (expected %s)", r.Node.Key, r.Node.Value, expected.Node.Value)
		}
		if expected.Node.Dir != r.Node.Dir {
			t.Fatalf("Unexpected response node %s dir %v (expected %v)", r.Node.Key, r.Node.Dir, expected.Node.Dir)
		}
		if expected.Node.CreatedIndex != r.Node.CreatedIndex {
			t.Fatalf("Unexpected response node %s CreatedIndex %v (expected %v)", r.Node.Key, r.Node.CreatedIndex, expected.Node.CreatedIndex)
		}
		if expected.Node.ModifiedIndex > r.Node.ModifiedIndex {
			t.Fatalf("Unexpected response node %s ModifiedIndex %v (expected %v)", r.Node.Key, r.Node.ModifiedIndex, expected.Node.ModifiedIndex)
		}
	}
	if expectedValue != "" {
		if r.Node == nil {
			t.Fatalf("Unexpected empty response node")
		}
		if r.Node.Value != expectedValue {
			t.Fatalf("Unexpected response node %s value %s (expected %s)", r.Node.Key, r.Node.Value, expectedValue)
		}
	}
}

func watchMockEtcd(ctx context.Context, watcher etcd.Watcher, result chan error) {
	type evt struct {
		key      string
		event    string
		received bool
	}

	expected := []evt{
		{"/coreos.com/network/foobar/config", "create", false},
		{"/coreos.com/network/blah/config", "create", false},
		{"/coreos.com/network/blah/config", "update", false},
		{"/coreos.com/network/foobar/config", "delete", false},
		{"/coreos.com/network/foobar", "delete", false},
	}

	// Wait for delete events on /coreos.com/network/foobar and its
	// 'config' child, and for the update event on
	// /coreos.com/network/foobar (for 'config' delete) and on
	// /coreos.com/network (for 'foobar' delete)
	numEvents := 0
	for {
		resp, err := watcher.Next(ctx)

		if err != nil {
			if err == context.Canceled {
				break
			}
			result <- fmt.Errorf("Unexpected error watching for event: %v", err)
			break
		}
		if resp.Node == nil {
			result <- fmt.Errorf("Unexpected empty node watching for event")
			break
		}
		found := false
		for i, e := range expected {
			if e.key == resp.Node.Key && e.event == resp.Action {
				if expected[i].received != true {
					expected[i].received = true
					found = true
					numEvents += 1
				}
				break
			}
		}
		if found == false {
			result <- fmt.Errorf("Received unexpected or already received event %v", resp)
			break
		}

		if numEvents == len(expected) {
			result <- nil
			break
		}
	}
}

func TestMockEtcd(t *testing.T) {
	m := newMockEtcd()

	ctx, _ := context.WithCancel(context.Background())

	// Sanity tests for our mock etcd

	// Ensure no entries yet exist
	opts := &etcd.GetOptions{Recursive: true}
	r, err := m.Get(ctx, "/", opts)
	e := &etcd.Response{Action: "get", Index: 1000, Node: m.nodes["/"]}
	expectSuccess(t, r, err, e, "")

	// Create base test keys
	sopts := &etcd.SetOptions{Dir: true}
	r, err = m.Set(ctx, "/coreos.com/network", "", sopts)
	e = &etcd.Response{Action: "create", Index: 1002}
	expectSuccess(t, r, err, e, "")

	wopts := &etcd.WatcherOptions{AfterIndex: m.index, Recursive: true}
	watcher := m.Watcher("/coreos.com/network", wopts)

	result := make(chan error, 1)
	go watchMockEtcd(ctx, watcher, result)

	// Populate etcd with some keys
	netKey1 := "/coreos.com/network/foobar/config"
	netValue := "{ \"Network\": \"10.1.0.0/16\", \"Backend\": { \"Type\": \"host-gw\" } }"
	r, err = m.Create(ctx, netKey1, netValue)
	e = &etcd.Response{Action: "create", Index: 1004}
	expectSuccess(t, r, err, e, netValue)

	netKey2 := "/coreos.com/network/blah/config"
	netValue = "{ \"Network\": \"10.1.1.0/16\", \"Backend\": { \"Type\": \"host-gw\" } }"
	r, err = m.Create(ctx, netKey2, netValue)
	e = &etcd.Response{Action: "create", Index: 1006}
	expectSuccess(t, r, err, e, netValue)

	// Get it again
	expectedNode := r.Node
	opts = &etcd.GetOptions{Recursive: false}
	r, err = m.Get(ctx, netKey2, opts)
	e = &etcd.Response{Action: "get", Index: m.index, Node: expectedNode}
	expectSuccess(t, r, err, e, netValue)

	// Update it
	netValue = "ReallyCoolValue"
	r, err = m.Update(ctx, netKey2, netValue)
	e = &etcd.Response{Action: "update", Index: m.index}
	expectSuccess(t, r, err, e, netValue)

	// Get it again
	opts = &etcd.GetOptions{Recursive: false}
	r, err = m.Get(ctx, netKey2, opts)
	e = &etcd.Response{Action: "get", Index: m.index}
	expectSuccess(t, r, err, e, netValue)

	// test directory listing
	opts = &etcd.GetOptions{Recursive: true}
	r, err = m.Get(ctx, "/coreos.com/network/", opts)
	e = &etcd.Response{Action: "get", Index: 1007}
	expectSuccess(t, r, err, e, "")

	if len(r.Node.Nodes) != 2 {
		t.Fatalf("Unexpected %d children in response (expected 2)", len(r.Node.Nodes))
	}
	node1Found := false
	node2Found := false
	for _, child := range r.Node.Nodes {
		if child.Dir != true {
			t.Fatalf("Unexpected non-directory child %s", child.Key)
		}
		if child.Key == "/coreos.com/network/foobar" {
			node1Found = true
		} else if child.Key == "/coreos.com/network/blah" {
			node2Found = true
		} else {
			t.Fatalf("Unexpected child %s found", child.Key)
		}
		if len(child.Nodes) != 1 {
			t.Fatalf("Unexpected %d children in response (expected 2)", len(r.Node.Nodes))
		}
	}
	if node1Found == false || node2Found == false {
		t.Fatalf("Failed to find expected children")
	}

	// Delete a key
	dopts := &etcd.DeleteOptions{Recursive: true, Dir: false}
	r, err = m.Delete(ctx, "/coreos.com/network/foobar", dopts)
	if err == nil {
		t.Fatalf("Unexpected success deleting a directory")
	}

	// Delete a key
	dopts = &etcd.DeleteOptions{Recursive: true, Dir: true}
	r, err = m.Delete(ctx, "/coreos.com/network/foobar", dopts)
	e = &etcd.Response{Action: "delete", Index: 1010}
	expectSuccess(t, r, err, e, "")

	// Get it again; should fail
	opts = &etcd.GetOptions{Recursive: false}
	r, err = m.Get(ctx, netKey1, opts)
	if err == nil {
		t.Fatalf("Get of %s after delete unexpectedly succeeded", netKey1)
	}
	if r != nil {
		t.Fatalf("Unexpected non-nil response to get after delete %v", r)
	}

	// Check errors from watch goroutine
	watchResult := <-result
	if watchResult != nil {
		t.Fatalf("Error watching keys: %v", watchResult)
	}
}
