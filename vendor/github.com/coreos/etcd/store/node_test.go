// Copyright 2015 The etcd Authors
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

package store

import (
	"testing"
	"time"

	"github.com/jonboulle/clockwork"
)

var (
	key, val   = "foo", "bar"
	val1, val2 = "bar1", "bar2"
	expiration = time.Minute
)

func TestNewKVIs(t *testing.T) {
	nd := newTestNode()

	if nd.IsHidden() {
		t.Errorf("nd.Hidden() = %v, want = false", nd.IsHidden())
	}

	if nd.IsPermanent() {
		t.Errorf("nd.IsPermanent() = %v, want = false", nd.IsPermanent())
	}

	if nd.IsDir() {
		t.Errorf("nd.IsDir() = %v, want = false", nd.IsDir())
	}
}

func TestNewKVReadWriteCompare(t *testing.T) {
	nd := newTestNode()

	if v, err := nd.Read(); v != val || err != nil {
		t.Errorf("value = %s and err = %v, want value = %s and err = nil", v, err, val)
	}

	if err := nd.Write(val1, nd.CreatedIndex+1); err != nil {
		t.Errorf("nd.Write error = %v, want = nil", err)
	} else {
		if v, err := nd.Read(); v != val1 || err != nil {
			t.Errorf("value = %s and err = %v, want value = %s and err = nil", v, err, val1)
		}
	}
	if err := nd.Write(val2, nd.CreatedIndex+2); err != nil {
		t.Errorf("nd.Write error = %v, want = nil", err)
	} else {
		if v, err := nd.Read(); v != val2 || err != nil {
			t.Errorf("value = %s and err = %v, want value = %s and err = nil", v, err, val2)
		}
	}

	if ok, which := nd.Compare(val2, 2); !ok || which != 0 {
		t.Errorf("ok = %v and which = %d, want ok = true and which = 0", ok, which)
	}
}

func TestNewKVExpiration(t *testing.T) {
	nd := newTestNode()

	if _, ttl := nd.expirationAndTTL(clockwork.NewFakeClock()); ttl > expiration.Nanoseconds() {
		t.Errorf("ttl = %d, want %d < %d", ttl, ttl, expiration.Nanoseconds())
	}

	newExpiration := time.Hour
	nd.UpdateTTL(time.Now().Add(newExpiration))
	if _, ttl := nd.expirationAndTTL(clockwork.NewFakeClock()); ttl > newExpiration.Nanoseconds() {
		t.Errorf("ttl = %d, want %d < %d", ttl, ttl, newExpiration.Nanoseconds())
	}
	if ns, err := nd.List(); ns != nil || err == nil {
		t.Errorf("nodes = %v and err = %v, want nodes = nil and err != nil", ns, err)
	}

	en := nd.Repr(false, false, clockwork.NewFakeClock())
	if en.Key != nd.Path {
		t.Errorf("en.Key = %s, want = %s", en.Key, nd.Path)
	}
	if *(en.Value) != nd.Value {
		t.Errorf("*(en.Key) = %s, want = %s", *(en.Value), nd.Value)
	}
}

func TestNewKVListReprCompareClone(t *testing.T) {
	nd := newTestNode()

	if ns, err := nd.List(); ns != nil || err == nil {
		t.Errorf("nodes = %v and err = %v, want nodes = nil and err != nil", ns, err)
	}

	en := nd.Repr(false, false, clockwork.NewFakeClock())
	if en.Key != nd.Path {
		t.Errorf("en.Key = %s, want = %s", en.Key, nd.Path)
	}
	if *(en.Value) != nd.Value {
		t.Errorf("*(en.Key) = %s, want = %s", *(en.Value), nd.Value)
	}

	cn := nd.Clone()
	if cn.Path != nd.Path {
		t.Errorf("cn.Path = %s, want = %s", cn.Path, nd.Path)
	}
	if cn.Value != nd.Value {
		t.Errorf("cn.Value = %s, want = %s", cn.Value, nd.Value)
	}
}

func TestNewKVRemove(t *testing.T) {
	nd := newTestNode()

	if v, err := nd.Read(); v != val || err != nil {
		t.Errorf("value = %s and err = %v, want value = %s and err = nil", v, err, val)
	}

	if err := nd.Write(val1, nd.CreatedIndex+1); err != nil {
		t.Errorf("nd.Write error = %v, want = nil", err)
	} else {
		if v, err := nd.Read(); v != val1 || err != nil {
			t.Errorf("value = %s and err = %v, want value = %s and err = nil", v, err, val1)
		}
	}
	if err := nd.Write(val2, nd.CreatedIndex+2); err != nil {
		t.Errorf("nd.Write error = %v, want = nil", err)
	} else {
		if v, err := nd.Read(); v != val2 || err != nil {
			t.Errorf("value = %s and err = %v, want value = %s and err = nil", v, err, val2)
		}
	}

	if err := nd.Remove(false, false, nil); err != nil {
		t.Errorf("nd.Remove err = %v, want = nil", err)
	} else {
		// still readable
		if v, err := nd.Read(); v != val2 || err != nil {
			t.Errorf("value = %s and err = %v, want value = %s and err = nil", v, err, val2)
		}
		if len(nd.store.ttlKeyHeap.array) != 0 {
			t.Errorf("len(nd.store.ttlKeyHeap.array) = %d, want = 0", len(nd.store.ttlKeyHeap.array))
		}
		if len(nd.store.ttlKeyHeap.keyMap) != 0 {
			t.Errorf("len(nd.store.ttlKeyHeap.keyMap) = %d, want = 0", len(nd.store.ttlKeyHeap.keyMap))
		}
	}
}

func TestNewDirIs(t *testing.T) {
	nd, _ := newTestNodeDir()
	if nd.IsHidden() {
		t.Errorf("nd.Hidden() = %v, want = false", nd.IsHidden())
	}

	if nd.IsPermanent() {
		t.Errorf("nd.IsPermanent() = %v, want = false", nd.IsPermanent())
	}

	if !nd.IsDir() {
		t.Errorf("nd.IsDir() = %v, want = true", nd.IsDir())
	}
}

func TestNewDirReadWriteListReprClone(t *testing.T) {
	nd, _ := newTestNodeDir()

	if _, err := nd.Read(); err == nil {
		t.Errorf("err = %v, want err != nil", err)
	}

	if err := nd.Write(val, nd.CreatedIndex+1); err == nil {
		t.Errorf("err = %v, want err != nil", err)
	}

	if ns, err := nd.List(); ns == nil && err != nil {
		t.Errorf("nodes = %v and err = %v, want nodes = nil and err == nil", ns, err)
	}

	en := nd.Repr(false, false, clockwork.NewFakeClock())
	if en.Key != nd.Path {
		t.Errorf("en.Key = %s, want = %s", en.Key, nd.Path)
	}

	cn := nd.Clone()
	if cn.Path != nd.Path {
		t.Errorf("cn.Path = %s, want = %s", cn.Path, nd.Path)
	}
}

func TestNewDirExpirationTTL(t *testing.T) {
	nd, _ := newTestNodeDir()

	if _, ttl := nd.expirationAndTTL(clockwork.NewFakeClock()); ttl > expiration.Nanoseconds() {
		t.Errorf("ttl = %d, want %d < %d", ttl, ttl, expiration.Nanoseconds())
	}

	newExpiration := time.Hour
	nd.UpdateTTL(time.Now().Add(newExpiration))
	if _, ttl := nd.expirationAndTTL(clockwork.NewFakeClock()); ttl > newExpiration.Nanoseconds() {
		t.Errorf("ttl = %d, want %d < %d", ttl, ttl, newExpiration.Nanoseconds())
	}
}

func TestNewDirChild(t *testing.T) {
	nd, child := newTestNodeDir()

	if err := nd.Add(child); err != nil {
		t.Errorf("nd.Add(child) err = %v, want = nil", err)
	} else {
		if len(nd.Children) == 0 {
			t.Errorf("len(nd.Children) = %d, want = 1", len(nd.Children))
		}
	}

	if err := child.Remove(true, true, nil); err != nil {
		t.Errorf("child.Remove err = %v, want = nil", err)
	} else {
		if len(nd.Children) != 0 {
			t.Errorf("len(nd.Children) = %d, want = 0", len(nd.Children))
		}
	}
}

func newTestNode() *node {
	nd := newKV(newStore(), key, val, 0, nil, time.Now().Add(expiration))
	return nd
}

func newTestNodeDir() (*node, *node) {
	s := newStore()
	nd := newDir(s, key, 0, nil, time.Now().Add(expiration))
	cKey, cVal := "hello", "world"
	child := newKV(s, cKey, cVal, 0, nd, time.Now().Add(expiration))
	return nd, child
}
