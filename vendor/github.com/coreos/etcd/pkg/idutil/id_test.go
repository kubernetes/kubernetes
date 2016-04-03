// Copyright 2015 CoreOS, Inc.
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

package idutil

import (
	"testing"
	"time"
)

func TestNewGenerator(t *testing.T) {
	g := NewGenerator(0x12, time.Unix(0, 0).Add(0x3456*time.Millisecond))
	id := g.Next()
	wid := uint64(0x12000000345601)
	if id != wid {
		t.Errorf("id = %x, want %x", id, wid)
	}
}

func TestNewGeneratorUnique(t *testing.T) {
	g := NewGenerator(0, time.Time{})
	id := g.Next()
	// different server generates different ID
	g1 := NewGenerator(1, time.Time{})
	if gid := g1.Next(); id == gid {
		t.Errorf("generate the same id %x using different server ID", id)
	}
	// restarted server generates different ID
	g2 := NewGenerator(0, time.Now())
	if gid := g2.Next(); id == gid {
		t.Errorf("generate the same id %x after restart", id)
	}
}

func TestNext(t *testing.T) {
	g := NewGenerator(0x12, time.Unix(0, 0).Add(0x3456*time.Millisecond))
	wid := uint64(0x12000000345601)
	for i := 0; i < 1000; i++ {
		id := g.Next()
		if id != wid+uint64(i) {
			t.Errorf("id = %x, want %x", id, wid+uint64(i))
		}
	}
}
