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

package store

import (
	"fmt"
	"testing"
	"time"
)

func TestHeapPushPop(t *testing.T) {
	h := newTtlKeyHeap()

	// add from older expire time to earlier expire time
	// the path is equal to ttl from now
	for i := 0; i < 10; i++ {
		path := fmt.Sprintf("%v", 10-i)
		m := time.Duration(10 - i)
		n := newKV(nil, path, path, 0, nil, time.Now().Add(time.Second*m))
		h.push(n)
	}

	min := time.Now()

	for i := 0; i < 10; i++ {
		node := h.pop()
		if node.ExpireTime.Before(min) {
			t.Fatal("heap sort wrong!")
		}
		min = node.ExpireTime
	}

}

func TestHeapUpdate(t *testing.T) {
	h := newTtlKeyHeap()

	kvs := make([]*node, 10)

	// add from older expire time to earlier expire time
	// the path is equal to ttl from now
	for i := range kvs {
		path := fmt.Sprintf("%v", 10-i)
		m := time.Duration(10 - i)
		n := newKV(nil, path, path, 0, nil, time.Now().Add(time.Second*m))
		kvs[i] = n
		h.push(n)
	}

	// Path 7
	kvs[3].ExpireTime = time.Now().Add(time.Second * 11)

	// Path 5
	kvs[5].ExpireTime = time.Now().Add(time.Second * 12)

	h.update(kvs[3])
	h.update(kvs[5])

	min := time.Now()

	for i := 0; i < 10; i++ {
		node := h.pop()
		if node.ExpireTime.Before(min) {
			t.Fatal("heap sort wrong!")
		}
		min = node.ExpireTime

		if i == 8 {
			if node.Path != "7" {
				t.Fatal("heap sort wrong!", node.Path)
			}
		}

		if i == 9 {
			if node.Path != "5" {
				t.Fatal("heap sort wrong!")
			}
		}

	}

}
