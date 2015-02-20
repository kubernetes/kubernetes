// Copyright 2012 The Gorilla Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package context

import (
	"net/http"
	"testing"
)

type keyType int

const (
	key1 keyType = iota
	key2
)

func TestContext(t *testing.T) {
	assertEqual := func(val interface{}, exp interface{}) {
		if val != exp {
			t.Errorf("Expected %v, got %v.", exp, val)
		}
	}

	r, _ := http.NewRequest("GET", "http://localhost:8080/", nil)

	// Get()
	assertEqual(Get(r, key1), nil)

	// Set()
	Set(r, key1, "1")
	assertEqual(Get(r, key1), "1")
	assertEqual(len(data[r]), 1)

	Set(r, key2, "2")
	assertEqual(Get(r, key2), "2")
	assertEqual(len(data[r]), 2)

	// Delete()
	Delete(r, key1)
	assertEqual(Get(r, key1), nil)
	assertEqual(len(data[r]), 1)

	Delete(r, key2)
	assertEqual(Get(r, key2), nil)
	assertEqual(len(data[r]), 0)

	// Clear()
	Clear(r)
	assertEqual(len(data), 0)
}
