// Copyright 2014 The Gorilla WebSocket Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

import (
	"net/http"
	"testing"
)

var tokenListContainsValueTests = []struct {
	value string
	ok    bool
}{
	{"WebSocket", true},
	{"WEBSOCKET", true},
	{"websocket", true},
	{"websockets", false},
	{"x websocket", false},
	{"websocket x", false},
	{"other,websocket,more", true},
	{"other, websocket, more", true},
}

func TestTokenListContainsValue(t *testing.T) {
	for _, tt := range tokenListContainsValueTests {
		h := http.Header{"Upgrade": {tt.value}}
		ok := tokenListContainsValue(h, "Upgrade", "websocket")
		if ok != tt.ok {
			t.Errorf("tokenListContainsValue(h, n, %q) = %v, want %v", tt.value, ok, tt.ok)
		}
	}
}
