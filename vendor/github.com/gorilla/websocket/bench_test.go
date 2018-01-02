// Copyright 2014 The Gorilla WebSocket Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

import (
	"testing"
)

func BenchmarkMaskBytes(b *testing.B) {
	var key [4]byte
	data := make([]byte, 1024)
	pos := 0
	for i := 0; i < b.N; i++ {
		pos = maskBytes(key, pos, data)
	}
	b.SetBytes(int64(len(data)))
}
