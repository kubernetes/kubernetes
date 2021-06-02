// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !race
// +build !race

package socket

func (m *Message) raceRead() {
}
func (m *Message) raceWrite() {
}
