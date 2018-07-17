//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.

package utils

// From http://www.ashishbanerjee.com/home/go/go-generate-uuid

import (
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"io"
	"sync"

	"github.com/lpabon/godbc"
)

type IdSource struct {
	io.Reader
}

var (
	Randomness = rand.Reader
)

func (s IdSource) ReadUUID() string {
	uuid := make([]byte, 16)
	n, err := s.Read(uuid)
	godbc.Check(n == len(uuid), n, len(uuid))
	godbc.Check(err == nil, err)

	return hex.EncodeToString(uuid)
}

// Return a 16-byte uuid
func GenUUID() string {
	return IdSource{Randomness}.ReadUUID()
}

type NonRandom struct {
	count uint64
	lock  sync.Mutex
}

func (n *NonRandom) Count() (curr uint64) {
	n.lock.Lock()
	defer n.lock.Unlock()
	curr = n.count
	n.count++
	return
}

func (n *NonRandom) Read(p []byte) (s int, err error) {
	offset := 0
	if len(p) > 8 {
		offset = len(p) - 8
	}

	binary.BigEndian.PutUint64(p[offset:], n.Count())
	s = len(p)
	return
}
