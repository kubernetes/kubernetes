// Copyright (c) 2011 CZ.NIC z.s.p.o. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// blame: jnml, labs.nic.cz

package storage

import (
	"os"
	"testing"
)

func (p *Probe) assert(t *testing.T, msg int, opsRd, opsWr, bytesRd, bytesWr, sectorsRd, sectorsWr int64) {
	if n := p.OpsRd; n != opsRd {
		t.Fatal(msg, n, opsRd)
	}

	if n := p.OpsWr; n != opsWr {
		t.Fatal(msg+1, n, opsWr)
	}

	if n := p.BytesRd; n != bytesRd {
		t.Fatal(msg+2, n, bytesRd)
	}

	if n := p.BytesWr; n != bytesWr {
		t.Fatal(msg+3, n, bytesWr)
	}

	if n := p.SectorsRd; n != sectorsRd {
		t.Fatal(msg+4, n, sectorsRd)
	}

	if n := p.SectorsWr; n != sectorsWr {
		t.Fatal(msg+5, n, sectorsWr)
	}
}

func TestProbe(t *testing.T) {
	return //TODO disabled due to atomic.AddInt64 failing on W32
	const fn = "test.tmp"

	store, err := NewFile(fn, os.O_CREATE|os.O_RDWR|os.O_CREATE, 0666)
	if err != nil {
		t.Fatal(10, err)
	}

	defer func() {
		ec := store.Close()
		er := os.Remove(fn)
		if ec != nil {
			t.Fatal(10000, ec)
		}
		if er != nil {
			t.Fatal(10001, er)
		}
	}()

	probe := NewProbe(store, nil)
	if n, err := probe.WriteAt([]byte{1}, 0); n != 1 {
		t.Fatal(20, err)
	}

	probe.assert(t, 30, 0, 1, 0, 1, 0, 1)
	b := []byte{0}
	if n, err := probe.ReadAt(b, 0); n != 1 {
		t.Fatal(40, err)
	}

	if n := b[0]; n != 1 {
		t.Fatal(50, n, 1)
	}

	probe.assert(t, 60, 1, 1, 1, 1, 1, 1)
	if n, err := probe.WriteAt([]byte{2, 3}, 510); n != 2 {
		t.Fatal(70, err)
	}

	probe.assert(t, 80, 1, 2, 1, 3, 1, 2)
	if n, err := probe.WriteAt([]byte{2, 3}, 511); n != 2 {
		t.Fatal(90, err)
	}

	probe.assert(t, 100, 1, 3, 1, 5, 1, 4)
}
