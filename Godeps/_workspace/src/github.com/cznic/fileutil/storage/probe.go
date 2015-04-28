// Copyright (c) 2011 CZ.NIC z.s.p.o. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// blame: jnml, labs.nic.cz

package storage

import "sync/atomic"

// Probe collects usage statistics of the embeded Accessor.
// Probe itself IS an Accessor.
type Probe struct {
	Accessor
	Chain     *Probe
	OpsRd     int64
	OpsWr     int64
	BytesRd   int64
	BytesWr   int64
	SectorsRd int64 // Assuming 512 byte sector size
	SectorsWr int64
}

// NewProbe returns a newly created probe which embedes the src Accessor.
// The retuned *Probe satisfies Accessor. if chain != nil then Reset()
// is cascaded down the chained Probes.
func NewProbe(src Accessor, chain *Probe) *Probe {
	return &Probe{Accessor: src, Chain: chain}
}

func reset(n *int64) {
	atomic.AddInt64(n, -atomic.AddInt64(n, 0))
}

// Reset zeroes the collected statistics of p.
func (p *Probe) Reset() {
	if p.Chain != nil {
		p.Chain.Reset()
	}
	reset(&p.OpsRd)
	reset(&p.OpsWr)
	reset(&p.BytesRd)
	reset(&p.BytesWr)
	reset(&p.SectorsRd)
	reset(&p.SectorsWr)
}

func (p *Probe) ReadAt(b []byte, off int64) (n int, err error) {
	n, err = p.Accessor.ReadAt(b, off)
	atomic.AddInt64(&p.OpsRd, 1)
	atomic.AddInt64(&p.BytesRd, int64(n))
	if n <= 0 {
		return
	}

	sectorFirst := off >> 9
	sectorLast := (off + int64(n) - 1) >> 9
	atomic.AddInt64(&p.SectorsRd, sectorLast-sectorFirst+1)
	return
}

func (p *Probe) WriteAt(b []byte, off int64) (n int, err error) {
	n, err = p.Accessor.WriteAt(b, off)
	atomic.AddInt64(&p.OpsWr, 1)
	atomic.AddInt64(&p.BytesWr, int64(n))
	if n <= 0 {
		return
	}

	sectorFirst := off >> 9
	sectorLast := (off + int64(n) - 1) >> 9
	atomic.AddInt64(&p.SectorsWr, sectorLast-sectorFirst+1)
	return
}
