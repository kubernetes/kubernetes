// Copyright 2016 The etcd Authors
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

package main

import (
	"io"
	"math/rand"
	"sync"
	"time"
)

var (
	// dispatchPoolDelay is the time to wait before flushing all buffered packets
	dispatchPoolDelay = 100 * time.Millisecond
	// dispatchPacketBytes is how many bytes to send until choosing a new connection
	dispatchPacketBytes = 32
)

type dispatcher interface {
	// Copy works like io.Copy using buffers provided by fetchFunc
	Copy(io.Writer, fetchFunc) error
}

type fetchFunc func() ([]byte, error)

type dispatcherPool struct {
	// mu protects the dispatch packet queue 'q'
	mu sync.Mutex
	q  []dispatchPacket
}

type dispatchPacket struct {
	buf []byte
	out io.Writer
}

func newDispatcherPool() dispatcher {
	d := &dispatcherPool{}
	go d.writeLoop()
	return d
}

func (d *dispatcherPool) writeLoop() {
	for {
		time.Sleep(dispatchPoolDelay)
		d.flush()
	}
}

func (d *dispatcherPool) flush() {
	d.mu.Lock()
	pkts := d.q
	d.q = nil
	d.mu.Unlock()
	if len(pkts) == 0 {
		return
	}

	// sort by sockets; preserve the packet ordering within a socket
	pktmap := make(map[io.Writer][]dispatchPacket)
	outs := []io.Writer{}
	for _, pkt := range pkts {
		opkts, ok := pktmap[pkt.out]
		if !ok {
			outs = append(outs, pkt.out)
		}
		pktmap[pkt.out] = append(opkts, pkt)
	}

	// send all packets in pkts
	for len(outs) != 0 {
		// randomize writer on every write
		r := rand.Intn(len(outs))
		rpkts := pktmap[outs[r]]
		rpkts[0].out.Write(rpkts[0].buf)
		// dequeue packet
		rpkts = rpkts[1:]
		if len(rpkts) == 0 {
			delete(pktmap, outs[r])
			outs = append(outs[:r], outs[r+1:]...)
		} else {
			pktmap[outs[r]] = rpkts
		}
	}
}

func (d *dispatcherPool) Copy(w io.Writer, f fetchFunc) error {
	for {
		b, err := f()
		if err != nil {
			return err
		}

		pkts := []dispatchPacket{}
		for len(b) > 0 {
			pkt := b
			if len(b) > dispatchPacketBytes {
				pkt = pkt[:dispatchPacketBytes]
				b = b[dispatchPacketBytes:]
			} else {
				b = nil
			}
			pkts = append(pkts, dispatchPacket{pkt, w})
		}

		d.mu.Lock()
		d.q = append(d.q, pkts...)
		d.mu.Unlock()
	}
}

type dispatcherImmediate struct{}

func newDispatcherImmediate() dispatcher {
	return &dispatcherImmediate{}
}

func (d *dispatcherImmediate) Copy(w io.Writer, f fetchFunc) error {
	for {
		b, err := f()
		if err != nil {
			return err
		}
		if _, err := w.Write(b); err != nil {
			return err
		}
	}
}
