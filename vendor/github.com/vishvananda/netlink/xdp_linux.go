package netlink

import (
	"bytes"
	"fmt"
)

const (
	xdrDiagUmemLen  = 8 + 8*4
	xdrDiagStatsLen = 6 * 8
)

func (x *XDPDiagUmem) deserialize(b []byte) error {
	if len(b) < xdrDiagUmemLen {
		return fmt.Errorf("XDP umem diagnosis data short read (%d); want %d", len(b), xdrDiagUmemLen)
	}

	rb := bytes.NewBuffer(b)
	x.Size = native.Uint64(rb.Next(8))
	x.ID = native.Uint32(rb.Next(4))
	x.NumPages = native.Uint32(rb.Next(4))
	x.ChunkSize = native.Uint32(rb.Next(4))
	x.Headroom = native.Uint32(rb.Next(4))
	x.Ifindex = native.Uint32(rb.Next(4))
	x.QueueID = native.Uint32(rb.Next(4))
	x.Flags = native.Uint32(rb.Next(4))
	x.Refs = native.Uint32(rb.Next(4))

	return nil
}

func (x *XDPDiagStats) deserialize(b []byte) error {
	if len(b) < xdrDiagStatsLen {
		return fmt.Errorf("XDP diagnosis statistics short read (%d); want %d", len(b), xdrDiagStatsLen)
	}

	rb := bytes.NewBuffer(b)
	x.RxDropped = native.Uint64(rb.Next(8))
	x.RxInvalid = native.Uint64(rb.Next(8))
	x.RxFull = native.Uint64(rb.Next(8))
	x.FillRingEmpty = native.Uint64(rb.Next(8))
	x.TxInvalid = native.Uint64(rb.Next(8))
	x.TxRingEmpty = native.Uint64(rb.Next(8))

	return nil
}
