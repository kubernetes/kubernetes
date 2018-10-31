// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzma

import (
	"fmt"
	"io"
)

// opLenMargin provides the upper limit of the number of bytes required
// to encode a single operation.
const opLenMargin = 10

// compressFlags control the compression process.
type compressFlags uint32

// Values for compressFlags.
const (
	// all data should be compressed, even if compression is not
	// optimal.
	all compressFlags = 1 << iota
)

// encoderFlags provide the flags for an encoder.
type encoderFlags uint32

// Flags for the encoder.
const (
	// eosMarker requests an EOS marker to be written.
	eosMarker encoderFlags = 1 << iota
)

// Encoder compresses data buffered in the encoder dictionary and writes
// it into a byte writer.
type encoder struct {
	dict  *encoderDict
	state *state
	re    *rangeEncoder
	start int64
	// generate eos marker
	marker bool
	limit  bool
	margin int
}

// newEncoder creates a new encoder. If the byte writer must be
// limited use LimitedByteWriter provided by this package. The flags
// argument supports the eosMarker flag, controlling whether a
// terminating end-of-stream marker must be written.
func newEncoder(bw io.ByteWriter, state *state, dict *encoderDict,
	flags encoderFlags) (e *encoder, err error) {

	re, err := newRangeEncoder(bw)
	if err != nil {
		return nil, err
	}
	e = &encoder{
		dict:   dict,
		state:  state,
		re:     re,
		marker: flags&eosMarker != 0,
		start:  dict.Pos(),
		margin: opLenMargin,
	}
	if e.marker {
		e.margin += 5
	}
	return e, nil
}

// Write writes the bytes from p into the dictionary. If not enough
// space is available the data in the dictionary buffer will be
// compressed to make additional space available. If the limit of the
// underlying writer has been reached ErrLimit will be returned.
func (e *encoder) Write(p []byte) (n int, err error) {
	for {
		k, err := e.dict.Write(p[n:])
		n += k
		if err == ErrNoSpace {
			if err = e.compress(0); err != nil {
				return n, err
			}
			continue
		}
		return n, err
	}
}

// Reopen reopens the encoder with a new byte writer.
func (e *encoder) Reopen(bw io.ByteWriter) error {
	var err error
	if e.re, err = newRangeEncoder(bw); err != nil {
		return err
	}
	e.start = e.dict.Pos()
	e.limit = false
	return nil
}

// writeLiteral writes a literal into the LZMA stream
func (e *encoder) writeLiteral(l lit) error {
	var err error
	state, state2, _ := e.state.states(e.dict.Pos())
	if err = e.state.isMatch[state2].Encode(e.re, 0); err != nil {
		return err
	}
	litState := e.state.litState(e.dict.ByteAt(1), e.dict.Pos())
	match := e.dict.ByteAt(int(e.state.rep[0]) + 1)
	err = e.state.litCodec.Encode(e.re, l.b, state, match, litState)
	if err != nil {
		return err
	}
	e.state.updateStateLiteral()
	return nil
}

// iverson implements the Iverson operator as proposed by Donald Knuth in his
// book Concrete Mathematics.
func iverson(ok bool) uint32 {
	if ok {
		return 1
	}
	return 0
}

// writeMatch writes a repetition operation into the operation stream
func (e *encoder) writeMatch(m match) error {
	var err error
	if !(minDistance <= m.distance && m.distance <= maxDistance) {
		panic(fmt.Errorf("match distance %d out of range", m.distance))
	}
	dist := uint32(m.distance - minDistance)
	if !(minMatchLen <= m.n && m.n <= maxMatchLen) &&
		!(dist == e.state.rep[0] && m.n == 1) {
		panic(fmt.Errorf(
			"match length %d out of range; dist %d rep[0] %d",
			m.n, dist, e.state.rep[0]))
	}
	state, state2, posState := e.state.states(e.dict.Pos())
	if err = e.state.isMatch[state2].Encode(e.re, 1); err != nil {
		return err
	}
	g := 0
	for ; g < 4; g++ {
		if e.state.rep[g] == dist {
			break
		}
	}
	b := iverson(g < 4)
	if err = e.state.isRep[state].Encode(e.re, b); err != nil {
		return err
	}
	n := uint32(m.n - minMatchLen)
	if b == 0 {
		// simple match
		e.state.rep[3], e.state.rep[2], e.state.rep[1], e.state.rep[0] =
			e.state.rep[2], e.state.rep[1], e.state.rep[0], dist
		e.state.updateStateMatch()
		if err = e.state.lenCodec.Encode(e.re, n, posState); err != nil {
			return err
		}
		return e.state.distCodec.Encode(e.re, dist, n)
	}
	b = iverson(g != 0)
	if err = e.state.isRepG0[state].Encode(e.re, b); err != nil {
		return err
	}
	if b == 0 {
		// g == 0
		b = iverson(m.n != 1)
		if err = e.state.isRepG0Long[state2].Encode(e.re, b); err != nil {
			return err
		}
		if b == 0 {
			e.state.updateStateShortRep()
			return nil
		}
	} else {
		// g in {1,2,3}
		b = iverson(g != 1)
		if err = e.state.isRepG1[state].Encode(e.re, b); err != nil {
			return err
		}
		if b == 1 {
			// g in {2,3}
			b = iverson(g != 2)
			err = e.state.isRepG2[state].Encode(e.re, b)
			if err != nil {
				return err
			}
			if b == 1 {
				e.state.rep[3] = e.state.rep[2]
			}
			e.state.rep[2] = e.state.rep[1]
		}
		e.state.rep[1] = e.state.rep[0]
		e.state.rep[0] = dist
	}
	e.state.updateStateRep()
	return e.state.repLenCodec.Encode(e.re, n, posState)
}

// writeOp writes a single operation to the range encoder. The function
// checks whether there is enough space available to close the LZMA
// stream.
func (e *encoder) writeOp(op operation) error {
	if e.re.Available() < int64(e.margin) {
		return ErrLimit
	}
	switch x := op.(type) {
	case lit:
		return e.writeLiteral(x)
	case match:
		return e.writeMatch(x)
	default:
		panic("unexpected operation")
	}
}

// compress compressed data from the dictionary buffer. If the flag all
// is set, all data in the dictionary buffer will be compressed. The
// function returns ErrLimit if the underlying writer has reached its
// limit.
func (e *encoder) compress(flags compressFlags) error {
	n := 0
	if flags&all == 0 {
		n = maxMatchLen - 1
	}
	d := e.dict
	m := d.m
	for d.Buffered() > n {
		op := m.NextOp(e.state.rep)
		if err := e.writeOp(op); err != nil {
			return err
		}
		d.Discard(op.Len())
	}
	return nil
}

// eosMatch is a pseudo operation that indicates the end of the stream.
var eosMatch = match{distance: maxDistance, n: minMatchLen}

// Close terminates the LZMA stream. If requested the end-of-stream
// marker will be written. If the byte writer limit has been or will be
// reached during compression of the remaining data in the buffer the
// LZMA stream will be closed and data will remain in the buffer.
func (e *encoder) Close() error {
	err := e.compress(all)
	if err != nil && err != ErrLimit {
		return err
	}
	if e.marker {
		if err := e.writeMatch(eosMatch); err != nil {
			return err
		}
	}
	err = e.re.Close()
	return err
}

// Compressed returns the number bytes of the input data that been
// compressed.
func (e *encoder) Compressed() int64 {
	return e.dict.Pos() - e.start
}
