// Copyright 2014 The ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ql

import (
	"bytes"
	"encoding/gob"
	"log"
	"math/big"
	"sync"
	"time"
)

const shortBlob = 256 // bytes

var (
	gobInitDuration = time.Duration(278)
	gobInitInt      = big.NewInt(42)
	gobInitRat      = big.NewRat(355, 113)
	gobInitTime     time.Time
)

func init() {
	var err error
	if gobInitTime, err = time.ParseInLocation(
		"Jan 2, 2006 at 3:04pm (MST)",
		"Jul 9, 2012 at 5:02am (CEST)",
		time.FixedZone("XYZ", 1234),
	); err != nil {
		log.Panic(err)
	}
	newGobCoder()
}

type gobCoder struct {
	buf bytes.Buffer
	dec *gob.Decoder
	enc *gob.Encoder
	mu  sync.Mutex
}

func newGobCoder() (g *gobCoder) {
	g = &gobCoder{}
	g.enc = gob.NewEncoder(&g.buf)
	if err := g.enc.Encode(gobInitInt); err != nil {
		log.Panic(err)
	}

	if err := g.enc.Encode(gobInitRat); err != nil {
		log.Panic(err)
	}

	if err := g.enc.Encode(gobInitTime); err != nil {
		log.Panic(err)
	}

	if err := g.enc.Encode(gobInitDuration); err != nil {
		log.Panic(err)
	}

	g.dec = gob.NewDecoder(&g.buf)
	i := big.NewInt(0)
	if err := g.dec.Decode(i); err != nil {
		log.Panic(err)
	}

	r := big.NewRat(3, 5)
	if err := g.dec.Decode(r); err != nil {
		log.Panic(err)
	}

	t := time.Now()
	if err := g.dec.Decode(&t); err != nil {
		log.Panic(err)
	}

	var d time.Duration
	if err := g.dec.Decode(&d); err != nil {
		log.Panic(err)
	}

	return
}

func (g *gobCoder) encode(v interface{}) (b []byte, err error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.buf.Reset()
	switch x := v.(type) {
	case []byte:
		return x, nil
	case *big.Int:
		err = g.enc.Encode(x)
	case *big.Rat:
		err = g.enc.Encode(x)
	case time.Time:
		err = g.enc.Encode(x)
	case time.Duration:
		err = g.enc.Encode(int64(x))
	default:
		//dbg("%T(%v)", v, v)
		log.Panic("internal error 002")
	}
	b = g.buf.Bytes()
	return
}

func (g *gobCoder) decode(b []byte, typ int) (v interface{}, err error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.buf.Reset()
	g.buf.Write(b)
	switch typ {
	case qBlob:
		return b, nil
	case qBigInt:
		x := big.NewInt(0)
		err = g.dec.Decode(&x)
		v = x
	case qBigRat:
		x := big.NewRat(1, 1)
		err = g.dec.Decode(&x)
		v = x
	case qTime:
		var x time.Time
		err = g.dec.Decode(&x)
		v = x
	case qDuration:
		var x int64
		err = g.dec.Decode(&x)
		v = time.Duration(x)
	default:
		log.Panic("internal error 003")
	}
	return
}
