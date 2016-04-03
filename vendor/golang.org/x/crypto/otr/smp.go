// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the Socialist Millionaires Protocol as described in
// http://www.cypherpunks.ca/otr/Protocol-v2-3.1.0.html. The protocol
// specification is required in order to understand this code and, where
// possible, the variable names in the code match up with the spec.

package otr

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"hash"
	"math/big"
)

type smpFailure string

func (s smpFailure) Error() string {
	return string(s)
}

var smpFailureError = smpFailure("otr: SMP protocol failed")
var smpSecretMissingError = smpFailure("otr: mutual secret needed")

const smpVersion = 1

const (
	smpState1 = iota
	smpState2
	smpState3
	smpState4
)

type smpState struct {
	state                  int
	a2, a3, b2, b3, pb, qb *big.Int
	g2a, g3a               *big.Int
	g2, g3                 *big.Int
	g3b, papb, qaqb, ra    *big.Int
	saved                  *tlv
	secret                 *big.Int
	question               string
}

func (c *Conversation) startSMP(question string) (tlvs []tlv) {
	if c.smp.state != smpState1 {
		tlvs = append(tlvs, c.generateSMPAbort())
	}
	tlvs = append(tlvs, c.generateSMP1(question))
	c.smp.question = ""
	c.smp.state = smpState2
	return
}

func (c *Conversation) resetSMP() {
	c.smp.state = smpState1
	c.smp.secret = nil
	c.smp.question = ""
}

func (c *Conversation) processSMP(in tlv) (out tlv, complete bool, err error) {
	data := in.data

	switch in.typ {
	case tlvTypeSMPAbort:
		if c.smp.state != smpState1 {
			err = smpFailureError
		}
		c.resetSMP()
		return
	case tlvTypeSMP1WithQuestion:
		// We preprocess this into a SMP1 message.
		nulPos := bytes.IndexByte(data, 0)
		if nulPos == -1 {
			err = errors.New("otr: SMP message with question didn't contain a NUL byte")
			return
		}
		c.smp.question = string(data[:nulPos])
		data = data[nulPos+1:]
	}

	numMPIs, data, ok := getU32(data)
	if !ok || numMPIs > 20 {
		err = errors.New("otr: corrupt SMP message")
		return
	}

	mpis := make([]*big.Int, numMPIs)
	for i := range mpis {
		var ok bool
		mpis[i], data, ok = getMPI(data)
		if !ok {
			err = errors.New("otr: corrupt SMP message")
			return
		}
	}

	switch in.typ {
	case tlvTypeSMP1, tlvTypeSMP1WithQuestion:
		if c.smp.state != smpState1 {
			c.resetSMP()
			out = c.generateSMPAbort()
			return
		}
		if c.smp.secret == nil {
			err = smpSecretMissingError
			return
		}
		if err = c.processSMP1(mpis); err != nil {
			return
		}
		c.smp.state = smpState3
		out = c.generateSMP2()
	case tlvTypeSMP2:
		if c.smp.state != smpState2 {
			c.resetSMP()
			out = c.generateSMPAbort()
			return
		}
		if out, err = c.processSMP2(mpis); err != nil {
			out = c.generateSMPAbort()
			return
		}
		c.smp.state = smpState4
	case tlvTypeSMP3:
		if c.smp.state != smpState3 {
			c.resetSMP()
			out = c.generateSMPAbort()
			return
		}
		if out, err = c.processSMP3(mpis); err != nil {
			return
		}
		c.smp.state = smpState1
		c.smp.secret = nil
		complete = true
	case tlvTypeSMP4:
		if c.smp.state != smpState4 {
			c.resetSMP()
			out = c.generateSMPAbort()
			return
		}
		if err = c.processSMP4(mpis); err != nil {
			out = c.generateSMPAbort()
			return
		}
		c.smp.state = smpState1
		c.smp.secret = nil
		complete = true
	default:
		panic("unknown SMP message")
	}

	return
}

func (c *Conversation) calcSMPSecret(mutualSecret []byte, weStarted bool) {
	h := sha256.New()
	h.Write([]byte{smpVersion})
	if weStarted {
		h.Write(c.PrivateKey.PublicKey.Fingerprint())
		h.Write(c.TheirPublicKey.Fingerprint())
	} else {
		h.Write(c.TheirPublicKey.Fingerprint())
		h.Write(c.PrivateKey.PublicKey.Fingerprint())
	}
	h.Write(c.SSID[:])
	h.Write(mutualSecret)
	c.smp.secret = new(big.Int).SetBytes(h.Sum(nil))
}

func (c *Conversation) generateSMP1(question string) tlv {
	var randBuf [16]byte
	c.smp.a2 = c.randMPI(randBuf[:])
	c.smp.a3 = c.randMPI(randBuf[:])
	g2a := new(big.Int).Exp(g, c.smp.a2, p)
	g3a := new(big.Int).Exp(g, c.smp.a3, p)
	h := sha256.New()

	r2 := c.randMPI(randBuf[:])
	r := new(big.Int).Exp(g, r2, p)
	c2 := new(big.Int).SetBytes(hashMPIs(h, 1, r))
	d2 := new(big.Int).Mul(c.smp.a2, c2)
	d2.Sub(r2, d2)
	d2.Mod(d2, q)
	if d2.Sign() < 0 {
		d2.Add(d2, q)
	}

	r3 := c.randMPI(randBuf[:])
	r.Exp(g, r3, p)
	c3 := new(big.Int).SetBytes(hashMPIs(h, 2, r))
	d3 := new(big.Int).Mul(c.smp.a3, c3)
	d3.Sub(r3, d3)
	d3.Mod(d3, q)
	if d3.Sign() < 0 {
		d3.Add(d3, q)
	}

	var ret tlv
	if len(question) > 0 {
		ret.typ = tlvTypeSMP1WithQuestion
		ret.data = append(ret.data, question...)
		ret.data = append(ret.data, 0)
	} else {
		ret.typ = tlvTypeSMP1
	}
	ret.data = appendU32(ret.data, 6)
	ret.data = appendMPIs(ret.data, g2a, c2, d2, g3a, c3, d3)
	return ret
}

func (c *Conversation) processSMP1(mpis []*big.Int) error {
	if len(mpis) != 6 {
		return errors.New("otr: incorrect number of arguments in SMP1 message")
	}
	g2a := mpis[0]
	c2 := mpis[1]
	d2 := mpis[2]
	g3a := mpis[3]
	c3 := mpis[4]
	d3 := mpis[5]
	h := sha256.New()

	r := new(big.Int).Exp(g, d2, p)
	s := new(big.Int).Exp(g2a, c2, p)
	r.Mul(r, s)
	r.Mod(r, p)
	t := new(big.Int).SetBytes(hashMPIs(h, 1, r))
	if c2.Cmp(t) != 0 {
		return errors.New("otr: ZKP c2 incorrect in SMP1 message")
	}
	r.Exp(g, d3, p)
	s.Exp(g3a, c3, p)
	r.Mul(r, s)
	r.Mod(r, p)
	t.SetBytes(hashMPIs(h, 2, r))
	if c3.Cmp(t) != 0 {
		return errors.New("otr: ZKP c3 incorrect in SMP1 message")
	}

	c.smp.g2a = g2a
	c.smp.g3a = g3a
	return nil
}

func (c *Conversation) generateSMP2() tlv {
	var randBuf [16]byte
	b2 := c.randMPI(randBuf[:])
	c.smp.b3 = c.randMPI(randBuf[:])
	r2 := c.randMPI(randBuf[:])
	r3 := c.randMPI(randBuf[:])
	r4 := c.randMPI(randBuf[:])
	r5 := c.randMPI(randBuf[:])
	r6 := c.randMPI(randBuf[:])

	g2b := new(big.Int).Exp(g, b2, p)
	g3b := new(big.Int).Exp(g, c.smp.b3, p)

	r := new(big.Int).Exp(g, r2, p)
	h := sha256.New()
	c2 := new(big.Int).SetBytes(hashMPIs(h, 3, r))
	d2 := new(big.Int).Mul(b2, c2)
	d2.Sub(r2, d2)
	d2.Mod(d2, q)
	if d2.Sign() < 0 {
		d2.Add(d2, q)
	}

	r.Exp(g, r3, p)
	c3 := new(big.Int).SetBytes(hashMPIs(h, 4, r))
	d3 := new(big.Int).Mul(c.smp.b3, c3)
	d3.Sub(r3, d3)
	d3.Mod(d3, q)
	if d3.Sign() < 0 {
		d3.Add(d3, q)
	}

	c.smp.g2 = new(big.Int).Exp(c.smp.g2a, b2, p)
	c.smp.g3 = new(big.Int).Exp(c.smp.g3a, c.smp.b3, p)
	c.smp.pb = new(big.Int).Exp(c.smp.g3, r4, p)
	c.smp.qb = new(big.Int).Exp(g, r4, p)
	r.Exp(c.smp.g2, c.smp.secret, p)
	c.smp.qb.Mul(c.smp.qb, r)
	c.smp.qb.Mod(c.smp.qb, p)

	s := new(big.Int)
	s.Exp(c.smp.g2, r6, p)
	r.Exp(g, r5, p)
	s.Mul(r, s)
	s.Mod(s, p)
	r.Exp(c.smp.g3, r5, p)
	cp := new(big.Int).SetBytes(hashMPIs(h, 5, r, s))

	// D5 = r5 - r4 cP mod q and D6 = r6 - y cP mod q

	s.Mul(r4, cp)
	r.Sub(r5, s)
	d5 := new(big.Int).Mod(r, q)
	if d5.Sign() < 0 {
		d5.Add(d5, q)
	}

	s.Mul(c.smp.secret, cp)
	r.Sub(r6, s)
	d6 := new(big.Int).Mod(r, q)
	if d6.Sign() < 0 {
		d6.Add(d6, q)
	}

	var ret tlv
	ret.typ = tlvTypeSMP2
	ret.data = appendU32(ret.data, 11)
	ret.data = appendMPIs(ret.data, g2b, c2, d2, g3b, c3, d3, c.smp.pb, c.smp.qb, cp, d5, d6)
	return ret
}

func (c *Conversation) processSMP2(mpis []*big.Int) (out tlv, err error) {
	if len(mpis) != 11 {
		err = errors.New("otr: incorrect number of arguments in SMP2 message")
		return
	}
	g2b := mpis[0]
	c2 := mpis[1]
	d2 := mpis[2]
	g3b := mpis[3]
	c3 := mpis[4]
	d3 := mpis[5]
	pb := mpis[6]
	qb := mpis[7]
	cp := mpis[8]
	d5 := mpis[9]
	d6 := mpis[10]
	h := sha256.New()

	r := new(big.Int).Exp(g, d2, p)
	s := new(big.Int).Exp(g2b, c2, p)
	r.Mul(r, s)
	r.Mod(r, p)
	s.SetBytes(hashMPIs(h, 3, r))
	if c2.Cmp(s) != 0 {
		err = errors.New("otr: ZKP c2 failed in SMP2 message")
		return
	}

	r.Exp(g, d3, p)
	s.Exp(g3b, c3, p)
	r.Mul(r, s)
	r.Mod(r, p)
	s.SetBytes(hashMPIs(h, 4, r))
	if c3.Cmp(s) != 0 {
		err = errors.New("otr: ZKP c3 failed in SMP2 message")
		return
	}

	c.smp.g2 = new(big.Int).Exp(g2b, c.smp.a2, p)
	c.smp.g3 = new(big.Int).Exp(g3b, c.smp.a3, p)

	r.Exp(g, d5, p)
	s.Exp(c.smp.g2, d6, p)
	r.Mul(r, s)
	s.Exp(qb, cp, p)
	r.Mul(r, s)
	r.Mod(r, p)

	s.Exp(c.smp.g3, d5, p)
	t := new(big.Int).Exp(pb, cp, p)
	s.Mul(s, t)
	s.Mod(s, p)
	t.SetBytes(hashMPIs(h, 5, s, r))
	if cp.Cmp(t) != 0 {
		err = errors.New("otr: ZKP cP failed in SMP2 message")
		return
	}

	var randBuf [16]byte
	r4 := c.randMPI(randBuf[:])
	r5 := c.randMPI(randBuf[:])
	r6 := c.randMPI(randBuf[:])
	r7 := c.randMPI(randBuf[:])

	pa := new(big.Int).Exp(c.smp.g3, r4, p)
	r.Exp(c.smp.g2, c.smp.secret, p)
	qa := new(big.Int).Exp(g, r4, p)
	qa.Mul(qa, r)
	qa.Mod(qa, p)

	r.Exp(g, r5, p)
	s.Exp(c.smp.g2, r6, p)
	r.Mul(r, s)
	r.Mod(r, p)

	s.Exp(c.smp.g3, r5, p)
	cp.SetBytes(hashMPIs(h, 6, s, r))

	r.Mul(r4, cp)
	d5 = new(big.Int).Sub(r5, r)
	d5.Mod(d5, q)
	if d5.Sign() < 0 {
		d5.Add(d5, q)
	}

	r.Mul(c.smp.secret, cp)
	d6 = new(big.Int).Sub(r6, r)
	d6.Mod(d6, q)
	if d6.Sign() < 0 {
		d6.Add(d6, q)
	}

	r.ModInverse(qb, p)
	qaqb := new(big.Int).Mul(qa, r)
	qaqb.Mod(qaqb, p)

	ra := new(big.Int).Exp(qaqb, c.smp.a3, p)
	r.Exp(qaqb, r7, p)
	s.Exp(g, r7, p)
	cr := new(big.Int).SetBytes(hashMPIs(h, 7, s, r))

	r.Mul(c.smp.a3, cr)
	d7 := new(big.Int).Sub(r7, r)
	d7.Mod(d7, q)
	if d7.Sign() < 0 {
		d7.Add(d7, q)
	}

	c.smp.g3b = g3b
	c.smp.qaqb = qaqb

	r.ModInverse(pb, p)
	c.smp.papb = new(big.Int).Mul(pa, r)
	c.smp.papb.Mod(c.smp.papb, p)
	c.smp.ra = ra

	out.typ = tlvTypeSMP3
	out.data = appendU32(out.data, 8)
	out.data = appendMPIs(out.data, pa, qa, cp, d5, d6, ra, cr, d7)
	return
}

func (c *Conversation) processSMP3(mpis []*big.Int) (out tlv, err error) {
	if len(mpis) != 8 {
		err = errors.New("otr: incorrect number of arguments in SMP3 message")
		return
	}
	pa := mpis[0]
	qa := mpis[1]
	cp := mpis[2]
	d5 := mpis[3]
	d6 := mpis[4]
	ra := mpis[5]
	cr := mpis[6]
	d7 := mpis[7]
	h := sha256.New()

	r := new(big.Int).Exp(g, d5, p)
	s := new(big.Int).Exp(c.smp.g2, d6, p)
	r.Mul(r, s)
	s.Exp(qa, cp, p)
	r.Mul(r, s)
	r.Mod(r, p)

	s.Exp(c.smp.g3, d5, p)
	t := new(big.Int).Exp(pa, cp, p)
	s.Mul(s, t)
	s.Mod(s, p)
	t.SetBytes(hashMPIs(h, 6, s, r))
	if t.Cmp(cp) != 0 {
		err = errors.New("otr: ZKP cP failed in SMP3 message")
		return
	}

	r.ModInverse(c.smp.qb, p)
	qaqb := new(big.Int).Mul(qa, r)
	qaqb.Mod(qaqb, p)

	r.Exp(qaqb, d7, p)
	s.Exp(ra, cr, p)
	r.Mul(r, s)
	r.Mod(r, p)

	s.Exp(g, d7, p)
	t.Exp(c.smp.g3a, cr, p)
	s.Mul(s, t)
	s.Mod(s, p)
	t.SetBytes(hashMPIs(h, 7, s, r))
	if t.Cmp(cr) != 0 {
		err = errors.New("otr: ZKP cR failed in SMP3 message")
		return
	}

	var randBuf [16]byte
	r7 := c.randMPI(randBuf[:])
	rb := new(big.Int).Exp(qaqb, c.smp.b3, p)

	r.Exp(qaqb, r7, p)
	s.Exp(g, r7, p)
	cr = new(big.Int).SetBytes(hashMPIs(h, 8, s, r))

	r.Mul(c.smp.b3, cr)
	d7 = new(big.Int).Sub(r7, r)
	d7.Mod(d7, q)
	if d7.Sign() < 0 {
		d7.Add(d7, q)
	}

	out.typ = tlvTypeSMP4
	out.data = appendU32(out.data, 3)
	out.data = appendMPIs(out.data, rb, cr, d7)

	r.ModInverse(c.smp.pb, p)
	r.Mul(pa, r)
	r.Mod(r, p)
	s.Exp(ra, c.smp.b3, p)
	if r.Cmp(s) != 0 {
		err = smpFailureError
	}

	return
}

func (c *Conversation) processSMP4(mpis []*big.Int) error {
	if len(mpis) != 3 {
		return errors.New("otr: incorrect number of arguments in SMP4 message")
	}
	rb := mpis[0]
	cr := mpis[1]
	d7 := mpis[2]
	h := sha256.New()

	r := new(big.Int).Exp(c.smp.qaqb, d7, p)
	s := new(big.Int).Exp(rb, cr, p)
	r.Mul(r, s)
	r.Mod(r, p)

	s.Exp(g, d7, p)
	t := new(big.Int).Exp(c.smp.g3b, cr, p)
	s.Mul(s, t)
	s.Mod(s, p)
	t.SetBytes(hashMPIs(h, 8, s, r))
	if t.Cmp(cr) != 0 {
		return errors.New("otr: ZKP cR failed in SMP4 message")
	}

	r.Exp(rb, c.smp.a3, p)
	if r.Cmp(c.smp.papb) != 0 {
		return smpFailureError
	}

	return nil
}

func (c *Conversation) generateSMPAbort() tlv {
	return tlv{typ: tlvTypeSMPAbort}
}

func hashMPIs(h hash.Hash, magic byte, mpis ...*big.Int) []byte {
	if h != nil {
		h.Reset()
	} else {
		h = sha256.New()
	}

	h.Write([]byte{magic})
	for _, mpi := range mpis {
		h.Write(appendMPI(nil, mpi))
	}
	return h.Sum(nil)
}
