// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzma

// literalCodec supports the encoding of literal. It provides 768 probability
// values per literal state. The upper 512 probabilities are used with the
// context of a match bit.
type literalCodec struct {
	probs []prob
}

// deepcopy initializes literal codec c as a deep copy of the source.
func (c *literalCodec) deepcopy(src *literalCodec) {
	if c == src {
		return
	}
	c.probs = make([]prob, len(src.probs))
	copy(c.probs, src.probs)
}

// init initializes the literal codec.
func (c *literalCodec) init(lc, lp int) {
	switch {
	case !(minLC <= lc && lc <= maxLC):
		panic("lc out of range")
	case !(minLP <= lp && lp <= maxLP):
		panic("lp out of range")
	}
	c.probs = make([]prob, 0x300<<uint(lc+lp))
	for i := range c.probs {
		c.probs[i] = probInit
	}
}

// Encode encodes the byte s using a range encoder as well as the current LZMA
// encoder state, a match byte and the literal state.
func (c *literalCodec) Encode(e *rangeEncoder, s byte,
	state uint32, match byte, litState uint32,
) (err error) {
	k := litState * 0x300
	probs := c.probs[k : k+0x300]
	symbol := uint32(1)
	r := uint32(s)
	if state >= 7 {
		m := uint32(match)
		for {
			matchBit := (m >> 7) & 1
			m <<= 1
			bit := (r >> 7) & 1
			r <<= 1
			i := ((1 + matchBit) << 8) | symbol
			if err = probs[i].Encode(e, bit); err != nil {
				return
			}
			symbol = (symbol << 1) | bit
			if matchBit != bit {
				break
			}
			if symbol >= 0x100 {
				break
			}
		}
	}
	for symbol < 0x100 {
		bit := (r >> 7) & 1
		r <<= 1
		if err = probs[symbol].Encode(e, bit); err != nil {
			return
		}
		symbol = (symbol << 1) | bit
	}
	return nil
}

// Decode decodes a literal byte using the range decoder as well as the LZMA
// state, a match byte, and the literal state.
func (c *literalCodec) Decode(d *rangeDecoder,
	state uint32, match byte, litState uint32,
) (s byte, err error) {
	k := litState * 0x300
	probs := c.probs[k : k+0x300]
	symbol := uint32(1)
	if state >= 7 {
		m := uint32(match)
		for {
			matchBit := (m >> 7) & 1
			m <<= 1
			i := ((1 + matchBit) << 8) | symbol
			bit, err := d.DecodeBit(&probs[i])
			if err != nil {
				return 0, err
			}
			symbol = (symbol << 1) | bit
			if matchBit != bit {
				break
			}
			if symbol >= 0x100 {
				break
			}
		}
	}
	for symbol < 0x100 {
		bit, err := d.DecodeBit(&probs[symbol])
		if err != nil {
			return 0, err
		}
		symbol = (symbol << 1) | bit
	}
	s = byte(symbol - 0x100)
	return s, nil
}

// minLC and maxLC define the range for LC values.
const (
	minLC = 0
	maxLC = 8
)

// minLC and maxLC define the range for LP values.
const (
	minLP = 0
	maxLP = 4
)

// minState and maxState define a range for the state values stored in
// the State values.
const (
	minState = 0
	maxState = 11
)
