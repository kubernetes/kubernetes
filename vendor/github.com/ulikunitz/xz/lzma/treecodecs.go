// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzma

// treeCodec encodes or decodes values with a fixed bit size. It is using a
// tree of probability value. The root of the tree is the most-significant bit.
type treeCodec struct {
	probTree
}

// makeTreeCodec makes a tree codec. The bits value must be inside the range
// [1,32].
func makeTreeCodec(bits int) treeCodec {
	return treeCodec{makeProbTree(bits)}
}

// deepcopy initializes tc as a deep copy of the source.
func (tc *treeCodec) deepcopy(src *treeCodec) {
	tc.probTree.deepcopy(&src.probTree)
}

// Encode uses the range encoder to encode a fixed-bit-size value.
func (tc *treeCodec) Encode(e *rangeEncoder, v uint32) (err error) {
	m := uint32(1)
	for i := int(tc.bits) - 1; i >= 0; i-- {
		b := (v >> uint(i)) & 1
		if err := e.EncodeBit(b, &tc.probs[m]); err != nil {
			return err
		}
		m = (m << 1) | b
	}
	return nil
}

// Decodes uses the range decoder to decode a fixed-bit-size value. Errors may
// be caused by the range decoder.
func (tc *treeCodec) Decode(d *rangeDecoder) (v uint32, err error) {
	m := uint32(1)
	for j := 0; j < int(tc.bits); j++ {
		b, err := d.DecodeBit(&tc.probs[m])
		if err != nil {
			return 0, err
		}
		m = (m << 1) | b
	}
	return m - (1 << uint(tc.bits)), nil
}

// treeReverseCodec is another tree codec, where the least-significant bit is
// the start of the probability tree.
type treeReverseCodec struct {
	probTree
}

// deepcopy initializes the treeReverseCodec as a deep copy of the
// source.
func (tc *treeReverseCodec) deepcopy(src *treeReverseCodec) {
	tc.probTree.deepcopy(&src.probTree)
}

// makeTreeReverseCodec creates treeReverseCodec value. The bits argument must
// be in the range [1,32].
func makeTreeReverseCodec(bits int) treeReverseCodec {
	return treeReverseCodec{makeProbTree(bits)}
}

// Encode uses range encoder to encode a fixed-bit-size value. The range
// encoder may cause errors.
func (tc *treeReverseCodec) Encode(v uint32, e *rangeEncoder) (err error) {
	m := uint32(1)
	for i := uint(0); i < uint(tc.bits); i++ {
		b := (v >> i) & 1
		if err := e.EncodeBit(b, &tc.probs[m]); err != nil {
			return err
		}
		m = (m << 1) | b
	}
	return nil
}

// Decodes uses the range decoder to decode a fixed-bit-size value. Errors
// returned by the range decoder will be returned.
func (tc *treeReverseCodec) Decode(d *rangeDecoder) (v uint32, err error) {
	m := uint32(1)
	for j := uint(0); j < uint(tc.bits); j++ {
		b, err := d.DecodeBit(&tc.probs[m])
		if err != nil {
			return 0, err
		}
		m = (m << 1) | b
		v |= b << j
	}
	return v, nil
}

// probTree stores enough probability values to be used by the treeEncode and
// treeDecode methods of the range coder types.
type probTree struct {
	probs []prob
	bits  byte
}

// deepcopy initializes the probTree value as a deep copy of the source.
func (t *probTree) deepcopy(src *probTree) {
	if t == src {
		return
	}
	t.probs = make([]prob, len(src.probs))
	copy(t.probs, src.probs)
	t.bits = src.bits
}

// makeProbTree initializes a probTree structure.
func makeProbTree(bits int) probTree {
	if !(1 <= bits && bits <= 32) {
		panic("bits outside of range [1,32]")
	}
	t := probTree{
		bits:  byte(bits),
		probs: make([]prob, 1<<uint(bits)),
	}
	for i := range t.probs {
		t.probs[i] = probInit
	}
	return t
}

// Bits provides the number of bits for the values to de- or encode.
func (t *probTree) Bits() int {
	return int(t.bits)
}
