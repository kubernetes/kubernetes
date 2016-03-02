// Copyright 2014 The Go Authors.
// See https://code.google.com/p/go/source/browse/CONTRIBUTORS
// Licensed under the same terms as Go itself:
// https://code.google.com/p/go/source/browse/LICENSE

package hpack

import (
	"bytes"
	"io"
	"sync"
)

var bufPool = sync.Pool{
	New: func() interface{} { return new(bytes.Buffer) },
}

// HuffmanDecode decodes the string in v and writes the expanded
// result to w, returning the number of bytes written to w and the
// Write call's return value. At most one Write call is made.
func HuffmanDecode(w io.Writer, v []byte) (int, error) {
	buf := bufPool.Get().(*bytes.Buffer)
	buf.Reset()
	defer bufPool.Put(buf)

	n := rootHuffmanNode
	cur, nbits := uint(0), uint8(0)
	for _, b := range v {
		cur = cur<<8 | uint(b)
		nbits += 8
		for nbits >= 8 {
			n = n.children[byte(cur>>(nbits-8))]
			if n.children == nil {
				buf.WriteByte(n.sym)
				nbits -= n.codeLen
				n = rootHuffmanNode
			} else {
				nbits -= 8
			}
		}
	}
	for nbits > 0 {
		n = n.children[byte(cur<<(8-nbits))]
		if n.children != nil || n.codeLen > nbits {
			break
		}
		buf.WriteByte(n.sym)
		nbits -= n.codeLen
		n = rootHuffmanNode
	}
	return w.Write(buf.Bytes())
}

type node struct {
	// children is non-nil for internal nodes
	children []*node

	// The following are only valid if children is nil:
	codeLen uint8 // number of bits that led to the output of sym
	sym     byte  // output symbol
}

func newInternalNode() *node {
	return &node{children: make([]*node, 256)}
}

var rootHuffmanNode = newInternalNode()

func init() {
	for i, code := range huffmanCodes {
		if i > 255 {
			panic("too many huffman codes")
		}
		addDecoderNode(byte(i), code, huffmanCodeLen[i])
	}
}

func addDecoderNode(sym byte, code uint32, codeLen uint8) {
	cur := rootHuffmanNode
	for codeLen > 8 {
		codeLen -= 8
		i := uint8(code >> codeLen)
		if cur.children[i] == nil {
			cur.children[i] = newInternalNode()
		}
		cur = cur.children[i]
	}
	shift := 8 - codeLen
	start, end := int(uint8(code<<shift)), int(1<<shift)
	for i := start; i < start+end; i++ {
		cur.children[i] = &node{sym: sym, codeLen: codeLen}
	}
}

// AppendHuffmanString appends s, as encoded in Huffman codes, to dst
// and returns the extended buffer.
func AppendHuffmanString(dst []byte, s string) []byte {
	rembits := uint8(8)

	for i := 0; i < len(s); i++ {
		if rembits == 8 {
			dst = append(dst, 0)
		}
		dst, rembits = appendByteToHuffmanCode(dst, rembits, s[i])
	}

	if rembits < 8 {
		// special EOS symbol
		code := uint32(0x3fffffff)
		nbits := uint8(30)

		t := uint8(code >> (nbits - rembits))
		dst[len(dst)-1] |= t
	}

	return dst
}

// HuffmanEncodeLength returns the number of bytes required to encode
// s in Huffman codes. The result is round up to byte boundary.
func HuffmanEncodeLength(s string) uint64 {
	n := uint64(0)
	for i := 0; i < len(s); i++ {
		n += uint64(huffmanCodeLen[s[i]])
	}
	return (n + 7) / 8
}

// appendByteToHuffmanCode appends Huffman code for c to dst and
// returns the extended buffer and the remaining bits in the last
// element. The appending is not byte aligned and the remaining bits
// in the last element of dst is given in rembits.
func appendByteToHuffmanCode(dst []byte, rembits uint8, c byte) ([]byte, uint8) {
	code := huffmanCodes[c]
	nbits := huffmanCodeLen[c]

	for {
		if rembits > nbits {
			t := uint8(code << (rembits - nbits))
			dst[len(dst)-1] |= t
			rembits -= nbits
			break
		}

		t := uint8(code >> (nbits - rembits))
		dst[len(dst)-1] |= t

		nbits -= rembits
		rembits = 8

		if nbits == 0 {
			break
		}

		dst = append(dst, 0)
	}

	return dst, rembits
}
