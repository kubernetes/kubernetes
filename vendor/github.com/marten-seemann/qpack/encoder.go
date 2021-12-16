package qpack

import (
	"io"

	"golang.org/x/net/http2/hpack"
)

// An Encoder performs QPACK encoding.
type Encoder struct {
	wrotePrefix bool

	w   io.Writer
	buf []byte
}

// NewEncoder returns a new Encoder which performs QPACK encoding. An
// encoded data is written to w.
func NewEncoder(w io.Writer) *Encoder {
	return &Encoder{w: w}
}

// WriteField encodes f into a single Write to e's underlying Writer.
// This function may also produce bytes for the Header Block Prefix
// if necessary. If produced, it is done before encoding f.
func (e *Encoder) WriteField(f HeaderField) error {
	// write the Header Block Prefix
	if !e.wrotePrefix {
		e.buf = appendVarInt(e.buf, 8, 0)
		e.buf = appendVarInt(e.buf, 7, 0)
		e.wrotePrefix = true
	}

	idxAndVals, nameFound := encoderMap[f.Name]
	if nameFound {
		if idxAndVals.values == nil {
			if len(f.Value) == 0 {
				e.writeIndexedField(idxAndVals.idx)
			} else {
				e.writeLiteralFieldWithNameReference(&f, idxAndVals.idx)
			}
		} else {
			valIdx, valueFound := idxAndVals.values[f.Value]
			if valueFound {
				e.writeIndexedField(valIdx)
			} else {
				e.writeLiteralFieldWithNameReference(&f, idxAndVals.idx)
			}
		}
	} else {
		e.writeLiteralFieldWithoutNameReference(f)
	}

	e.w.Write(e.buf)
	e.buf = e.buf[:0]
	return nil
}

// Close declares that the encoding is complete and resets the Encoder
// to be reused again for a new header block.
func (e *Encoder) Close() error {
	e.wrotePrefix = false
	return nil
}

func (e *Encoder) writeLiteralFieldWithoutNameReference(f HeaderField) {
	offset := len(e.buf)
	e.buf = appendVarInt(e.buf, 3, hpack.HuffmanEncodeLength(f.Name))
	e.buf[offset] ^= 0x20 ^ 0x8
	e.buf = hpack.AppendHuffmanString(e.buf, f.Name)
	offset = len(e.buf)
	e.buf = appendVarInt(e.buf, 7, hpack.HuffmanEncodeLength(f.Value))
	e.buf[offset] ^= 0x80
	e.buf = hpack.AppendHuffmanString(e.buf, f.Value)
}

// Encodes a header field whose name is present in one of the tables.
func (e *Encoder) writeLiteralFieldWithNameReference(f *HeaderField, id uint8) {
	offset := len(e.buf)
	e.buf = appendVarInt(e.buf, 4, uint64(id))
	// Set the 01NTxxxx pattern, forcing N to 0 and T to 1
	e.buf[offset] ^= 0x50
	offset = len(e.buf)
	e.buf = appendVarInt(e.buf, 7, hpack.HuffmanEncodeLength(f.Value))
	e.buf[offset] ^= 0x80
	e.buf = hpack.AppendHuffmanString(e.buf, f.Value)
}

// Encodes an indexed field, meaning it's entirely defined in one of the tables.
func (e *Encoder) writeIndexedField(id uint8) {
	offset := len(e.buf)
	e.buf = appendVarInt(e.buf, 6, uint64(id))
	// Set the 1Txxxxxx pattern, forcing T to 1
	e.buf[offset] ^= 0xc0
}
