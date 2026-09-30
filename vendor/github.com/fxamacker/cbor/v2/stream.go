// Copyright (c) Faye Amacker. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root for license information.

package cbor

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"reflect"
)

// Decoder reads and decodes CBOR values from io.Reader.
type Decoder struct {
	r         io.Reader
	d         decoder
	buf       []byte
	off       int // next read offset in buf
	bytesRead int
}

// NewDecoder returns a new decoder that reads and decodes from r using
// the default decoding options.
func NewDecoder(r io.Reader) *Decoder {
	return defaultDecMode.NewDecoder(r)
}

// Decode reads CBOR value and decodes it into the value pointed to by v.
func (dec *Decoder) Decode(v any) error {
	_, err := dec.readNext()
	if err != nil {
		// Return validation error or read error.
		return err
	}

	dec.d.reset(dec.buf[dec.off:])
	err = dec.d.value(v)

	// Increment dec.off even if decoding err is not nil because
	// dec.d.off points to the next CBOR data item if current
	// CBOR data item is valid but failed to be decoded into v.
	// This allows next CBOR data item to be decoded in next
	// call to this function.
	dec.off += dec.d.off
	dec.bytesRead += dec.d.off

	return err
}

// Skip skips to the next CBOR data item (if there is any),
// otherwise it returns error such as io.EOF, io.UnexpectedEOF, etc.
func (dec *Decoder) Skip() error {
	n, err := dec.readNext()
	if err != nil {
		// Return validation error or read error.
		return err
	}

	dec.off += n
	dec.bytesRead += n
	return nil
}

// NumBytesRead returns the number of bytes read.
func (dec *Decoder) NumBytesRead() int {
	return dec.bytesRead
}

// Buffered returns a reader for data remaining in Decoder's buffer.
// Returned reader is valid until the next call to Decode or Skip.
func (dec *Decoder) Buffered() io.Reader {
	return bytes.NewReader(dec.buf[dec.off:])
}

// readNext() reads next CBOR data item from Reader to buffer.
// It returns the size of next CBOR data item.
// It also returns validation error or read error if any.
func (dec *Decoder) readNext() (int, error) {
	var readErr error
	var validErr error

	for {
		// Process any unread data in dec.buf.
		if dec.off < len(dec.buf) {
			dec.d.reset(dec.buf[dec.off:])
			off := dec.off // Save offset before data validation
			validErr = dec.d.wellformed(true, false)
			dec.off = off // Restore offset

			if validErr == nil {
				return dec.d.off, nil
			}

			if validErr != io.ErrUnexpectedEOF {
				return 0, validErr
			}

			// Process last read error on io.ErrUnexpectedEOF.
			if readErr != nil {
				if readErr == io.EOF {
					// current CBOR data item is incomplete.
					return 0, io.ErrUnexpectedEOF
				}
				return 0, readErr
			}
		}

		// More data is needed and there was no read error.
		var n int
		for n == 0 {
			n, readErr = dec.read()
			if n == 0 && readErr != nil {
				// No more data can be read and read error is encountered.
				// At this point, validErr is either nil or io.ErrUnexpectedEOF.
				if readErr == io.EOF {
					if validErr == io.ErrUnexpectedEOF {
						// current CBOR data item is incomplete.
						return 0, io.ErrUnexpectedEOF
					}
				}
				return 0, readErr
			}
		}

		// At this point, dec.buf contains new data from last read (n > 0).
	}
}

// read() reads data from Reader to buffer.
// It returns number of bytes read and any read error encountered.
// Postconditions:
// - dec.buf contains previously unread data and new data.
// - dec.off is 0.
func (dec *Decoder) read() (int, error) {
	// Grow buf if needed.
	const minRead = 512
	if cap(dec.buf)-len(dec.buf)+dec.off < minRead {
		oldUnreadBuf := dec.buf[dec.off:]
		dec.buf = make([]byte, len(dec.buf)-dec.off, 2*cap(dec.buf)+minRead)
		dec.overwriteBuf(oldUnreadBuf)
	}

	// Copy unread data over read data and reset off to 0.
	if dec.off > 0 {
		dec.overwriteBuf(dec.buf[dec.off:])
	}

	// Read from reader and reslice buf.
	n, err := dec.r.Read(dec.buf[len(dec.buf):cap(dec.buf)])
	dec.buf = dec.buf[0 : len(dec.buf)+n]
	return n, err
}

func (dec *Decoder) overwriteBuf(newBuf []byte) {
	n := copy(dec.buf, newBuf)
	dec.buf = dec.buf[:n]
	dec.off = 0
}

// indefDataItem tracks open indefinite-length data item during encoding.
// typ is the CBOR type of the indefinite-length data item.
// count is the number of items written so far.
// For indefinite-length maps, count must be even (key/value pairs) when
// the container is closed.
type indefDataItem struct {
	typ   cborType
	count int
}

// IndefiniteLengthMapOddItemCountError indicates that EndIndefinite was
// called on an open indefinite-length map with an odd number of items.
type IndefiniteLengthMapOddItemCountError struct {
	count int
}

func (e *IndefiniteLengthMapOddItemCountError) Error() string {
	return fmt.Sprintf("cbor: cannot end indefinite-length map with %d item(s)", e.count)
}

// Encoder writes CBOR values to io.Writer.
type Encoder struct {
	w       io.Writer
	em      *encMode
	indefs  []indefDataItem
	scratch [1]byte // reused for single-byte writes (indefinite-length head and break code)
}

// NewEncoder returns a new encoder that writes to w using the default encoding options.
func NewEncoder(w io.Writer) *Encoder {
	return defaultEncMode.NewEncoder(w)
}

// Encode writes the CBOR encoding of v.
func (enc *Encoder) Encode(v any) error {
	if len(enc.indefs) > 0 {
		err := validateIndefiniteLengthChunkByType(enc.indefs[len(enc.indefs)-1].typ, v)
		if err != nil {
			return err
		}
	}

	buf := getEncodeBuffer()

	err := encode(buf, enc.em, reflect.ValueOf(v))

	// Validate the encoded chunk against the indefinite-length data item using a byte-based check.
	// This reliably detects chunks from cbor.Marshaler, registered tags, StringToByteString, etc.,
	// which may produce a chunk inconsistent with the parent's major type.
	// Applies only to indefinite-length byte/text string parents (RFC 8949 Section 3.2.3).
	if err == nil && len(enc.indefs) > 0 {
		err = validateIndefiniteLengthChunkByData(enc.indefs[len(enc.indefs)-1].typ, buf.Bytes(), v)
	}

	if err == nil {
		_, err = enc.w.Write(buf.Bytes())
	}

	putEncodeBuffer(buf)

	if err != nil {
		return err
	}

	if len(enc.indefs) > 0 {
		enc.indefs[len(enc.indefs)-1].count++
	}

	return nil
}

// StartIndefiniteByteString starts indefinite-length byte string encoding.
// Subsequent calls of (*Encoder).Encode() encodes definite length byte strings
// ("chunks") as one contiguous string until EndIndefinite is called.
func (enc *Encoder) StartIndefiniteByteString() error {
	return enc.startIndefinite(cborTypeByteString)
}

// StartIndefiniteTextString starts indefinite-length text string encoding.
// Subsequent calls of (*Encoder).Encode() encodes definite length text strings
// ("chunks") as one contiguous string until EndIndefinite is called.
func (enc *Encoder) StartIndefiniteTextString() error {
	return enc.startIndefinite(cborTypeTextString)
}

// StartIndefiniteArray starts indefinite-length array encoding.
// Subsequent calls of (*Encoder).Encode() encodes elements of the array
// until EndIndefinite is called.
func (enc *Encoder) StartIndefiniteArray() error {
	return enc.startIndefinite(cborTypeArray)
}

// StartIndefiniteMap starts indefinite-length map encoding.
// Subsequent calls of (*Encoder).Encode() encodes elements of the map
// until EndIndefinite is called.
func (enc *Encoder) StartIndefiniteMap() error {
	return enc.startIndefinite(cborTypeMap)
}

// EndIndefinite closes last opened indefinite-length value.
// It returns *IndefiniteLengthMapOddItemCountError without writing the
// "break" code if the open indefinite-length map has an odd number of
// items; the encoder state is unchanged so the caller can write the
// missing value and retry.
func (enc *Encoder) EndIndefinite() error {
	if len(enc.indefs) == 0 {
		return errors.New("cbor: cannot encode \"break\" code outside indefinite length values")
	}

	// Verify that indefinite-length map has even number of elements
	top := enc.indefs[len(enc.indefs)-1]
	if top.typ == cborTypeMap && top.count%2 != 0 {
		return &IndefiniteLengthMapOddItemCountError{count: top.count}
	}

	// Write break code
	enc.scratch[0] = cborBreakFlag
	_, err := enc.w.Write(enc.scratch[:])
	if err != nil {
		return err
	}

	enc.indefs = enc.indefs[:len(enc.indefs)-1]

	// Increment parent container's item count because the child
	// (indefinite-length data item) is fully written to the stream.
	if len(enc.indefs) > 0 {
		enc.indefs[len(enc.indefs)-1].count++
	}

	return nil
}

func (enc *Encoder) startIndefinite(typ cborType) error {
	if enc.em.indefLength == IndefLengthForbidden {
		return &IndefiniteLengthError{typ}
	}

	// Verify that new indefinite-length data item is not a chunk in indefinite-length byte/text string.
	if len(enc.indefs) > 0 {
		parent := enc.indefs[len(enc.indefs)-1].typ
		if parent == cborTypeByteString || parent == cborTypeTextString {
			return errors.New("cbor: cannot encode indefinite-length " + typ.String() +
				" as chunk of indefinite-length " + parent.String())
		}
	}

	// Write indefinite-length head.
	enc.scratch[0] = byte(typ) | additionalInformationAsIndefiniteLengthFlag
	_, err := enc.w.Write(enc.scratch[:])
	if err != nil {
		return err
	}

	enc.indefs = append(enc.indefs, indefDataItem{typ: typ})
	return nil
}

// validateIndefiniteLengthChunkByType rejects chunks based solely on their Go type.
func validateIndefiniteLengthChunkByType(indefiniteLengthCborType cborType, v any) error {
	if indefiniteLengthCborType != cborTypeByteString &&
		indefiniteLengthCborType != cborTypeTextString {
		return nil
	}
	if v == nil {
		return errors.New("cbor: cannot encode nil for indefinite-length " + indefiniteLengthCborType.String())
	}
	return nil
}

// validateIndefiniteLengthChunkByData checks that chunk is a definite-length
// CBOR data item with a matching major type.
// No-op for indefinite-length array/map, where any data item is valid.
func validateIndefiniteLengthChunkByData(indefiniteLengthCborType cborType, chunk []byte, v any) error {
	if indefiniteLengthCborType != cborTypeByteString &&
		indefiniteLengthCborType != cborTypeTextString {
		return nil
	}

	if len(chunk) == 0 {
		return errors.New("cbor: cannot encode item type " + reflect.TypeOf(v).Kind().String() +
			" for indefinite-length " + indefiniteLengthCborType.String())
	}

	t, ai := parseInitialByte(chunk[0])
	if t != indefiniteLengthCborType {
		return errors.New("cbor: cannot encode item type " + reflect.TypeOf(v).Kind().String() +
			" for indefinite-length " + indefiniteLengthCborType.String())
	}

	if ai == additionalInformationAsIndefiniteLengthFlag {
		return errors.New("cbor: cannot encode indefinite-length " + indefiniteLengthCborType.String() +
			" as chunk of indefinite-length " + indefiniteLengthCborType.String())
	}
	return nil
}

// RawMessage is a raw encoded CBOR value.
type RawMessage []byte

// MarshalCBOR returns m or CBOR nil if m is nil.
func (m RawMessage) MarshalCBOR() ([]byte, error) {
	if len(m) == 0 {
		b := make([]byte, len(cborNil))
		copy(b, cborNil)
		return b, nil
	}
	return m, nil
}

// UnmarshalCBOR creates a copy of data and saves to *m.
func (m *RawMessage) UnmarshalCBOR(data []byte) error {
	if m == nil {
		return errors.New("cbor.RawMessage: UnmarshalCBOR on nil pointer")
	}
	*m = append((*m)[0:0], data...)
	return nil
}
