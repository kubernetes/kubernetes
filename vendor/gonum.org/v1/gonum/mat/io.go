// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// version is the current on-disk codec version.
const version uint32 = 0x1

// maxLen is the biggest slice/array len one can create on a 32/64b platform.
const maxLen = int64(int(^uint(0) >> 1))

var (
	headerSize  = binary.Size(storage{})
	sizeInt64   = binary.Size(int64(0))
	sizeFloat64 = binary.Size(float64(0))

	errWrongType = errors.New("mat: wrong data type")

	errTooBig    = errors.New("mat: resulting data slice too big")
	errTooSmall  = errors.New("mat: input slice too small")
	errBadBuffer = errors.New("mat: data buffer size mismatch")
	errBadSize   = errors.New("mat: invalid dimension")
)

// Type encoding scheme:
//
// Type 		Form 	Packing 	Uplo 		Unit 		Rows 	Columns kU 	kL
// uint8 		[GST] 	uint8 [BPF] 	uint8 [AUL] 	bool 		int64 	int64 	int64 	int64
// General 		'G' 	'F' 		'A' 		false 		r 	c 	0 	0
// Band 		'G' 	'B' 		'A' 		false 		r 	c 	kU 	kL
// Symmetric 		'S' 	'F' 		ul 		false 		n 	n 	0 	0
// SymmetricBand 	'S' 	'B' 		ul 		false 		n 	n 	k 	k
// SymmetricPacked 	'S' 	'P' 		ul 		false 		n 	n 	0 	0
// Triangular 		'T' 	'F' 		ul 		Diag==Unit 	n 	n 	0 	0
// TriangularBand 	'T' 	'B' 		ul 		Diag==Unit 	n 	n 	k 	k
// TriangularPacked 	'T' 	'P' 		ul	 	Diag==Unit 	n 	n 	0 	0
//
// G - general, S - symmetric, T - triangular
// F - full, B - band, P - packed
// A - all, U - upper, L - lower

// MarshalBinary encodes the receiver into a binary form and returns the result.
//
// Dense is little-endian encoded as follows:
//   0 -  3  Version = 1          (uint32)
//   4       'G'                  (byte)
//   5       'F'                  (byte)
//   6       'A'                  (byte)
//   7       0                    (byte)
//   8 - 15  number of rows       (int64)
//  16 - 23  number of columns    (int64)
//  24 - 31  0                    (int64)
//  32 - 39  0                    (int64)
//  40 - ..  matrix data elements (float64)
//           [0,0] [0,1] ... [0,ncols-1]
//           [1,0] [1,1] ... [1,ncols-1]
//           ...
//           [nrows-1,0] ... [nrows-1,ncols-1]
func (m Dense) MarshalBinary() ([]byte, error) {
	bufLen := int64(headerSize) + int64(m.mat.Rows)*int64(m.mat.Cols)*int64(sizeFloat64)
	if bufLen <= 0 {
		// bufLen is too big and has wrapped around.
		return nil, errTooBig
	}

	header := storage{
		Form: 'G', Packing: 'F', Uplo: 'A',
		Rows: int64(m.mat.Rows), Cols: int64(m.mat.Cols),
		Version: version,
	}
	buf := make([]byte, bufLen)
	n, err := header.marshalBinaryTo(bytes.NewBuffer(buf[:0]))
	if err != nil {
		return buf[:n], err
	}

	p := headerSize
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			binary.LittleEndian.PutUint64(buf[p:p+sizeFloat64], math.Float64bits(m.at(i, j)))
			p += sizeFloat64
		}
	}

	return buf, nil
}

// MarshalBinaryTo encodes the receiver into a binary form and writes it into w.
// MarshalBinaryTo returns the number of bytes written into w and an error, if any.
//
// See MarshalBinary for the on-disk layout.
func (m Dense) MarshalBinaryTo(w io.Writer) (int, error) {
	header := storage{
		Form: 'G', Packing: 'F', Uplo: 'A',
		Rows: int64(m.mat.Rows), Cols: int64(m.mat.Cols),
		Version: version,
	}
	n, err := header.marshalBinaryTo(w)
	if err != nil {
		return n, err
	}

	r, c := m.Dims()
	var b [8]byte
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			binary.LittleEndian.PutUint64(b[:], math.Float64bits(m.at(i, j)))
			nn, err := w.Write(b[:])
			n += nn
			if err != nil {
				return n, err
			}
		}
	}

	return n, nil
}

// UnmarshalBinary decodes the binary form into the receiver.
// It panics if the receiver is a non-zero Dense matrix.
//
// See MarshalBinary for the on-disk layout.
//
// Limited checks on the validity of the binary input are performed:
//  - matrix.ErrShape is returned if the number of rows or columns is negative,
//  - an error is returned if the resulting Dense matrix is too
//  big for the current architecture (e.g. a 16GB matrix written by a
//  64b application and read back from a 32b application.)
// UnmarshalBinary does not limit the size of the unmarshaled matrix, and so
// it should not be used on untrusted data.
func (m *Dense) UnmarshalBinary(data []byte) error {
	if !m.IsZero() {
		panic("mat: unmarshal into non-zero matrix")
	}

	if len(data) < headerSize {
		return errTooSmall
	}

	var header storage
	err := header.unmarshalBinary(data[:headerSize])
	if err != nil {
		return err
	}
	rows := header.Rows
	cols := header.Cols
	header.Version = 0
	header.Rows = 0
	header.Cols = 0
	if (header != storage{Form: 'G', Packing: 'F', Uplo: 'A'}) {
		return errWrongType
	}
	if rows < 0 || cols < 0 {
		return errBadSize
	}
	size := rows * cols
	if size == 0 {
		return ErrZeroLength
	}
	if int(size) < 0 || size > maxLen {
		return errTooBig
	}
	if len(data) != headerSize+int(rows*cols)*sizeFloat64 {
		return errBadBuffer
	}

	p := headerSize
	m.reuseAs(int(rows), int(cols))
	for i := range m.mat.Data {
		m.mat.Data[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[p : p+sizeFloat64]))
		p += sizeFloat64
	}

	return nil
}

// UnmarshalBinaryFrom decodes the binary form into the receiver and returns
// the number of bytes read and an error if any.
// It panics if the receiver is a non-zero Dense matrix.
//
// See MarshalBinary for the on-disk layout.
//
// Limited checks on the validity of the binary input are performed:
//  - matrix.ErrShape is returned if the number of rows or columns is negative,
//  - an error is returned if the resulting Dense matrix is too
//  big for the current architecture (e.g. a 16GB matrix written by a
//  64b application and read back from a 32b application.)
// UnmarshalBinary does not limit the size of the unmarshaled matrix, and so
// it should not be used on untrusted data.
func (m *Dense) UnmarshalBinaryFrom(r io.Reader) (int, error) {
	if !m.IsZero() {
		panic("mat: unmarshal into non-zero matrix")
	}

	var header storage
	n, err := header.unmarshalBinaryFrom(r)
	if err != nil {
		return n, err
	}
	rows := header.Rows
	cols := header.Cols
	header.Version = 0
	header.Rows = 0
	header.Cols = 0
	if (header != storage{Form: 'G', Packing: 'F', Uplo: 'A'}) {
		return n, errWrongType
	}
	if rows < 0 || cols < 0 {
		return n, errBadSize
	}
	size := rows * cols
	if size == 0 {
		return n, ErrZeroLength
	}
	if int(size) < 0 || size > maxLen {
		return n, errTooBig
	}

	m.reuseAs(int(rows), int(cols))
	var b [8]byte
	for i := range m.mat.Data {
		nn, err := readFull(r, b[:])
		n += nn
		if err != nil {
			if err == io.EOF {
				return n, io.ErrUnexpectedEOF
			}
			return n, err
		}
		m.mat.Data[i] = math.Float64frombits(binary.LittleEndian.Uint64(b[:]))
	}

	return n, nil
}

// MarshalBinary encodes the receiver into a binary form and returns the result.
//
// VecDense is little-endian encoded as follows:
//
//   0 -  3  Version = 1            (uint32)
//   4       'G'                    (byte)
//   5       'F'                    (byte)
//   6       'A'                    (byte)
//   7       0                      (byte)
//   8 - 15  number of elements     (int64)
//  16 - 23  1                      (int64)
//  24 - 31  0                      (int64)
//  32 - 39  0                      (int64)
//  40 - ..  vector's data elements (float64)
func (v VecDense) MarshalBinary() ([]byte, error) {
	bufLen := int64(headerSize) + int64(v.n)*int64(sizeFloat64)
	if bufLen <= 0 {
		// bufLen is too big and has wrapped around.
		return nil, errTooBig
	}

	header := storage{
		Form: 'G', Packing: 'F', Uplo: 'A',
		Rows: int64(v.n), Cols: 1,
		Version: version,
	}
	buf := make([]byte, bufLen)
	n, err := header.marshalBinaryTo(bytes.NewBuffer(buf[:0]))
	if err != nil {
		return buf[:n], err
	}

	p := headerSize
	for i := 0; i < v.n; i++ {
		binary.LittleEndian.PutUint64(buf[p:p+sizeFloat64], math.Float64bits(v.at(i)))
		p += sizeFloat64
	}

	return buf, nil
}

// MarshalBinaryTo encodes the receiver into a binary form, writes it to w and
// returns the number of bytes written and an error if any.
//
// See MarshalBainry for the on-disk format.
func (v VecDense) MarshalBinaryTo(w io.Writer) (int, error) {
	header := storage{
		Form: 'G', Packing: 'F', Uplo: 'A',
		Rows: int64(v.n), Cols: 1,
		Version: version,
	}
	n, err := header.marshalBinaryTo(w)
	if err != nil {
		return n, err
	}

	var buf [8]byte
	for i := 0; i < v.n; i++ {
		binary.LittleEndian.PutUint64(buf[:], math.Float64bits(v.at(i)))
		nn, err := w.Write(buf[:])
		n += nn
		if err != nil {
			return n, err
		}
	}

	return n, nil
}

// UnmarshalBinary decodes the binary form into the receiver.
// It panics if the receiver is a non-zero VecDense.
//
// See MarshalBinary for the on-disk layout.
//
// Limited checks on the validity of the binary input are performed:
//  - matrix.ErrShape is returned if the number of rows is negative,
//  - an error is returned if the resulting VecDense is too
//  big for the current architecture (e.g. a 16GB vector written by a
//  64b application and read back from a 32b application.)
// UnmarshalBinary does not limit the size of the unmarshaled vector, and so
// it should not be used on untrusted data.
func (v *VecDense) UnmarshalBinary(data []byte) error {
	if !v.IsZero() {
		panic("mat: unmarshal into non-zero vector")
	}

	if len(data) < headerSize {
		return errTooSmall
	}

	var header storage
	err := header.unmarshalBinary(data[:headerSize])
	if err != nil {
		return err
	}
	if header.Cols != 1 {
		return ErrShape
	}
	n := header.Rows
	header.Version = 0
	header.Rows = 0
	header.Cols = 0
	if (header != storage{Form: 'G', Packing: 'F', Uplo: 'A'}) {
		return errWrongType
	}
	if n == 0 {
		return ErrZeroLength
	}
	if n < 0 {
		return errBadSize
	}
	if int64(maxLen) < n {
		return errTooBig
	}
	if len(data) != headerSize+int(n)*sizeFloat64 {
		return errBadBuffer
	}

	p := headerSize
	v.reuseAs(int(n))
	for i := range v.mat.Data {
		v.mat.Data[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[p : p+sizeFloat64]))
		p += sizeFloat64
	}

	return nil
}

// UnmarshalBinaryFrom decodes the binary form into the receiver, from the
// io.Reader and returns the number of bytes read and an error if any.
// It panics if the receiver is a non-zero VecDense.
//
// See MarshalBinary for the on-disk layout.
// See UnmarshalBinary for the list of sanity checks performed on the input.
func (v *VecDense) UnmarshalBinaryFrom(r io.Reader) (int, error) {
	if !v.IsZero() {
		panic("mat: unmarshal into non-zero vector")
	}

	var header storage
	n, err := header.unmarshalBinaryFrom(r)
	if err != nil {
		return n, err
	}
	if header.Cols != 1 {
		return n, ErrShape
	}
	l := header.Rows
	header.Version = 0
	header.Rows = 0
	header.Cols = 0
	if (header != storage{Form: 'G', Packing: 'F', Uplo: 'A'}) {
		return n, errWrongType
	}
	if l == 0 {
		return n, ErrZeroLength
	}
	if l < 0 {
		return n, errBadSize
	}
	if int64(maxLen) < l {
		return n, errTooBig
	}

	v.reuseAs(int(l))
	var b [8]byte
	for i := range v.mat.Data {
		nn, err := readFull(r, b[:])
		n += nn
		if err != nil {
			if err == io.EOF {
				return n, io.ErrUnexpectedEOF
			}
			return n, err
		}
		v.mat.Data[i] = math.Float64frombits(binary.LittleEndian.Uint64(b[:]))
	}

	return n, nil
}

// storage is the internal representation of the storage format of a
// serialised matrix.
type storage struct {
	Version uint32 // Keep this first.
	Form    byte   // [GST]
	Packing byte   // [BPF]
	Uplo    byte   // [AUL]
	Unit    bool
	Rows    int64
	Cols    int64
	KU      int64
	KL      int64
}

// TODO(kortschak): Consider replacing these with calls to direct
// encoding/decoding of fields rather than to binary.Write/binary.Read.

func (s storage) marshalBinaryTo(w io.Writer) (int, error) {
	buf := bytes.NewBuffer(make([]byte, 0, headerSize))
	err := binary.Write(buf, binary.LittleEndian, s)
	if err != nil {
		return 0, err
	}
	return w.Write(buf.Bytes())
}

func (s *storage) unmarshalBinary(buf []byte) error {
	err := binary.Read(bytes.NewReader(buf), binary.LittleEndian, s)
	if err != nil {
		return err
	}
	if s.Version != version {
		return fmt.Errorf("mat: incorrect version: %d", s.Version)
	}
	return nil
}

func (s *storage) unmarshalBinaryFrom(r io.Reader) (int, error) {
	buf := make([]byte, headerSize)
	n, err := readFull(r, buf)
	if err != nil {
		return n, err
	}
	return n, s.unmarshalBinary(buf[:n])
}

// readFull reads from r into buf until it has read len(buf).
// It returns the number of bytes copied and an error if fewer bytes were read.
// If an EOF happens after reading fewer than len(buf) bytes, io.ErrUnexpectedEOF is returned.
func readFull(r io.Reader, buf []byte) (int, error) {
	var n int
	var err error
	for n < len(buf) && err == nil {
		var nn int
		nn, err = r.Read(buf[n:])
		n += nn
	}
	if n == len(buf) {
		return n, nil
	}
	if err == io.EOF {
		return n, io.ErrUnexpectedEOF
	}
	return n, err
}
