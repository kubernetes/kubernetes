// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xz

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"hash"
	"hash/crc32"
	"io"

	"github.com/ulikunitz/xz/lzma"
)

// allZeros checks whether a given byte slice has only zeros.
func allZeros(p []byte) bool {
	for _, c := range p {
		if c != 0 {
			return false
		}
	}
	return true
}

// padLen returns the length of the padding required for the given
// argument.
func padLen(n int64) int {
	k := int(n % 4)
	if k > 0 {
		k = 4 - k
	}
	return k
}

/*** Header ***/

// headerMagic stores the magic bytes for the header
var headerMagic = []byte{0xfd, '7', 'z', 'X', 'Z', 0x00}

// HeaderLen provides the length of the xz file header.
const HeaderLen = 12

// Constants for the checksum methods supported by xz.
const (
	CRC32  byte = 0x1
	CRC64       = 0x4
	SHA256      = 0xa
)

// errInvalidFlags indicates that flags are invalid.
var errInvalidFlags = errors.New("xz: invalid flags")

// verifyFlags returns the error errInvalidFlags if the value is
// invalid.
func verifyFlags(flags byte) error {
	switch flags {
	case CRC32, CRC64, SHA256:
		return nil
	default:
		return errInvalidFlags
	}
}

// flagstrings maps flag values to strings.
var flagstrings = map[byte]string{
	CRC32:  "CRC-32",
	CRC64:  "CRC-64",
	SHA256: "SHA-256",
}

// flagString returns the string representation for the given flags.
func flagString(flags byte) string {
	s, ok := flagstrings[flags]
	if !ok {
		return "invalid"
	}
	return s
}

// newHashFunc returns a function that creates hash instances for the
// hash method encoded in flags.
func newHashFunc(flags byte) (newHash func() hash.Hash, err error) {
	switch flags {
	case CRC32:
		newHash = newCRC32
	case CRC64:
		newHash = newCRC64
	case SHA256:
		newHash = sha256.New
	default:
		err = errInvalidFlags
	}
	return
}

// header provides the actual content of the xz file header: the flags.
type header struct {
	flags byte
}

// Errors returned by readHeader.
var errHeaderMagic = errors.New("xz: invalid header magic bytes")

// ValidHeader checks whether data is a correct xz file header. The
// length of data must be HeaderLen.
func ValidHeader(data []byte) bool {
	var h header
	err := h.UnmarshalBinary(data)
	return err == nil
}

// String returns a string representation of the flags.
func (h header) String() string {
	return flagString(h.flags)
}

// UnmarshalBinary reads header from the provided data slice.
func (h *header) UnmarshalBinary(data []byte) error {
	// header length
	if len(data) != HeaderLen {
		return errors.New("xz: wrong file header length")
	}

	// magic header
	if !bytes.Equal(headerMagic, data[:6]) {
		return errHeaderMagic
	}

	// checksum
	crc := crc32.NewIEEE()
	crc.Write(data[6:8])
	if uint32LE(data[8:]) != crc.Sum32() {
		return errors.New("xz: invalid checksum for file header")
	}

	// stream flags
	if data[6] != 0 {
		return errInvalidFlags
	}
	flags := data[7]
	if err := verifyFlags(flags); err != nil {
		return err
	}

	h.flags = flags
	return nil
}

// MarshalBinary generates the xz file header.
func (h *header) MarshalBinary() (data []byte, err error) {
	if err = verifyFlags(h.flags); err != nil {
		return nil, err
	}

	data = make([]byte, 12)
	copy(data, headerMagic)
	data[7] = h.flags

	crc := crc32.NewIEEE()
	crc.Write(data[6:8])
	putUint32LE(data[8:], crc.Sum32())

	return data, nil
}

/*** Footer ***/

// footerLen defines the length of the footer.
const footerLen = 12

// footerMagic contains the footer magic bytes.
var footerMagic = []byte{'Y', 'Z'}

// footer represents the content of the xz file footer.
type footer struct {
	indexSize int64
	flags     byte
}

// String prints a string representation of the footer structure.
func (f footer) String() string {
	return fmt.Sprintf("%s index size %d", flagString(f.flags), f.indexSize)
}

// Minimum and maximum for the size of the index (backward size).
const (
	minIndexSize = 4
	maxIndexSize = (1 << 32) * 4
)

// MarshalBinary converts footer values into an xz file footer. Note
// that the footer value is checked for correctness.
func (f *footer) MarshalBinary() (data []byte, err error) {
	if err = verifyFlags(f.flags); err != nil {
		return nil, err
	}
	if !(minIndexSize <= f.indexSize && f.indexSize <= maxIndexSize) {
		return nil, errors.New("xz: index size out of range")
	}
	if f.indexSize%4 != 0 {
		return nil, errors.New(
			"xz: index size not aligned to four bytes")
	}

	data = make([]byte, footerLen)

	// backward size (index size)
	s := (f.indexSize / 4) - 1
	putUint32LE(data[4:], uint32(s))
	// flags
	data[9] = f.flags
	// footer magic
	copy(data[10:], footerMagic)

	// CRC-32
	crc := crc32.NewIEEE()
	crc.Write(data[4:10])
	putUint32LE(data, crc.Sum32())

	return data, nil
}

// UnmarshalBinary sets the footer value by unmarshalling an xz file
// footer.
func (f *footer) UnmarshalBinary(data []byte) error {
	if len(data) != footerLen {
		return errors.New("xz: wrong footer length")
	}

	// magic bytes
	if !bytes.Equal(data[10:], footerMagic) {
		return errors.New("xz: footer magic invalid")
	}

	// CRC-32
	crc := crc32.NewIEEE()
	crc.Write(data[4:10])
	if uint32LE(data) != crc.Sum32() {
		return errors.New("xz: footer checksum error")
	}

	var g footer
	// backward size (index size)
	g.indexSize = (int64(uint32LE(data[4:])) + 1) * 4

	// flags
	if data[8] != 0 {
		return errInvalidFlags
	}
	g.flags = data[9]
	if err := verifyFlags(g.flags); err != nil {
		return err
	}

	*f = g
	return nil
}

/*** Block Header ***/

// blockHeader represents the content of an xz block header.
type blockHeader struct {
	compressedSize   int64
	uncompressedSize int64
	filters          []filter
}

// String converts the block header into a string.
func (h blockHeader) String() string {
	var buf bytes.Buffer
	first := true
	if h.compressedSize >= 0 {
		fmt.Fprintf(&buf, "compressed size %d", h.compressedSize)
		first = false
	}
	if h.uncompressedSize >= 0 {
		if !first {
			buf.WriteString(" ")
		}
		fmt.Fprintf(&buf, "uncompressed size %d", h.uncompressedSize)
		first = false
	}
	for _, f := range h.filters {
		if !first {
			buf.WriteString(" ")
		}
		fmt.Fprintf(&buf, "filter %s", f)
		first = false
	}
	return buf.String()
}

// Masks for the block flags.
const (
	filterCountMask         = 0x03
	compressedSizePresent   = 0x40
	uncompressedSizePresent = 0x80
	reservedBlockFlags      = 0x3C
)

// errIndexIndicator signals that an index indicator (0x00) has been found
// instead of an expected block header indicator.
var errIndexIndicator = errors.New("xz: found index indicator")

// readBlockHeader reads the block header.
func readBlockHeader(r io.Reader) (h *blockHeader, n int, err error) {
	var buf bytes.Buffer
	buf.Grow(20)

	// block header size
	z, err := io.CopyN(&buf, r, 1)
	n = int(z)
	if err != nil {
		return nil, n, err
	}
	s := buf.Bytes()[0]
	if s == 0 {
		return nil, n, errIndexIndicator
	}

	// read complete header
	headerLen := (int(s) + 1) * 4
	buf.Grow(headerLen - 1)
	z, err = io.CopyN(&buf, r, int64(headerLen-1))
	n += int(z)
	if err != nil {
		return nil, n, err
	}

	// unmarshal block header
	h = new(blockHeader)
	if err = h.UnmarshalBinary(buf.Bytes()); err != nil {
		return nil, n, err
	}

	return h, n, nil
}

// readSizeInBlockHeader reads the uncompressed or compressed size
// fields in the block header. The present value informs the function
// whether the respective field is actually present in the header.
func readSizeInBlockHeader(r io.ByteReader, present bool) (n int64, err error) {
	if !present {
		return -1, nil
	}
	x, _, err := readUvarint(r)
	if err != nil {
		return 0, err
	}
	if x >= 1<<63 {
		return 0, errors.New("xz: size overflow in block header")
	}
	return int64(x), nil
}

// UnmarshalBinary unmarshals the block header.
func (h *blockHeader) UnmarshalBinary(data []byte) error {
	// Check header length
	s := data[0]
	if data[0] == 0 {
		return errIndexIndicator
	}
	headerLen := (int(s) + 1) * 4
	if len(data) != headerLen {
		return fmt.Errorf("xz: data length %d; want %d", len(data),
			headerLen)
	}
	n := headerLen - 4

	// Check CRC-32
	crc := crc32.NewIEEE()
	crc.Write(data[:n])
	if crc.Sum32() != uint32LE(data[n:]) {
		return errors.New("xz: checksum error for block header")
	}

	// Block header flags
	flags := data[1]
	if flags&reservedBlockFlags != 0 {
		return errors.New("xz: reserved block header flags set")
	}

	r := bytes.NewReader(data[2:n])

	// Compressed size
	var err error
	h.compressedSize, err = readSizeInBlockHeader(
		r, flags&compressedSizePresent != 0)
	if err != nil {
		return err
	}

	// Uncompressed size
	h.uncompressedSize, err = readSizeInBlockHeader(
		r, flags&uncompressedSizePresent != 0)
	if err != nil {
		return err
	}

	h.filters, err = readFilters(r, int(flags&filterCountMask)+1)
	if err != nil {
		return err
	}

	// Check padding
	// Since headerLen is a multiple of 4 we don't need to check
	// alignment.
	k := r.Len()
	// The standard spec says that the padding should have not more
	// than 3 bytes. However we found paddings of 4 or 5 in the
	// wild. See https://github.com/ulikunitz/xz/pull/11 and
	// https://github.com/ulikunitz/xz/issues/15
	//
	// The only reasonable approach seems to be to ignore the
	// padding size. We still check that all padding bytes are zero.
	if !allZeros(data[n-k : n]) {
		return errPadding
	}
	return nil
}

// MarshalBinary marshals the binary header.
func (h *blockHeader) MarshalBinary() (data []byte, err error) {
	if !(minFilters <= len(h.filters) && len(h.filters) <= maxFilters) {
		return nil, errors.New("xz: filter count wrong")
	}
	for i, f := range h.filters {
		if i < len(h.filters)-1 {
			if f.id() == lzmaFilterID {
				return nil, errors.New(
					"xz: LZMA2 filter is not the last")
			}
		} else {
			// last filter
			if f.id() != lzmaFilterID {
				return nil, errors.New("xz: " +
					"last filter must be the LZMA2 filter")
			}
		}
	}

	var buf bytes.Buffer
	// header size must set at the end
	buf.WriteByte(0)

	// flags
	flags := byte(len(h.filters) - 1)
	if h.compressedSize >= 0 {
		flags |= compressedSizePresent
	}
	if h.uncompressedSize >= 0 {
		flags |= uncompressedSizePresent
	}
	buf.WriteByte(flags)

	p := make([]byte, 10)
	if h.compressedSize >= 0 {
		k := putUvarint(p, uint64(h.compressedSize))
		buf.Write(p[:k])
	}
	if h.uncompressedSize >= 0 {
		k := putUvarint(p, uint64(h.uncompressedSize))
		buf.Write(p[:k])
	}

	for _, f := range h.filters {
		fp, err := f.MarshalBinary()
		if err != nil {
			return nil, err
		}
		buf.Write(fp)
	}

	// padding
	for i := padLen(int64(buf.Len())); i > 0; i-- {
		buf.WriteByte(0)
	}

	// crc place holder
	buf.Write(p[:4])

	data = buf.Bytes()
	if len(data)%4 != 0 {
		panic("data length not aligned")
	}
	s := len(data)/4 - 1
	if !(1 < s && s <= 255) {
		panic("wrong block header size")
	}
	data[0] = byte(s)

	crc := crc32.NewIEEE()
	crc.Write(data[:len(data)-4])
	putUint32LE(data[len(data)-4:], crc.Sum32())

	return data, nil
}

// Constants used for marshalling and unmarshalling filters in the xz
// block header.
const (
	minFilters    = 1
	maxFilters    = 4
	minReservedID = 1 << 62
)

// filter represents a filter in the block header.
type filter interface {
	id() uint64
	UnmarshalBinary(data []byte) error
	MarshalBinary() (data []byte, err error)
	reader(r io.Reader, c *ReaderConfig) (fr io.Reader, err error)
	writeCloser(w io.WriteCloser, c *WriterConfig) (fw io.WriteCloser, err error)
	// filter must be last filter
	last() bool
}

// readFilter reads a block filter from the block header. At this point
// in time only the LZMA2 filter is supported.
func readFilter(r io.Reader) (f filter, err error) {
	br := lzma.ByteReader(r)

	// index
	id, _, err := readUvarint(br)
	if err != nil {
		return nil, err
	}

	var data []byte
	switch id {
	case lzmaFilterID:
		data = make([]byte, lzmaFilterLen)
		data[0] = lzmaFilterID
		if _, err = io.ReadFull(r, data[1:]); err != nil {
			return nil, err
		}
		f = new(lzmaFilter)
	default:
		if id >= minReservedID {
			return nil, errors.New(
				"xz: reserved filter id in block stream header")
		}
		return nil, errors.New("xz: invalid filter id")
	}
	if err = f.UnmarshalBinary(data); err != nil {
		return nil, err
	}
	return f, err
}

// readFilters reads count filters. At this point in time only the count
// 1 is supported.
func readFilters(r io.Reader, count int) (filters []filter, err error) {
	if count != 1 {
		return nil, errors.New("xz: unsupported filter count")
	}
	f, err := readFilter(r)
	if err != nil {
		return nil, err
	}
	return []filter{f}, err
}

// writeFilters writes the filters.
func writeFilters(w io.Writer, filters []filter) (n int, err error) {
	for _, f := range filters {
		p, err := f.MarshalBinary()
		if err != nil {
			return n, err
		}
		k, err := w.Write(p)
		n += k
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

/*** Index ***/

// record describes a block in the xz file index.
type record struct {
	unpaddedSize     int64
	uncompressedSize int64
}

// readRecord reads an index record.
func readRecord(r io.ByteReader) (rec record, n int, err error) {
	u, k, err := readUvarint(r)
	n += k
	if err != nil {
		return rec, n, err
	}
	rec.unpaddedSize = int64(u)
	if rec.unpaddedSize < 0 {
		return rec, n, errors.New("xz: unpadded size negative")
	}

	u, k, err = readUvarint(r)
	n += k
	if err != nil {
		return rec, n, err
	}
	rec.uncompressedSize = int64(u)
	if rec.uncompressedSize < 0 {
		return rec, n, errors.New("xz: uncompressed size negative")
	}

	return rec, n, nil
}

// MarshalBinary converts an index record in its binary encoding.
func (rec *record) MarshalBinary() (data []byte, err error) {
	// maximum length of a uvarint is 10
	p := make([]byte, 20)
	n := putUvarint(p, uint64(rec.unpaddedSize))
	n += putUvarint(p[n:], uint64(rec.uncompressedSize))
	return p[:n], nil
}

// writeIndex writes the index, a sequence of records.
func writeIndex(w io.Writer, index []record) (n int64, err error) {
	crc := crc32.NewIEEE()
	mw := io.MultiWriter(w, crc)

	// index indicator
	k, err := mw.Write([]byte{0})
	n += int64(k)
	if err != nil {
		return n, err
	}

	// number of records
	p := make([]byte, 10)
	k = putUvarint(p, uint64(len(index)))
	k, err = mw.Write(p[:k])
	n += int64(k)
	if err != nil {
		return n, err
	}

	// list of records
	for _, rec := range index {
		p, err := rec.MarshalBinary()
		if err != nil {
			return n, err
		}
		k, err = mw.Write(p)
		n += int64(k)
		if err != nil {
			return n, err
		}
	}

	// index padding
	k, err = mw.Write(make([]byte, padLen(int64(n))))
	n += int64(k)
	if err != nil {
		return n, err
	}

	// crc32 checksum
	putUint32LE(p, crc.Sum32())
	k, err = w.Write(p[:4])
	n += int64(k)

	return n, err
}

// readIndexBody reads the index from the reader. It assumes that the
// index indicator has already been read.
func readIndexBody(r io.Reader) (records []record, n int64, err error) {
	crc := crc32.NewIEEE()
	// index indicator
	crc.Write([]byte{0})

	br := lzma.ByteReader(io.TeeReader(r, crc))

	// number of records
	u, k, err := readUvarint(br)
	n += int64(k)
	if err != nil {
		return nil, n, err
	}
	recLen := int(u)
	if recLen < 0 || uint64(recLen) != u {
		return nil, n, errors.New("xz: record number overflow")
	}

	// list of records
	records = make([]record, recLen)
	for i := range records {
		records[i], k, err = readRecord(br)
		n += int64(k)
		if err != nil {
			return nil, n, err
		}
	}

	p := make([]byte, padLen(int64(n+1)), 4)
	k, err = io.ReadFull(br.(io.Reader), p)
	n += int64(k)
	if err != nil {
		return nil, n, err
	}
	if !allZeros(p) {
		return nil, n, errors.New("xz: non-zero byte in index padding")
	}

	// crc32
	s := crc.Sum32()
	p = p[:4]
	k, err = io.ReadFull(br.(io.Reader), p)
	n += int64(k)
	if err != nil {
		return records, n, err
	}
	if uint32LE(p) != s {
		return nil, n, errors.New("xz: wrong checksum for index")
	}

	return records, n, nil
}
