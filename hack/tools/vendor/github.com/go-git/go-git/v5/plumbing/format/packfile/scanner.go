package packfile

import (
	"bufio"
	"bytes"
	"compress/zlib"
	"fmt"
	"hash"
	"hash/crc32"
	"io"
	stdioutil "io/ioutil"
	"sync"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/utils/binary"
	"github.com/go-git/go-git/v5/utils/ioutil"
)

var (
	// ErrEmptyPackfile is returned by ReadHeader when no data is found in the packfile
	ErrEmptyPackfile = NewError("empty packfile")
	// ErrBadSignature is returned by ReadHeader when the signature in the packfile is incorrect.
	ErrBadSignature = NewError("malformed pack file signature")
	// ErrUnsupportedVersion is returned by ReadHeader when the packfile version is
	// different than VersionSupported.
	ErrUnsupportedVersion = NewError("unsupported packfile version")
	// ErrSeekNotSupported returned if seek is not support
	ErrSeekNotSupported = NewError("not seek support")
)

// ObjectHeader contains the information related to the object, this information
// is collected from the previous bytes to the content of the object.
type ObjectHeader struct {
	Type            plumbing.ObjectType
	Offset          int64
	Length          int64
	Reference       plumbing.Hash
	OffsetReference int64
}

type Scanner struct {
	r   *scannerReader
	crc hash.Hash32

	// pendingObject is used to detect if an object has been read, or still
	// is waiting to be read
	pendingObject    *ObjectHeader
	version, objects uint32

	// lsSeekable says if this scanner can do Seek or not, to have a Scanner
	// seekable a r implementing io.Seeker is required
	IsSeekable bool
}

// NewScanner returns a new Scanner based on a reader, if the given reader
// implements io.ReadSeeker the Scanner will be also Seekable
func NewScanner(r io.Reader) *Scanner {
	_, ok := r.(io.ReadSeeker)

	crc := crc32.NewIEEE()
	return &Scanner{
		r:          newScannerReader(r, crc),
		crc:        crc,
		IsSeekable: ok,
	}
}

func (s *Scanner) Reset(r io.Reader) {
	_, ok := r.(io.ReadSeeker)

	s.r.Reset(r)
	s.crc.Reset()
	s.IsSeekable = ok
	s.pendingObject = nil
	s.version = 0
	s.objects = 0
}

// Header reads the whole packfile header (signature, version and object count).
// It returns the version and the object count and performs checks on the
// validity of the signature and the version fields.
func (s *Scanner) Header() (version, objects uint32, err error) {
	if s.version != 0 {
		return s.version, s.objects, nil
	}

	sig, err := s.readSignature()
	if err != nil {
		if err == io.EOF {
			err = ErrEmptyPackfile
		}

		return
	}

	if !s.isValidSignature(sig) {
		err = ErrBadSignature
		return
	}

	version, err = s.readVersion()
	s.version = version
	if err != nil {
		return
	}

	if !s.isSupportedVersion(version) {
		err = ErrUnsupportedVersion.AddDetails("%d", version)
		return
	}

	objects, err = s.readCount()
	s.objects = objects
	return
}

// readSignature reads an returns the signature field in the packfile.
func (s *Scanner) readSignature() ([]byte, error) {
	var sig = make([]byte, 4)
	if _, err := io.ReadFull(s.r, sig); err != nil {
		return []byte{}, err
	}

	return sig, nil
}

// isValidSignature returns if sig is a valid packfile signature.
func (s *Scanner) isValidSignature(sig []byte) bool {
	return bytes.Equal(sig, signature)
}

// readVersion reads and returns the version field of a packfile.
func (s *Scanner) readVersion() (uint32, error) {
	return binary.ReadUint32(s.r)
}

// isSupportedVersion returns whether version v is supported by the parser.
// The current supported version is VersionSupported, defined above.
func (s *Scanner) isSupportedVersion(v uint32) bool {
	return v == VersionSupported
}

// readCount reads and returns the count of objects field of a packfile.
func (s *Scanner) readCount() (uint32, error) {
	return binary.ReadUint32(s.r)
}

// SeekObjectHeader seeks to specified offset and returns the ObjectHeader
// for the next object in the reader
func (s *Scanner) SeekObjectHeader(offset int64) (*ObjectHeader, error) {
	// if seeking we assume that you are not interested in the header
	if s.version == 0 {
		s.version = VersionSupported
	}

	if _, err := s.r.Seek(offset, io.SeekStart); err != nil {
		return nil, err
	}

	h, err := s.nextObjectHeader()
	if err != nil {
		return nil, err
	}

	h.Offset = offset
	return h, nil
}

// NextObjectHeader returns the ObjectHeader for the next object in the reader
func (s *Scanner) NextObjectHeader() (*ObjectHeader, error) {
	if err := s.doPending(); err != nil {
		return nil, err
	}

	offset, err := s.r.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, err
	}

	h, err := s.nextObjectHeader()
	if err != nil {
		return nil, err
	}

	h.Offset = offset
	return h, nil
}

// nextObjectHeader returns the ObjectHeader for the next object in the reader
// without the Offset field
func (s *Scanner) nextObjectHeader() (*ObjectHeader, error) {
	s.r.Flush()
	s.crc.Reset()

	h := &ObjectHeader{}
	s.pendingObject = h

	var err error
	h.Offset, err = s.r.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, err
	}

	h.Type, h.Length, err = s.readObjectTypeAndLength()
	if err != nil {
		return nil, err
	}

	switch h.Type {
	case plumbing.OFSDeltaObject:
		no, err := binary.ReadVariableWidthInt(s.r)
		if err != nil {
			return nil, err
		}

		h.OffsetReference = h.Offset - no
	case plumbing.REFDeltaObject:
		var err error
		h.Reference, err = binary.ReadHash(s.r)
		if err != nil {
			return nil, err
		}
	}

	return h, nil
}

func (s *Scanner) doPending() error {
	if s.version == 0 {
		var err error
		s.version, s.objects, err = s.Header()
		if err != nil {
			return err
		}
	}

	return s.discardObjectIfNeeded()
}

func (s *Scanner) discardObjectIfNeeded() error {
	if s.pendingObject == nil {
		return nil
	}

	h := s.pendingObject
	n, _, err := s.NextObject(stdioutil.Discard)
	if err != nil {
		return err
	}

	if n != h.Length {
		return fmt.Errorf(
			"error discarding object, discarded %d, expected %d",
			n, h.Length,
		)
	}

	return nil
}

// ReadObjectTypeAndLength reads and returns the object type and the
// length field from an object entry in a packfile.
func (s *Scanner) readObjectTypeAndLength() (plumbing.ObjectType, int64, error) {
	t, c, err := s.readType()
	if err != nil {
		return t, 0, err
	}

	l, err := s.readLength(c)

	return t, l, err
}

func (s *Scanner) readType() (plumbing.ObjectType, byte, error) {
	var c byte
	var err error
	if c, err = s.r.ReadByte(); err != nil {
		return plumbing.ObjectType(0), 0, err
	}

	typ := parseType(c)

	return typ, c, nil
}

func parseType(b byte) plumbing.ObjectType {
	return plumbing.ObjectType((b & maskType) >> firstLengthBits)
}

// the length is codified in the last 4 bits of the first byte and in
// the last 7 bits of subsequent bytes.  Last byte has a 0 MSB.
func (s *Scanner) readLength(first byte) (int64, error) {
	length := int64(first & maskFirstLength)

	c := first
	shift := firstLengthBits
	var err error
	for c&maskContinue > 0 {
		if c, err = s.r.ReadByte(); err != nil {
			return 0, err
		}

		length += int64(c&maskLength) << shift
		shift += lengthBits
	}

	return length, nil
}

// NextObject writes the content of the next object into the reader, returns
// the number of bytes written, the CRC32 of the content and an error, if any
func (s *Scanner) NextObject(w io.Writer) (written int64, crc32 uint32, err error) {
	s.pendingObject = nil
	written, err = s.copyObject(w)

	s.r.Flush()
	crc32 = s.crc.Sum32()
	s.crc.Reset()

	return
}

// ReadRegularObject reads and write a non-deltified object
// from it zlib stream in an object entry in the packfile.
func (s *Scanner) copyObject(w io.Writer) (n int64, err error) {
	zr := zlibReaderPool.Get().(io.ReadCloser)
	defer zlibReaderPool.Put(zr)

	if err = zr.(zlib.Resetter).Reset(s.r, nil); err != nil {
		return 0, fmt.Errorf("zlib reset error: %s", err)
	}

	defer ioutil.CheckClose(zr, &err)
	buf := byteSlicePool.Get().([]byte)
	n, err = io.CopyBuffer(w, zr, buf)
	byteSlicePool.Put(buf)
	return
}

var byteSlicePool = sync.Pool{
	New: func() interface{} {
		return make([]byte, 32*1024)
	},
}

// SeekFromStart sets a new offset from start, returns the old position before
// the change.
func (s *Scanner) SeekFromStart(offset int64) (previous int64, err error) {
	// if seeking we assume that you are not interested in the header
	if s.version == 0 {
		s.version = VersionSupported
	}

	previous, err = s.r.Seek(0, io.SeekCurrent)
	if err != nil {
		return -1, err
	}

	_, err = s.r.Seek(offset, io.SeekStart)
	return previous, err
}

// Checksum returns the checksum of the packfile
func (s *Scanner) Checksum() (plumbing.Hash, error) {
	err := s.discardObjectIfNeeded()
	if err != nil {
		return plumbing.ZeroHash, err
	}

	return binary.ReadHash(s.r)
}

// Close reads the reader until io.EOF
func (s *Scanner) Close() error {
	buf := byteSlicePool.Get().([]byte)
	_, err := io.CopyBuffer(stdioutil.Discard, s.r, buf)
	byteSlicePool.Put(buf)
	return err
}

// Flush is a no-op (deprecated)
func (s *Scanner) Flush() error {
	return nil
}

// scannerReader has the following characteristics:
// - Provides an io.SeekReader impl for bufio.Reader, when the underlying
//   reader supports it.
// - Keeps track of the current read position, for when the underlying reader
//   isn't an io.SeekReader, but we still want to know the current offset.
// - Writes to the hash writer what it reads, with the aid of a smaller buffer.
//   The buffer helps avoid a performance penality for performing small writes
//   to the crc32 hash writer.
type scannerReader struct {
	reader io.Reader
	crc    io.Writer
	rbuf   *bufio.Reader
	wbuf   *bufio.Writer
	offset int64
}

func newScannerReader(r io.Reader, h io.Writer) *scannerReader {
	sr := &scannerReader{
		rbuf: bufio.NewReader(nil),
		wbuf: bufio.NewWriterSize(nil, 64),
		crc:  h,
	}
	sr.Reset(r)

	return sr
}

func (r *scannerReader) Reset(reader io.Reader) {
	r.reader = reader
	r.rbuf.Reset(r.reader)
	r.wbuf.Reset(r.crc)

	r.offset = 0
	if seeker, ok := r.reader.(io.ReadSeeker); ok {
		r.offset, _ = seeker.Seek(0, io.SeekCurrent)
	}
}

func (r *scannerReader) Read(p []byte) (n int, err error) {
	n, err = r.rbuf.Read(p)

	r.offset += int64(n)
	if _, err := r.wbuf.Write(p[:n]); err != nil {
		return n, err
	}
	return
}

func (r *scannerReader) ReadByte() (b byte, err error) {
	b, err = r.rbuf.ReadByte()
	if err == nil {
		r.offset++
		return b, r.wbuf.WriteByte(b)
	}
	return
}

func (r *scannerReader) Flush() error {
	return r.wbuf.Flush()
}

// Seek seeks to a location. If the underlying reader is not an io.ReadSeeker,
// then only whence=io.SeekCurrent is supported, any other operation fails.
func (r *scannerReader) Seek(offset int64, whence int) (int64, error) {
	var err error

	if seeker, ok := r.reader.(io.ReadSeeker); !ok {
		if whence != io.SeekCurrent || offset != 0 {
			return -1, ErrSeekNotSupported
		}
	} else {
		if whence == io.SeekCurrent && offset == 0 {
			return r.offset, nil
		}

		r.offset, err = seeker.Seek(offset, whence)
		r.rbuf.Reset(r.reader)
	}

	return r.offset, err
}
