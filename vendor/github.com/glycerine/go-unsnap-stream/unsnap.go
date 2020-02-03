package unsnap

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strings"

	"hash/crc32"

	snappy "github.com/golang/snappy"
	// The C library can be used, but this makes the binary dependent
	// lots of extraneous c-libraries; it is no longer stand-alone. Yuck.
	//
	// Therefore we comment out the "dgryski/go-csnappy" path and use the
	// "github.com/golang/snappy/snappy" above instead. If you are
	// performance limited and can deal with distributing more libraries,
	// then this is easy to swap.
	//
	// If you swap, note that some of the tests won't pass
	// because snappy-go produces slightly different (but still
	// conformant) encodings on some data. Here are bindings
	// to the C-snappy:
	// snappy "github.com/dgryski/go-csnappy"
)

// SnappyFile: create a drop-in-replacement/wrapper for an *os.File that handles doing the unsnappification online as more is read from it

type SnappyFile struct {
	Fname string

	Reader io.Reader
	Writer io.Writer

	// allow clients to substitute us for an os.File and just switch
	// off compression if they don't want it.
	SnappyEncodeDecodeOff bool // if true, we bypass straight to Filep

	EncBuf FixedSizeRingBuf // holds any extra that isn't yet returned, encoded
	DecBuf FixedSizeRingBuf // holds any extra that isn't yet returned, decoded

	// for writing to stream-framed snappy
	HeaderChunkWritten bool

	// Sanity check: we can only read, or only write, to one SnappyFile.
	// EncBuf and DecBuf are used differently in each mode. Verify
	// that we are consistent with this flag.
	Writing bool
}

var total int

// for debugging, show state of buffers
func (f *SnappyFile) Dump() {
	fmt.Printf("EncBuf has length %d and contents:\n%s\n", len(f.EncBuf.Bytes()), string(f.EncBuf.Bytes()))
	fmt.Printf("DecBuf has length %d and contents:\n%s\n", len(f.DecBuf.Bytes()), string(f.DecBuf.Bytes()))
}

func (f *SnappyFile) Read(p []byte) (n int, err error) {

	if f.SnappyEncodeDecodeOff {
		return f.Reader.Read(p)
	}

	if f.Writing {
		panic("Reading on a write-only SnappyFile")
	}

	// before we unencrypt more, try to drain the DecBuf first
	n, _ = f.DecBuf.Read(p)
	if n > 0 {
		total += n
		return n, nil
	}

	//nEncRead, nDecAdded, err := UnsnapOneFrame(f.Filep, &f.EncBuf, &f.DecBuf, f.Fname)
	_, _, err = UnsnapOneFrame(f.Reader, &f.EncBuf, &f.DecBuf, f.Fname)
	if err != nil && err != io.EOF {
		panic(err)
	}

	n, _ = f.DecBuf.Read(p)

	if n > 0 {
		total += n
		return n, nil
	}
	if f.DecBuf.Readable == 0 {
		if f.DecBuf.Readable == 0 && f.EncBuf.Readable == 0 {
			// only now (when EncBuf is empty) can we give io.EOF.
			// Any earlier, and we leave stuff un-decoded!
			return 0, io.EOF
		}
	}
	return 0, nil
}

func Open(name string) (file *SnappyFile, err error) {
	fp, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	// encoding in snappy can apparently go beyond the original size, so
	// we make our buffers big enough, 2*max snappy chunk => 2 * CHUNK_MAX(65536)

	snap := NewReader(fp)
	snap.Fname = name
	return snap, nil
}

func NewReader(r io.Reader) *SnappyFile {
	return &SnappyFile{
		Reader:  r,
		EncBuf:  *NewFixedSizeRingBuf(CHUNK_MAX * 2), // buffer of snappy encoded bytes
		DecBuf:  *NewFixedSizeRingBuf(CHUNK_MAX * 2), // buffer of snapppy decoded bytes
		Writing: false,
	}
}

func NewWriter(w io.Writer) *SnappyFile {
	return &SnappyFile{
		Writer:  w,
		EncBuf:  *NewFixedSizeRingBuf(65536),     // on writing: temp for testing compression
		DecBuf:  *NewFixedSizeRingBuf(65536 * 2), // on writing: final buffer of snappy framed and encoded bytes
		Writing: true,
	}
}

func Create(name string) (file *SnappyFile, err error) {
	fp, err := os.Create(name)
	if err != nil {
		return nil, err
	}
	snap := NewWriter(fp)
	snap.Fname = name
	return snap, nil
}

func (f *SnappyFile) Close() error {
	if f.Writing {
		wc, ok := f.Writer.(io.WriteCloser)
		if ok {
			return wc.Close()
		}
		return nil
	}
	rc, ok := f.Reader.(io.ReadCloser)
	if ok {
		return rc.Close()
	}
	return nil
}

func (f *SnappyFile) Sync() error {
	file, ok := f.Writer.(*os.File)
	if ok {
		return file.Sync()
	}
	return nil
}

// for an increment of a frame at a time:
// read from r into encBuf (encBuf is still encoded, thus the name), and write unsnappified frames into outDecodedBuf
//  the returned n: number of bytes read from the encrypted encBuf
func UnsnapOneFrame(r io.Reader, encBuf *FixedSizeRingBuf, outDecodedBuf *FixedSizeRingBuf, fname string) (nEnc int64, nDec int64, err error) {
	//	b, err := ioutil.ReadAll(r)
	//	if err != nil {
	//		panic(err)
	//	}

	nEnc = 0
	nDec = 0

	// read up to 65536 bytes from r into encBuf, at least a snappy frame
	nread, err := io.CopyN(encBuf, r, 65536) // returns nwrotebytes, err
	nEnc += nread
	if err != nil {
		if err == io.EOF {
			if nread == 0 {
				if encBuf.Readable == 0 {
					return nEnc, nDec, io.EOF
				}
				// else we have bytes in encBuf, so decode them!
				err = nil
			} else {
				// continue below, processing the nread bytes
				err = nil
			}
		} else {
			// may be an odd already closed... don't panic on that
			if strings.Contains(err.Error(), "file already closed") {
				err = nil
			} else {
				panic(err)
			}
		}
	}

	// flag for printing chunk size alignment messages
	verbose := false

	const snappyStreamHeaderSz = 10
	const headerSz = 4
	const crc32Sz = 4
	// the magic 18 bytes accounts for the snappy streaming header and the first chunks size and checksum
	// http://code.google.com/p/snappy/source/browse/trunk/framing_format.txt

	chunk := (*encBuf).Bytes()

	// however we exit, advance as
	//	defer func() { (*encBuf).Next(N) }()

	// 65536 is the max size of a snappy framed chunk. See
	// http://code.google.com/p/snappy/source/browse/trunk/framing_format.txt:91
	// buf := make([]byte, 65536)

	//	fmt.Printf("read from file, b is len:%d with value: %#v\n", len(b), b)
	//	fmt.Printf("read from file, bcut is len:%d with value: %#v\n", len(bcut), bcut)

	//fmt.Printf("raw bytes of chunksz are: %v\n", b[11:14])

	fourbytes := make([]byte, 4)
	chunkCount := 0

	for nDec < 65536 {
		if len(chunk) == 0 {
			break
		}
		chunkCount++
		fourbytes[3] = 0
		copy(fourbytes, chunk[1:4])
		chunksz := binary.LittleEndian.Uint32(fourbytes)
		chunk_type := chunk[0]

		switch true {
		case chunk_type == 0xff:
			{ // stream identifier

				streamHeader := chunk[:snappyStreamHeaderSz]
				if 0 != bytes.Compare(streamHeader, []byte{0xff, 0x06, 0x00, 0x00, 0x73, 0x4e, 0x61, 0x50, 0x70, 0x59}) {
					panic("file had chunk starting with 0xff but then no magic snappy streaming protocol bytes, aborting.")
				} else {
					//fmt.Printf("got streaming snappy magic header just fine.\n")
				}
				chunk = chunk[snappyStreamHeaderSz:]
				(*encBuf).Advance(snappyStreamHeaderSz)
				nEnc += snappyStreamHeaderSz
				continue
			}
		case chunk_type == 0x00:
			{ // compressed data
				if verbose {
					fmt.Fprintf(os.Stderr, "chunksz is %d  while  total bytes avail are: %d\n", int(chunksz), len(chunk)-4)
				}

				crc := binary.LittleEndian.Uint32(chunk[headerSz:(headerSz + crc32Sz)])
				section := chunk[(headerSz + crc32Sz):(headerSz + chunksz)]

				dec, ok := snappy.Decode(nil, section)
				if ok != nil {
					// we've probably truncated a snappy frame at this point
					// ok=snappy: corrupt input
					// len(dec) == 0
					//
					panic(fmt.Sprintf("could not decode snappy stream: '%s' and len dec=%d and ok=%v\n", fname, len(dec), ok))

					// get back to caller with what we've got so far
					return nEnc, nDec, nil
				}
				//	fmt.Printf("ok, b is %#v , %#v\n", ok, dec)

				// spit out decoded text
				// n, err := w.Write(dec)
				//fmt.Printf("len(dec) = %d,   outDecodedBuf.Readable=%d\n", len(dec), outDecodedBuf.Readable)
				bnb := bytes.NewBuffer(dec)
				n, err := io.Copy(outDecodedBuf, bnb)
				if err != nil {
					//fmt.Printf("got n=%d, err= %s ; when trying to io.Copy(outDecodedBuf: N=%d, Readable=%d)\n", n, err, outDecodedBuf.N, outDecodedBuf.Readable)
					panic(err)
				}
				if n != int64(len(dec)) {
					panic("could not write all bytes to outDecodedBuf")
				}
				nDec += n

				// verify the crc32 rotated checksum
				m32 := masked_crc32c(dec)
				if m32 != crc {
					panic(fmt.Sprintf("crc32 masked failiure. expected: %v but got: %v", crc, m32))
				} else {
					//fmt.Printf("\nchecksums match: %v == %v\n", crc, m32)
				}

				// move to next header
				inc := (headerSz + int(chunksz))
				chunk = chunk[inc:]
				(*encBuf).Advance(inc)
				nEnc += int64(inc)
				continue
			}
		case chunk_type == 0x01:
			{ // uncompressed data

				//n, err := w.Write(chunk[(headerSz+crc32Sz):(headerSz + int(chunksz))])
				n, err := io.Copy(outDecodedBuf, bytes.NewBuffer(chunk[(headerSz+crc32Sz):(headerSz+int(chunksz))]))
				if verbose {
					//fmt.Printf("debug: n=%d  err=%v  chunksz=%d  outDecodedBuf='%v'\n", n, err, chunksz, outDecodedBuf)
				}
				if err != nil {
					panic(err)
				}
				if n != int64(chunksz-crc32Sz) {
					panic("could not write all bytes to stdout")
				}
				nDec += n

				inc := (headerSz + int(chunksz))
				chunk = chunk[inc:]
				(*encBuf).Advance(inc)
				nEnc += int64(inc)
				continue
			}
		case chunk_type == 0xfe:
			fallthrough // padding, just skip it
		case chunk_type >= 0x80 && chunk_type <= 0xfd:
			{ //  Reserved skippable chunks
				//fmt.Printf("\nin reserved skippable chunks, at nEnc=%v\n", nEnc)
				inc := (headerSz + int(chunksz))
				chunk = chunk[inc:]
				nEnc += int64(inc)
				(*encBuf).Advance(inc)
				continue
			}

		default:
			panic(fmt.Sprintf("unrecognized/unsupported chunk type %#v", chunk_type))
		}

	} // end for{}

	return nEnc, nDec, err
	//return int64(N), nil
}

// for whole file at once:
//
// receive on stdin a stream of bytes in the snappy-streaming framed
//  format, defined here: http://code.google.com/p/snappy/source/browse/trunk/framing_format.txt
// Grab each frame, run it through the snappy decoder, and spit out
//  each frame all joined back-to-back on stdout.
//
func Unsnappy(r io.Reader, w io.Writer) (err error) {
	b, err := ioutil.ReadAll(r)
	if err != nil {
		panic(err)
	}

	// flag for printing chunk size alignment messages
	verbose := false

	const snappyStreamHeaderSz = 10
	const headerSz = 4
	const crc32Sz = 4
	// the magic 18 bytes accounts for the snappy streaming header and the first chunks size and checksum
	// http://code.google.com/p/snappy/source/browse/trunk/framing_format.txt

	chunk := b[:]

	// 65536 is the max size of a snappy framed chunk. See
	// http://code.google.com/p/snappy/source/browse/trunk/framing_format.txt:91
	//buf := make([]byte, 65536)

	//	fmt.Printf("read from file, b is len:%d with value: %#v\n", len(b), b)
	//	fmt.Printf("read from file, bcut is len:%d with value: %#v\n", len(bcut), bcut)

	//fmt.Printf("raw bytes of chunksz are: %v\n", b[11:14])

	fourbytes := make([]byte, 4)
	chunkCount := 0

	for {
		if len(chunk) == 0 {
			break
		}
		chunkCount++
		fourbytes[3] = 0
		copy(fourbytes, chunk[1:4])
		chunksz := binary.LittleEndian.Uint32(fourbytes)
		chunk_type := chunk[0]

		switch true {
		case chunk_type == 0xff:
			{ // stream identifier

				streamHeader := chunk[:snappyStreamHeaderSz]
				if 0 != bytes.Compare(streamHeader, []byte{0xff, 0x06, 0x00, 0x00, 0x73, 0x4e, 0x61, 0x50, 0x70, 0x59}) {
					panic("file had chunk starting with 0xff but then no magic snappy streaming protocol bytes, aborting.")
				} else {
					//fmt.Printf("got streaming snappy magic header just fine.\n")
				}
				chunk = chunk[snappyStreamHeaderSz:]
				continue
			}
		case chunk_type == 0x00:
			{ // compressed data
				if verbose {
					fmt.Fprintf(os.Stderr, "chunksz is %d  while  total bytes avail are: %d\n", int(chunksz), len(chunk)-4)
				}

				//crc := binary.LittleEndian.Uint32(chunk[headerSz:(headerSz + crc32Sz)])
				section := chunk[(headerSz + crc32Sz):(headerSz + chunksz)]

				dec, ok := snappy.Decode(nil, section)
				if ok != nil {
					panic("could not decode snappy stream")
				}
				//	fmt.Printf("ok, b is %#v , %#v\n", ok, dec)

				// spit out decoded text
				n, err := w.Write(dec)
				if err != nil {
					panic(err)
				}
				if n != len(dec) {
					panic("could not write all bytes to stdout")
				}

				// TODO: verify the crc32 rotated checksum?

				// move to next header
				chunk = chunk[(headerSz + int(chunksz)):]
				continue
			}
		case chunk_type == 0x01:
			{ // uncompressed data

				//crc := binary.LittleEndian.Uint32(chunk[headerSz:(headerSz + crc32Sz)])
				section := chunk[(headerSz + crc32Sz):(headerSz + chunksz)]

				n, err := w.Write(section)
				if err != nil {
					panic(err)
				}
				if n != int(chunksz-crc32Sz) {
					panic("could not write all bytes to stdout")
				}

				chunk = chunk[(headerSz + int(chunksz)):]
				continue
			}
		case chunk_type == 0xfe:
			fallthrough // padding, just skip it
		case chunk_type >= 0x80 && chunk_type <= 0xfd:
			{ //  Reserved skippable chunks
				chunk = chunk[(headerSz + int(chunksz)):]
				continue
			}

		default:
			panic(fmt.Sprintf("unrecognized/unsupported chunk type %#v", chunk_type))
		}

	} // end for{}

	return nil
}

// 0xff 0x06 0x00 0x00 sNaPpY
var SnappyStreamHeaderMagic = []byte{0xff, 0x06, 0x00, 0x00, 0x73, 0x4e, 0x61, 0x50, 0x70, 0x59}

const CHUNK_MAX = 65536
const _STREAM_TO_STREAM_BLOCK_SIZE = CHUNK_MAX
const _STREAM_IDENTIFIER = `sNaPpY`
const _COMPRESSED_CHUNK = 0x00
const _UNCOMPRESSED_CHUNK = 0x01
const _IDENTIFIER_CHUNK = 0xff
const _RESERVED_UNSKIPPABLE0 = 0x02 // chunk ranges are [inclusive, exclusive)
const _RESERVED_UNSKIPPABLE1 = 0x80
const _RESERVED_SKIPPABLE0 = 0x80
const _RESERVED_SKIPPABLE1 = 0xff

// the minimum percent of bytes compression must save to be enabled in automatic
// mode
const _COMPRESSION_THRESHOLD = .125

var crctab *crc32.Table

func init() {
	crctab = crc32.MakeTable(crc32.Castagnoli) // this is correct table, matches the crc32c.c code used by python
}

func masked_crc32c(data []byte) uint32 {

	// see the framing format specification, http://code.google.com/p/snappy/source/browse/trunk/framing_format.txt
	var crc uint32 = crc32.Checksum(data, crctab)
	return (uint32((crc>>15)|(crc<<17)) + 0xa282ead8)
}

func ReadSnappyStreamCompressedFile(filename string) ([]byte, error) {

	snappyFile, err := Open(filename)
	if err != nil {
		return []byte{}, err
	}

	var bb bytes.Buffer
	_, err = bb.ReadFrom(snappyFile)
	if err == io.EOF {
		err = nil
	}
	if err != nil {
		panic(err)
	}

	return bb.Bytes(), err
}
