package unsnap

import (
	"encoding/binary"

	// no c lib dependency
	snappy "github.com/golang/snappy"
	// or, use the C wrapper for speed
	//snappy "github.com/dgryski/go-csnappy"
)

// add Write() method for SnappyFile (see unsnap.go)

// reference for snappy framing/streaming format:
//         http://code.google.com/p/snappy/source/browse/trunk/framing_format.txt
//             ?spec=svn68&r=71

//
// Write writes len(p) bytes from p to the underlying data stream.
// It returns the number of bytes written from p (0 <= n <= len(p)) and
// any error encountered that caused the write to stop early. Write
// must return a non-nil error if it returns n < len(p).
//
func (sf *SnappyFile) Write(p []byte) (n int, err error) {

	if sf.SnappyEncodeDecodeOff {
		return sf.Writer.Write(p)
	}

	if !sf.Writing {
		panic("Writing on a read-only SnappyFile")
	}

	// encoding in snappy can apparently go beyond the original size, beware.
	// so our buffers must be sized 2*max snappy chunk => 2 * CHUNK_MAX(65536)

	sf.DecBuf.Reset()
	sf.EncBuf.Reset()

	if !sf.HeaderChunkWritten {
		sf.HeaderChunkWritten = true
		_, err = sf.Writer.Write(SnappyStreamHeaderMagic)
		if err != nil {
			return
		}
	}
	var chunk []byte
	var chunk_type byte
	var crc uint32

	for len(p) > 0 {

		// chunk points to input p by default, unencoded input.
		chunk = p[:IntMin(len(p), CHUNK_MAX)]
		crc = masked_crc32c(chunk)

		writeme := chunk[:]

		// first write to EncBuf, as a temp, in case we want
		// to discard and send uncompressed instead.
		compressed_chunk := snappy.Encode(sf.EncBuf.GetEndmostWritableSlice(), chunk)

		if len(compressed_chunk) <= int((1-_COMPRESSION_THRESHOLD)*float64(len(chunk))) {
			writeme = compressed_chunk
			chunk_type = _COMPRESSED_CHUNK
		} else {
			// keep writeme pointing at original chunk (uncompressed)
			chunk_type = _UNCOMPRESSED_CHUNK
		}

		const crc32Sz = 4
		var tag32 uint32 = uint32(chunk_type) + (uint32(len(writeme)+crc32Sz) << 8)

		err = binary.Write(sf.Writer, binary.LittleEndian, tag32)
		if err != nil {
			return
		}

		err = binary.Write(sf.Writer, binary.LittleEndian, crc)
		if err != nil {
			return
		}

		_, err = sf.Writer.Write(writeme)
		if err != nil {
			return
		}

		n += len(chunk)
		p = p[len(chunk):]
	}
	return n, nil
}

func IntMin(a int, b int) int {
	if a < b {
		return a
	}
	return b
}
