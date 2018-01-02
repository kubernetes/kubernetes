package wim

import (
	"encoding/binary"
	"io"
	"io/ioutil"

	"github.com/Microsoft/go-winio/wim/lzx"
)

const chunkSize = 32768 // Compressed resource chunk size

type compressedReader struct {
	r            *io.SectionReader
	d            io.ReadCloser
	chunks       []int64
	curChunk     int
	originalSize int64
}

func newCompressedReader(r *io.SectionReader, originalSize int64, offset int64) (*compressedReader, error) {
	nchunks := (originalSize + chunkSize - 1) / chunkSize
	var base int64
	chunks := make([]int64, nchunks)
	if originalSize <= 0xffffffff {
		// 32-bit chunk offsets
		base = (nchunks - 1) * 4
		chunks32 := make([]uint32, nchunks-1)
		err := binary.Read(r, binary.LittleEndian, chunks32)
		if err != nil {
			return nil, err
		}
		for i, n := range chunks32 {
			chunks[i+1] = int64(n)
		}

	} else {
		// 64-bit chunk offsets
		base = (nchunks - 1) * 8
		err := binary.Read(r, binary.LittleEndian, chunks[1:])
		if err != nil {
			return nil, err
		}
	}

	for i, c := range chunks {
		chunks[i] = c + base
	}

	cr := &compressedReader{
		r:            r,
		chunks:       chunks,
		originalSize: originalSize,
	}

	err := cr.reset(int(offset / chunkSize))
	if err != nil {
		return nil, err
	}

	suboff := offset % chunkSize
	if suboff != 0 {
		_, err := io.CopyN(ioutil.Discard, cr.d, suboff)
		if err != nil {
			return nil, err
		}
	}
	return cr, nil
}

func (r *compressedReader) chunkOffset(n int) int64 {
	if n == len(r.chunks) {
		return r.r.Size()
	}
	return r.chunks[n]
}

func (r *compressedReader) chunkSize(n int) int {
	return int(r.chunkOffset(n+1) - r.chunkOffset(n))
}

func (r *compressedReader) uncompressedSize(n int) int {
	if n < len(r.chunks)-1 {
		return chunkSize
	}
	size := int(r.originalSize % chunkSize)
	if size == 0 {
		size = chunkSize
	}
	return size
}

func (r *compressedReader) reset(n int) error {
	if n >= len(r.chunks) {
		return io.EOF
	}
	if r.d != nil {
		r.d.Close()
	}
	r.curChunk = n
	size := r.chunkSize(n)
	uncompressedSize := r.uncompressedSize(n)
	section := io.NewSectionReader(r.r, r.chunkOffset(n), int64(size))
	if size != uncompressedSize {
		d, err := lzx.NewReader(section, uncompressedSize)
		if err != nil {
			return err
		}
		r.d = d
	} else {
		r.d = ioutil.NopCloser(section)
	}

	return nil
}

func (r *compressedReader) Read(b []byte) (int, error) {
	for {
		n, err := r.d.Read(b)
		if err != io.EOF {
			return n, err
		}

		err = r.reset(r.curChunk + 1)
		if err != nil {
			return n, err
		}
	}
}

func (r *compressedReader) Close() error {
	var err error
	if r.d != nil {
		err = r.d.Close()
		r.d = nil
	}
	return err
}
