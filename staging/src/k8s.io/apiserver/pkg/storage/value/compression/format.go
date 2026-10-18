/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package compression compresses the serialized form of an API object on its
// way to etcd, ahead of any encryption at rest. A frame starts with 0x00, which
// no storage serializer emits as a first byte, so the first byte alone tells a
// frame from an unframed value. Reading is therefore unconditional and only
// writing is configurable. See KEP-6264.
package compression

import (
	"bytes"
	"compress/flate"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"runtime"
	"sync"
	"time"
)

// Frame layout. This is the at-rest format and must never change. The 9-byte
// header is laid out the same for every algorithm, so a reader can verify the
// checksum even on a frame whose algorithm it doesn't support.
//
//	offset  size  meaning
//	0       1     0x00, the discriminator
//	1       3     "k8s"
//	4       1     algorithm
//	5       4     CRC32C, little-endian, over every frame byte except these four
//	9       ...   body, defined by the algorithm
//
// The algStored (0x00) body:
//
//	offset  size  meaning
//	9       n     the plaintext, verbatim; n is the frame length minus 9
//
// The algDeflate (0x01) body:
//
//	offset  size  meaning
//	9       w     declared plaintext length n, unsigned LEB128; 1-3 bytes, as n <= MaxPlaintextBytes
//	9+w     ...   raw DEFLATE stream (RFC 1951) that inflates to exactly n bytes and ends
const (
	frameDiscriminator byte = 0x00

	// algStored marks a value kept verbatim inside a frame. That's either a value
	// the policy selected that DEFLATE couldn't shrink, which stays framed so reads
	// don't report it stale, or an escaped value that begins with 0x00.
	algStored  byte = 0x00
	algDeflate byte = 0x01

	crcOffset  = 5
	headerSize = 9

	// DefaultMinSize is the write threshold when an entry sets none. Below it the
	// ratio doesn't repay the CPU spent on every read and write.
	DefaultMinSize = 1024

	// MinConfigurableMinSize is the smallest legal threshold.
	MinConfigurableMinSize = 64

	// MaxPlaintextBytes is the largest value the policy frames and the largest declared
	// length a reader inflates. etcd's default 1.5 MiB limit applies to the whole request,
	// so 64 KiB is kept for the key, the transaction and any encryption envelope, which
	// for KMS v2 can hold 32 KiB of annotations. Anything compressed can then be rewritten
	// uncompressed. An escaped value is framed at any size, since it can't be stored bare.
	MaxPlaintextBytes = 1536*1024 - 64*1024

	deflateLevel = 1

	minInflations = 4
	maxInflations = 384
)

var (
	frameMagic = [4]byte{frameDiscriminator, 'k', '8', 's'}
	crcTable   = crc32.MakeTable(crc32.Castagnoli)
)

var (
	// errChecksum is the only sentinel that means the stored bytes are damaged.
	errChecksum = errors.New("storage compression frame failed its checksum")
	// errFormat means a frame passed its checksum but doesn't parse.
	// Either the writer has a bug, or someone with write access to etcd crafted the
	// bytes and recomputed the checksum. aescbc, KMS v1 and no encryption at all
	// can't detect that, because none of them authenticates the bytes it reads.
	errFormat = errors.New("malformed storage compression frame")
	// errUnsupported means an intact frame uses an algorithm this apiserver doesn't
	// implement, most likely written by a newer apiserver during version skew.
	errUnsupported = errors.New("unsupported storage compression algorithm")
	// errCanceled means the read's context ended while it waited for an inflation slot.
	errCanceled = errors.New("storage compression read gave up waiting to decompress")
)

// readError is the type of every error unframe returns. Its message never
// includes the reason, which only goes to the log: a client or a metric that
// showed it would tell someone tampering with etcd which check their bytes failed.
type readError struct {
	sentinel error
	reason   string
	cause    error
}

func (e *readError) Error() string {
	if e.cause != nil {
		return e.sentinel.Error() + ": " + e.cause.Error()
	}
	return e.sentinel.Error()
}

// Unwrap lets errors.Is match the sentinel and, for a canceled read, the
// context's error too.
func (e *readError) Unwrap() []error {
	if e.cause == nil {
		return []error{e.sentinel}
	}
	return []error{e.sentinel, e.cause}
}

// DataCorruption tells value.IsDataCorruption that only a failed checksum means
// damaged bytes. Unsafe deletion of an intact frame could destroy a value that a
// newer or fixed apiserver can still read.
func (e *readError) DataCorruption() bool { return errors.Is(e.sentinel, errChecksum) }

func rejectFrame(sentinel error, reason string) error {
	return &readError{sentinel: sentinel, reason: reason}
}

type frameKind int

const (
	unframed frameKind = iota
	framedStored
	framedDeflate
)

var (
	writerPool = sync.Pool{
		New: func() any {
			w, err := flate.NewWriter(io.Discard, deflateLevel)
			if err != nil {
				panic(fmt.Sprintf("compression: invalid deflate level %d: %v", deflateLevel, err))
			}
			return w
		},
	}
	readerPool = sync.Pool{
		New: func() any { return flate.NewReader(bytes.NewReader(nil)) },
	}
)

// inflateBudget caps concurrent inflations across the whole process. A watch
// stream can decode several events at once and there can be many streams, so a
// per-stream cap wouldn't bound the total.
var inflateBudget = make(chan struct{}, maxInflightInflations())

func maxInflightInflations() int {
	return min(max(runtime.GOMAXPROCS(0)*2, minInflations), maxInflations)
}

// acquireInflateBudget blocks until a slot frees up or the caller's context
// ends, so a full budget delays a read instead of failing it.
func acquireInflateBudget(ctx context.Context) (release func(), err error) {
	start := time.Now()
	select {
	case inflateBudget <- struct{}{}:
		recordInflationWait(time.Since(start))
		return func() { <-inflateBudget }, nil
	case <-ctx.Done():
		recordInflationWait(time.Since(start))
		return nil, ctx.Err()
	}
}

func isFramed(data []byte) bool {
	return len(data) > 0 && data[0] == frameDiscriminator
}

func checksum(frame []byte) uint32 {
	c := crc32.Update(0, crcTable, frame[:crcOffset])
	return crc32.Update(c, crcTable, frame[headerSize:])
}

// seal writes the frame's checksum into bytes 5-8, in place, and returns the frame.
func seal(frame []byte) []byte {
	binary.LittleEndian.PutUint32(frame[crcOffset:headerSize], checksum(frame))
	return frame
}

func storedFrame(plaintext []byte) []byte {
	out := make([]byte, headerSize, headerSize+len(plaintext))
	copy(out, frameMagic[:])
	out[4] = algStored
	return seal(append(out, plaintext...))
}

// deflateFrame makes no size judgment; the caller decides whether the result
// beats a stored frame.
func deflateFrame(plaintext []byte) ([]byte, error) {
	var hdr [headerSize + binary.MaxVarintLen64]byte
	copy(hdr[:], frameMagic[:])
	hdr[4] = algDeflate
	w := binary.PutUvarint(hdr[headerSize:], uint64(len(plaintext)))

	// The capacity guesses 2:1 plus 64 bytes of DEFLATE block overhead. It's only
	// a starting size: a value that compresses worse, such as a small object or an
	// incompressible one, costs one regrowth and copy.
	buf := bytes.NewBuffer(make([]byte, 0, headerSize+w+len(plaintext)/2+64))
	buf.Write(hdr[:headerSize+w])

	zw := writerPool.Get().(*flate.Writer)
	defer writerPool.Put(zw)
	zw.Reset(buf)
	if _, err := zw.Write(plaintext); err != nil {
		return nil, fmt.Errorf("compression: deflate write failed: %w", err)
	}
	if err := zw.Close(); err != nil {
		return nil, fmt.Errorf("compression: deflate close failed: %w", err)
	}
	return seal(buf.Bytes()), nil
}

// unframe decodes a stored value. A value that doesn't start with 0x00 was
// stored without a frame and comes back as is. When the provider is aescbc,
// KMS v1 or no encryption, anyone who can write to etcd can choose these bytes,
// so unframe checks the declared length before it allocates anything.
func unframe(ctx context.Context, data []byte) ([]byte, frameKind, error) {
	if !isFramed(data) {
		return data, unframed, nil
	}
	if len(data) < headerSize {
		return nil, unframed, rejectFrame(errChecksum, "frame shorter than its header")
	}
	if binary.LittleEndian.Uint32(data[crcOffset:headerSize]) != checksum(data) {
		return nil, unframed, rejectFrame(errChecksum, "checksum mismatch")
	}
	if !bytes.Equal(data[:len(frameMagic)], frameMagic[:]) {
		return nil, unframed, rejectFrame(errFormat, "bad magic")
	}

	switch alg := data[4]; alg {
	case algStored:
		return data[headerSize:], framedStored, nil

	case algDeflate:
		n, w := binary.Uvarint(data[headerSize:])
		if w <= 0 || n == 0 || n > MaxPlaintextBytes {
			return nil, framedDeflate, rejectFrame(errFormat, "invalid declared length")
		}

		release, err := acquireInflateBudget(ctx)
		if err != nil {
			return nil, framedDeflate, &readError{sentinel: errCanceled, cause: err}
		}
		defer release()

		start := time.Now()
		out, err := inflateExact(bytes.NewReader(data[headerSize+w:]), int(n))
		if err != nil {
			return nil, framedDeflate, err
		}
		recordDuration(fromStorage, time.Since(start))
		return out, framedDeflate, nil

	default:
		return nil, unframed, rejectFrame(errUnsupported, fmt.Sprintf("%#x", alg))
	}
}

// inflateExact requires the stream to produce exactly n bytes and then end. It
// reads only one byte past n, so a stream that inflates further costs at most
// one 32 KiB inflater window instead of being drained.
func inflateExact(src io.Reader, n int) ([]byte, error) {
	out := make([]byte, n)
	zr := readerPool.Get().(io.ReadCloser)
	defer readerPool.Put(zr)
	if err := zr.(flate.Resetter).Reset(src, nil); err != nil {
		return nil, rejectFrame(errFormat, "inflater reset failed")
	}
	if _, err := io.ReadFull(zr, out); err != nil {
		if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
			return nil, rejectFrame(errFormat, "stream shorter than declared length")
		}
		return nil, rejectFrame(errFormat, "corrupt deflate stream")
	}
	// io.Copy treats io.EOF as success, so a stream that ends cleanly at n gives no error.
	extra, err := io.Copy(io.Discard, io.LimitReader(zr, 1))
	switch {
	case extra != 0:
		return nil, rejectFrame(errFormat, "stream longer than declared length")
	case err != nil:
		return nil, rejectFrame(errFormat, "stream does not end at its declared length")
	}
	return out, nil
}
