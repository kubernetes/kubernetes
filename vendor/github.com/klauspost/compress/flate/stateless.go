package flate

import (
	"io"
	"math"
	"sync"
)

const (
	maxStatelessBlock = math.MaxInt16
	// dictionary will be taken from maxStatelessBlock, so limit it.
	maxStatelessDict = 8 << 10

	slTableBits  = 13
	slTableSize  = 1 << slTableBits
	slTableShift = 32 - slTableBits
)

type statelessWriter struct {
	dst    io.Writer
	closed bool
}

func (s *statelessWriter) Close() error {
	if s.closed {
		return nil
	}
	s.closed = true
	// Emit EOF block
	return StatelessDeflate(s.dst, nil, true, nil)
}

func (s *statelessWriter) Write(p []byte) (n int, err error) {
	err = StatelessDeflate(s.dst, p, false, nil)
	if err != nil {
		return 0, err
	}
	return len(p), nil
}

func (s *statelessWriter) Reset(w io.Writer) {
	s.dst = w
	s.closed = false
}

// NewStatelessWriter will do compression but without maintaining any state
// between Write calls.
// There will be no memory kept between Write calls,
// but compression and speed will be suboptimal.
// Because of this, the size of actual Write calls will affect output size.
func NewStatelessWriter(dst io.Writer) io.WriteCloser {
	return &statelessWriter{dst: dst}
}

// bitWriterPool contains bit writers that can be reused.
var bitWriterPool = sync.Pool{
	New: func() interface{} {
		return newHuffmanBitWriter(nil)
	},
}

// StatelessDeflate allows to compress directly to a Writer without retaining state.
// When returning everything will be flushed.
// Up to 8KB of an optional dictionary can be given which is presumed to presumed to precede the block.
// Longer dictionaries will be truncated and will still produce valid output.
// Sending nil dictionary is perfectly fine.
func StatelessDeflate(out io.Writer, in []byte, eof bool, dict []byte) error {
	var dst tokens
	bw := bitWriterPool.Get().(*huffmanBitWriter)
	bw.reset(out)
	defer func() {
		// don't keep a reference to our output
		bw.reset(nil)
		bitWriterPool.Put(bw)
	}()
	if eof && len(in) == 0 {
		// Just write an EOF block.
		// Could be faster...
		bw.writeStoredHeader(0, true)
		bw.flush()
		return bw.err
	}

	// Truncate dict
	if len(dict) > maxStatelessDict {
		dict = dict[len(dict)-maxStatelessDict:]
	}

	for len(in) > 0 {
		todo := in
		if len(todo) > maxStatelessBlock-len(dict) {
			todo = todo[:maxStatelessBlock-len(dict)]
		}
		in = in[len(todo):]
		uncompressed := todo
		if len(dict) > 0 {
			// combine dict and source
			bufLen := len(todo) + len(dict)
			combined := make([]byte, bufLen)
			copy(combined, dict)
			copy(combined[len(dict):], todo)
			todo = combined
		}
		// Compress
		statelessEnc(&dst, todo, int16(len(dict)))
		isEof := eof && len(in) == 0

		if dst.n == 0 {
			bw.writeStoredHeader(len(uncompressed), isEof)
			if bw.err != nil {
				return bw.err
			}
			bw.writeBytes(uncompressed)
		} else if int(dst.n) > len(uncompressed)-len(uncompressed)>>4 {
			// If we removed less than 1/16th, huffman compress the block.
			bw.writeBlockHuff(isEof, uncompressed, len(in) == 0)
		} else {
			bw.writeBlockDynamic(&dst, isEof, uncompressed, len(in) == 0)
		}
		if len(in) > 0 {
			// Retain a dict if we have more
			dict = todo[len(todo)-maxStatelessDict:]
			dst.Reset()
		}
		if bw.err != nil {
			return bw.err
		}
	}
	if !eof {
		// Align, only a stored block can do that.
		bw.writeStoredHeader(0, false)
	}
	bw.flush()
	return bw.err
}

func hashSL(u uint32) uint32 {
	return (u * 0x1e35a7bd) >> slTableShift
}

func load3216(b []byte, i int16) uint32 {
	// Help the compiler eliminate bounds checks on the read so it can be done in a single read.
	b = b[i:]
	b = b[:4]
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

func load6416(b []byte, i int16) uint64 {
	// Help the compiler eliminate bounds checks on the read so it can be done in a single read.
	b = b[i:]
	b = b[:8]
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
}

func statelessEnc(dst *tokens, src []byte, startAt int16) {
	const (
		inputMargin            = 12 - 1
		minNonLiteralBlockSize = 1 + 1 + inputMargin
	)

	type tableEntry struct {
		offset int16
	}

	var table [slTableSize]tableEntry

	// This check isn't in the Snappy implementation, but there, the caller
	// instead of the callee handles this case.
	if len(src)-int(startAt) < minNonLiteralBlockSize {
		// We do not fill the token table.
		// This will be picked up by caller.
		dst.n = 0
		return
	}
	// Index until startAt
	if startAt > 0 {
		cv := load3232(src, 0)
		for i := int16(0); i < startAt; i++ {
			table[hashSL(cv)] = tableEntry{offset: i}
			cv = (cv >> 8) | (uint32(src[i+4]) << 24)
		}
	}

	s := startAt + 1
	nextEmit := startAt
	// sLimit is when to stop looking for offset/length copies. The inputMargin
	// lets us use a fast path for emitLiteral in the main loop, while we are
	// looking for copies.
	sLimit := int16(len(src) - inputMargin)

	// nextEmit is where in src the next emitLiteral should start from.
	cv := load3216(src, s)

	for {
		const skipLog = 5
		const doEvery = 2

		nextS := s
		var candidate tableEntry
		for {
			nextHash := hashSL(cv)
			candidate = table[nextHash]
			nextS = s + doEvery + (s-nextEmit)>>skipLog
			if nextS > sLimit || nextS <= 0 {
				goto emitRemainder
			}

			now := load6416(src, nextS)
			table[nextHash] = tableEntry{offset: s}
			nextHash = hashSL(uint32(now))

			if cv == load3216(src, candidate.offset) {
				table[nextHash] = tableEntry{offset: nextS}
				break
			}

			// Do one right away...
			cv = uint32(now)
			s = nextS
			nextS++
			candidate = table[nextHash]
			now >>= 8
			table[nextHash] = tableEntry{offset: s}

			if cv == load3216(src, candidate.offset) {
				table[nextHash] = tableEntry{offset: nextS}
				break
			}
			cv = uint32(now)
			s = nextS
		}

		// A 4-byte match has been found. We'll later see if more than 4 bytes
		// match. But, prior to the match, src[nextEmit:s] are unmatched. Emit
		// them as literal bytes.
		for {
			// Invariant: we have a 4-byte match at s, and no need to emit any
			// literal bytes prior to s.

			// Extend the 4-byte match as long as possible.
			t := candidate.offset
			l := int16(matchLen(src[s+4:], src[t+4:]) + 4)

			// Extend backwards
			for t > 0 && s > nextEmit && src[t-1] == src[s-1] {
				s--
				t--
				l++
			}
			if nextEmit < s {
				emitLiteral(dst, src[nextEmit:s])
			}

			// Save the match found
			dst.AddMatchLong(int32(l), uint32(s-t-baseMatchOffset))
			s += l
			nextEmit = s
			if nextS >= s {
				s = nextS + 1
			}
			if s >= sLimit {
				goto emitRemainder
			}

			// We could immediately start working at s now, but to improve
			// compression we first update the hash table at s-2 and at s. If
			// another emitCopy is not our next move, also calculate nextHash
			// at s+1. At least on GOARCH=amd64, these three hash calculations
			// are faster as one load64 call (with some shifts) instead of
			// three load32 calls.
			x := load6416(src, s-2)
			o := s - 2
			prevHash := hashSL(uint32(x))
			table[prevHash] = tableEntry{offset: o}
			x >>= 16
			currHash := hashSL(uint32(x))
			candidate = table[currHash]
			table[currHash] = tableEntry{offset: o + 2}

			if uint32(x) != load3216(src, candidate.offset) {
				cv = uint32(x >> 8)
				s++
				break
			}
		}
	}

emitRemainder:
	if int(nextEmit) < len(src) {
		// If nothing was added, don't encode literals.
		if dst.n == 0 {
			return
		}
		emitLiteral(dst, src[nextEmit:])
	}
}
