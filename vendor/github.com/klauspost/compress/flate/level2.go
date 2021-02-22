package flate

import "fmt"

// fastGen maintains the table for matches,
// and the previous byte block for level 2.
// This is the generic implementation.
type fastEncL2 struct {
	fastGen
	table [bTableSize]tableEntry
}

// EncodeL2 uses a similar algorithm to level 1, but is capable
// of matching across blocks giving better compression at a small slowdown.
func (e *fastEncL2) Encode(dst *tokens, src []byte) {
	const (
		inputMargin            = 12 - 1
		minNonLiteralBlockSize = 1 + 1 + inputMargin
	)

	if debugDeflate && e.cur < 0 {
		panic(fmt.Sprint("e.cur < 0: ", e.cur))
	}

	// Protect against e.cur wraparound.
	for e.cur >= bufferReset {
		if len(e.hist) == 0 {
			for i := range e.table[:] {
				e.table[i] = tableEntry{}
			}
			e.cur = maxMatchOffset
			break
		}
		// Shift down everything in the table that isn't already too far away.
		minOff := e.cur + int32(len(e.hist)) - maxMatchOffset
		for i := range e.table[:] {
			v := e.table[i].offset
			if v <= minOff {
				v = 0
			} else {
				v = v - e.cur + maxMatchOffset
			}
			e.table[i].offset = v
		}
		e.cur = maxMatchOffset
	}

	s := e.addBlock(src)

	// This check isn't in the Snappy implementation, but there, the caller
	// instead of the callee handles this case.
	if len(src) < minNonLiteralBlockSize {
		// We do not fill the token table.
		// This will be picked up by caller.
		dst.n = uint16(len(src))
		return
	}

	// Override src
	src = e.hist
	nextEmit := s

	// sLimit is when to stop looking for offset/length copies. The inputMargin
	// lets us use a fast path for emitLiteral in the main loop, while we are
	// looking for copies.
	sLimit := int32(len(src) - inputMargin)

	// nextEmit is where in src the next emitLiteral should start from.
	cv := load3232(src, s)
	for {
		// When should we start skipping if we haven't found matches in a long while.
		const skipLog = 5
		const doEvery = 2

		nextS := s
		var candidate tableEntry
		for {
			nextHash := hash4u(cv, bTableBits)
			s = nextS
			nextS = s + doEvery + (s-nextEmit)>>skipLog
			if nextS > sLimit {
				goto emitRemainder
			}
			candidate = e.table[nextHash]
			now := load6432(src, nextS)
			e.table[nextHash] = tableEntry{offset: s + e.cur}
			nextHash = hash4u(uint32(now), bTableBits)

			offset := s - (candidate.offset - e.cur)
			if offset < maxMatchOffset && cv == load3232(src, candidate.offset-e.cur) {
				e.table[nextHash] = tableEntry{offset: nextS + e.cur}
				break
			}

			// Do one right away...
			cv = uint32(now)
			s = nextS
			nextS++
			candidate = e.table[nextHash]
			now >>= 8
			e.table[nextHash] = tableEntry{offset: s + e.cur}

			offset = s - (candidate.offset - e.cur)
			if offset < maxMatchOffset && cv == load3232(src, candidate.offset-e.cur) {
				break
			}
			cv = uint32(now)
		}

		// A 4-byte match has been found. We'll later see if more than 4 bytes
		// match. But, prior to the match, src[nextEmit:s] are unmatched. Emit
		// them as literal bytes.

		// Call emitCopy, and then see if another emitCopy could be our next
		// move. Repeat until we find no match for the input immediately after
		// what was consumed by the last emitCopy call.
		//
		// If we exit this loop normally then we need to call emitLiteral next,
		// though we don't yet know how big the literal will be. We handle that
		// by proceeding to the next iteration of the main loop. We also can
		// exit this loop via goto if we get close to exhausting the input.
		for {
			// Invariant: we have a 4-byte match at s, and no need to emit any
			// literal bytes prior to s.

			// Extend the 4-byte match as long as possible.
			t := candidate.offset - e.cur
			l := e.matchlenLong(s+4, t+4, src) + 4

			// Extend backwards
			for t > 0 && s > nextEmit && src[t-1] == src[s-1] {
				s--
				t--
				l++
			}
			if nextEmit < s {
				emitLiteral(dst, src[nextEmit:s])
			}

			dst.AddMatchLong(l, uint32(s-t-baseMatchOffset))
			s += l
			nextEmit = s
			if nextS >= s {
				s = nextS + 1
			}

			if s >= sLimit {
				// Index first pair after match end.
				if int(s+l+4) < len(src) {
					cv := load3232(src, s)
					e.table[hash4u(cv, bTableBits)] = tableEntry{offset: s + e.cur}
				}
				goto emitRemainder
			}

			// Store every second hash in-between, but offset by 1.
			for i := s - l + 2; i < s-5; i += 7 {
				x := load6432(src, int32(i))
				nextHash := hash4u(uint32(x), bTableBits)
				e.table[nextHash] = tableEntry{offset: e.cur + i}
				// Skip one
				x >>= 16
				nextHash = hash4u(uint32(x), bTableBits)
				e.table[nextHash] = tableEntry{offset: e.cur + i + 2}
				// Skip one
				x >>= 16
				nextHash = hash4u(uint32(x), bTableBits)
				e.table[nextHash] = tableEntry{offset: e.cur + i + 4}
			}

			// We could immediately start working at s now, but to improve
			// compression we first update the hash table at s-2 to s. If
			// another emitCopy is not our next move, also calculate nextHash
			// at s+1. At least on GOARCH=amd64, these three hash calculations
			// are faster as one load64 call (with some shifts) instead of
			// three load32 calls.
			x := load6432(src, s-2)
			o := e.cur + s - 2
			prevHash := hash4u(uint32(x), bTableBits)
			prevHash2 := hash4u(uint32(x>>8), bTableBits)
			e.table[prevHash] = tableEntry{offset: o}
			e.table[prevHash2] = tableEntry{offset: o + 1}
			currHash := hash4u(uint32(x>>16), bTableBits)
			candidate = e.table[currHash]
			e.table[currHash] = tableEntry{offset: o + 2}

			offset := s - (candidate.offset - e.cur)
			if offset > maxMatchOffset || uint32(x>>16) != load3232(src, candidate.offset-e.cur) {
				cv = uint32(x >> 24)
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
