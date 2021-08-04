package flate

import "fmt"

type fastEncL6 struct {
	fastGen
	table  [tableSize]tableEntry
	bTable [tableSize]tableEntryPrev
}

func (e *fastEncL6) Encode(dst *tokens, src []byte) {
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
			for i := range e.bTable[:] {
				e.bTable[i] = tableEntryPrev{}
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
		for i := range e.bTable[:] {
			v := e.bTable[i]
			if v.Cur.offset <= minOff {
				v.Cur.offset = 0
				v.Prev.offset = 0
			} else {
				v.Cur.offset = v.Cur.offset - e.cur + maxMatchOffset
				if v.Prev.offset <= minOff {
					v.Prev.offset = 0
				} else {
					v.Prev.offset = v.Prev.offset - e.cur + maxMatchOffset
				}
			}
			e.bTable[i] = v
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
	cv := load6432(src, s)
	// Repeat MUST be > 1 and within range
	repeat := int32(1)
	for {
		const skipLog = 7
		const doEvery = 1

		nextS := s
		var l int32
		var t int32
		for {
			nextHashS := hash4x64(cv, tableBits)
			nextHashL := hash7(cv, tableBits)
			s = nextS
			nextS = s + doEvery + (s-nextEmit)>>skipLog
			if nextS > sLimit {
				goto emitRemainder
			}
			// Fetch a short+long candidate
			sCandidate := e.table[nextHashS]
			lCandidate := e.bTable[nextHashL]
			next := load6432(src, nextS)
			entry := tableEntry{offset: s + e.cur}
			e.table[nextHashS] = entry
			eLong := &e.bTable[nextHashL]
			eLong.Cur, eLong.Prev = entry, eLong.Cur

			// Calculate hashes of 'next'
			nextHashS = hash4x64(next, tableBits)
			nextHashL = hash7(next, tableBits)

			t = lCandidate.Cur.offset - e.cur
			if s-t < maxMatchOffset {
				if uint32(cv) == load3232(src, lCandidate.Cur.offset-e.cur) {
					// Long candidate matches at least 4 bytes.

					// Store the next match
					e.table[nextHashS] = tableEntry{offset: nextS + e.cur}
					eLong := &e.bTable[nextHashL]
					eLong.Cur, eLong.Prev = tableEntry{offset: nextS + e.cur}, eLong.Cur

					// Check the previous long candidate as well.
					t2 := lCandidate.Prev.offset - e.cur
					if s-t2 < maxMatchOffset && uint32(cv) == load3232(src, lCandidate.Prev.offset-e.cur) {
						l = e.matchlen(s+4, t+4, src) + 4
						ml1 := e.matchlen(s+4, t2+4, src) + 4
						if ml1 > l {
							t = t2
							l = ml1
							break
						}
					}
					break
				}
				// Current value did not match, but check if previous long value does.
				t = lCandidate.Prev.offset - e.cur
				if s-t < maxMatchOffset && uint32(cv) == load3232(src, lCandidate.Prev.offset-e.cur) {
					// Store the next match
					e.table[nextHashS] = tableEntry{offset: nextS + e.cur}
					eLong := &e.bTable[nextHashL]
					eLong.Cur, eLong.Prev = tableEntry{offset: nextS + e.cur}, eLong.Cur
					break
				}
			}

			t = sCandidate.offset - e.cur
			if s-t < maxMatchOffset && uint32(cv) == load3232(src, sCandidate.offset-e.cur) {
				// Found a 4 match...
				l = e.matchlen(s+4, t+4, src) + 4

				// Look up next long candidate (at nextS)
				lCandidate = e.bTable[nextHashL]

				// Store the next match
				e.table[nextHashS] = tableEntry{offset: nextS + e.cur}
				eLong := &e.bTable[nextHashL]
				eLong.Cur, eLong.Prev = tableEntry{offset: nextS + e.cur}, eLong.Cur

				// Check repeat at s + repOff
				const repOff = 1
				t2 := s - repeat + repOff
				if load3232(src, t2) == uint32(cv>>(8*repOff)) {
					ml := e.matchlen(s+4+repOff, t2+4, src) + 4
					if ml > l {
						t = t2
						l = ml
						s += repOff
						// Not worth checking more.
						break
					}
				}

				// If the next long is a candidate, use that...
				t2 = lCandidate.Cur.offset - e.cur
				if nextS-t2 < maxMatchOffset {
					if load3232(src, lCandidate.Cur.offset-e.cur) == uint32(next) {
						ml := e.matchlen(nextS+4, t2+4, src) + 4
						if ml > l {
							t = t2
							s = nextS
							l = ml
							// This is ok, but check previous as well.
						}
					}
					// If the previous long is a candidate, use that...
					t2 = lCandidate.Prev.offset - e.cur
					if nextS-t2 < maxMatchOffset && load3232(src, lCandidate.Prev.offset-e.cur) == uint32(next) {
						ml := e.matchlen(nextS+4, t2+4, src) + 4
						if ml > l {
							t = t2
							s = nextS
							l = ml
							break
						}
					}
				}
				break
			}
			cv = next
		}

		// A 4-byte match has been found. We'll later see if more than 4 bytes
		// match. But, prior to the match, src[nextEmit:s] are unmatched. Emit
		// them as literal bytes.

		// Extend the 4-byte match as long as possible.
		if l == 0 {
			l = e.matchlenLong(s+4, t+4, src) + 4
		} else if l == maxMatchLength {
			l += e.matchlenLong(s+l, t+l, src)
		}

		// Try to locate a better match by checking the end-of-match...
		if sAt := s + l; sAt < sLimit {
			eLong := &e.bTable[hash7(load6432(src, sAt), tableBits)]
			// Test current
			t2 := eLong.Cur.offset - e.cur - l
			off := s - t2
			if off < maxMatchOffset {
				if off > 0 && t2 >= 0 {
					if l2 := e.matchlenLong(s, t2, src); l2 > l {
						t = t2
						l = l2
					}
				}
				// Test next:
				t2 = eLong.Prev.offset - e.cur - l
				off := s - t2
				if off > 0 && off < maxMatchOffset && t2 >= 0 {
					if l2 := e.matchlenLong(s, t2, src); l2 > l {
						t = t2
						l = l2
					}
				}
			}
		}

		// Extend backwards
		for t > 0 && s > nextEmit && src[t-1] == src[s-1] {
			s--
			t--
			l++
		}
		if nextEmit < s {
			emitLiteral(dst, src[nextEmit:s])
		}
		if false {
			if t >= s {
				panic(fmt.Sprintln("s-t", s, t))
			}
			if (s - t) > maxMatchOffset {
				panic(fmt.Sprintln("mmo", s-t))
			}
			if l < baseMatchLength {
				panic("bml")
			}
		}

		dst.AddMatchLong(l, uint32(s-t-baseMatchOffset))
		repeat = s - t
		s += l
		nextEmit = s
		if nextS >= s {
			s = nextS + 1
		}

		if s >= sLimit {
			// Index after match end.
			for i := nextS + 1; i < int32(len(src))-8; i += 2 {
				cv := load6432(src, i)
				e.table[hash4x64(cv, tableBits)] = tableEntry{offset: i + e.cur}
				eLong := &e.bTable[hash7(cv, tableBits)]
				eLong.Cur, eLong.Prev = tableEntry{offset: i + e.cur}, eLong.Cur
			}
			goto emitRemainder
		}

		// Store every long hash in-between and every second short.
		if true {
			for i := nextS + 1; i < s-1; i += 2 {
				cv := load6432(src, i)
				t := tableEntry{offset: i + e.cur}
				t2 := tableEntry{offset: t.offset + 1}
				eLong := &e.bTable[hash7(cv, tableBits)]
				eLong2 := &e.bTable[hash7(cv>>8, tableBits)]
				e.table[hash4x64(cv, tableBits)] = t
				eLong.Cur, eLong.Prev = t, eLong.Cur
				eLong2.Cur, eLong2.Prev = t2, eLong2.Cur
			}
		}

		// We could immediately start working at s now, but to improve
		// compression we first update the hash table at s-1 and at s.
		cv = load6432(src, s)
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
