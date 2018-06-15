package nuts

// KeyLen returns the minimum number of bytes required to represent x; the result is 1 for x == 0.
// Returns 1-8.
func KeyLen(x uint64) int {
	n := 1
	if x >= 1<<32 {
		x >>= 32
		n += 4
	}
	if x >= 1<<16 {
		x >>= 16
		n += 2
	}
	if x >= 1<<8 {
		x >>= 8
		n += 1
	}
	return n
}

// Key is a byte slice with methods for serializing uint64 (big endian).
// Length can minimized (<8) with KeyLen.
//  make(Key, KeyLen(uint64(max)))
// Large Keys can constructed by slicing.
//  uuid := make(Key, 16)
//  uuid[:8].Put(a)
//  uuid[8:].Put(b)
type Key []byte

// Put serializes x into the buffer (big endian). Behavior is undefined when x
// does not fit, so the caller must ensure c is large enough.
func (c Key) Put(x uint64) {
	s := uint(8 * (len(c) - 1))
	for i := range c {
		c[i] = byte(x >> s)
		s -= 8
	}
}
