package msgp

import (
	"math"
)

// Locate returns a []byte pointing to the field
// in a messagepack map with the provided key. (The returned []byte
// points to a sub-slice of 'raw'; Locate does no allocations.) If the
// key doesn't exist in the map, a zero-length []byte will be returned.
func Locate(key string, raw []byte) []byte {
	s, n := locate(raw, key)
	return raw[s:n]
}

// Replace takes a key ("key") in a messagepack map ("raw")
// and replaces its value with the one provided and returns
// the new []byte. The returned []byte may point to the same
// memory as "raw". Replace makes no effort to evaluate the validity
// of the contents of 'val'. It may use up to the full capacity of 'raw.'
// Replace returns 'nil' if the field doesn't exist or if the object in 'raw'
// is not a map.
func Replace(key string, raw []byte, val []byte) []byte {
	start, end := locate(raw, key)
	if start == end {
		return nil
	}
	return replace(raw, start, end, val, true)
}

// CopyReplace works similarly to Replace except that the returned
// byte slice does not point to the same memory as 'raw'. CopyReplace
// returns 'nil' if the field doesn't exist or 'raw' isn't a map.
func CopyReplace(key string, raw []byte, val []byte) []byte {
	start, end := locate(raw, key)
	if start == end {
		return nil
	}
	return replace(raw, start, end, val, false)
}

// Remove removes a key-value pair from 'raw'. It returns
// 'raw' unchanged if the key didn't exist.
func Remove(key string, raw []byte) []byte {
	start, end := locateKV(raw, key)
	if start == end {
		return raw
	}
	raw = raw[:start+copy(raw[start:], raw[end:])]
	return resizeMap(raw, -1)
}

// HasKey returns whether the map in 'raw' has
// a field with key 'key'
func HasKey(key string, raw []byte) bool {
	sz, bts, err := ReadMapHeaderBytes(raw)
	if err != nil {
		return false
	}
	var field []byte
	for i := uint32(0); i < sz; i++ {
		field, bts, err = ReadStringZC(bts)
		if err != nil {
			return false
		}
		if UnsafeString(field) == key {
			return true
		}
	}
	return false
}

func replace(raw []byte, start int, end int, val []byte, inplace bool) []byte {
	ll := end - start // length of segment to replace
	lv := len(val)

	if inplace {
		extra := lv - ll

		// fastest case: we're doing
		// a 1:1 replacement
		if extra == 0 {
			copy(raw[start:], val)
			return raw

		} else if extra < 0 {
			// 'val' smaller than replaced value
			// copy in place and shift back

			x := copy(raw[start:], val)
			y := copy(raw[start+x:], raw[end:])
			return raw[:start+x+y]

		} else if extra < cap(raw)-len(raw) {
			// 'val' less than (cap-len) extra bytes
			// copy in place and shift forward
			raw = raw[0 : len(raw)+extra]
			// shift end forward
			copy(raw[end+extra:], raw[end:])
			copy(raw[start:], val)
			return raw
		}
	}

	// we have to allocate new space
	out := make([]byte, len(raw)+len(val)-ll)
	x := copy(out, raw[:start])
	y := copy(out[x:], val)
	copy(out[x+y:], raw[end:])
	return out
}

// locate does a naive O(n) search for the map key; returns start, end
// (returns 0,0 on error)
func locate(raw []byte, key string) (start int, end int) {
	var (
		sz    uint32
		bts   []byte
		field []byte
		err   error
	)
	sz, bts, err = ReadMapHeaderBytes(raw)
	if err != nil {
		return
	}

	// loop and locate field
	for i := uint32(0); i < sz; i++ {
		field, bts, err = ReadStringZC(bts)
		if err != nil {
			return 0, 0
		}
		if UnsafeString(field) == key {
			// start location
			l := len(raw)
			start = l - len(bts)
			bts, err = Skip(bts)
			if err != nil {
				return 0, 0
			}
			end = l - len(bts)
			return
		}
		bts, err = Skip(bts)
		if err != nil {
			return 0, 0
		}
	}
	return 0, 0
}

// locate key AND value
func locateKV(raw []byte, key string) (start int, end int) {
	var (
		sz    uint32
		bts   []byte
		field []byte
		err   error
	)
	sz, bts, err = ReadMapHeaderBytes(raw)
	if err != nil {
		return 0, 0
	}

	for i := uint32(0); i < sz; i++ {
		tmp := len(bts)
		field, bts, err = ReadStringZC(bts)
		if err != nil {
			return 0, 0
		}
		if UnsafeString(field) == key {
			start = len(raw) - tmp
			bts, err = Skip(bts)
			if err != nil {
				return 0, 0
			}
			end = len(raw) - len(bts)
			return
		}
		bts, err = Skip(bts)
		if err != nil {
			return 0, 0
		}
	}
	return 0, 0
}

// delta is delta on map size
func resizeMap(raw []byte, delta int64) []byte {
	var sz int64
	switch raw[0] {
	case mmap16:
		sz = int64(big.Uint16(raw[1:]))
		if sz+delta <= math.MaxUint16 {
			big.PutUint16(raw[1:], uint16(sz+delta))
			return raw
		}
		if cap(raw)-len(raw) >= 2 {
			raw = raw[0 : len(raw)+2]
			copy(raw[5:], raw[3:])
			raw[0] = mmap32
			big.PutUint32(raw[1:], uint32(sz+delta))
			return raw
		}
		n := make([]byte, 0, len(raw)+5)
		n = AppendMapHeader(n, uint32(sz+delta))
		return append(n, raw[3:]...)

	case mmap32:
		sz = int64(big.Uint32(raw[1:]))
		big.PutUint32(raw[1:], uint32(sz+delta))
		return raw

	default:
		sz = int64(rfixmap(raw[0]))
		if sz+delta < 16 {
			raw[0] = wfixmap(uint8(sz + delta))
			return raw
		} else if sz+delta <= math.MaxUint16 {
			if cap(raw)-len(raw) >= 2 {
				raw = raw[0 : len(raw)+2]
				copy(raw[3:], raw[1:])
				raw[0] = mmap16
				big.PutUint16(raw[1:], uint16(sz+delta))
				return raw
			}
			n := make([]byte, 0, len(raw)+5)
			n = AppendMapHeader(n, uint32(sz+delta))
			return append(n, raw[1:]...)
		}
		if cap(raw)-len(raw) >= 4 {
			raw = raw[0 : len(raw)+4]
			copy(raw[5:], raw[1:])
			raw[0] = mmap32
			big.PutUint32(raw[1:], uint32(sz+delta))
			return raw
		}
		n := make([]byte, 0, len(raw)+5)
		n = AppendMapHeader(n, uint32(sz+delta))
		return append(n, raw[1:]...)
	}
}
