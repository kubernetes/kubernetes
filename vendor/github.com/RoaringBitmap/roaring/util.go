package roaring

import (
	"math/rand"
	"sort"
)

const (
	arrayDefaultMaxSize        = 4096 // containers with 4096 or fewer integers should be array containers.
	arrayLazyLowerBound        = 1024
	maxCapacity                = 1 << 16
	serialCookieNoRunContainer = 12346 // only arrays and bitmaps
	invalidCardinality         = -1
	serialCookie               = 12347 // runs, arrays, and bitmaps
	noOffsetThreshold          = 4

	// MaxUint32 is the largest uint32 value.
	MaxUint32 = 4294967295

	// MaxRange is One more than the maximum allowed bitmap bit index. For use as an upper
	// bound for ranges.
	MaxRange uint64 = MaxUint32 + 1

	// MaxUint16 is the largest 16 bit unsigned int.
	// This is the largest value an interval16 can store.
	MaxUint16 = 65535

	// Compute wordSizeInBytes, the size of a word in bytes.
	_m              = ^uint64(0)
	_logS           = _m>>8&1 + _m>>16&1 + _m>>32&1
	wordSizeInBytes = 1 << _logS

	// other constants used in ctz_generic.go
	wordSizeInBits = wordSizeInBytes << 3 // word size in bits
)

const maxWord = 1<<wordSizeInBits - 1

// doesn't apply to runContainers
func getSizeInBytesFromCardinality(card int) int {
	if card > arrayDefaultMaxSize {
		// bitmapContainer
		return maxCapacity / 8
	}
	// arrayContainer
	return 2 * card
}

func fill(arr []uint64, val uint64) {
	for i := range arr {
		arr[i] = val
	}
}
func fillRange(arr []uint64, start, end int, val uint64) {
	for i := start; i < end; i++ {
		arr[i] = val
	}
}

func fillArrayAND(container []uint16, bitmap1, bitmap2 []uint64) {
	if len(bitmap1) != len(bitmap2) {
		panic("array lengths don't match")
	}
	// TODO: rewrite in assembly
	pos := 0
	for k := range bitmap1 {
		bitset := bitmap1[k] & bitmap2[k]
		for bitset != 0 {
			t := bitset & -bitset
			container[pos] = uint16((k*64 + int(popcount(t-1))))
			pos = pos + 1
			bitset ^= t
		}
	}
}

func fillArrayANDNOT(container []uint16, bitmap1, bitmap2 []uint64) {
	if len(bitmap1) != len(bitmap2) {
		panic("array lengths don't match")
	}
	// TODO: rewrite in assembly
	pos := 0
	for k := range bitmap1 {
		bitset := bitmap1[k] &^ bitmap2[k]
		for bitset != 0 {
			t := bitset & -bitset
			container[pos] = uint16((k*64 + int(popcount(t-1))))
			pos = pos + 1
			bitset ^= t
		}
	}
}

func fillArrayXOR(container []uint16, bitmap1, bitmap2 []uint64) {
	if len(bitmap1) != len(bitmap2) {
		panic("array lengths don't match")
	}
	// TODO: rewrite in assembly
	pos := 0
	for k := 0; k < len(bitmap1); k++ {
		bitset := bitmap1[k] ^ bitmap2[k]
		for bitset != 0 {
			t := bitset & -bitset
			container[pos] = uint16((k*64 + int(popcount(t-1))))
			pos = pos + 1
			bitset ^= t
		}
	}
}

func highbits(x uint32) uint16 {
	return uint16(x >> 16)
}
func lowbits(x uint32) uint16 {
	return uint16(x & 0xFFFF)
}

const maxLowBit = 0xFFFF

func flipBitmapRange(bitmap []uint64, start int, end int) {
	if start >= end {
		return
	}
	firstword := start / 64
	endword := (end - 1) / 64
	bitmap[firstword] ^= ^(^uint64(0) << uint(start%64))
	for i := firstword; i < endword; i++ {
		bitmap[i] = ^bitmap[i]
	}
	bitmap[endword] ^= ^uint64(0) >> (uint(-end) % 64)
}

func resetBitmapRange(bitmap []uint64, start int, end int) {
	if start >= end {
		return
	}
	firstword := start / 64
	endword := (end - 1) / 64
	if firstword == endword {
		bitmap[firstword] &= ^((^uint64(0) << uint(start%64)) & (^uint64(0) >> (uint(-end) % 64)))
		return
	}
	bitmap[firstword] &= ^(^uint64(0) << uint(start%64))
	for i := firstword + 1; i < endword; i++ {
		bitmap[i] = 0
	}
	bitmap[endword] &= ^(^uint64(0) >> (uint(-end) % 64))

}

func setBitmapRange(bitmap []uint64, start int, end int) {
	if start >= end {
		return
	}
	firstword := start / 64
	endword := (end - 1) / 64
	if firstword == endword {
		bitmap[firstword] |= (^uint64(0) << uint(start%64)) & (^uint64(0) >> (uint(-end) % 64))
		return
	}
	bitmap[firstword] |= ^uint64(0) << uint(start%64)
	for i := firstword + 1; i < endword; i++ {
		bitmap[i] = ^uint64(0)
	}
	bitmap[endword] |= ^uint64(0) >> (uint(-end) % 64)
}

func flipBitmapRangeAndCardinalityChange(bitmap []uint64, start int, end int) int {
	before := wordCardinalityForBitmapRange(bitmap, start, end)
	flipBitmapRange(bitmap, start, end)
	after := wordCardinalityForBitmapRange(bitmap, start, end)
	return int(after - before)
}

func resetBitmapRangeAndCardinalityChange(bitmap []uint64, start int, end int) int {
	before := wordCardinalityForBitmapRange(bitmap, start, end)
	resetBitmapRange(bitmap, start, end)
	after := wordCardinalityForBitmapRange(bitmap, start, end)
	return int(after - before)
}

func setBitmapRangeAndCardinalityChange(bitmap []uint64, start int, end int) int {
	before := wordCardinalityForBitmapRange(bitmap, start, end)
	setBitmapRange(bitmap, start, end)
	after := wordCardinalityForBitmapRange(bitmap, start, end)
	return int(after - before)
}

func wordCardinalityForBitmapRange(bitmap []uint64, start int, end int) uint64 {
	answer := uint64(0)
	if start >= end {
		return answer
	}
	firstword := start / 64
	endword := (end - 1) / 64
	for i := firstword; i <= endword; i++ {
		answer += popcount(bitmap[i])
	}
	return answer
}

func selectBitPosition(w uint64, j int) int {
	seen := 0

	// Divide 64bit
	part := w & 0xFFFFFFFF
	n := popcount(part)
	if n <= uint64(j) {
		part = w >> 32
		seen += 32
		j -= int(n)
	}
	w = part

	// Divide 32bit
	part = w & 0xFFFF
	n = popcount(part)
	if n <= uint64(j) {
		part = w >> 16
		seen += 16
		j -= int(n)
	}
	w = part

	// Divide 16bit
	part = w & 0xFF
	n = popcount(part)
	if n <= uint64(j) {
		part = w >> 8
		seen += 8
		j -= int(n)
	}
	w = part

	// Lookup in final byte
	var counter uint
	for counter = 0; counter < 8; counter++ {
		j -= int((w >> counter) & 1)
		if j < 0 {
			break
		}
	}
	return seen + int(counter)

}

func panicOn(err error) {
	if err != nil {
		panic(err)
	}
}

type ph struct {
	orig int
	rand int
}

type pha []ph

func (p pha) Len() int           { return len(p) }
func (p pha) Less(i, j int) bool { return p[i].rand < p[j].rand }
func (p pha) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

func getRandomPermutation(n int) []int {
	r := make([]ph, n)
	for i := 0; i < n; i++ {
		r[i].orig = i
		r[i].rand = rand.Intn(1 << 29)
	}
	sort.Sort(pha(r))
	m := make([]int, n)
	for i := range m {
		m[i] = r[i].orig
	}
	return m
}

func minOfInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxOfInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func maxOfUint16(a, b uint16) uint16 {
	if a > b {
		return a
	}
	return b
}

func minOfUint16(a, b uint16) uint16 {
	if a < b {
		return a
	}
	return b
}
