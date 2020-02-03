/*
Package bitset implements bitsets, a mapping
between non-negative integers and boolean values. It should be more
efficient than map[uint] bool.

It provides methods for setting, clearing, flipping, and testing
individual integers.

But it also provides set intersection, union, difference,
complement, and symmetric operations, as well as tests to
check whether any, all, or no bits are set, and querying a
bitset's current length and number of positive bits.

BitSets are expanded to the size of the largest set bit; the
memory allocation is approximately Max bits, where Max is
the largest set bit. BitSets are never shrunk. On creation,
a hint can be given for the number of bits that will be used.

Many of the methods, including Set,Clear, and Flip, return
a BitSet pointer, which allows for chaining.

Example use:

	import "bitset"
	var b BitSet
	b.Set(10).Set(11)
	if b.Test(1000) {
		b.Clear(1000)
	}
	if B.Intersection(bitset.New(100).Set(10)).Count() > 1 {
		fmt.Println("Intersection works.")
	}

As an alternative to BitSets, one should check out the 'big' package,
which provides a (less set-theoretical) view of bitsets.

*/
package bitset

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
)

// the wordSize of a bit set
const wordSize = uint(64)

// log2WordSize is lg(wordSize)
const log2WordSize = uint(6)

// allBits has every bit set
const allBits uint64 = 0xffffffffffffffff

// default binary BigEndian
var binaryOrder binary.ByteOrder = binary.BigEndian

// default json encoding base64.URLEncoding
var base64Encoding = base64.URLEncoding

// Base64StdEncoding Marshal/Unmarshal BitSet with base64.StdEncoding(Default: base64.URLEncoding)
func Base64StdEncoding() { base64Encoding = base64.StdEncoding }

// LittleEndian Marshal/Unmarshal Binary as Little Endian(Default: binary.BigEndian)
func LittleEndian() { binaryOrder = binary.LittleEndian }

// A BitSet is a set of bits. The zero value of a BitSet is an empty set of length 0.
type BitSet struct {
	length uint
	set    []uint64
}

// Error is used to distinguish errors (panics) generated in this package.
type Error string

// safeSet will fixup b.set to be non-nil and return the field value
func (b *BitSet) safeSet() []uint64 {
	if b.set == nil {
		b.set = make([]uint64, wordsNeeded(0))
	}
	return b.set
}

// From is a constructor used to create a BitSet from an array of integers
func From(buf []uint64) *BitSet {
	return &BitSet{uint(len(buf)) * 64, buf}
}

// Bytes returns the bitset as array of integers
func (b *BitSet) Bytes() []uint64 {
	return b.set
}

// wordsNeeded calculates the number of words needed for i bits
func wordsNeeded(i uint) int {
	if i > (Cap() - wordSize + 1) {
		return int(Cap() >> log2WordSize)
	}
	return int((i + (wordSize - 1)) >> log2WordSize)
}

// New creates a new BitSet with a hint that length bits will be required
func New(length uint) (bset *BitSet) {
	defer func() {
		if r := recover(); r != nil {
			bset = &BitSet{
				0,
				make([]uint64, 0),
			}
		}
	}()

	bset = &BitSet{
		length,
		make([]uint64, wordsNeeded(length)),
	}

	return bset
}

// Cap returns the total possible capacity, or number of bits
func Cap() uint {
	return ^uint(0)
}

// Len returns the length of the BitSet in words
func (b *BitSet) Len() uint {
	return b.length
}

// extendSetMaybe adds additional words to incorporate new bits if needed
func (b *BitSet) extendSetMaybe(i uint) {
	if i >= b.length { // if we need more bits, make 'em
		nsize := wordsNeeded(i + 1)
		if b.set == nil {
			b.set = make([]uint64, nsize)
		} else if cap(b.set) >= nsize {
			b.set = b.set[:nsize] // fast resize
		} else if len(b.set) < nsize {
			newset := make([]uint64, nsize, 2*nsize) // increase capacity 2x
			copy(newset, b.set)
			b.set = newset
		}
		b.length = i + 1
	}
}

// Test whether bit i is set.
func (b *BitSet) Test(i uint) bool {
	if i >= b.length {
		return false
	}
	return b.set[i>>log2WordSize]&(1<<(i&(wordSize-1))) != 0
}

// Set bit i to 1
func (b *BitSet) Set(i uint) *BitSet {
	b.extendSetMaybe(i)
	b.set[i>>log2WordSize] |= 1 << (i & (wordSize - 1))
	return b
}

// Clear bit i to 0
func (b *BitSet) Clear(i uint) *BitSet {
	if i >= b.length {
		return b
	}
	b.set[i>>log2WordSize] &^= 1 << (i & (wordSize - 1))
	return b
}

// SetTo sets bit i to value
func (b *BitSet) SetTo(i uint, value bool) *BitSet {
	if value {
		return b.Set(i)
	}
	return b.Clear(i)
}

// Flip bit at i
func (b *BitSet) Flip(i uint) *BitSet {
	if i >= b.length {
		return b.Set(i)
	}
	b.set[i>>log2WordSize] ^= 1 << (i & (wordSize - 1))
	return b
}

// Shrink shrinks BitSet to desired length in bits. It clears all bits > length
// and reduces the size and length of the set.
//
// A new slice is allocated to store the new bits, so you may see an increase in
// memory usage until the GC runs. Normally this should not be a problem, but if you
// have an extremely large BitSet its important to understand that the old BitSet will
// remain in memory until the GC frees it.
func (b *BitSet) Shrink(length uint) *BitSet {
	idx := wordsNeeded(length + 1)
	if idx > len(b.set) {
		return b
	}
	shrunk := make([]uint64, idx)
	copy(shrunk, b.set[:idx])
	b.set = shrunk
	b.length = length + 1
	b.set[idx-1] &= (allBits >> (uint64(64) - uint64(length&(wordSize-1)) - 1))
	return b
}

// InsertAt takes an index which indicates where a bit should be
// inserted. Then it shifts all the bits in the set to the left by 1, starting
// from the given index position, and sets the index position to 0.
//
// Depending on the size of your BitSet, and where you are inserting the new entry,
// this method could be extremely slow and in some cases might cause the entire BitSet
// to be recopied.
func (b *BitSet) InsertAt(idx uint) *BitSet {
	insertAtElement := (idx >> log2WordSize)

	// if length of set is a multiple of wordSize we need to allocate more space first
	if b.isLenExactMultiple() {
		b.set = append(b.set, uint64(0))
	}

	var i uint
	for i = uint(len(b.set) - 1); i > insertAtElement; i-- {
		// all elements above the position where we want to insert can simply by shifted
		b.set[i] <<= 1

		// we take the most significant bit of the previous element and set it as
		// the least significant bit of the current element
		b.set[i] |= (b.set[i-1] & 0x8000000000000000) >> 63
	}

	// generate a mask to extract the data that we need to shift left
	// within the element where we insert a bit
	dataMask := ^(uint64(1)<<uint64(idx&(wordSize-1)) - 1)

	// extract that data that we'll shift
	data := b.set[i] & dataMask

	// set the positions of the data mask to 0 in the element where we insert
	b.set[i] &= ^dataMask

	// shift data mask to the left and insert its data to the slice element
	b.set[i] |= data << 1

	// add 1 to length of BitSet
	b.length++

	return b
}

// String creates a string representation of the Bitmap
func (b *BitSet) String() string {
	// follows code from https://github.com/RoaringBitmap/roaring
	var buffer bytes.Buffer
	start := []byte("{")
	buffer.Write(start)
	counter := 0
	i, e := b.NextSet(0)
	for e {
		counter = counter + 1
		// to avoid exhausting the memory
		if counter > 0x40000 {
			buffer.WriteString("...")
			break
		}
		buffer.WriteString(strconv.FormatInt(int64(i), 10))
		i, e = b.NextSet(i + 1)
		if e {
			buffer.WriteString(",")
		}
	}
	buffer.WriteString("}")
	return buffer.String()
}

// DeleteAt deletes the bit at the given index position from
// within the bitset
// All the bits residing on the left of the deleted bit get
// shifted right by 1
// The running time of this operation may potentially be
// relatively slow, O(length)
func (b *BitSet) DeleteAt(i uint) *BitSet {
	// the index of the slice element where we'll delete a bit
	deleteAtElement := i >> log2WordSize

	// generate a mask for the data that needs to be shifted right
	// within that slice element that gets modified
	dataMask := ^((uint64(1) << (i & (wordSize - 1))) - 1)

	// extract the data that we'll shift right from the slice element
	data := b.set[deleteAtElement] & dataMask

	// set the masked area to 0 while leaving the rest as it is
	b.set[deleteAtElement] &= ^dataMask

	// shift the previously extracted data to the right and then
	// set it in the previously masked area
	b.set[deleteAtElement] |= (data >> 1) & dataMask

	// loop over all the consecutive slice elements to copy each
	// lowest bit into the highest position of the previous element,
	// then shift the entire content to the right by 1
	for i := int(deleteAtElement) + 1; i < len(b.set); i++ {
		b.set[i-1] |= (b.set[i] & 1) << 63
		b.set[i] >>= 1
	}

	b.length = b.length - 1

	return b
}

// NextSet returns the next bit set from the specified index,
// including possibly the current index
// along with an error code (true = valid, false = no set bit found)
// for i,e := v.NextSet(0); e; i,e = v.NextSet(i + 1) {...}
func (b *BitSet) NextSet(i uint) (uint, bool) {
	x := int(i >> log2WordSize)
	if x >= len(b.set) {
		return 0, false
	}
	w := b.set[x]
	w = w >> (i & (wordSize - 1))
	if w != 0 {
		return i + trailingZeroes64(w), true
	}
	x = x + 1
	for x < len(b.set) {
		if b.set[x] != 0 {
			return uint(x)*wordSize + trailingZeroes64(b.set[x]), true
		}
		x = x + 1

	}
	return 0, false
}

// NextSetMany returns many next bit sets from the specified index,
// including possibly the current index and up to cap(buffer).
// If the returned slice has len zero, then no more set bits were found
//
//    buffer := make([]uint, 256) // this should be reused
//    j := uint(0)
//    j, buffer = bitmap.NextSetMany(j, buffer)
//    for ; len(buffer) > 0; j, buffer = bitmap.NextSetMany(j,buffer) {
//     for k := range buffer {
//      do something with buffer[k]
//     }
//     j += 1
//    }
//
func (b *BitSet) NextSetMany(i uint, buffer []uint) (uint, []uint) {
	myanswer := buffer
	capacity := cap(buffer)
	x := int(i >> log2WordSize)
	if x >= len(b.set) || capacity == 0 {
		return 0, myanswer[:0]
	}
	skip := i & (wordSize - 1)
	word := b.set[x] >> skip
	myanswer = myanswer[:capacity]
	size := int(0)
	for word != 0 {
		r := trailingZeroes64(word)
		t := word & ((^word) + 1)
		myanswer[size] = r + i
		size++
		if size == capacity {
			goto End
		}
		word = word ^ t
	}
	x++
	for idx, word := range b.set[x:] {
		for word != 0 {
			r := trailingZeroes64(word)
			t := word & ((^word) + 1)
			myanswer[size] = r + (uint(x+idx) << 6)
			size++
			if size == capacity {
				goto End
			}
			word = word ^ t
		}
	}
End:
	if size > 0 {
		return myanswer[size-1], myanswer[:size]
	}
	return 0, myanswer[:0]
}

// NextClear returns the next clear bit from the specified index,
// including possibly the current index
// along with an error code (true = valid, false = no bit found i.e. all bits are set)
func (b *BitSet) NextClear(i uint) (uint, bool) {
	x := int(i >> log2WordSize)
	if x >= len(b.set) {
		return 0, false
	}
	w := b.set[x]
	w = w >> (i & (wordSize - 1))
	wA := allBits >> (i & (wordSize - 1))
	index := i + trailingZeroes64(^w)
	if w != wA && index < b.length {
		return index, true
	}
	x++
	for x < len(b.set) {
		index = uint(x)*wordSize + trailingZeroes64(^b.set[x])
		if b.set[x] != allBits && index < b.length {
			return index, true
		}
		x++
	}
	return 0, false
}

// ClearAll clears the entire BitSet
func (b *BitSet) ClearAll() *BitSet {
	if b != nil && b.set != nil {
		for i := range b.set {
			b.set[i] = 0
		}
	}
	return b
}

// wordCount returns the number of words used in a bit set
func (b *BitSet) wordCount() int {
	return len(b.set)
}

// Clone this BitSet
func (b *BitSet) Clone() *BitSet {
	c := New(b.length)
	if b.set != nil { // Clone should not modify current object
		copy(c.set, b.set)
	}
	return c
}

// Copy into a destination BitSet
// Returning the size of the destination BitSet
// like array copy
func (b *BitSet) Copy(c *BitSet) (count uint) {
	if c == nil {
		return
	}
	if b.set != nil { // Copy should not modify current object
		copy(c.set, b.set)
	}
	count = c.length
	if b.length < c.length {
		count = b.length
	}
	return
}

// Count (number of set bits)
func (b *BitSet) Count() uint {
	if b != nil && b.set != nil {
		return uint(popcntSlice(b.set))
	}
	return 0
}

// Equal tests the equvalence of two BitSets.
// False if they are of different sizes, otherwise true
// only if all the same bits are set
func (b *BitSet) Equal(c *BitSet) bool {
	if c == nil {
		return false
	}
	if b.length != c.length {
		return false
	}
	if b.length == 0 { // if they have both length == 0, then could have nil set
		return true
	}
	// testing for equality shoud not transform the bitset (no call to safeSet)

	for p, v := range b.set {
		if c.set[p] != v {
			return false
		}
	}
	return true
}

func panicIfNull(b *BitSet) {
	if b == nil {
		panic(Error("BitSet must not be null"))
	}
}

// Difference of base set and other set
// This is the BitSet equivalent of &^ (and not)
func (b *BitSet) Difference(compare *BitSet) (result *BitSet) {
	panicIfNull(b)
	panicIfNull(compare)
	result = b.Clone() // clone b (in case b is bigger than compare)
	l := int(compare.wordCount())
	if l > int(b.wordCount()) {
		l = int(b.wordCount())
	}
	for i := 0; i < l; i++ {
		result.set[i] = b.set[i] &^ compare.set[i]
	}
	return
}

// DifferenceCardinality computes the cardinality of the differnce
func (b *BitSet) DifferenceCardinality(compare *BitSet) uint {
	panicIfNull(b)
	panicIfNull(compare)
	l := int(compare.wordCount())
	if l > int(b.wordCount()) {
		l = int(b.wordCount())
	}
	cnt := uint64(0)
	cnt += popcntMaskSlice(b.set[:l], compare.set[:l])
	cnt += popcntSlice(b.set[l:])
	return uint(cnt)
}

// InPlaceDifference computes the difference of base set and other set
// This is the BitSet equivalent of &^ (and not)
func (b *BitSet) InPlaceDifference(compare *BitSet) {
	panicIfNull(b)
	panicIfNull(compare)
	l := int(compare.wordCount())
	if l > int(b.wordCount()) {
		l = int(b.wordCount())
	}
	for i := 0; i < l; i++ {
		b.set[i] &^= compare.set[i]
	}
}

// Convenience function: return two bitsets ordered by
// increasing length. Note: neither can be nil
func sortByLength(a *BitSet, b *BitSet) (ap *BitSet, bp *BitSet) {
	if a.length <= b.length {
		ap, bp = a, b
	} else {
		ap, bp = b, a
	}
	return
}

// Intersection of base set and other set
// This is the BitSet equivalent of & (and)
func (b *BitSet) Intersection(compare *BitSet) (result *BitSet) {
	panicIfNull(b)
	panicIfNull(compare)
	b, compare = sortByLength(b, compare)
	result = New(b.length)
	for i, word := range b.set {
		result.set[i] = word & compare.set[i]
	}
	return
}

// IntersectionCardinality computes the cardinality of the union
func (b *BitSet) IntersectionCardinality(compare *BitSet) uint {
	panicIfNull(b)
	panicIfNull(compare)
	b, compare = sortByLength(b, compare)
	cnt := popcntAndSlice(b.set, compare.set)
	return uint(cnt)
}

// InPlaceIntersection destructively computes the intersection of
// base set and the compare set.
// This is the BitSet equivalent of & (and)
func (b *BitSet) InPlaceIntersection(compare *BitSet) {
	panicIfNull(b)
	panicIfNull(compare)
	l := int(compare.wordCount())
	if l > int(b.wordCount()) {
		l = int(b.wordCount())
	}
	for i := 0; i < l; i++ {
		b.set[i] &= compare.set[i]
	}
	for i := l; i < len(b.set); i++ {
		b.set[i] = 0
	}
	if compare.length > 0 {
		b.extendSetMaybe(compare.length - 1)
	}
}

// Union of base set and other set
// This is the BitSet equivalent of | (or)
func (b *BitSet) Union(compare *BitSet) (result *BitSet) {
	panicIfNull(b)
	panicIfNull(compare)
	b, compare = sortByLength(b, compare)
	result = compare.Clone()
	for i, word := range b.set {
		result.set[i] = word | compare.set[i]
	}
	return
}

// UnionCardinality computes the cardinality of the uniton of the base set
// and the compare set.
func (b *BitSet) UnionCardinality(compare *BitSet) uint {
	panicIfNull(b)
	panicIfNull(compare)
	b, compare = sortByLength(b, compare)
	cnt := popcntOrSlice(b.set, compare.set)
	if len(compare.set) > len(b.set) {
		cnt += popcntSlice(compare.set[len(b.set):])
	}
	return uint(cnt)
}

// InPlaceUnion creates the destructive union of base set and compare set.
// This is the BitSet equivalent of | (or).
func (b *BitSet) InPlaceUnion(compare *BitSet) {
	panicIfNull(b)
	panicIfNull(compare)
	l := int(compare.wordCount())
	if l > int(b.wordCount()) {
		l = int(b.wordCount())
	}
	if compare.length > 0 {
		b.extendSetMaybe(compare.length - 1)
	}
	for i := 0; i < l; i++ {
		b.set[i] |= compare.set[i]
	}
	if len(compare.set) > l {
		for i := l; i < len(compare.set); i++ {
			b.set[i] = compare.set[i]
		}
	}
}

// SymmetricDifference of base set and other set
// This is the BitSet equivalent of ^ (xor)
func (b *BitSet) SymmetricDifference(compare *BitSet) (result *BitSet) {
	panicIfNull(b)
	panicIfNull(compare)
	b, compare = sortByLength(b, compare)
	// compare is bigger, so clone it
	result = compare.Clone()
	for i, word := range b.set {
		result.set[i] = word ^ compare.set[i]
	}
	return
}

// SymmetricDifferenceCardinality computes the cardinality of the symmetric difference
func (b *BitSet) SymmetricDifferenceCardinality(compare *BitSet) uint {
	panicIfNull(b)
	panicIfNull(compare)
	b, compare = sortByLength(b, compare)
	cnt := popcntXorSlice(b.set, compare.set)
	if len(compare.set) > len(b.set) {
		cnt += popcntSlice(compare.set[len(b.set):])
	}
	return uint(cnt)
}

// InPlaceSymmetricDifference creates the destructive SymmetricDifference of base set and other set
// This is the BitSet equivalent of ^ (xor)
func (b *BitSet) InPlaceSymmetricDifference(compare *BitSet) {
	panicIfNull(b)
	panicIfNull(compare)
	l := int(compare.wordCount())
	if l > int(b.wordCount()) {
		l = int(b.wordCount())
	}
	if compare.length > 0 {
		b.extendSetMaybe(compare.length - 1)
	}
	for i := 0; i < l; i++ {
		b.set[i] ^= compare.set[i]
	}
	if len(compare.set) > l {
		for i := l; i < len(compare.set); i++ {
			b.set[i] = compare.set[i]
		}
	}
}

// Is the length an exact multiple of word sizes?
func (b *BitSet) isLenExactMultiple() bool {
	return b.length%wordSize == 0
}

// Clean last word by setting unused bits to 0
func (b *BitSet) cleanLastWord() {
	if !b.isLenExactMultiple() {
		b.set[len(b.set)-1] &= allBits >> (wordSize - b.length%wordSize)
	}
}

// Complement computes the (local) complement of a biset (up to length bits)
func (b *BitSet) Complement() (result *BitSet) {
	panicIfNull(b)
	result = New(b.length)
	for i, word := range b.set {
		result.set[i] = ^word
	}
	result.cleanLastWord()
	return
}

// All returns true if all bits are set, false otherwise. Returns true for
// empty sets.
func (b *BitSet) All() bool {
	panicIfNull(b)
	return b.Count() == b.length
}

// None returns true if no bit is set, false otherwise. Retursn true for
// empty sets.
func (b *BitSet) None() bool {
	panicIfNull(b)
	if b != nil && b.set != nil {
		for _, word := range b.set {
			if word > 0 {
				return false
			}
		}
		return true
	}
	return true
}

// Any returns true if any bit is set, false otherwise
func (b *BitSet) Any() bool {
	panicIfNull(b)
	return !b.None()
}

// IsSuperSet returns true if this is a superset of the other set
func (b *BitSet) IsSuperSet(other *BitSet) bool {
	for i, e := other.NextSet(0); e; i, e = other.NextSet(i + 1) {
		if !b.Test(i) {
			return false
		}
	}
	return true
}

// IsStrictSuperSet returns true if this is a strict superset of the other set
func (b *BitSet) IsStrictSuperSet(other *BitSet) bool {
	return b.Count() > other.Count() && b.IsSuperSet(other)
}

// DumpAsBits dumps a bit set as a string of bits
func (b *BitSet) DumpAsBits() string {
	if b.set == nil {
		return "."
	}
	buffer := bytes.NewBufferString("")
	i := len(b.set) - 1
	for ; i >= 0; i-- {
		fmt.Fprintf(buffer, "%064b.", b.set[i])
	}
	return buffer.String()
}

// BinaryStorageSize returns the binary storage requirements
func (b *BitSet) BinaryStorageSize() int {
	return binary.Size(uint64(0)) + binary.Size(b.set)
}

// WriteTo writes a BitSet to a stream
func (b *BitSet) WriteTo(stream io.Writer) (int64, error) {
	length := uint64(b.length)

	// Write length
	err := binary.Write(stream, binaryOrder, length)
	if err != nil {
		return 0, err
	}

	// Write set
	err = binary.Write(stream, binaryOrder, b.set)
	return int64(b.BinaryStorageSize()), err
}

// ReadFrom reads a BitSet from a stream written using WriteTo
func (b *BitSet) ReadFrom(stream io.Reader) (int64, error) {
	var length uint64

	// Read length first
	err := binary.Read(stream, binaryOrder, &length)
	if err != nil {
		return 0, err
	}
	newset := New(uint(length))

	if uint64(newset.length) != length {
		return 0, errors.New("Unmarshalling error: type mismatch")
	}

	// Read remaining bytes as set
	err = binary.Read(stream, binaryOrder, newset.set)
	if err != nil {
		return 0, err
	}

	*b = *newset
	return int64(b.BinaryStorageSize()), nil
}

// MarshalBinary encodes a BitSet into a binary form and returns the result.
func (b *BitSet) MarshalBinary() ([]byte, error) {
	var buf bytes.Buffer
	writer := bufio.NewWriter(&buf)

	_, err := b.WriteTo(writer)
	if err != nil {
		return []byte{}, err
	}

	err = writer.Flush()

	return buf.Bytes(), err
}

// UnmarshalBinary decodes the binary form generated by MarshalBinary.
func (b *BitSet) UnmarshalBinary(data []byte) error {
	buf := bytes.NewReader(data)
	reader := bufio.NewReader(buf)

	_, err := b.ReadFrom(reader)

	return err
}

// MarshalJSON marshals a BitSet as a JSON structure
func (b *BitSet) MarshalJSON() ([]byte, error) {
	buffer := bytes.NewBuffer(make([]byte, 0, b.BinaryStorageSize()))
	_, err := b.WriteTo(buffer)
	if err != nil {
		return nil, err
	}

	// URLEncode all bytes
	return json.Marshal(base64Encoding.EncodeToString(buffer.Bytes()))
}

// UnmarshalJSON unmarshals a BitSet from JSON created using MarshalJSON
func (b *BitSet) UnmarshalJSON(data []byte) error {
	// Unmarshal as string
	var s string
	err := json.Unmarshal(data, &s)
	if err != nil {
		return err
	}

	// URLDecode string
	buf, err := base64Encoding.DecodeString(s)
	if err != nil {
		return err
	}

	_, err = b.ReadFrom(bytes.NewReader(buf))
	return err
}
