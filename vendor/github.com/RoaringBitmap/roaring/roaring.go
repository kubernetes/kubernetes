// Package roaring is an implementation of Roaring Bitmaps in Go.
// They provide fast compressed bitmap data structures (also called bitset).
// They are ideally suited to represent sets of integers over
// relatively small ranges.
// See http://roaringbitmap.org for details.
package roaring

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"io"
	"strconv"
	"sync"
)

// Bitmap represents a compressed bitmap where you can add integers.
type Bitmap struct {
	highlowcontainer roaringArray
}

// ToBase64 serializes a bitmap as Base64
func (rb *Bitmap) ToBase64() (string, error) {
	buf := new(bytes.Buffer)
	_, err := rb.WriteTo(buf)
	return base64.StdEncoding.EncodeToString(buf.Bytes()), err

}

// FromBase64 deserializes a bitmap from Base64
func (rb *Bitmap) FromBase64(str string) (int64, error) {
	data, err := base64.StdEncoding.DecodeString(str)
	if err != nil {
		return 0, err
	}
	buf := bytes.NewBuffer(data)

	return rb.ReadFrom(buf)
}

// WriteTo writes a serialized version of this bitmap to stream.
// The format is compatible with other RoaringBitmap
// implementations (Java, C) and is documented here:
// https://github.com/RoaringBitmap/RoaringFormatSpec
func (rb *Bitmap) WriteTo(stream io.Writer) (int64, error) {
	return rb.highlowcontainer.writeTo(stream)
}

// ToBytes returns an array of bytes corresponding to what is written
// when calling WriteTo
func (rb *Bitmap) ToBytes() ([]byte, error) {
	return rb.highlowcontainer.toBytes()
}

// Deprecated: WriteToMsgpack writes a msgpack2/snappy-streaming compressed serialized
// version of this bitmap to stream. The format is not
// compatible with the WriteTo() format, and is
// experimental: it may produce smaller on disk
// footprint and/or be faster to read, depending
// on your content. Currently only the Go roaring
// implementation supports this format.
func (rb *Bitmap) WriteToMsgpack(stream io.Writer) (int64, error) {
	return 0, rb.highlowcontainer.writeToMsgpack(stream)
}

// ReadFrom reads a serialized version of this bitmap from stream.
// The format is compatible with other RoaringBitmap
// implementations (Java, C) and is documented here:
// https://github.com/RoaringBitmap/RoaringFormatSpec
func (rb *Bitmap) ReadFrom(reader io.Reader) (p int64, err error) {
	stream := byteInputAdapterPool.Get().(*byteInputAdapter)
	stream.reset(reader)

	p, err = rb.highlowcontainer.readFrom(stream)
	byteInputAdapterPool.Put(stream)

	return
}

// FromBuffer creates a bitmap from its serialized version stored in buffer
//
// The format specification is available here:
// https://github.com/RoaringBitmap/RoaringFormatSpec
//
// The provided byte array (buf) is expected to be a constant.
// The function makes the best effort attempt not to copy data.
// You should take care not to modify buff as it will
// likely result in unexpected program behavior.
//
// Resulting bitmaps are effectively immutable in the following sense:
// a copy-on-write marker is used so that when you modify the resulting
// bitmap, copies of selected data (containers) are made.
// You should *not* change the copy-on-write status of the resulting
// bitmaps (SetCopyOnWrite).
//
// If buf becomes unavailable, then a bitmap created with
// FromBuffer would be effectively broken. Furthermore, any
// bitmap derived from this bitmap (e.g., via Or, And) might
// also be broken. Thus, before making buf unavailable, you should
// call CloneCopyOnWriteContainers on all such bitmaps.
//
func (rb *Bitmap) FromBuffer(buf []byte) (p int64, err error) {
	stream := byteBufferPool.Get().(*byteBuffer)
	stream.reset(buf)

	p, err = rb.highlowcontainer.readFrom(stream)
	byteBufferPool.Put(stream)

	return
}

var (
	byteBufferPool = sync.Pool{
		New: func() interface{} {
			return &byteBuffer{}
		},
	}

	byteInputAdapterPool = sync.Pool{
		New: func() interface{} {
			return &byteInputAdapter{}
		},
	}
)

// RunOptimize attempts to further compress the runs of consecutive values found in the bitmap
func (rb *Bitmap) RunOptimize() {
	rb.highlowcontainer.runOptimize()
}

// HasRunCompression returns true if the bitmap benefits from run compression
func (rb *Bitmap) HasRunCompression() bool {
	return rb.highlowcontainer.hasRunCompression()
}

// Deprecated: ReadFromMsgpack reads a msgpack2/snappy-streaming serialized
// version of this bitmap from stream. The format is
// expected is that written by the WriteToMsgpack()
// call; see additional notes there.
func (rb *Bitmap) ReadFromMsgpack(stream io.Reader) (int64, error) {
	return 0, rb.highlowcontainer.readFromMsgpack(stream)
}

// MarshalBinary implements the encoding.BinaryMarshaler interface for the bitmap
// (same as ToBytes)
func (rb *Bitmap) MarshalBinary() ([]byte, error) {
	return rb.ToBytes()
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface for the bitmap
func (rb *Bitmap) UnmarshalBinary(data []byte) error {
	r := bytes.NewReader(data)
	_, err := rb.ReadFrom(r)
	return err
}

// NewBitmap creates a new empty Bitmap (see also New)
func NewBitmap() *Bitmap {
	return &Bitmap{}
}

// New creates a new empty Bitmap (same as NewBitmap)
func New() *Bitmap {
	return &Bitmap{}
}

// Clear resets the Bitmap to be logically empty, but may retain
// some memory allocations that may speed up future operations
func (rb *Bitmap) Clear() {
	rb.highlowcontainer.clear()
}

// ToArray creates a new slice containing all of the integers stored in the Bitmap in sorted order
func (rb *Bitmap) ToArray() []uint32 {
	array := make([]uint32, rb.GetCardinality())
	pos := 0
	pos2 := 0

	for pos < rb.highlowcontainer.size() {
		hs := uint32(rb.highlowcontainer.getKeyAtIndex(pos)) << 16
		c := rb.highlowcontainer.getContainerAtIndex(pos)
		pos++
		c.fillLeastSignificant16bits(array, pos2, hs)
		pos2 += c.getCardinality()
	}
	return array
}

// GetSizeInBytes estimates the memory usage of the Bitmap. Note that this
// might differ slightly from the amount of bytes required for persistent storage
func (rb *Bitmap) GetSizeInBytes() uint64 {
	size := uint64(8)
	for _, c := range rb.highlowcontainer.containers {
		size += uint64(2) + uint64(c.getSizeInBytes())
	}
	return size
}

// GetSerializedSizeInBytes computes the serialized size in bytes
// of the Bitmap. It should correspond to the
// number of bytes written when invoking WriteTo. You can expect
// that this function is much cheaper computationally than WriteTo.
func (rb *Bitmap) GetSerializedSizeInBytes() uint64 {
	return rb.highlowcontainer.serializedSizeInBytes()
}

// BoundSerializedSizeInBytes returns an upper bound on the serialized size in bytes
// assuming that one wants to store "cardinality" integers in [0, universe_size)
func BoundSerializedSizeInBytes(cardinality uint64, universeSize uint64) uint64 {
	contnbr := (universeSize + uint64(65535)) / uint64(65536)
	if contnbr > cardinality {
		contnbr = cardinality
		// we can't have more containers than we have values
	}
	headermax := 8*contnbr + 4
	if 4 > (contnbr+7)/8 {
		headermax += 4
	} else {
		headermax += (contnbr + 7) / 8
	}
	valsarray := uint64(arrayContainerSizeInBytes(int(cardinality)))
	valsbitmap := contnbr * uint64(bitmapContainerSizeInBytes())
	valsbest := valsarray
	if valsbest > valsbitmap {
		valsbest = valsbitmap
	}
	return valsbest + headermax
}

// IntIterable allows you to iterate over the values in a Bitmap
type IntIterable interface {
	HasNext() bool
	Next() uint32
}

// IntPeekable allows you to look at the next value without advancing and
// advance as long as the next value is smaller than minval
type IntPeekable interface {
	IntIterable
	// PeekNext peeks the next value without advancing the iterator
	PeekNext() uint32
	// AdvanceIfNeeded advances as long as the next value is smaller than minval
	AdvanceIfNeeded(minval uint32)
}

type intIterator struct {
	pos              int
	hs               uint32
	iter             shortPeekable
	highlowcontainer *roaringArray
}

// HasNext returns true if there are more integers to iterate over
func (ii *intIterator) HasNext() bool {
	return ii.pos < ii.highlowcontainer.size()
}

func (ii *intIterator) init() {
	if ii.highlowcontainer.size() > ii.pos {
		ii.iter = ii.highlowcontainer.getContainerAtIndex(ii.pos).getShortIterator()
		ii.hs = uint32(ii.highlowcontainer.getKeyAtIndex(ii.pos)) << 16
	}
}

// Next returns the next integer
func (ii *intIterator) Next() uint32 {
	x := uint32(ii.iter.next()) | ii.hs
	if !ii.iter.hasNext() {
		ii.pos = ii.pos + 1
		ii.init()
	}
	return x
}

// PeekNext peeks the next value without advancing the iterator
func (ii *intIterator) PeekNext() uint32 {
	return uint32(ii.iter.peekNext()&maxLowBit) | ii.hs
}

// AdvanceIfNeeded advances as long as the next value is smaller than minval
func (ii *intIterator) AdvanceIfNeeded(minval uint32) {
	to := minval >> 16

	for ii.HasNext() && (ii.hs>>16) < to {
		ii.pos++
		ii.init()
	}

	if ii.HasNext() && (ii.hs>>16) == to {
		ii.iter.advanceIfNeeded(lowbits(minval))

		if !ii.iter.hasNext() {
			ii.pos++
			ii.init()
		}
	}
}

func newIntIterator(a *Bitmap) *intIterator {
	p := new(intIterator)
	p.pos = 0
	p.highlowcontainer = &a.highlowcontainer
	p.init()
	return p
}

type intReverseIterator struct {
	pos              int
	hs               uint32
	iter             shortIterable
	highlowcontainer *roaringArray
}

// HasNext returns true if there are more integers to iterate over
func (ii *intReverseIterator) HasNext() bool {
	return ii.pos >= 0
}

func (ii *intReverseIterator) init() {
	if ii.pos >= 0 {
		ii.iter = ii.highlowcontainer.getContainerAtIndex(ii.pos).getReverseIterator()
		ii.hs = uint32(ii.highlowcontainer.getKeyAtIndex(ii.pos)) << 16
	} else {
		ii.iter = nil
	}
}

// Next returns the next integer
func (ii *intReverseIterator) Next() uint32 {
	x := uint32(ii.iter.next()) | ii.hs
	if !ii.iter.hasNext() {
		ii.pos = ii.pos - 1
		ii.init()
	}
	return x
}

func newIntReverseIterator(a *Bitmap) *intReverseIterator {
	p := new(intReverseIterator)
	p.highlowcontainer = &a.highlowcontainer
	p.pos = a.highlowcontainer.size() - 1
	p.init()
	return p
}

// ManyIntIterable allows you to iterate over the values in a Bitmap
type ManyIntIterable interface {
	// pass in a buffer to fill up with values, returns how many values were returned
	NextMany([]uint32) int
}

type manyIntIterator struct {
	pos              int
	hs               uint32
	iter             manyIterable
	highlowcontainer *roaringArray
}

func (ii *manyIntIterator) init() {
	if ii.highlowcontainer.size() > ii.pos {
		ii.iter = ii.highlowcontainer.getContainerAtIndex(ii.pos).getManyIterator()
		ii.hs = uint32(ii.highlowcontainer.getKeyAtIndex(ii.pos)) << 16
	} else {
		ii.iter = nil
	}
}

func (ii *manyIntIterator) NextMany(buf []uint32) int {
	n := 0
	for n < len(buf) {
		if ii.iter == nil {
			break
		}
		moreN := ii.iter.nextMany(ii.hs, buf[n:])
		n += moreN
		if moreN == 0 {
			ii.pos = ii.pos + 1
			ii.init()
		}
	}

	return n
}

func newManyIntIterator(a *Bitmap) *manyIntIterator {
	p := new(manyIntIterator)
	p.pos = 0
	p.highlowcontainer = &a.highlowcontainer
	p.init()
	return p
}

// String creates a string representation of the Bitmap
func (rb *Bitmap) String() string {
	// inspired by https://github.com/fzandona/goroar/
	var buffer bytes.Buffer
	start := []byte("{")
	buffer.Write(start)
	i := rb.Iterator()
	counter := 0
	if i.HasNext() {
		counter = counter + 1
		buffer.WriteString(strconv.FormatInt(int64(i.Next()), 10))
	}
	for i.HasNext() {
		buffer.WriteString(",")
		counter = counter + 1
		// to avoid exhausting the memory
		if counter > 0x40000 {
			buffer.WriteString("...")
			break
		}
		buffer.WriteString(strconv.FormatInt(int64(i.Next()), 10))
	}
	buffer.WriteString("}")
	return buffer.String()
}

// Iterator creates a new IntPeekable to iterate over the integers contained in the bitmap, in sorted order;
// the iterator becomes invalid if the bitmap is modified (e.g., with Add or Remove).
func (rb *Bitmap) Iterator() IntPeekable {
	return newIntIterator(rb)
}

// ReverseIterator creates a new IntIterable to iterate over the integers contained in the bitmap, in sorted order;
// the iterator becomes invalid if the bitmap is modified (e.g., with Add or Remove).
func (rb *Bitmap) ReverseIterator() IntIterable {
	return newIntReverseIterator(rb)
}

// ManyIterator creates a new ManyIntIterable to iterate over the integers contained in the bitmap, in sorted order;
// the iterator becomes invalid if the bitmap is modified (e.g., with Add or Remove).
func (rb *Bitmap) ManyIterator() ManyIntIterable {
	return newManyIntIterator(rb)
}

// Clone creates a copy of the Bitmap
func (rb *Bitmap) Clone() *Bitmap {
	ptr := new(Bitmap)
	ptr.highlowcontainer = *rb.highlowcontainer.clone()
	return ptr
}

// Minimum get the smallest value stored in this roaring bitmap, assumes that it is not empty
func (rb *Bitmap) Minimum() uint32 {
	return uint32(rb.highlowcontainer.containers[0].minimum()) | (uint32(rb.highlowcontainer.keys[0]) << 16)
}

// Maximum get the largest value stored in this roaring bitmap, assumes that it is not empty
func (rb *Bitmap) Maximum() uint32 {
	lastindex := len(rb.highlowcontainer.containers) - 1
	return uint32(rb.highlowcontainer.containers[lastindex].maximum()) | (uint32(rb.highlowcontainer.keys[lastindex]) << 16)
}

// Contains returns true if the integer is contained in the bitmap
func (rb *Bitmap) Contains(x uint32) bool {
	hb := highbits(x)
	c := rb.highlowcontainer.getContainer(hb)
	return c != nil && c.contains(lowbits(x))
}

// ContainsInt returns true if the integer is contained in the bitmap (this is a convenience method, the parameter is casted to uint32 and Contains is called)
func (rb *Bitmap) ContainsInt(x int) bool {
	return rb.Contains(uint32(x))
}

// Equals returns true if the two bitmaps contain the same integers
func (rb *Bitmap) Equals(o interface{}) bool {
	srb, ok := o.(*Bitmap)
	if ok {
		return srb.highlowcontainer.equals(rb.highlowcontainer)
	}
	return false
}

// AddOffset adds the value 'offset' to each and every value in a bitmap, generating a new bitmap in the process
func AddOffset(x *Bitmap, offset uint32) (answer *Bitmap) {
	containerOffset := highbits(offset)
	inOffset := lowbits(offset)
	if inOffset == 0 {
		answer = x.Clone()
		for pos := 0; pos < answer.highlowcontainer.size(); pos++ {
			key := answer.highlowcontainer.getKeyAtIndex(pos)
			key += containerOffset
			answer.highlowcontainer.keys[pos] = key
		}
	} else {
		answer = New()
		for pos := 0; pos < x.highlowcontainer.size(); pos++ {
			key := x.highlowcontainer.getKeyAtIndex(pos)
			key += containerOffset
			c := x.highlowcontainer.getContainerAtIndex(pos)
			offsetted := c.addOffset(inOffset)
			if offsetted[0].getCardinality() > 0 {
				curSize := answer.highlowcontainer.size()
				lastkey := uint16(0)
				if curSize > 0 {
					lastkey = answer.highlowcontainer.getKeyAtIndex(curSize - 1)
				}
				if curSize > 0 && lastkey == key {
					prev := answer.highlowcontainer.getContainerAtIndex(curSize - 1)
					orrseult := prev.ior(offsetted[0])
					answer.highlowcontainer.setContainerAtIndex(curSize-1, orrseult)
				} else {
					answer.highlowcontainer.appendContainer(key, offsetted[0], false)
				}
			}
			if offsetted[1].getCardinality() > 0 {
				answer.highlowcontainer.appendContainer(key+1, offsetted[1], false)
			}
		}
	}
	return answer
}

// Add the integer x to the bitmap
func (rb *Bitmap) Add(x uint32) {
	hb := highbits(x)
	ra := &rb.highlowcontainer
	i := ra.getIndex(hb)
	if i >= 0 {
		var c container
		c = ra.getWritableContainerAtIndex(i).iaddReturnMinimized(lowbits(x))
		rb.highlowcontainer.setContainerAtIndex(i, c)
	} else {
		newac := newArrayContainer()
		rb.highlowcontainer.insertNewKeyValueAt(-i-1, hb, newac.iaddReturnMinimized(lowbits(x)))
	}
}

// add the integer x to the bitmap, return the container and its index
func (rb *Bitmap) addwithptr(x uint32) (int, container) {
	hb := highbits(x)
	ra := &rb.highlowcontainer
	i := ra.getIndex(hb)
	var c container
	if i >= 0 {
		c = ra.getWritableContainerAtIndex(i).iaddReturnMinimized(lowbits(x))
		rb.highlowcontainer.setContainerAtIndex(i, c)
		return i, c
	}
	newac := newArrayContainer()
	c = newac.iaddReturnMinimized(lowbits(x))
	rb.highlowcontainer.insertNewKeyValueAt(-i-1, hb, c)
	return -i - 1, c
}

// CheckedAdd adds the integer x to the bitmap and return true  if it was added (false if the integer was already present)
func (rb *Bitmap) CheckedAdd(x uint32) bool {
	// TODO: add unit tests for this method
	hb := highbits(x)
	i := rb.highlowcontainer.getIndex(hb)
	if i >= 0 {
		C := rb.highlowcontainer.getWritableContainerAtIndex(i)
		oldcard := C.getCardinality()
		C = C.iaddReturnMinimized(lowbits(x))
		rb.highlowcontainer.setContainerAtIndex(i, C)
		return C.getCardinality() > oldcard
	}
	newac := newArrayContainer()
	rb.highlowcontainer.insertNewKeyValueAt(-i-1, hb, newac.iaddReturnMinimized(lowbits(x)))
	return true

}

// AddInt adds the integer x to the bitmap (convenience method: the parameter is casted to uint32 and we call Add)
func (rb *Bitmap) AddInt(x int) {
	rb.Add(uint32(x))
}

// Remove the integer x from the bitmap
func (rb *Bitmap) Remove(x uint32) {
	hb := highbits(x)
	i := rb.highlowcontainer.getIndex(hb)
	if i >= 0 {
		c := rb.highlowcontainer.getWritableContainerAtIndex(i).iremoveReturnMinimized(lowbits(x))
		rb.highlowcontainer.setContainerAtIndex(i, c)
		if rb.highlowcontainer.getContainerAtIndex(i).getCardinality() == 0 {
			rb.highlowcontainer.removeAtIndex(i)
		}
	}
}

// CheckedRemove removes the integer x from the bitmap and return true if the integer was effectively remove (and false if the integer was not present)
func (rb *Bitmap) CheckedRemove(x uint32) bool {
	// TODO: add unit tests for this method
	hb := highbits(x)
	i := rb.highlowcontainer.getIndex(hb)
	if i >= 0 {
		C := rb.highlowcontainer.getWritableContainerAtIndex(i)
		oldcard := C.getCardinality()
		C = C.iremoveReturnMinimized(lowbits(x))
		rb.highlowcontainer.setContainerAtIndex(i, C)
		if rb.highlowcontainer.getContainerAtIndex(i).getCardinality() == 0 {
			rb.highlowcontainer.removeAtIndex(i)
			return true
		}
		return C.getCardinality() < oldcard
	}
	return false

}

// IsEmpty returns true if the Bitmap is empty (it is faster than doing (GetCardinality() == 0))
func (rb *Bitmap) IsEmpty() bool {
	return rb.highlowcontainer.size() == 0
}

// GetCardinality returns the number of integers contained in the bitmap
func (rb *Bitmap) GetCardinality() uint64 {
	size := uint64(0)
	for _, c := range rb.highlowcontainer.containers {
		size += uint64(c.getCardinality())
	}
	return size
}

// Rank returns the number of integers that are smaller or equal to x (Rank(infinity) would be GetCardinality())
func (rb *Bitmap) Rank(x uint32) uint64 {
	size := uint64(0)
	for i := 0; i < rb.highlowcontainer.size(); i++ {
		key := rb.highlowcontainer.getKeyAtIndex(i)
		if key > highbits(x) {
			return size
		}
		if key < highbits(x) {
			size += uint64(rb.highlowcontainer.getContainerAtIndex(i).getCardinality())
		} else {
			return size + uint64(rb.highlowcontainer.getContainerAtIndex(i).rank(lowbits(x)))
		}
	}
	return size
}

// Select returns the xth integer in the bitmap
func (rb *Bitmap) Select(x uint32) (uint32, error) {
	if rb.GetCardinality() <= uint64(x) {
		return 0, fmt.Errorf("can't find %dth integer in a bitmap with only %d items", x, rb.GetCardinality())
	}

	remaining := x
	for i := 0; i < rb.highlowcontainer.size(); i++ {
		c := rb.highlowcontainer.getContainerAtIndex(i)
		if remaining >= uint32(c.getCardinality()) {
			remaining -= uint32(c.getCardinality())
		} else {
			key := rb.highlowcontainer.getKeyAtIndex(i)
			return uint32(key)<<16 + uint32(c.selectInt(uint16(remaining))), nil
		}
	}
	return 0, fmt.Errorf("can't find %dth integer in a bitmap with only %d items", x, rb.GetCardinality())
}

// And computes the intersection between two bitmaps and stores the result in the current bitmap
func (rb *Bitmap) And(x2 *Bitmap) {
	pos1 := 0
	pos2 := 0
	intersectionsize := 0
	length1 := rb.highlowcontainer.size()
	length2 := x2.highlowcontainer.size()

main:
	for {
		if pos1 < length1 && pos2 < length2 {
			s1 := rb.highlowcontainer.getKeyAtIndex(pos1)
			s2 := x2.highlowcontainer.getKeyAtIndex(pos2)
			for {
				if s1 == s2 {
					c1 := rb.highlowcontainer.getWritableContainerAtIndex(pos1)
					c2 := x2.highlowcontainer.getContainerAtIndex(pos2)
					diff := c1.iand(c2)
					if diff.getCardinality() > 0 {
						rb.highlowcontainer.replaceKeyAndContainerAtIndex(intersectionsize, s1, diff, false)
						intersectionsize++
					}
					pos1++
					pos2++
					if (pos1 == length1) || (pos2 == length2) {
						break main
					}
					s1 = rb.highlowcontainer.getKeyAtIndex(pos1)
					s2 = x2.highlowcontainer.getKeyAtIndex(pos2)
				} else if s1 < s2 {
					pos1 = rb.highlowcontainer.advanceUntil(s2, pos1)
					if pos1 == length1 {
						break main
					}
					s1 = rb.highlowcontainer.getKeyAtIndex(pos1)
				} else { //s1 > s2
					pos2 = x2.highlowcontainer.advanceUntil(s1, pos2)
					if pos2 == length2 {
						break main
					}
					s2 = x2.highlowcontainer.getKeyAtIndex(pos2)
				}
			}
		} else {
			break
		}
	}
	rb.highlowcontainer.resize(intersectionsize)
}

// OrCardinality  returns the cardinality of the union between two bitmaps, bitmaps are not modified
func (rb *Bitmap) OrCardinality(x2 *Bitmap) uint64 {
	pos1 := 0
	pos2 := 0
	length1 := rb.highlowcontainer.size()
	length2 := x2.highlowcontainer.size()
	answer := uint64(0)
main:
	for {
		if (pos1 < length1) && (pos2 < length2) {
			s1 := rb.highlowcontainer.getKeyAtIndex(pos1)
			s2 := x2.highlowcontainer.getKeyAtIndex(pos2)

			for {
				if s1 < s2 {
					answer += uint64(rb.highlowcontainer.getContainerAtIndex(pos1).getCardinality())
					pos1++
					if pos1 == length1 {
						break main
					}
					s1 = rb.highlowcontainer.getKeyAtIndex(pos1)
				} else if s1 > s2 {
					answer += uint64(x2.highlowcontainer.getContainerAtIndex(pos2).getCardinality())
					pos2++
					if pos2 == length2 {
						break main
					}
					s2 = x2.highlowcontainer.getKeyAtIndex(pos2)
				} else {
					// TODO: could be faster if we did not have to materialize the container
					answer += uint64(rb.highlowcontainer.getContainerAtIndex(pos1).or(x2.highlowcontainer.getContainerAtIndex(pos2)).getCardinality())
					pos1++
					pos2++
					if (pos1 == length1) || (pos2 == length2) {
						break main
					}
					s1 = rb.highlowcontainer.getKeyAtIndex(pos1)
					s2 = x2.highlowcontainer.getKeyAtIndex(pos2)
				}
			}
		} else {
			break
		}
	}
	for ; pos1 < length1; pos1++ {
		answer += uint64(rb.highlowcontainer.getContainerAtIndex(pos1).getCardinality())
	}
	for ; pos2 < length2; pos2++ {
		answer += uint64(x2.highlowcontainer.getContainerAtIndex(pos2).getCardinality())
	}
	return answer
}

// AndCardinality returns the cardinality of the intersection between two bitmaps, bitmaps are not modified
func (rb *Bitmap) AndCardinality(x2 *Bitmap) uint64 {
	pos1 := 0
	pos2 := 0
	answer := uint64(0)
	length1 := rb.highlowcontainer.size()
	length2 := x2.highlowcontainer.size()

main:
	for {
		if pos1 < length1 && pos2 < length2 {
			s1 := rb.highlowcontainer.getKeyAtIndex(pos1)
			s2 := x2.highlowcontainer.getKeyAtIndex(pos2)
			for {
				if s1 == s2 {
					c1 := rb.highlowcontainer.getContainerAtIndex(pos1)
					c2 := x2.highlowcontainer.getContainerAtIndex(pos2)
					answer += uint64(c1.andCardinality(c2))
					pos1++
					pos2++
					if (pos1 == length1) || (pos2 == length2) {
						break main
					}
					s1 = rb.highlowcontainer.getKeyAtIndex(pos1)
					s2 = x2.highlowcontainer.getKeyAtIndex(pos2)
				} else if s1 < s2 {
					pos1 = rb.highlowcontainer.advanceUntil(s2, pos1)
					if pos1 == length1 {
						break main
					}
					s1 = rb.highlowcontainer.getKeyAtIndex(pos1)
				} else { //s1 > s2
					pos2 = x2.highlowcontainer.advanceUntil(s1, pos2)
					if pos2 == length2 {
						break main
					}
					s2 = x2.highlowcontainer.getKeyAtIndex(pos2)
				}
			}
		} else {
			break
		}
	}
	return answer
}

// Intersects checks whether two bitmap intersects, bitmaps are not modified
func (rb *Bitmap) Intersects(x2 *Bitmap) bool {
	pos1 := 0
	pos2 := 0
	length1 := rb.highlowcontainer.size()
	length2 := x2.highlowcontainer.size()

main:
	for {
		if pos1 < length1 && pos2 < length2 {
			s1 := rb.highlowcontainer.getKeyAtIndex(pos1)
			s2 := x2.highlowcontainer.getKeyAtIndex(pos2)
			for {
				if s1 == s2 {
					c1 := rb.highlowcontainer.getContainerAtIndex(pos1)
					c2 := x2.highlowcontainer.getContainerAtIndex(pos2)
					if c1.intersects(c2) {
						return true
					}
					pos1++
					pos2++
					if (pos1 == length1) || (pos2 == length2) {
						break main
					}
					s1 = rb.highlowcontainer.getKeyAtIndex(pos1)
					s2 = x2.highlowcontainer.getKeyAtIndex(pos2)
				} else if s1 < s2 {
					pos1 = rb.highlowcontainer.advanceUntil(s2, pos1)
					if pos1 == length1 {
						break main
					}
					s1 = rb.highlowcontainer.getKeyAtIndex(pos1)
				} else { //s1 > s2
					pos2 = x2.highlowcontainer.advanceUntil(s1, pos2)
					if pos2 == length2 {
						break main
					}
					s2 = x2.highlowcontainer.getKeyAtIndex(pos2)
				}
			}
		} else {
			break
		}
	}
	return false
}

// Xor computes the symmetric difference between two bitmaps and stores the result in the current bitmap
func (rb *Bitmap) Xor(x2 *Bitmap) {
	pos1 := 0
	pos2 := 0
	length1 := rb.highlowcontainer.size()
	length2 := x2.highlowcontainer.size()
	for {
		if (pos1 < length1) && (pos2 < length2) {
			s1 := rb.highlowcontainer.getKeyAtIndex(pos1)
			s2 := x2.highlowcontainer.getKeyAtIndex(pos2)
			if s1 < s2 {
				pos1 = rb.highlowcontainer.advanceUntil(s2, pos1)
				if pos1 == length1 {
					break
				}
			} else if s1 > s2 {
				c := x2.highlowcontainer.getWritableContainerAtIndex(pos2)
				rb.highlowcontainer.insertNewKeyValueAt(pos1, x2.highlowcontainer.getKeyAtIndex(pos2), c)
				length1++
				pos1++
				pos2++
			} else {
				// TODO: couple be computed in-place for reduced memory usage
				c := rb.highlowcontainer.getContainerAtIndex(pos1).xor(x2.highlowcontainer.getContainerAtIndex(pos2))
				if c.getCardinality() > 0 {
					rb.highlowcontainer.setContainerAtIndex(pos1, c)
					pos1++
				} else {
					rb.highlowcontainer.removeAtIndex(pos1)
					length1--
				}
				pos2++
			}
		} else {
			break
		}
	}
	if pos1 == length1 {
		rb.highlowcontainer.appendCopyMany(x2.highlowcontainer, pos2, length2)
	}
}

// Or computes the union between two bitmaps and stores the result in the current bitmap
func (rb *Bitmap) Or(x2 *Bitmap) {
	pos1 := 0
	pos2 := 0
	length1 := rb.highlowcontainer.size()
	length2 := x2.highlowcontainer.size()
main:
	for (pos1 < length1) && (pos2 < length2) {
		s1 := rb.highlowcontainer.getKeyAtIndex(pos1)
		s2 := x2.highlowcontainer.getKeyAtIndex(pos2)

		for {
			if s1 < s2 {
				pos1++
				if pos1 == length1 {
					break main
				}
				s1 = rb.highlowcontainer.getKeyAtIndex(pos1)
			} else if s1 > s2 {
				rb.highlowcontainer.insertNewKeyValueAt(pos1, s2, x2.highlowcontainer.getContainerAtIndex(pos2).clone())
				pos1++
				length1++
				pos2++
				if pos2 == length2 {
					break main
				}
				s2 = x2.highlowcontainer.getKeyAtIndex(pos2)
			} else {
				rb.highlowcontainer.replaceKeyAndContainerAtIndex(pos1, s1, rb.highlowcontainer.getWritableContainerAtIndex(pos1).ior(x2.highlowcontainer.getContainerAtIndex(pos2)), false)
				pos1++
				pos2++
				if (pos1 == length1) || (pos2 == length2) {
					break main
				}
				s1 = rb.highlowcontainer.getKeyAtIndex(pos1)
				s2 = x2.highlowcontainer.getKeyAtIndex(pos2)
			}
		}
	}
	if pos1 == length1 {
		rb.highlowcontainer.appendCopyMany(x2.highlowcontainer, pos2, length2)
	}
}

// AndNot computes the difference between two bitmaps and stores the result in the current bitmap
func (rb *Bitmap) AndNot(x2 *Bitmap) {
	pos1 := 0
	pos2 := 0
	intersectionsize := 0
	length1 := rb.highlowcontainer.size()
	length2 := x2.highlowcontainer.size()

main:
	for {
		if pos1 < length1 && pos2 < length2 {
			s1 := rb.highlowcontainer.getKeyAtIndex(pos1)
			s2 := x2.highlowcontainer.getKeyAtIndex(pos2)
			for {
				if s1 == s2 {
					c1 := rb.highlowcontainer.getWritableContainerAtIndex(pos1)
					c2 := x2.highlowcontainer.getContainerAtIndex(pos2)
					diff := c1.iandNot(c2)
					if diff.getCardinality() > 0 {
						rb.highlowcontainer.replaceKeyAndContainerAtIndex(intersectionsize, s1, diff, false)
						intersectionsize++
					}
					pos1++
					pos2++
					if (pos1 == length1) || (pos2 == length2) {
						break main
					}
					s1 = rb.highlowcontainer.getKeyAtIndex(pos1)
					s2 = x2.highlowcontainer.getKeyAtIndex(pos2)
				} else if s1 < s2 {
					c1 := rb.highlowcontainer.getContainerAtIndex(pos1)
					mustCopyOnWrite := rb.highlowcontainer.needsCopyOnWrite(pos1)
					rb.highlowcontainer.replaceKeyAndContainerAtIndex(intersectionsize, s1, c1, mustCopyOnWrite)
					intersectionsize++
					pos1++
					if pos1 == length1 {
						break main
					}
					s1 = rb.highlowcontainer.getKeyAtIndex(pos1)
				} else { //s1 > s2
					pos2 = x2.highlowcontainer.advanceUntil(s1, pos2)
					if pos2 == length2 {
						break main
					}
					s2 = x2.highlowcontainer.getKeyAtIndex(pos2)
				}
			}
		} else {
			break
		}
	}
	// TODO:implement as a copy
	for pos1 < length1 {
		c1 := rb.highlowcontainer.getContainerAtIndex(pos1)
		s1 := rb.highlowcontainer.getKeyAtIndex(pos1)
		mustCopyOnWrite := rb.highlowcontainer.needsCopyOnWrite(pos1)
		rb.highlowcontainer.replaceKeyAndContainerAtIndex(intersectionsize, s1, c1, mustCopyOnWrite)
		intersectionsize++
		pos1++
	}
	rb.highlowcontainer.resize(intersectionsize)
}

// Or computes the union between two bitmaps and returns the result
func Or(x1, x2 *Bitmap) *Bitmap {
	answer := NewBitmap()
	pos1 := 0
	pos2 := 0
	length1 := x1.highlowcontainer.size()
	length2 := x2.highlowcontainer.size()
main:
	for (pos1 < length1) && (pos2 < length2) {
		s1 := x1.highlowcontainer.getKeyAtIndex(pos1)
		s2 := x2.highlowcontainer.getKeyAtIndex(pos2)

		for {
			if s1 < s2 {
				answer.highlowcontainer.appendCopy(x1.highlowcontainer, pos1)
				pos1++
				if pos1 == length1 {
					break main
				}
				s1 = x1.highlowcontainer.getKeyAtIndex(pos1)
			} else if s1 > s2 {
				answer.highlowcontainer.appendCopy(x2.highlowcontainer, pos2)
				pos2++
				if pos2 == length2 {
					break main
				}
				s2 = x2.highlowcontainer.getKeyAtIndex(pos2)
			} else {

				answer.highlowcontainer.appendContainer(s1, x1.highlowcontainer.getContainerAtIndex(pos1).or(x2.highlowcontainer.getContainerAtIndex(pos2)), false)
				pos1++
				pos2++
				if (pos1 == length1) || (pos2 == length2) {
					break main
				}
				s1 = x1.highlowcontainer.getKeyAtIndex(pos1)
				s2 = x2.highlowcontainer.getKeyAtIndex(pos2)
			}
		}
	}
	if pos1 == length1 {
		answer.highlowcontainer.appendCopyMany(x2.highlowcontainer, pos2, length2)
	} else if pos2 == length2 {
		answer.highlowcontainer.appendCopyMany(x1.highlowcontainer, pos1, length1)
	}
	return answer
}

// And computes the intersection between two bitmaps and returns the result
func And(x1, x2 *Bitmap) *Bitmap {
	answer := NewBitmap()
	pos1 := 0
	pos2 := 0
	length1 := x1.highlowcontainer.size()
	length2 := x2.highlowcontainer.size()
main:
	for pos1 < length1 && pos2 < length2 {
		s1 := x1.highlowcontainer.getKeyAtIndex(pos1)
		s2 := x2.highlowcontainer.getKeyAtIndex(pos2)
		for {
			if s1 == s2 {
				C := x1.highlowcontainer.getContainerAtIndex(pos1)
				C = C.and(x2.highlowcontainer.getContainerAtIndex(pos2))

				if C.getCardinality() > 0 {
					answer.highlowcontainer.appendContainer(s1, C, false)
				}
				pos1++
				pos2++
				if (pos1 == length1) || (pos2 == length2) {
					break main
				}
				s1 = x1.highlowcontainer.getKeyAtIndex(pos1)
				s2 = x2.highlowcontainer.getKeyAtIndex(pos2)
			} else if s1 < s2 {
				pos1 = x1.highlowcontainer.advanceUntil(s2, pos1)
				if pos1 == length1 {
					break main
				}
				s1 = x1.highlowcontainer.getKeyAtIndex(pos1)
			} else { // s1 > s2
				pos2 = x2.highlowcontainer.advanceUntil(s1, pos2)
				if pos2 == length2 {
					break main
				}
				s2 = x2.highlowcontainer.getKeyAtIndex(pos2)
			}
		}
	}
	return answer
}

// Xor computes the symmetric difference between two bitmaps and returns the result
func Xor(x1, x2 *Bitmap) *Bitmap {
	answer := NewBitmap()
	pos1 := 0
	pos2 := 0
	length1 := x1.highlowcontainer.size()
	length2 := x2.highlowcontainer.size()
	for {
		if (pos1 < length1) && (pos2 < length2) {
			s1 := x1.highlowcontainer.getKeyAtIndex(pos1)
			s2 := x2.highlowcontainer.getKeyAtIndex(pos2)
			if s1 < s2 {
				answer.highlowcontainer.appendCopy(x1.highlowcontainer, pos1)
				pos1++
			} else if s1 > s2 {
				answer.highlowcontainer.appendCopy(x2.highlowcontainer, pos2)
				pos2++
			} else {
				c := x1.highlowcontainer.getContainerAtIndex(pos1).xor(x2.highlowcontainer.getContainerAtIndex(pos2))
				if c.getCardinality() > 0 {
					answer.highlowcontainer.appendContainer(s1, c, false)
				}
				pos1++
				pos2++
			}
		} else {
			break
		}
	}
	if pos1 == length1 {
		answer.highlowcontainer.appendCopyMany(x2.highlowcontainer, pos2, length2)
	} else if pos2 == length2 {
		answer.highlowcontainer.appendCopyMany(x1.highlowcontainer, pos1, length1)
	}
	return answer
}

// AndNot computes the difference between two bitmaps and returns the result
func AndNot(x1, x2 *Bitmap) *Bitmap {
	answer := NewBitmap()
	pos1 := 0
	pos2 := 0
	length1 := x1.highlowcontainer.size()
	length2 := x2.highlowcontainer.size()

main:
	for {
		if pos1 < length1 && pos2 < length2 {
			s1 := x1.highlowcontainer.getKeyAtIndex(pos1)
			s2 := x2.highlowcontainer.getKeyAtIndex(pos2)
			for {
				if s1 < s2 {
					answer.highlowcontainer.appendCopy(x1.highlowcontainer, pos1)
					pos1++
					if pos1 == length1 {
						break main
					}
					s1 = x1.highlowcontainer.getKeyAtIndex(pos1)
				} else if s1 == s2 {
					c1 := x1.highlowcontainer.getContainerAtIndex(pos1)
					c2 := x2.highlowcontainer.getContainerAtIndex(pos2)
					diff := c1.andNot(c2)
					if diff.getCardinality() > 0 {
						answer.highlowcontainer.appendContainer(s1, diff, false)
					}
					pos1++
					pos2++
					if (pos1 == length1) || (pos2 == length2) {
						break main
					}
					s1 = x1.highlowcontainer.getKeyAtIndex(pos1)
					s2 = x2.highlowcontainer.getKeyAtIndex(pos2)
				} else { //s1 > s2
					pos2 = x2.highlowcontainer.advanceUntil(s1, pos2)
					if pos2 == length2 {
						break main
					}
					s2 = x2.highlowcontainer.getKeyAtIndex(pos2)
				}
			}
		} else {
			break
		}
	}
	if pos2 == length2 {
		answer.highlowcontainer.appendCopyMany(x1.highlowcontainer, pos1, length1)
	}
	return answer
}

// AddMany add all of the values in dat
func (rb *Bitmap) AddMany(dat []uint32) {
	if len(dat) == 0 {
		return
	}
	prev := dat[0]
	idx, c := rb.addwithptr(prev)
	for _, i := range dat[1:] {
		if highbits(prev) == highbits(i) {
			c = c.iaddReturnMinimized(lowbits(i))
			rb.highlowcontainer.setContainerAtIndex(idx, c)
		} else {
			idx, c = rb.addwithptr(i)
		}
		prev = i
	}
}

// BitmapOf generates a new bitmap filled with the specified integers
func BitmapOf(dat ...uint32) *Bitmap {
	ans := NewBitmap()
	ans.AddMany(dat)
	return ans
}

// Flip negates the bits in the given range (i.e., [rangeStart,rangeEnd)), any integer present in this range and in the bitmap is removed,
// and any integer present in the range and not in the bitmap is added.
// The function uses 64-bit parameters even though a Bitmap stores 32-bit values because it is allowed and meaningful to use [0,uint64(0x100000000)) as a range
// while uint64(0x100000000) cannot be represented as a 32-bit value.
func (rb *Bitmap) Flip(rangeStart, rangeEnd uint64) {

	if rangeEnd > MaxUint32+1 {
		panic("rangeEnd > MaxUint32+1")
	}
	if rangeStart > MaxUint32+1 {
		panic("rangeStart > MaxUint32+1")
	}

	if rangeStart >= rangeEnd {
		return
	}

	hbStart := uint32(highbits(uint32(rangeStart)))
	lbStart := uint32(lowbits(uint32(rangeStart)))
	hbLast := uint32(highbits(uint32(rangeEnd - 1)))
	lbLast := uint32(lowbits(uint32(rangeEnd - 1)))

	var max uint32 = maxLowBit
	for hb := hbStart; hb <= hbLast; hb++ {
		var containerStart uint32
		if hb == hbStart {
			containerStart = uint32(lbStart)
		}
		containerLast := max
		if hb == hbLast {
			containerLast = uint32(lbLast)
		}

		i := rb.highlowcontainer.getIndex(uint16(hb))

		if i >= 0 {
			c := rb.highlowcontainer.getWritableContainerAtIndex(i).inot(int(containerStart), int(containerLast)+1)
			if c.getCardinality() > 0 {
				rb.highlowcontainer.setContainerAtIndex(i, c)
			} else {
				rb.highlowcontainer.removeAtIndex(i)
			}
		} else { // *think* the range of ones must never be
			// empty.
			rb.highlowcontainer.insertNewKeyValueAt(-i-1, uint16(hb), rangeOfOnes(int(containerStart), int(containerLast)))
		}
	}
}

// FlipInt calls Flip after casting the parameters  (convenience method)
func (rb *Bitmap) FlipInt(rangeStart, rangeEnd int) {
	rb.Flip(uint64(rangeStart), uint64(rangeEnd))
}

// AddRange adds the integers in [rangeStart, rangeEnd) to the bitmap.
// The function uses 64-bit parameters even though a Bitmap stores 32-bit values because it is allowed and meaningful to use [0,uint64(0x100000000)) as a range
// while uint64(0x100000000) cannot be represented as a 32-bit value.
func (rb *Bitmap) AddRange(rangeStart, rangeEnd uint64) {
	if rangeStart >= rangeEnd {
		return
	}
	if rangeEnd-1 > MaxUint32 {
		panic("rangeEnd-1 > MaxUint32")
	}
	hbStart := uint32(highbits(uint32(rangeStart)))
	lbStart := uint32(lowbits(uint32(rangeStart)))
	hbLast := uint32(highbits(uint32(rangeEnd - 1)))
	lbLast := uint32(lowbits(uint32(rangeEnd - 1)))

	var max uint32 = maxLowBit
	for hb := hbStart; hb <= hbLast; hb++ {
		containerStart := uint32(0)
		if hb == hbStart {
			containerStart = lbStart
		}
		containerLast := max
		if hb == hbLast {
			containerLast = lbLast
		}

		i := rb.highlowcontainer.getIndex(uint16(hb))

		if i >= 0 {
			c := rb.highlowcontainer.getWritableContainerAtIndex(i).iaddRange(int(containerStart), int(containerLast)+1)
			rb.highlowcontainer.setContainerAtIndex(i, c)
		} else { // *think* the range of ones must never be
			// empty.
			rb.highlowcontainer.insertNewKeyValueAt(-i-1, uint16(hb), rangeOfOnes(int(containerStart), int(containerLast)))
		}
	}
}

// RemoveRange removes the integers in [rangeStart, rangeEnd) from the bitmap.
// The function uses 64-bit parameters even though a Bitmap stores 32-bit values because it is allowed and meaningful to use [0,uint64(0x100000000)) as a range
// while uint64(0x100000000) cannot be represented as a 32-bit value.
func (rb *Bitmap) RemoveRange(rangeStart, rangeEnd uint64) {
	if rangeStart >= rangeEnd {
		return
	}
	if rangeEnd-1 > MaxUint32 {
		// logically, we should assume that the user wants to
		// remove all values from rangeStart to infinity
		// see https://github.com/RoaringBitmap/roaring/issues/141
		rangeEnd = uint64(0x100000000)
	}
	hbStart := uint32(highbits(uint32(rangeStart)))
	lbStart := uint32(lowbits(uint32(rangeStart)))
	hbLast := uint32(highbits(uint32(rangeEnd - 1)))
	lbLast := uint32(lowbits(uint32(rangeEnd - 1)))

	var max uint32 = maxLowBit

	if hbStart == hbLast {
		i := rb.highlowcontainer.getIndex(uint16(hbStart))
		if i < 0 {
			return
		}
		c := rb.highlowcontainer.getWritableContainerAtIndex(i).iremoveRange(int(lbStart), int(lbLast+1))
		if c.getCardinality() > 0 {
			rb.highlowcontainer.setContainerAtIndex(i, c)
		} else {
			rb.highlowcontainer.removeAtIndex(i)
		}
		return
	}
	ifirst := rb.highlowcontainer.getIndex(uint16(hbStart))
	ilast := rb.highlowcontainer.getIndex(uint16(hbLast))

	if ifirst >= 0 {
		if lbStart != 0 {
			c := rb.highlowcontainer.getWritableContainerAtIndex(ifirst).iremoveRange(int(lbStart), int(max+1))
			if c.getCardinality() > 0 {
				rb.highlowcontainer.setContainerAtIndex(ifirst, c)
				ifirst++
			}
		}
	} else {
		ifirst = -ifirst - 1
	}
	if ilast >= 0 {
		if lbLast != max {
			c := rb.highlowcontainer.getWritableContainerAtIndex(ilast).iremoveRange(int(0), int(lbLast+1))
			if c.getCardinality() > 0 {
				rb.highlowcontainer.setContainerAtIndex(ilast, c)
			} else {
				ilast++
			}
		} else {
			ilast++
		}
	} else {
		ilast = -ilast - 1
	}
	rb.highlowcontainer.removeIndexRange(ifirst, ilast)
}

// Flip negates the bits in the given range  (i.e., [rangeStart,rangeEnd)), any integer present in this range and in the bitmap is removed,
// and any integer present in the range and not in the bitmap is added, a new bitmap is returned leaving
// the current bitmap unchanged.
// The function uses 64-bit parameters even though a Bitmap stores 32-bit values because it is allowed and meaningful to use [0,uint64(0x100000000)) as a range
// while uint64(0x100000000) cannot be represented as a 32-bit value.
func Flip(bm *Bitmap, rangeStart, rangeEnd uint64) *Bitmap {
	if rangeStart >= rangeEnd {
		return bm.Clone()
	}

	if rangeStart > MaxUint32 {
		panic("rangeStart > MaxUint32")
	}
	if rangeEnd-1 > MaxUint32 {
		panic("rangeEnd-1 > MaxUint32")
	}

	answer := NewBitmap()
	hbStart := uint32(highbits(uint32(rangeStart)))
	lbStart := uint32(lowbits(uint32(rangeStart)))
	hbLast := uint32(highbits(uint32(rangeEnd - 1)))
	lbLast := uint32(lowbits(uint32(rangeEnd - 1)))

	// copy the containers before the active area
	answer.highlowcontainer.appendCopiesUntil(bm.highlowcontainer, uint16(hbStart))

	var max uint32 = maxLowBit
	for hb := hbStart; hb <= hbLast; hb++ {
		var containerStart uint32
		if hb == hbStart {
			containerStart = uint32(lbStart)
		}
		containerLast := max
		if hb == hbLast {
			containerLast = uint32(lbLast)
		}

		i := bm.highlowcontainer.getIndex(uint16(hb))
		j := answer.highlowcontainer.getIndex(uint16(hb))

		if i >= 0 {
			c := bm.highlowcontainer.getContainerAtIndex(i).not(int(containerStart), int(containerLast)+1)
			if c.getCardinality() > 0 {
				answer.highlowcontainer.insertNewKeyValueAt(-j-1, uint16(hb), c)
			}

		} else { // *think* the range of ones must never be
			// empty.
			answer.highlowcontainer.insertNewKeyValueAt(-j-1, uint16(hb),
				rangeOfOnes(int(containerStart), int(containerLast)))
		}
	}
	// copy the containers after the active area.
	answer.highlowcontainer.appendCopiesAfter(bm.highlowcontainer, uint16(hbLast))

	return answer
}

// SetCopyOnWrite sets this bitmap to use copy-on-write so that copies are fast and memory conscious
// if the parameter is true, otherwise we leave the default where hard copies are made
// (copy-on-write requires extra care in a threaded context).
// Calling SetCopyOnWrite(true) on a bitmap created with FromBuffer is unsafe.
func (rb *Bitmap) SetCopyOnWrite(val bool) {
	rb.highlowcontainer.copyOnWrite = val
}

// GetCopyOnWrite gets this bitmap's copy-on-write property
func (rb *Bitmap) GetCopyOnWrite() (val bool) {
	return rb.highlowcontainer.copyOnWrite
}

// CloneCopyOnWriteContainers clones all containers which have
// needCopyOnWrite set to true.
// This can be used to make sure it is safe to munmap a []byte
// that the roaring array may still have a reference to, after
// calling FromBuffer.
// More generally this function is useful if you call FromBuffer
// to construct a bitmap with a backing array buf
// and then later discard the buf array. Note that you should call
// CloneCopyOnWriteContainers on all bitmaps that were derived
// from the 'FromBuffer' bitmap since they map have dependencies
// on the buf array as well.
func (rb *Bitmap) CloneCopyOnWriteContainers() {
	rb.highlowcontainer.cloneCopyOnWriteContainers()
}

// FlipInt calls Flip after casting the parameters (convenience method)
func FlipInt(bm *Bitmap, rangeStart, rangeEnd int) *Bitmap {
	return Flip(bm, uint64(rangeStart), uint64(rangeEnd))
}

// Statistics provides details on the container types in use.
type Statistics struct {
	Cardinality uint64
	Containers  uint64

	ArrayContainers      uint64
	ArrayContainerBytes  uint64
	ArrayContainerValues uint64

	BitmapContainers      uint64
	BitmapContainerBytes  uint64
	BitmapContainerValues uint64

	RunContainers      uint64
	RunContainerBytes  uint64
	RunContainerValues uint64
}

// Stats returns details on container type usage in a Statistics struct.
func (rb *Bitmap) Stats() Statistics {
	stats := Statistics{}
	stats.Containers = uint64(len(rb.highlowcontainer.containers))
	for _, c := range rb.highlowcontainer.containers {
		stats.Cardinality += uint64(c.getCardinality())

		switch c.(type) {
		case *arrayContainer:
			stats.ArrayContainers++
			stats.ArrayContainerBytes += uint64(c.getSizeInBytes())
			stats.ArrayContainerValues += uint64(c.getCardinality())
		case *bitmapContainer:
			stats.BitmapContainers++
			stats.BitmapContainerBytes += uint64(c.getSizeInBytes())
			stats.BitmapContainerValues += uint64(c.getCardinality())
		case *runContainer16:
			stats.RunContainers++
			stats.RunContainerBytes += uint64(c.getSizeInBytes())
			stats.RunContainerValues += uint64(c.getCardinality())
		}
	}
	return stats
}
