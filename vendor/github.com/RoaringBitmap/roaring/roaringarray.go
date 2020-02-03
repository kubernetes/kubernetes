package roaring

import (
	"bytes"
	"encoding/binary"
	"fmt"
	snappy "github.com/glycerine/go-unsnap-stream"
	"github.com/tinylib/msgp/msgp"
	"io"
)

//go:generate msgp -unexported

type container interface {
	addOffset(uint16) []container

	clone() container
	and(container) container
	andCardinality(container) int
	iand(container) container // i stands for inplace
	andNot(container) container
	iandNot(container) container // i stands for inplace
	getCardinality() int
	// rank returns the number of integers that are
	// smaller or equal to x. rank(infinity) would be getCardinality().
	rank(uint16) int

	iadd(x uint16) bool                   // inplace, returns true if x was new.
	iaddReturnMinimized(uint16) container // may change return type to minimize storage.

	//addRange(start, final int) container  // range is [firstOfRange,lastOfRange) (unused)
	iaddRange(start, endx int) container // i stands for inplace, range is [firstOfRange,endx)

	iremove(x uint16) bool                   // inplace, returns true if x was present.
	iremoveReturnMinimized(uint16) container // may change return type to minimize storage.

	not(start, final int) container        // range is [firstOfRange,lastOfRange)
	inot(firstOfRange, endx int) container // i stands for inplace, range is [firstOfRange,endx)
	xor(r container) container
	getShortIterator() shortPeekable
	getReverseIterator() shortIterable
	getManyIterator() manyIterable
	contains(i uint16) bool
	maximum() uint16
	minimum() uint16

	// equals is now logical equals; it does not require the
	// same underlying container types, but compares across
	// any of the implementations.
	equals(r container) bool

	fillLeastSignificant16bits(array []uint32, i int, mask uint32)
	or(r container) container
	orCardinality(r container) int
	isFull() bool
	ior(r container) container   // i stands for inplace
	intersects(r container) bool // whether the two containers intersect
	lazyOR(r container) container
	lazyIOR(r container) container
	getSizeInBytes() int
	//removeRange(start, final int) container  // range is [firstOfRange,lastOfRange) (unused)
	iremoveRange(start, final int) container // i stands for inplace, range is [firstOfRange,lastOfRange)
	selectInt(x uint16) int                  // selectInt returns the xth integer in the container
	serializedSizeInBytes() int
	writeTo(io.Writer) (int, error)

	numberOfRuns() int
	toEfficientContainer() container
	String() string
	containerType() contype
}

type contype uint8

const (
	bitmapContype contype = iota
	arrayContype
	run16Contype
	run32Contype
)

// careful: range is [firstOfRange,lastOfRange]
func rangeOfOnes(start, last int) container {
	if start > MaxUint16 {
		panic("rangeOfOnes called with start > MaxUint16")
	}
	if last > MaxUint16 {
		panic("rangeOfOnes called with last > MaxUint16")
	}
	if start < 0 {
		panic("rangeOfOnes called with start < 0")
	}
	if last < 0 {
		panic("rangeOfOnes called with last < 0")
	}
	return newRunContainer16Range(uint16(start), uint16(last))
}

type roaringArray struct {
	keys            []uint16
	containers      []container `msg:"-"` // don't try to serialize directly.
	needCopyOnWrite []bool
	copyOnWrite     bool

	// conserz is used at serialization time
	// to serialize containers. Otherwise empty.
	conserz []containerSerz
}

// containerSerz facilitates serializing container (tricky to
// serialize because it is an interface) by providing a
// light wrapper with a type identifier.
type containerSerz struct {
	t contype  `msg:"t"` // type
	r msgp.Raw `msg:"r"` // Raw msgpack of the actual container type
}

func newRoaringArray() *roaringArray {
	return &roaringArray{}
}

// runOptimize compresses the element containers to minimize space consumed.
// Q: how does this interact with copyOnWrite and needCopyOnWrite?
// A: since we aren't changing the logical content, just the representation,
//    we don't bother to check the needCopyOnWrite bits. We replace
//    (possibly all) elements of ra.containers in-place with space
//    optimized versions.
func (ra *roaringArray) runOptimize() {
	for i := range ra.containers {
		ra.containers[i] = ra.containers[i].toEfficientContainer()
	}
}

func (ra *roaringArray) appendContainer(key uint16, value container, mustCopyOnWrite bool) {
	ra.keys = append(ra.keys, key)
	ra.containers = append(ra.containers, value)
	ra.needCopyOnWrite = append(ra.needCopyOnWrite, mustCopyOnWrite)
}

func (ra *roaringArray) appendWithoutCopy(sa roaringArray, startingindex int) {
	mustCopyOnWrite := sa.needCopyOnWrite[startingindex]
	ra.appendContainer(sa.keys[startingindex], sa.containers[startingindex], mustCopyOnWrite)
}

func (ra *roaringArray) appendCopy(sa roaringArray, startingindex int) {
	// cow only if the two request it, or if we already have a lightweight copy
	copyonwrite := (ra.copyOnWrite && sa.copyOnWrite) || sa.needsCopyOnWrite(startingindex)
	if !copyonwrite {
		// since there is no copy-on-write, we need to clone the container (this is important)
		ra.appendContainer(sa.keys[startingindex], sa.containers[startingindex].clone(), copyonwrite)
	} else {
		ra.appendContainer(sa.keys[startingindex], sa.containers[startingindex], copyonwrite)
		if !sa.needsCopyOnWrite(startingindex) {
			sa.setNeedsCopyOnWrite(startingindex)
		}
	}
}

func (ra *roaringArray) appendWithoutCopyMany(sa roaringArray, startingindex, end int) {
	for i := startingindex; i < end; i++ {
		ra.appendWithoutCopy(sa, i)
	}
}

func (ra *roaringArray) appendCopyMany(sa roaringArray, startingindex, end int) {
	for i := startingindex; i < end; i++ {
		ra.appendCopy(sa, i)
	}
}

func (ra *roaringArray) appendCopiesUntil(sa roaringArray, stoppingKey uint16) {
	// cow only if the two request it, or if we already have a lightweight copy
	copyonwrite := ra.copyOnWrite && sa.copyOnWrite

	for i := 0; i < sa.size(); i++ {
		if sa.keys[i] >= stoppingKey {
			break
		}
		thiscopyonewrite := copyonwrite || sa.needsCopyOnWrite(i)
		if thiscopyonewrite {
			ra.appendContainer(sa.keys[i], sa.containers[i], thiscopyonewrite)
			if !sa.needsCopyOnWrite(i) {
				sa.setNeedsCopyOnWrite(i)
			}

		} else {
			// since there is no copy-on-write, we need to clone the container (this is important)
			ra.appendContainer(sa.keys[i], sa.containers[i].clone(), thiscopyonewrite)

		}
	}
}

func (ra *roaringArray) appendCopiesAfter(sa roaringArray, beforeStart uint16) {
	// cow only if the two request it, or if we already have a lightweight copy
	copyonwrite := ra.copyOnWrite && sa.copyOnWrite

	startLocation := sa.getIndex(beforeStart)
	if startLocation >= 0 {
		startLocation++
	} else {
		startLocation = -startLocation - 1
	}

	for i := startLocation; i < sa.size(); i++ {
		thiscopyonewrite := copyonwrite || sa.needsCopyOnWrite(i)
		if thiscopyonewrite {
			ra.appendContainer(sa.keys[i], sa.containers[i], thiscopyonewrite)
			if !sa.needsCopyOnWrite(i) {
				sa.setNeedsCopyOnWrite(i)
			}
		} else {
			// since there is no copy-on-write, we need to clone the container (this is important)
			ra.appendContainer(sa.keys[i], sa.containers[i].clone(), thiscopyonewrite)

		}
	}
}

func (ra *roaringArray) removeIndexRange(begin, end int) {
	if end <= begin {
		return
	}

	r := end - begin

	copy(ra.keys[begin:], ra.keys[end:])
	copy(ra.containers[begin:], ra.containers[end:])
	copy(ra.needCopyOnWrite[begin:], ra.needCopyOnWrite[end:])

	ra.resize(len(ra.keys) - r)
}

func (ra *roaringArray) resize(newsize int) {
	for k := newsize; k < len(ra.containers); k++ {
		ra.containers[k] = nil
	}

	ra.keys = ra.keys[:newsize]
	ra.containers = ra.containers[:newsize]
	ra.needCopyOnWrite = ra.needCopyOnWrite[:newsize]
}

func (ra *roaringArray) clear() {
	ra.resize(0)
	ra.copyOnWrite = false
	ra.conserz = nil
}

func (ra *roaringArray) clone() *roaringArray {

	sa := roaringArray{}
	sa.copyOnWrite = ra.copyOnWrite

	// this is where copyOnWrite is used.
	if ra.copyOnWrite {
		sa.keys = make([]uint16, len(ra.keys))
		copy(sa.keys, ra.keys)
		sa.containers = make([]container, len(ra.containers))
		copy(sa.containers, ra.containers)
		sa.needCopyOnWrite = make([]bool, len(ra.needCopyOnWrite))

		ra.markAllAsNeedingCopyOnWrite()
		sa.markAllAsNeedingCopyOnWrite()

		// sa.needCopyOnWrite is shared
	} else {
		// make a full copy

		sa.keys = make([]uint16, len(ra.keys))
		copy(sa.keys, ra.keys)

		sa.containers = make([]container, len(ra.containers))
		for i := range sa.containers {
			sa.containers[i] = ra.containers[i].clone()
		}

		sa.needCopyOnWrite = make([]bool, len(ra.needCopyOnWrite))
	}
	return &sa
}

// clone all containers which have needCopyOnWrite set to true
// This can be used to make sure it is safe to munmap a []byte
// that the roaring array may still have a reference to.
func (ra *roaringArray) cloneCopyOnWriteContainers() {
	for i, needCopyOnWrite := range ra.needCopyOnWrite {
		if needCopyOnWrite {
			ra.containers[i] = ra.containers[i].clone()
			ra.needCopyOnWrite[i] = false
		}
	}
}

// unused function:
//func (ra *roaringArray) containsKey(x uint16) bool {
//	return (ra.binarySearch(0, int64(len(ra.keys)), x) >= 0)
//}

func (ra *roaringArray) getContainer(x uint16) container {
	i := ra.binarySearch(0, int64(len(ra.keys)), x)
	if i < 0 {
		return nil
	}
	return ra.containers[i]
}

func (ra *roaringArray) getContainerAtIndex(i int) container {
	return ra.containers[i]
}

func (ra *roaringArray) getFastContainerAtIndex(i int, needsWriteable bool) container {
	c := ra.getContainerAtIndex(i)
	switch t := c.(type) {
	case *arrayContainer:
		c = t.toBitmapContainer()
	case *runContainer16:
		if !t.isFull() {
			c = t.toBitmapContainer()
		}
	case *bitmapContainer:
		if needsWriteable && ra.needCopyOnWrite[i] {
			c = ra.containers[i].clone()
		}
	}
	return c
}

func (ra *roaringArray) getWritableContainerAtIndex(i int) container {
	if ra.needCopyOnWrite[i] {
		ra.containers[i] = ra.containers[i].clone()
		ra.needCopyOnWrite[i] = false
	}
	return ra.containers[i]
}

func (ra *roaringArray) getIndex(x uint16) int {
	// before the binary search, we optimize for frequent cases
	size := len(ra.keys)
	if (size == 0) || (ra.keys[size-1] == x) {
		return size - 1
	}
	return ra.binarySearch(0, int64(size), x)
}

func (ra *roaringArray) getKeyAtIndex(i int) uint16 {
	return ra.keys[i]
}

func (ra *roaringArray) insertNewKeyValueAt(i int, key uint16, value container) {
	ra.keys = append(ra.keys, 0)
	ra.containers = append(ra.containers, nil)

	copy(ra.keys[i+1:], ra.keys[i:])
	copy(ra.containers[i+1:], ra.containers[i:])

	ra.keys[i] = key
	ra.containers[i] = value

	ra.needCopyOnWrite = append(ra.needCopyOnWrite, false)
	copy(ra.needCopyOnWrite[i+1:], ra.needCopyOnWrite[i:])
	ra.needCopyOnWrite[i] = false
}

func (ra *roaringArray) remove(key uint16) bool {
	i := ra.binarySearch(0, int64(len(ra.keys)), key)
	if i >= 0 { // if a new key
		ra.removeAtIndex(i)
		return true
	}
	return false
}

func (ra *roaringArray) removeAtIndex(i int) {
	copy(ra.keys[i:], ra.keys[i+1:])
	copy(ra.containers[i:], ra.containers[i+1:])

	copy(ra.needCopyOnWrite[i:], ra.needCopyOnWrite[i+1:])

	ra.resize(len(ra.keys) - 1)
}

func (ra *roaringArray) setContainerAtIndex(i int, c container) {
	ra.containers[i] = c
}

func (ra *roaringArray) replaceKeyAndContainerAtIndex(i int, key uint16, c container, mustCopyOnWrite bool) {
	ra.keys[i] = key
	ra.containers[i] = c
	ra.needCopyOnWrite[i] = mustCopyOnWrite
}

func (ra *roaringArray) size() int {
	return len(ra.keys)
}

func (ra *roaringArray) binarySearch(begin, end int64, ikey uint16) int {
	low := begin
	high := end - 1
	for low+16 <= high {
		middleIndex := low + (high-low)/2 // avoid overflow
		middleValue := ra.keys[middleIndex]

		if middleValue < ikey {
			low = middleIndex + 1
		} else if middleValue > ikey {
			high = middleIndex - 1
		} else {
			return int(middleIndex)
		}
	}
	for ; low <= high; low++ {
		val := ra.keys[low]
		if val >= ikey {
			if val == ikey {
				return int(low)
			}
			break
		}
	}
	return -int(low + 1)
}

func (ra *roaringArray) equals(o interface{}) bool {
	srb, ok := o.(roaringArray)
	if ok {

		if srb.size() != ra.size() {
			return false
		}
		for i, k := range ra.keys {
			if k != srb.keys[i] {
				return false
			}
		}

		for i, c := range ra.containers {
			if !c.equals(srb.containers[i]) {
				return false
			}
		}
		return true
	}
	return false
}

func (ra *roaringArray) headerSize() uint64 {
	size := uint64(len(ra.keys))
	if ra.hasRunCompression() {
		if size < noOffsetThreshold { // for small bitmaps, we omit the offsets
			return 4 + (size+7)/8 + 4*size
		}
		return 4 + (size+7)/8 + 8*size // - 4 because we pack the size with the cookie
	}
	return 4 + 4 + 8*size

}

// should be dirt cheap
func (ra *roaringArray) serializedSizeInBytes() uint64 {
	answer := ra.headerSize()
	for _, c := range ra.containers {
		answer += uint64(c.serializedSizeInBytes())
	}
	return answer
}

//
// spec: https://github.com/RoaringBitmap/RoaringFormatSpec
//
func (ra *roaringArray) writeTo(w io.Writer) (n int64, err error) {
	hasRun := ra.hasRunCompression()
	isRunSizeInBytes := 0
	cookieSize := 8
	if hasRun {
		cookieSize = 4
		isRunSizeInBytes = (len(ra.keys) + 7) / 8
	}
	descriptiveHeaderSize := 4 * len(ra.keys)
	preambleSize := cookieSize + isRunSizeInBytes + descriptiveHeaderSize

	buf := make([]byte, preambleSize+4*len(ra.keys))

	nw := 0

	if hasRun {
		binary.LittleEndian.PutUint16(buf[0:], uint16(serialCookie))
		nw += 2
		binary.LittleEndian.PutUint16(buf[2:], uint16(len(ra.keys)-1))
		nw += 2

		// compute isRun bitmap
		var ir []byte

		isRun := newBitmapContainer()
		for i, c := range ra.containers {
			switch c.(type) {
			case *runContainer16:
				isRun.iadd(uint16(i))
			}
		}
		// convert to little endian
		ir = isRun.asLittleEndianByteSlice()[:isRunSizeInBytes]
		nw += copy(buf[nw:], ir)
	} else {
		binary.LittleEndian.PutUint32(buf[0:], uint32(serialCookieNoRunContainer))
		nw += 4
		binary.LittleEndian.PutUint32(buf[4:], uint32(len(ra.keys)))
		nw += 4
	}

	// descriptive header
	for i, key := range ra.keys {
		binary.LittleEndian.PutUint16(buf[nw:], key)
		nw += 2
		c := ra.containers[i]
		binary.LittleEndian.PutUint16(buf[nw:], uint16(c.getCardinality()-1))
		nw += 2
	}

	startOffset := int64(preambleSize + 4*len(ra.keys))
	if !hasRun || (len(ra.keys) >= noOffsetThreshold) {
		// offset header
		for _, c := range ra.containers {
			binary.LittleEndian.PutUint32(buf[nw:], uint32(startOffset))
			nw += 4
			switch rc := c.(type) {
			case *runContainer16:
				startOffset += 2 + int64(len(rc.iv))*4
			default:
				startOffset += int64(getSizeInBytesFromCardinality(c.getCardinality()))
			}
		}
	}

	written, err := w.Write(buf[:nw])
	if err != nil {
		return n, err
	}
	n += int64(written)

	for _, c := range ra.containers {
		written, err := c.writeTo(w)
		if err != nil {
			return n, err
		}
		n += int64(written)
	}
	return n, nil
}

//
// spec: https://github.com/RoaringBitmap/RoaringFormatSpec
//
func (ra *roaringArray) toBytes() ([]byte, error) {
	var buf bytes.Buffer
	_, err := ra.writeTo(&buf)
	return buf.Bytes(), err
}

func (ra *roaringArray) readFrom(stream byteInput) (int64, error) {
	cookie, err := stream.readUInt32()

	if err != nil {
		return stream.getReadBytes(), fmt.Errorf("error in roaringArray.readFrom: could not read initial cookie: %s", err)
	}

	var size uint32
	var isRunBitmap []byte

	if cookie&0x0000FFFF == serialCookie {
		size = uint32(uint16(cookie>>16) + 1)
		// create is-run-container bitmap
		isRunBitmapSize := (int(size) + 7) / 8
		isRunBitmap, err = stream.next(isRunBitmapSize)

		if err != nil {
			return stream.getReadBytes(), fmt.Errorf("malformed bitmap, failed to read is-run bitmap, got: %s", err)
		}
	} else if cookie == serialCookieNoRunContainer {
		size, err = stream.readUInt32()

		if err != nil {
			return stream.getReadBytes(), fmt.Errorf("malformed bitmap, failed to read a bitmap size: %s", err)
		}
	} else {
		return stream.getReadBytes(), fmt.Errorf("error in roaringArray.readFrom: did not find expected serialCookie in header")
	}

	if size > (1 << 16) {
		return stream.getReadBytes(), fmt.Errorf("it is logically impossible to have more than (1<<16) containers")
	}

	// descriptive header
	buf, err := stream.next(2 * 2 * int(size))

	if err != nil {
		return stream.getReadBytes(), fmt.Errorf("failed to read descriptive header: %s", err)
	}

	keycard := byteSliceAsUint16Slice(buf)

	if isRunBitmap == nil || size >= noOffsetThreshold {
		if err := stream.skipBytes(int(size) * 4); err != nil {
			return stream.getReadBytes(), fmt.Errorf("failed to skip bytes: %s", err)
		}
	}

	// Allocate slices upfront as number of containers is known
	if cap(ra.containers) >= int(size) {
		ra.containers = ra.containers[:size]
	} else {
		ra.containers = make([]container, size)
	}

	if cap(ra.keys) >= int(size) {
		ra.keys = ra.keys[:size]
	} else {
		ra.keys = make([]uint16, size)
	}

	if cap(ra.needCopyOnWrite) >= int(size) {
		ra.needCopyOnWrite = ra.needCopyOnWrite[:size]
	} else {
		ra.needCopyOnWrite = make([]bool, size)
	}

	for i := uint32(0); i < size; i++ {
		key := keycard[2*i]
		card := int(keycard[2*i+1]) + 1
		ra.keys[i] = key
		ra.needCopyOnWrite[i] = true

		if isRunBitmap != nil && isRunBitmap[i/8]&(1<<(i%8)) != 0 {
			// run container
			nr, err := stream.readUInt16()

			if err != nil {
				return 0, fmt.Errorf("failed to read runtime container size: %s", err)
			}

			buf, err := stream.next(int(nr) * 4)

			if err != nil {
				return stream.getReadBytes(), fmt.Errorf("failed to read runtime container content: %s", err)
			}

			nb := runContainer16{
				iv:   byteSliceAsInterval16Slice(buf),
				card: int64(card),
			}

			ra.containers[i] = &nb
		} else if card > arrayDefaultMaxSize {
			// bitmap container
			buf, err := stream.next(arrayDefaultMaxSize * 2)

			if err != nil {
				return stream.getReadBytes(), fmt.Errorf("failed to read bitmap container: %s", err)
			}

			nb := bitmapContainer{
				cardinality: card,
				bitmap:      byteSliceAsUint64Slice(buf),
			}

			ra.containers[i] = &nb
		} else {
			// array container
			buf, err := stream.next(card * 2)

			if err != nil {
				return stream.getReadBytes(), fmt.Errorf("failed to read array container: %s", err)
			}

			nb := arrayContainer{
				byteSliceAsUint16Slice(buf),
			}

			ra.containers[i] = &nb
		}
	}

	return stream.getReadBytes(), nil
}

func (ra *roaringArray) hasRunCompression() bool {
	for _, c := range ra.containers {
		switch c.(type) {
		case *runContainer16:
			return true
		}
	}
	return false
}

func (ra *roaringArray) writeToMsgpack(stream io.Writer) error {

	ra.conserz = make([]containerSerz, len(ra.containers))
	for i, v := range ra.containers {
		switch cn := v.(type) {
		case *bitmapContainer:
			bts, err := cn.MarshalMsg(nil)
			if err != nil {
				return err
			}
			ra.conserz[i].t = bitmapContype
			ra.conserz[i].r = bts
		case *arrayContainer:
			bts, err := cn.MarshalMsg(nil)
			if err != nil {
				return err
			}
			ra.conserz[i].t = arrayContype
			ra.conserz[i].r = bts
		case *runContainer16:
			bts, err := cn.MarshalMsg(nil)
			if err != nil {
				return err
			}
			ra.conserz[i].t = run16Contype
			ra.conserz[i].r = bts
		default:
			panic(fmt.Errorf("Unrecognized container implementation: %T", cn))
		}
	}
	w := snappy.NewWriter(stream)
	err := msgp.Encode(w, ra)
	ra.conserz = nil
	return err
}

func (ra *roaringArray) readFromMsgpack(stream io.Reader) error {
	r := snappy.NewReader(stream)
	err := msgp.Decode(r, ra)
	if err != nil {
		return err
	}

	if len(ra.containers) != len(ra.keys) {
		ra.containers = make([]container, len(ra.keys))
	}

	for i, v := range ra.conserz {
		switch v.t {
		case bitmapContype:
			c := &bitmapContainer{}
			_, err = c.UnmarshalMsg(v.r)
			if err != nil {
				return err
			}
			ra.containers[i] = c
		case arrayContype:
			c := &arrayContainer{}
			_, err = c.UnmarshalMsg(v.r)
			if err != nil {
				return err
			}
			ra.containers[i] = c
		case run16Contype:
			c := &runContainer16{}
			_, err = c.UnmarshalMsg(v.r)
			if err != nil {
				return err
			}
			ra.containers[i] = c
		default:
			return fmt.Errorf("unrecognized contype serialization code: '%v'", v.t)
		}
	}
	ra.conserz = nil
	return nil
}

func (ra *roaringArray) advanceUntil(min uint16, pos int) int {
	lower := pos + 1

	if lower >= len(ra.keys) || ra.keys[lower] >= min {
		return lower
	}

	spansize := 1

	for lower+spansize < len(ra.keys) && ra.keys[lower+spansize] < min {
		spansize *= 2
	}
	var upper int
	if lower+spansize < len(ra.keys) {
		upper = lower + spansize
	} else {
		upper = len(ra.keys) - 1
	}

	if ra.keys[upper] == min {
		return upper
	}

	if ra.keys[upper] < min {
		// means
		// array
		// has no
		// item
		// >= min
		// pos = array.length;
		return len(ra.keys)
	}

	// we know that the next-smallest span was too small
	lower += (spansize >> 1)

	mid := 0
	for lower+1 != upper {
		mid = (lower + upper) >> 1
		if ra.keys[mid] == min {
			return mid
		} else if ra.keys[mid] < min {
			lower = mid
		} else {
			upper = mid
		}
	}
	return upper
}

func (ra *roaringArray) markAllAsNeedingCopyOnWrite() {
	for i := range ra.needCopyOnWrite {
		ra.needCopyOnWrite[i] = true
	}
}

func (ra *roaringArray) needsCopyOnWrite(i int) bool {
	return ra.needCopyOnWrite[i]
}

func (ra *roaringArray) setNeedsCopyOnWrite(i int) {
	ra.needCopyOnWrite[i] = true
}
