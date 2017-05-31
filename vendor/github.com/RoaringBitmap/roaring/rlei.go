package roaring

///////////////////////////////////////////////////
//
// container interface methods for runContainer16
//
///////////////////////////////////////////////////

import (
	"fmt"
)

// compile time verify we meet interface requirements
var _ container = &runContainer16{}

func (rc *runContainer16) clone() container {
	return newRunContainer16CopyIv(rc.iv)
}

func (rc *runContainer16) minimum() uint16 {
	return rc.iv[0].start // assume not empty
}

func (rc *runContainer16) maximum() uint16 {
	return rc.iv[len(rc.iv)-1].last // assume not empty
}

func (rc *runContainer16) isFull() bool {
	return (len(rc.iv) == 1) && ((rc.iv[0].start == 0) && (rc.iv[0].last == MaxUint16))
}

func (rc *runContainer16) and(a container) container {
	if rc.isFull() {
		return a.clone()
	}
	switch c := a.(type) {
	case *runContainer16:
		return rc.intersect(c)
	case *arrayContainer:
		return rc.andArray(c)
	case *bitmapContainer:
		return rc.andBitmapContainer(c)
	}
	panic("unsupported container type")
}

func (rc *runContainer16) andCardinality(a container) int {
	switch c := a.(type) {
	case *runContainer16:
		return int(rc.intersectCardinality(c))
	case *arrayContainer:
		return rc.andArrayCardinality(c)
	case *bitmapContainer:
		return rc.andBitmapContainerCardinality(c)
	}
	panic("unsupported container type")
}

// andBitmapContainer finds the intersection of rc and b.
func (rc *runContainer16) andBitmapContainer(bc *bitmapContainer) container {
	bc2 := newBitmapContainerFromRun(rc)
	return bc2.andBitmap(bc)
}

func (rc *runContainer16) andArray(ac *arrayContainer) container {
	bc1 := ac.toBitmapContainer()
	bc2 := newBitmapContainerFromRun(rc)
	return bc2.andBitmap(bc1)
}

func (rc *runContainer16) andArrayCardinality(ac *arrayContainer) int {
	pos := 0
	answer := 0
	maxpos := ac.getCardinality()
	if maxpos == 0 {
		return 0 // won't happen in actual code
	}
	v := ac.content[pos]
mainloop:
	for _, p := range rc.iv {
		for v < p.start {
			pos += 1
			if pos == maxpos {
				break mainloop
			}
			v = ac.content[pos]
		}
		for v <= p.last {
			answer += 1
			pos += 1
			if pos == maxpos {
				break mainloop
			}
			v = ac.content[pos]
		}
	}
	return answer
}

func (rc *runContainer16) iand(a container) container {
	if rc.isFull() {
		return a.clone()
	}
	switch c := a.(type) {
	case *runContainer16:
		return rc.inplaceIntersect(c)
	case *arrayContainer:
		return rc.iandArray(c)
	case *bitmapContainer:
		return rc.iandBitmapContainer(c)
	}
	panic("unsupported container type")
}

func (rc *runContainer16) inplaceIntersect(rc2 *runContainer16) container {
	// TODO: optimize by doing less allocation, possibly?

	// sect will be new
	sect := rc.intersect(rc2)
	*rc = *sect
	return rc
}

func (rc *runContainer16) iandBitmapContainer(bc *bitmapContainer) container {
	isect := rc.andBitmapContainer(bc)
	*rc = *newRunContainer16FromContainer(isect)
	return rc
}

func (rc *runContainer16) iandArray(ac *arrayContainer) container {

	bc1 := newBitmapContainerFromRun(rc)
	bc2 := ac.toBitmapContainer()
	and := bc1.andBitmap(bc2)
	var rc2 *runContainer16
	switch x := and.(type) {
	case *bitmapContainer:
		rc2 = newRunContainer16FromBitmapContainer(x)
	case *arrayContainer:
		rc2 = newRunContainer16FromArray(x)
	case *runContainer16:
		rc2 = x
	default:
		panic("unknown container type")
	}
	*rc = *rc2
	return rc

	/*
		// TODO: optimize by doing less allocation, possibly?
		out := newRunContainer16()
		for _, p := range rc.iv {
			for i := p.start; i <= p.last; i++ {
				if ac.contains(i) {
					out.Add(i)
				}
			}
		}
		*rc = *out
		return rc
	*/
}

func (rc *runContainer16) andNot(a container) container {
	switch c := a.(type) {
	case *arrayContainer:
		return rc.andNotArray(c)
	case *bitmapContainer:
		return rc.andNotBitmap(c)
	case *runContainer16:
		return rc.andNotRunContainer16(c)
	}
	panic("unsupported container type")
}

func (rc *runContainer16) fillLeastSignificant16bits(x []uint32, i int, mask uint32) {
	k := 0
	var val int64
	for _, p := range rc.iv {
		n := p.runlen()
		for j := int64(0); j < n; j++ {
			val = int64(p.start) + j
			x[k+i] = uint32(val) | mask
			k++
		}
	}
}

func (rc *runContainer16) getShortIterator() shortIterable {
	return rc.newRunIterator16()
}

// add the values in the range [firstOfRange, endx). endx
// is still abe to express 2^16 because it is an int not an uint16.
func (rc *runContainer16) iaddRange(firstOfRange, endx int) container {

	if firstOfRange >= endx {
		panic(fmt.Sprintf("invalid %v = endx >= firstOfRange", endx))
	}
	addme := newRunContainer16TakeOwnership([]interval16{
		{
			start: uint16(firstOfRange),
			last:  uint16(endx - 1),
		},
	})
	*rc = *rc.union(addme)
	return rc
}

// remove the values in the range [firstOfRange,endx)
func (rc *runContainer16) iremoveRange(firstOfRange, endx int) container {
	if firstOfRange >= endx {
		panic(fmt.Sprintf("request to iremove empty set [%v, %v),"+
			" nothing to do.", firstOfRange, endx))
		//return rc
	}
	x := interval16{start: uint16(firstOfRange), last: uint16(endx - 1)}
	rc.isubtract(x)
	return rc
}

// not flip the values in the range [firstOfRange,endx)
func (rc *runContainer16) not(firstOfRange, endx int) container {
	if firstOfRange >= endx {
		panic(fmt.Sprintf("invalid %v = endx >= firstOfRange = %v", endx, firstOfRange))
	}

	return rc.Not(firstOfRange, endx)
}

// Not flips the values in the range [firstOfRange,endx).
// This is not inplace. Only the returned value has the flipped bits.
//
// Currently implemented as (!A intersect B) union (A minus B),
// where A is rc, and B is the supplied [firstOfRange, endx) interval.
//
// TODO(time optimization): convert this to a single pass
// algorithm by copying AndNotRunContainer16() and modifying it.
// Current routine is correct but
// makes 2 more passes through the arrays than should be
// strictly necessary. Measure both ways though--this may not matter.
//
func (rc *runContainer16) Not(firstOfRange, endx int) *runContainer16 {

	if firstOfRange >= endx {
		panic(fmt.Sprintf("invalid %v = endx >= firstOfRange == %v", endx, firstOfRange))
	}

	if firstOfRange >= endx {
		return rc.Clone()
	}

	a := rc
	// algo:
	// (!A intersect B) union (A minus B)

	nota := a.invert()

	bs := []interval16{{start: uint16(firstOfRange), last: uint16(endx - 1)}}
	b := newRunContainer16TakeOwnership(bs)

	notAintersectB := nota.intersect(b)

	aMinusB := a.AndNotRunContainer16(b)

	rc2 := notAintersectB.union(aMinusB)
	return rc2
}

// equals is now logical equals; it does not require the
// same underlying container type.
func (rc *runContainer16) equals(o container) bool {
	srb, ok := o.(*runContainer16)

	if !ok {
		// maybe value instead of pointer
		val, valok := o.(*runContainer16)
		if valok {
			srb = val
			ok = true
		}
	}
	if ok {
		// Check if the containers are the same object.
		if rc == srb {
			return true
		}

		if len(srb.iv) != len(rc.iv) {
			return false
		}

		for i, v := range rc.iv {
			if v != srb.iv[i] {
				return false
			}
		}
		return true
	}

	// use generic comparison
	if o.getCardinality() != rc.getCardinality() {
		return false
	}
	rit := rc.getShortIterator()
	bit := o.getShortIterator()

	//k := 0
	for rit.hasNext() {
		if bit.next() != rit.next() {
			return false
		}
		//k++
	}
	return true
}

func (rc *runContainer16) iaddReturnMinimized(x uint16) container {
	rc.Add(x)
	return rc
}

func (rc *runContainer16) iadd(x uint16) (wasNew bool) {
	return rc.Add(x)
}

func (rc *runContainer16) iremoveReturnMinimized(x uint16) container {
	rc.removeKey(x)
	return rc
}

func (rc *runContainer16) iremove(x uint16) bool {
	return rc.removeKey(x)
}

func (rc *runContainer16) or(a container) container {
	if rc.isFull() {
		return rc.clone()
	}
	switch c := a.(type) {
	case *runContainer16:
		return rc.union(c)
	case *arrayContainer:
		return rc.orArray(c)
	case *bitmapContainer:
		return rc.orBitmapContainer(c)
	}
	panic("unsupported container type")
}

func (rc *runContainer16) orCardinality(a container) int {
	switch c := a.(type) {
	case *runContainer16:
		return int(rc.unionCardinality(c))
	case *arrayContainer:
		return rc.orArrayCardinality(c)
	case *bitmapContainer:
		return rc.orBitmapContainerCardinality(c)
	}
	panic("unsupported container type")
}

// orBitmapContainer finds the union of rc and bc.
func (rc *runContainer16) orBitmapContainer(bc *bitmapContainer) container {
	bc2 := newBitmapContainerFromRun(rc)
	return bc2.iorBitmap(bc)
}

func (rc *runContainer16) andBitmapContainerCardinality(bc *bitmapContainer) int {
	answer := 0
	for i := range rc.iv {
		answer += bc.getCardinalityInRange(uint(rc.iv[i].start), uint(rc.iv[i].last)+1)
	}
	//bc.computeCardinality()
	return answer
}

func (rc *runContainer16) orBitmapContainerCardinality(bc *bitmapContainer) int {
	return rc.getCardinality() + bc.getCardinality() - rc.andBitmapContainerCardinality(bc)
}

// orArray finds the union of rc and ac.
func (rc *runContainer16) orArray(ac *arrayContainer) container {
	bc1 := newBitmapContainerFromRun(rc)
	bc2 := ac.toBitmapContainer()
	return bc1.orBitmap(bc2)
}

// orArray finds the union of rc and ac.
func (rc *runContainer16) orArrayCardinality(ac *arrayContainer) int {
	return ac.getCardinality() + rc.getCardinality() - rc.andArrayCardinality(ac)
}

func (rc *runContainer16) ior(a container) container {
	if rc.isFull() {
		return rc
	}
	switch c := a.(type) {
	case *runContainer16:
		return rc.inplaceUnion(c)
	case *arrayContainer:
		return rc.iorArray(c)
	case *bitmapContainer:
		return rc.iorBitmapContainer(c)
	}
	panic("unsupported container type")
}

func (rc *runContainer16) inplaceUnion(rc2 *runContainer16) container {
	p("rc.inplaceUnion with len(rc2.iv)=%v", len(rc2.iv))
	for _, p := range rc2.iv {
		last := int64(p.last)
		for i := int64(p.start); i <= last; i++ {
			rc.Add(uint16(i))
		}
	}
	return rc
}

func (rc *runContainer16) iorBitmapContainer(bc *bitmapContainer) container {

	it := bc.getShortIterator()
	for it.hasNext() {
		rc.Add(it.next())
	}
	return rc
}

func (rc *runContainer16) iorArray(ac *arrayContainer) container {
	it := ac.getShortIterator()
	for it.hasNext() {
		rc.Add(it.next())
	}
	return rc
}

// lazyIOR is described (not yet implemented) in
// this nice note from @lemire on
// https://github.com/RoaringBitmap/roaring/pull/70#issuecomment-263613737
//
// Description of lazyOR and lazyIOR from @lemire:
//
// Lazy functions are optional and can be simply
// wrapper around non-lazy functions.
//
// The idea of "laziness" is as follows. It is
// inspired by the concept of lazy evaluation
// you might be familiar with (functional programming
// and all that). So a roaring bitmap is
// such that all its containers are, in some
// sense, chosen to use as little memory as
// possible. This is nice. Also, all bitsets
// are "cardinality aware" so that you can do
// fast rank/select queries, or query the
// cardinality of the whole bitmap... very fast,
// without latency.
//
// However, imagine that you are aggregating 100
// bitmaps together. So you OR the first two, then OR
// that with the third one and so forth. Clearly,
// intermediate bitmaps don't need to be as
// compressed as possible, right? They can be
// in a "dirty state". You only need the end
// result to be in a nice state... which you
// can achieve by calling repairAfterLazy at the end.
//
// The Java/C code does something special for
// the in-place lazy OR runs. The idea is that
// instead of taking two run containers and
// generating a new one, we actually try to
// do the computation in-place through a
// technique invented by @gssiyankai (pinging him!).
// What you do is you check whether the host
// run container has lots of extra capacity.
// If it does, you move its data at the end of
// the backing array, and then you write
// the answer at the beginning. What this
// trick does is minimize memory allocations.
//
func (rc *runContainer16) lazyIOR(a container) container {
	// not lazy at the moment
	// TODO: make it lazy
	return rc.ior(a)

	/*
		switch c := a.(type) {
		case *arrayContainer:
			return rc.lazyIorArray(c)
		case *bitmapContainer:
			return rc.lazyIorBitmap(c)
		case *runContainer16:
			return rc.lazyIorRun16(c)
		}
		panic("unsupported container type")
	*/
}

// lazyOR is described above in lazyIOR.
func (rc *runContainer16) lazyOR(a container) container {

	// not lazy at the moment
	// TODO: make it lazy
	return rc.or(a)

	/*
		switch c := a.(type) {
		case *arrayContainer:
			return rc.lazyOrArray(c)
		case *bitmapContainer:
			return rc.lazyOrBitmap(c)
		case *runContainer16:
			return rc.lazyOrRunContainer16(c)
		}
		panic("unsupported container type")
	*/
}

func (rc *runContainer16) intersects(a container) bool {
	// TODO: optimize by doing inplace/less allocation, possibly?
	isect := rc.and(a)
	return isect.getCardinality() > 0
}

func (rc *runContainer16) xor(a container) container {
	switch c := a.(type) {
	case *arrayContainer:
		return rc.xorArray(c)
	case *bitmapContainer:
		return rc.xorBitmap(c)
	case *runContainer16:
		return rc.xorRunContainer16(c)
	}
	panic("unsupported container type")
}

func (rc *runContainer16) iandNot(a container) container {
	switch c := a.(type) {
	case *arrayContainer:
		return rc.iandNotArray(c)
	case *bitmapContainer:
		return rc.iandNotBitmap(c)
	case *runContainer16:
		return rc.iandNotRunContainer16(c)
	}
	panic("unsupported container type")
}

// flip the values in the range [firstOfRange,endx)
func (rc *runContainer16) inot(firstOfRange, endx int) container {
	if firstOfRange >= endx {
		panic(fmt.Sprintf("invalid %v = endx >= firstOfRange = %v", endx, firstOfRange))
	}
	// TODO: minimize copies, do it all inplace; not() makes a copy.
	rc = rc.Not(firstOfRange, endx)
	return rc
}

func (rc *runContainer16) getCardinality() int {
	return int(rc.cardinality())
}

func (rc *runContainer16) rank(x uint16) int {
	n := int64(len(rc.iv))
	xx := int64(x)
	w, already, _ := rc.search(xx, nil)
	if w < 0 {
		return 0
	}
	if !already && w == n-1 {
		return rc.getCardinality()
	}
	var rnk int64
	if !already {
		for i := int64(0); i <= w; i++ {
			rnk += rc.iv[i].runlen()
		}
		return int(rnk)
	}
	for i := int64(0); i < w; i++ {
		rnk += rc.iv[i].runlen()
	}
	rnk += int64(x-rc.iv[w].start) + 1
	return int(rnk)
}

func (rc *runContainer16) selectInt(x uint16) int {
	return rc.selectInt16(x)
}

func (rc *runContainer16) andNotRunContainer16(b *runContainer16) container {
	return rc.AndNotRunContainer16(b)
}

func (rc *runContainer16) andNotArray(ac *arrayContainer) container {
	rcb := rc.toBitmapContainer()
	acb := ac.toBitmapContainer()
	return rcb.andNotBitmap(acb)
}

func (rc *runContainer16) andNotBitmap(bc *bitmapContainer) container {
	rcb := rc.toBitmapContainer()
	return rcb.andNotBitmap(bc)
}

func (rc *runContainer16) toBitmapContainer() *bitmapContainer {
	p("run16 toBitmap starting; rc has %v ranges", len(rc.iv))
	bc := newBitmapContainer()
	for i := range rc.iv {
		bc.iaddRange(int(rc.iv[i].start), int(rc.iv[i].last)+1)
	}
	bc.computeCardinality()
	return bc
}

func (rc *runContainer16) iandNotRunContainer16(x2 *runContainer16) container {
	rcb := rc.toBitmapContainer()
	x2b := x2.toBitmapContainer()
	rcb.iandNotBitmapSurely(x2b)
	// TODO: check size and optimize the return value
	// TODO: is inplace modification really required? If not, elide the copy.
	rc2 := newRunContainer16FromBitmapContainer(rcb)
	*rc = *rc2
	return rc
}

func (rc *runContainer16) iandNotArray(ac *arrayContainer) container {
	rcb := rc.toBitmapContainer()
	acb := ac.toBitmapContainer()
	rcb.iandNotBitmapSurely(acb)
	// TODO: check size and optimize the return value
	// TODO: is inplace modification really required? If not, elide the copy.
	rc2 := newRunContainer16FromBitmapContainer(rcb)
	*rc = *rc2
	return rc
}

func (rc *runContainer16) iandNotBitmap(bc *bitmapContainer) container {
	rcb := rc.toBitmapContainer()
	rcb.iandNotBitmapSurely(bc)
	// TODO: check size and optimize the return value
	// TODO: is inplace modification really required? If not, elide the copy.
	rc2 := newRunContainer16FromBitmapContainer(rcb)
	*rc = *rc2
	return rc
}

func (rc *runContainer16) xorRunContainer16(x2 *runContainer16) container {
	rcb := rc.toBitmapContainer()
	x2b := x2.toBitmapContainer()
	return rcb.xorBitmap(x2b)
}

func (rc *runContainer16) xorArray(ac *arrayContainer) container {
	rcb := rc.toBitmapContainer()
	acb := ac.toBitmapContainer()
	return rcb.xorBitmap(acb)
}

func (rc *runContainer16) xorBitmap(bc *bitmapContainer) container {
	rcb := rc.toBitmapContainer()
	return rcb.xorBitmap(bc)
}

// convert to bitmap or array *if needed*
func (rc *runContainer16) toEfficientContainer() container {

	// runContainer16SerializedSizeInBytes(numRuns)
	sizeAsRunContainer := rc.getSizeInBytes()
	sizeAsBitmapContainer := bitmapContainerSizeInBytes()
	card := int(rc.cardinality())
	sizeAsArrayContainer := arrayContainerSizeInBytes(card)
	if sizeAsRunContainer <= min(sizeAsBitmapContainer, sizeAsArrayContainer) {
		return rc
	}
	if card <= arrayDefaultMaxSize {
		return rc.toArrayContainer()
	}
	bc := newBitmapContainerFromRun(rc)
	return bc
}

func (rc *runContainer16) toArrayContainer() *arrayContainer {
	ac := newArrayContainer()
	for i := range rc.iv {
		ac.iaddRange(int(rc.iv[i].start), int(rc.iv[i].last)+1)
	}
	return ac
}

func newRunContainer16FromContainer(c container) *runContainer16 {

	switch x := c.(type) {
	case *runContainer16:
		return x.Clone()
	case *arrayContainer:
		return newRunContainer16FromArray(x)
	case *bitmapContainer:
		return newRunContainer16FromBitmapContainer(x)
	}
	panic("unsupported container type")
}
