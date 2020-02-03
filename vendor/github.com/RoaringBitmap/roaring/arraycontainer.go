package roaring

import (
	"fmt"
)

//go:generate msgp -unexported

type arrayContainer struct {
	content []uint16
}

func (ac *arrayContainer) String() string {
	s := "{"
	for it := ac.getShortIterator(); it.hasNext(); {
		s += fmt.Sprintf("%v, ", it.next())
	}
	return s + "}"
}

func (ac *arrayContainer) fillLeastSignificant16bits(x []uint32, i int, mask uint32) {
	for k := 0; k < len(ac.content); k++ {
		x[k+i] = uint32(ac.content[k]) | mask
	}
}

func (ac *arrayContainer) getShortIterator() shortPeekable {
	return &shortIterator{ac.content, 0}
}

func (ac *arrayContainer) getReverseIterator() shortIterable {
	return &reverseIterator{ac.content, len(ac.content) - 1}
}

func (ac *arrayContainer) getManyIterator() manyIterable {
	return &shortIterator{ac.content, 0}
}

func (ac *arrayContainer) minimum() uint16 {
	return ac.content[0] // assume not empty
}

func (ac *arrayContainer) maximum() uint16 {
	return ac.content[len(ac.content)-1] // assume not empty
}

func (ac *arrayContainer) getSizeInBytes() int {
	return ac.getCardinality() * 2
}

func (ac *arrayContainer) serializedSizeInBytes() int {
	return ac.getCardinality() * 2
}

func arrayContainerSizeInBytes(card int) int {
	return card * 2
}

// add the values in the range [firstOfRange,endx)
func (ac *arrayContainer) iaddRange(firstOfRange, endx int) container {
	if firstOfRange >= endx {
		return ac
	}
	indexstart := binarySearch(ac.content, uint16(firstOfRange))
	if indexstart < 0 {
		indexstart = -indexstart - 1
	}
	indexend := binarySearch(ac.content, uint16(endx-1))
	if indexend < 0 {
		indexend = -indexend - 1
	} else {
		indexend++
	}
	rangelength := endx - firstOfRange
	newcardinality := indexstart + (ac.getCardinality() - indexend) + rangelength
	if newcardinality > arrayDefaultMaxSize {
		a := ac.toBitmapContainer()
		return a.iaddRange(firstOfRange, endx)
	}
	if cap(ac.content) < newcardinality {
		tmp := make([]uint16, newcardinality, newcardinality)
		copy(tmp[:indexstart], ac.content[:indexstart])
		copy(tmp[indexstart+rangelength:], ac.content[indexend:])

		ac.content = tmp
	} else {
		ac.content = ac.content[:newcardinality]
		copy(ac.content[indexstart+rangelength:], ac.content[indexend:])

	}
	for k := 0; k < rangelength; k++ {
		ac.content[k+indexstart] = uint16(firstOfRange + k)
	}
	return ac
}

// remove the values in the range [firstOfRange,endx)
func (ac *arrayContainer) iremoveRange(firstOfRange, endx int) container {
	if firstOfRange >= endx {
		return ac
	}
	indexstart := binarySearch(ac.content, uint16(firstOfRange))
	if indexstart < 0 {
		indexstart = -indexstart - 1
	}
	indexend := binarySearch(ac.content, uint16(endx-1))
	if indexend < 0 {
		indexend = -indexend - 1
	} else {
		indexend++
	}
	rangelength := indexend - indexstart
	answer := ac
	copy(answer.content[indexstart:], ac.content[indexstart+rangelength:])
	answer.content = answer.content[:ac.getCardinality()-rangelength]
	return answer
}

// flip the values in the range [firstOfRange,endx)
func (ac *arrayContainer) not(firstOfRange, endx int) container {
	if firstOfRange >= endx {
		return ac.clone()
	}
	return ac.notClose(firstOfRange, endx-1) // remove everything in [firstOfRange,endx-1]
}

// flip the values in the range [firstOfRange,lastOfRange]
func (ac *arrayContainer) notClose(firstOfRange, lastOfRange int) container {
	if firstOfRange > lastOfRange { // unlike add and remove, not uses an inclusive range [firstOfRange,lastOfRange]
		return ac.clone()
	}

	// determine the span of array indices to be affected^M
	startIndex := binarySearch(ac.content, uint16(firstOfRange))
	if startIndex < 0 {
		startIndex = -startIndex - 1
	}
	lastIndex := binarySearch(ac.content, uint16(lastOfRange))
	if lastIndex < 0 {
		lastIndex = -lastIndex - 2
	}
	currentValuesInRange := lastIndex - startIndex + 1
	spanToBeFlipped := lastOfRange - firstOfRange + 1
	newValuesInRange := spanToBeFlipped - currentValuesInRange
	cardinalityChange := newValuesInRange - currentValuesInRange
	newCardinality := len(ac.content) + cardinalityChange
	if newCardinality > arrayDefaultMaxSize {
		return ac.toBitmapContainer().not(firstOfRange, lastOfRange+1)
	}
	answer := newArrayContainer()
	answer.content = make([]uint16, newCardinality, newCardinality) //a hack for sure

	copy(answer.content, ac.content[:startIndex])
	outPos := startIndex
	inPos := startIndex
	valInRange := firstOfRange
	for ; valInRange <= lastOfRange && inPos <= lastIndex; valInRange++ {
		if uint16(valInRange) != ac.content[inPos] {
			answer.content[outPos] = uint16(valInRange)
			outPos++
		} else {
			inPos++
		}
	}

	for ; valInRange <= lastOfRange; valInRange++ {
		answer.content[outPos] = uint16(valInRange)
		outPos++
	}

	for i := lastIndex + 1; i < len(ac.content); i++ {
		answer.content[outPos] = ac.content[i]
		outPos++
	}
	answer.content = answer.content[:newCardinality]
	return answer

}

func (ac *arrayContainer) equals(o container) bool {

	srb, ok := o.(*arrayContainer)
	if ok {
		// Check if the containers are the same object.
		if ac == srb {
			return true
		}

		if len(srb.content) != len(ac.content) {
			return false
		}

		for i, v := range ac.content {
			if v != srb.content[i] {
				return false
			}
		}
		return true
	}

	// use generic comparison
	bCard := o.getCardinality()
	aCard := ac.getCardinality()
	if bCard != aCard {
		return false
	}

	ait := ac.getShortIterator()
	bit := o.getShortIterator()
	for ait.hasNext() {
		if bit.next() != ait.next() {
			return false
		}
	}
	return true
}

func (ac *arrayContainer) toBitmapContainer() *bitmapContainer {
	bc := newBitmapContainer()
	bc.loadData(ac)
	return bc

}
func (ac *arrayContainer) iadd(x uint16) (wasNew bool) {
	// Special case adding to the end of the container.
	l := len(ac.content)
	if l > 0 && l < arrayDefaultMaxSize && ac.content[l-1] < x {
		ac.content = append(ac.content, x)
		return true
	}

	loc := binarySearch(ac.content, x)

	if loc < 0 {
		s := ac.content
		i := -loc - 1
		s = append(s, 0)
		copy(s[i+1:], s[i:])
		s[i] = x
		ac.content = s
		return true
	}
	return false
}

func (ac *arrayContainer) iaddReturnMinimized(x uint16) container {
	// Special case adding to the end of the container.
	l := len(ac.content)
	if l > 0 && l < arrayDefaultMaxSize && ac.content[l-1] < x {
		ac.content = append(ac.content, x)
		return ac
	}

	loc := binarySearch(ac.content, x)

	if loc < 0 {
		if len(ac.content) >= arrayDefaultMaxSize {
			a := ac.toBitmapContainer()
			a.iadd(x)
			return a
		}
		s := ac.content
		i := -loc - 1
		s = append(s, 0)
		copy(s[i+1:], s[i:])
		s[i] = x
		ac.content = s
	}
	return ac
}

// iremoveReturnMinimized is allowed to change the return type to minimize storage.
func (ac *arrayContainer) iremoveReturnMinimized(x uint16) container {
	ac.iremove(x)
	return ac
}

func (ac *arrayContainer) iremove(x uint16) bool {
	loc := binarySearch(ac.content, x)
	if loc >= 0 {
		s := ac.content
		s = append(s[:loc], s[loc+1:]...)
		ac.content = s
		return true
	}
	return false
}

func (ac *arrayContainer) remove(x uint16) container {
	out := &arrayContainer{make([]uint16, len(ac.content))}
	copy(out.content, ac.content[:])

	loc := binarySearch(out.content, x)
	if loc >= 0 {
		s := out.content
		s = append(s[:loc], s[loc+1:]...)
		out.content = s
	}
	return out
}

func (ac *arrayContainer) or(a container) container {
	switch x := a.(type) {
	case *arrayContainer:
		return ac.orArray(x)
	case *bitmapContainer:
		return x.orArray(ac)
	case *runContainer16:
		if x.isFull() {
			return x.clone()
		}
		return x.orArray(ac)
	}
	panic("unsupported container type")
}

func (ac *arrayContainer) orCardinality(a container) int {
	switch x := a.(type) {
	case *arrayContainer:
		return ac.orArrayCardinality(x)
	case *bitmapContainer:
		return x.orArrayCardinality(ac)
	case *runContainer16:
		return x.orArrayCardinality(ac)
	}
	panic("unsupported container type")
}

func (ac *arrayContainer) ior(a container) container {
	switch x := a.(type) {
	case *arrayContainer:
		return ac.iorArray(x)
	case *bitmapContainer:
		return a.(*bitmapContainer).orArray(ac)
		//return ac.iorBitmap(x) // note: this does not make sense
	case *runContainer16:
		if x.isFull() {
			return x.clone()
		}
		return ac.iorRun16(x)
	}
	panic("unsupported container type")
}

func (ac *arrayContainer) iorArray(value2 *arrayContainer) container {
	value1 := ac
	len1 := value1.getCardinality()
	len2 := value2.getCardinality()
	maxPossibleCardinality := len1 + len2
	if maxPossibleCardinality > arrayDefaultMaxSize { // it could be a bitmap!
		bc := newBitmapContainer()
		for k := 0; k < len(value2.content); k++ {
			v := value2.content[k]
			i := uint(v) >> 6
			mask := uint64(1) << (v % 64)
			bc.bitmap[i] |= mask
		}
		for k := 0; k < len(ac.content); k++ {
			v := ac.content[k]
			i := uint(v) >> 6
			mask := uint64(1) << (v % 64)
			bc.bitmap[i] |= mask
		}
		bc.cardinality = int(popcntSlice(bc.bitmap))
		if bc.cardinality <= arrayDefaultMaxSize {
			return bc.toArrayContainer()
		}
		return bc
	}
	if maxPossibleCardinality > cap(value1.content) {
		newcontent := make([]uint16, 0, maxPossibleCardinality)
		copy(newcontent[len2:maxPossibleCardinality], ac.content[0:len1])
		ac.content = newcontent
	} else {
		copy(ac.content[len2:maxPossibleCardinality], ac.content[0:len1])
	}
	nl := union2by2(value1.content[len2:maxPossibleCardinality], value2.content, ac.content)
	ac.content = ac.content[:nl] // reslice to match actual used capacity
	return ac
}

// Note: such code does not make practical sense, except for lazy evaluations
func (ac *arrayContainer) iorBitmap(bc2 *bitmapContainer) container {
	bc1 := ac.toBitmapContainer()
	bc1.iorBitmap(bc2)
	*ac = *newArrayContainerFromBitmap(bc1)
	return ac
}

func (ac *arrayContainer) iorRun16(rc *runContainer16) container {
	bc1 := ac.toBitmapContainer()
	bc2 := rc.toBitmapContainer()
	bc1.iorBitmap(bc2)
	*ac = *newArrayContainerFromBitmap(bc1)
	return ac
}

func (ac *arrayContainer) lazyIOR(a container) container {
	switch x := a.(type) {
	case *arrayContainer:
		return ac.lazyIorArray(x)
	case *bitmapContainer:
		return ac.lazyIorBitmap(x)
	case *runContainer16:
		if x.isFull() {
			return x.clone()
		}
		return ac.lazyIorRun16(x)

	}
	panic("unsupported container type")
}

func (ac *arrayContainer) lazyIorArray(ac2 *arrayContainer) container {
	// TODO actually make this lazy
	return ac.iorArray(ac2)
}

func (ac *arrayContainer) lazyIorBitmap(bc *bitmapContainer) container {
	// TODO actually make this lazy
	return ac.iorBitmap(bc)
}

func (ac *arrayContainer) lazyIorRun16(rc *runContainer16) container {
	// TODO actually make this lazy
	return ac.iorRun16(rc)
}

func (ac *arrayContainer) lazyOR(a container) container {
	switch x := a.(type) {
	case *arrayContainer:
		return ac.lazyorArray(x)
	case *bitmapContainer:
		return a.lazyOR(ac)
	case *runContainer16:
		if x.isFull() {
			return x.clone()
		}
		return x.orArray(ac)
	}
	panic("unsupported container type")
}

func (ac *arrayContainer) orArray(value2 *arrayContainer) container {
	value1 := ac
	maxPossibleCardinality := value1.getCardinality() + value2.getCardinality()
	if maxPossibleCardinality > arrayDefaultMaxSize { // it could be a bitmap!
		bc := newBitmapContainer()
		for k := 0; k < len(value2.content); k++ {
			v := value2.content[k]
			i := uint(v) >> 6
			mask := uint64(1) << (v % 64)
			bc.bitmap[i] |= mask
		}
		for k := 0; k < len(ac.content); k++ {
			v := ac.content[k]
			i := uint(v) >> 6
			mask := uint64(1) << (v % 64)
			bc.bitmap[i] |= mask
		}
		bc.cardinality = int(popcntSlice(bc.bitmap))
		if bc.cardinality <= arrayDefaultMaxSize {
			return bc.toArrayContainer()
		}
		return bc
	}
	answer := newArrayContainerCapacity(maxPossibleCardinality)
	nl := union2by2(value1.content, value2.content, answer.content)
	answer.content = answer.content[:nl] // reslice to match actual used capacity
	return answer
}

func (ac *arrayContainer) orArrayCardinality(value2 *arrayContainer) int {
	return union2by2Cardinality(ac.content, value2.content)
}

func (ac *arrayContainer) lazyorArray(value2 *arrayContainer) container {
	value1 := ac
	maxPossibleCardinality := value1.getCardinality() + value2.getCardinality()
	if maxPossibleCardinality > arrayLazyLowerBound { // it could be a bitmap!^M
		bc := newBitmapContainer()
		for k := 0; k < len(value2.content); k++ {
			v := value2.content[k]
			i := uint(v) >> 6
			mask := uint64(1) << (v % 64)
			bc.bitmap[i] |= mask
		}
		for k := 0; k < len(ac.content); k++ {
			v := ac.content[k]
			i := uint(v) >> 6
			mask := uint64(1) << (v % 64)
			bc.bitmap[i] |= mask
		}
		bc.cardinality = invalidCardinality
		return bc
	}
	answer := newArrayContainerCapacity(maxPossibleCardinality)
	nl := union2by2(value1.content, value2.content, answer.content)
	answer.content = answer.content[:nl] // reslice to match actual used capacity
	return answer
}

func (ac *arrayContainer) and(a container) container {
	switch x := a.(type) {
	case *arrayContainer:
		return ac.andArray(x)
	case *bitmapContainer:
		return x.and(ac)
	case *runContainer16:
		if x.isFull() {
			return ac.clone()
		}
		return x.andArray(ac)
	}
	panic("unsupported container type")
}

func (ac *arrayContainer) andCardinality(a container) int {
	switch x := a.(type) {
	case *arrayContainer:
		return ac.andArrayCardinality(x)
	case *bitmapContainer:
		return x.andCardinality(ac)
	case *runContainer16:
		return x.andArrayCardinality(ac)
	}
	panic("unsupported container type")
}

func (ac *arrayContainer) intersects(a container) bool {
	switch x := a.(type) {
	case *arrayContainer:
		return ac.intersectsArray(x)
	case *bitmapContainer:
		return x.intersects(ac)
	case *runContainer16:
		return x.intersects(ac)
	}
	panic("unsupported container type")
}

func (ac *arrayContainer) iand(a container) container {
	switch x := a.(type) {
	case *arrayContainer:
		return ac.iandArray(x)
	case *bitmapContainer:
		return ac.iandBitmap(x)
	case *runContainer16:
		if x.isFull() {
			return ac
		}
		return x.andArray(ac)
	}
	panic("unsupported container type")
}

func (ac *arrayContainer) iandBitmap(bc *bitmapContainer) container {
	pos := 0
	c := ac.getCardinality()
	for k := 0; k < c; k++ {
		// branchless
		v := ac.content[k]
		ac.content[pos] = v
		pos += int(bc.bitValue(v))
	}
	ac.content = ac.content[:pos]
	return ac

}

func (ac *arrayContainer) xor(a container) container {
	switch x := a.(type) {
	case *arrayContainer:
		return ac.xorArray(x)
	case *bitmapContainer:
		return a.xor(ac)
	case *runContainer16:
		return x.xorArray(ac)
	}
	panic("unsupported container type")
}

func (ac *arrayContainer) xorArray(value2 *arrayContainer) container {
	value1 := ac
	totalCardinality := value1.getCardinality() + value2.getCardinality()
	if totalCardinality > arrayDefaultMaxSize { // it could be a bitmap!
		bc := newBitmapContainer()
		for k := 0; k < len(value2.content); k++ {
			v := value2.content[k]
			i := uint(v) >> 6
			bc.bitmap[i] ^= (uint64(1) << (v % 64))
		}
		for k := 0; k < len(ac.content); k++ {
			v := ac.content[k]
			i := uint(v) >> 6
			bc.bitmap[i] ^= (uint64(1) << (v % 64))
		}
		bc.computeCardinality()
		if bc.cardinality <= arrayDefaultMaxSize {
			return bc.toArrayContainer()
		}
		return bc
	}
	desiredCapacity := totalCardinality
	answer := newArrayContainerCapacity(desiredCapacity)
	length := exclusiveUnion2by2(value1.content, value2.content, answer.content)
	answer.content = answer.content[:length]
	return answer

}

func (ac *arrayContainer) andNot(a container) container {
	switch x := a.(type) {
	case *arrayContainer:
		return ac.andNotArray(x)
	case *bitmapContainer:
		return ac.andNotBitmap(x)
	case *runContainer16:
		return ac.andNotRun16(x)
	}
	panic("unsupported container type")
}

func (ac *arrayContainer) andNotRun16(rc *runContainer16) container {
	acb := ac.toBitmapContainer()
	rcb := rc.toBitmapContainer()
	return acb.andNotBitmap(rcb)
}

func (ac *arrayContainer) iandNot(a container) container {
	switch x := a.(type) {
	case *arrayContainer:
		return ac.iandNotArray(x)
	case *bitmapContainer:
		return ac.iandNotBitmap(x)
	case *runContainer16:
		return ac.iandNotRun16(x)
	}
	panic("unsupported container type")
}

func (ac *arrayContainer) iandNotRun16(rc *runContainer16) container {
	rcb := rc.toBitmapContainer()
	acb := ac.toBitmapContainer()
	acb.iandNotBitmapSurely(rcb)
	*ac = *(acb.toArrayContainer())
	return ac
}

func (ac *arrayContainer) andNotArray(value2 *arrayContainer) container {
	value1 := ac
	desiredcapacity := value1.getCardinality()
	answer := newArrayContainerCapacity(desiredcapacity)
	length := difference(value1.content, value2.content, answer.content)
	answer.content = answer.content[:length]
	return answer
}

func (ac *arrayContainer) iandNotArray(value2 *arrayContainer) container {
	length := difference(ac.content, value2.content, ac.content)
	ac.content = ac.content[:length]
	return ac
}

func (ac *arrayContainer) andNotBitmap(value2 *bitmapContainer) container {
	desiredcapacity := ac.getCardinality()
	answer := newArrayContainerCapacity(desiredcapacity)
	answer.content = answer.content[:desiredcapacity]
	pos := 0
	for _, v := range ac.content {
		answer.content[pos] = v
		pos += 1 - int(value2.bitValue(v))
	}
	answer.content = answer.content[:pos]
	return answer
}

func (ac *arrayContainer) andBitmap(value2 *bitmapContainer) container {
	desiredcapacity := ac.getCardinality()
	answer := newArrayContainerCapacity(desiredcapacity)
	answer.content = answer.content[:desiredcapacity]
	pos := 0
	for _, v := range ac.content {
		answer.content[pos] = v
		pos += int(value2.bitValue(v))
	}
	answer.content = answer.content[:pos]
	return answer
}

func (ac *arrayContainer) iandNotBitmap(value2 *bitmapContainer) container {
	pos := 0
	for _, v := range ac.content {
		ac.content[pos] = v
		pos += 1 - int(value2.bitValue(v))
	}
	ac.content = ac.content[:pos]
	return ac
}

func copyOf(array []uint16, size int) []uint16 {
	result := make([]uint16, size)
	for i, x := range array {
		if i == size {
			break
		}
		result[i] = x
	}
	return result
}

// flip the values in the range [firstOfRange,endx)
func (ac *arrayContainer) inot(firstOfRange, endx int) container {
	if firstOfRange >= endx {
		return ac
	}
	return ac.inotClose(firstOfRange, endx-1) // remove everything in [firstOfRange,endx-1]
}

// flip the values in the range [firstOfRange,lastOfRange]
func (ac *arrayContainer) inotClose(firstOfRange, lastOfRange int) container {
	if firstOfRange > lastOfRange { // unlike add and remove, not uses an inclusive range [firstOfRange,lastOfRange]
		return ac
	}
	// determine the span of array indices to be affected
	startIndex := binarySearch(ac.content, uint16(firstOfRange))
	if startIndex < 0 {
		startIndex = -startIndex - 1
	}
	lastIndex := binarySearch(ac.content, uint16(lastOfRange))
	if lastIndex < 0 {
		lastIndex = -lastIndex - 1 - 1
	}
	currentValuesInRange := lastIndex - startIndex + 1
	spanToBeFlipped := lastOfRange - firstOfRange + 1

	newValuesInRange := spanToBeFlipped - currentValuesInRange
	buffer := make([]uint16, newValuesInRange)
	cardinalityChange := newValuesInRange - currentValuesInRange
	newCardinality := len(ac.content) + cardinalityChange
	if cardinalityChange > 0 {
		if newCardinality > len(ac.content) {
			if newCardinality > arrayDefaultMaxSize {
				bcRet := ac.toBitmapContainer()
				bcRet.inot(firstOfRange, lastOfRange+1)
				*ac = *bcRet.toArrayContainer()
				return bcRet
			}
			ac.content = copyOf(ac.content, newCardinality)
		}
		base := lastIndex + 1
		copy(ac.content[lastIndex+1+cardinalityChange:], ac.content[base:base+len(ac.content)-1-lastIndex])
		ac.negateRange(buffer, startIndex, lastIndex, firstOfRange, lastOfRange+1)
	} else { // no expansion needed
		ac.negateRange(buffer, startIndex, lastIndex, firstOfRange, lastOfRange+1)
		if cardinalityChange < 0 {

			for i := startIndex + newValuesInRange; i < newCardinality; i++ {
				ac.content[i] = ac.content[i-cardinalityChange]
			}
		}
	}
	ac.content = ac.content[:newCardinality]
	return ac
}

func (ac *arrayContainer) negateRange(buffer []uint16, startIndex, lastIndex, startRange, lastRange int) {
	// compute the negation into buffer
	outPos := 0
	inPos := startIndex // value here always >= valInRange,
	// until it is exhausted
	// n.b., we can start initially exhausted.

	valInRange := startRange
	for ; valInRange < lastRange && inPos <= lastIndex; valInRange++ {
		if uint16(valInRange) != ac.content[inPos] {
			buffer[outPos] = uint16(valInRange)
			outPos++
		} else {
			inPos++
		}
	}

	// if there are extra items (greater than the biggest
	// pre-existing one in range), buffer them
	for ; valInRange < lastRange; valInRange++ {
		buffer[outPos] = uint16(valInRange)
		outPos++
	}

	if outPos != len(buffer) {
		panic("negateRange: internal bug")
	}

	for i, item := range buffer {
		ac.content[i+startIndex] = item
	}
}

func (ac *arrayContainer) isFull() bool {
	return false
}

func (ac *arrayContainer) andArray(value2 *arrayContainer) container {
	desiredcapacity := minOfInt(ac.getCardinality(), value2.getCardinality())
	answer := newArrayContainerCapacity(desiredcapacity)
	length := intersection2by2(
		ac.content,
		value2.content,
		answer.content)
	answer.content = answer.content[:length]
	return answer
}

func (ac *arrayContainer) andArrayCardinality(value2 *arrayContainer) int {
	return intersection2by2Cardinality(
		ac.content,
		value2.content)
}

func (ac *arrayContainer) intersectsArray(value2 *arrayContainer) bool {
	return intersects2by2(
		ac.content,
		value2.content)
}

func (ac *arrayContainer) iandArray(value2 *arrayContainer) container {
	length := intersection2by2(
		ac.content,
		value2.content,
		ac.content)
	ac.content = ac.content[:length]
	return ac
}

func (ac *arrayContainer) getCardinality() int {
	return len(ac.content)
}

func (ac *arrayContainer) rank(x uint16) int {
	answer := binarySearch(ac.content, x)
	if answer >= 0 {
		return answer + 1
	}
	return -answer - 1

}

func (ac *arrayContainer) selectInt(x uint16) int {
	return int(ac.content[x])
}

func (ac *arrayContainer) clone() container {
	ptr := arrayContainer{make([]uint16, len(ac.content))}
	copy(ptr.content, ac.content[:])
	return &ptr
}

func (ac *arrayContainer) contains(x uint16) bool {
	return binarySearch(ac.content, x) >= 0
}

func (ac *arrayContainer) loadData(bitmapContainer *bitmapContainer) {
	ac.content = make([]uint16, bitmapContainer.cardinality, bitmapContainer.cardinality)
	bitmapContainer.fillArray(ac.content)
}
func newArrayContainer() *arrayContainer {
	p := new(arrayContainer)
	return p
}

func newArrayContainerFromBitmap(bc *bitmapContainer) *arrayContainer {
	ac := &arrayContainer{}
	ac.loadData(bc)
	return ac
}

func newArrayContainerCapacity(size int) *arrayContainer {
	p := new(arrayContainer)
	p.content = make([]uint16, 0, size)
	return p
}

func newArrayContainerSize(size int) *arrayContainer {
	p := new(arrayContainer)
	p.content = make([]uint16, size, size)
	return p
}

func newArrayContainerRange(firstOfRun, lastOfRun int) *arrayContainer {
	valuesInRange := lastOfRun - firstOfRun + 1
	this := newArrayContainerCapacity(valuesInRange)
	for i := 0; i < valuesInRange; i++ {
		this.content = append(this.content, uint16(firstOfRun+i))
	}
	return this
}

func (ac *arrayContainer) numberOfRuns() (nr int) {
	n := len(ac.content)
	var runlen uint16
	var cur, prev uint16

	switch n {
	case 0:
		return 0
	case 1:
		return 1
	default:
		for i := 1; i < n; i++ {
			prev = ac.content[i-1]
			cur = ac.content[i]

			if cur == prev+1 {
				runlen++
			} else {
				if cur < prev {
					panic("then fundamental arrayContainer assumption of sorted ac.content was broken")
				}
				if cur == prev {
					panic("then fundamental arrayContainer assumption of deduplicated content was broken")
				} else {
					nr++
					runlen = 0
				}
			}
		}
		nr++
	}
	return
}

// convert to run or array *if needed*
func (ac *arrayContainer) toEfficientContainer() container {

	numRuns := ac.numberOfRuns()

	sizeAsRunContainer := runContainer16SerializedSizeInBytes(numRuns)
	sizeAsBitmapContainer := bitmapContainerSizeInBytes()
	card := ac.getCardinality()
	sizeAsArrayContainer := arrayContainerSizeInBytes(card)

	if sizeAsRunContainer <= minOfInt(sizeAsBitmapContainer, sizeAsArrayContainer) {
		return newRunContainer16FromArray(ac)
	}
	if card <= arrayDefaultMaxSize {
		return ac
	}
	return ac.toBitmapContainer()
}

func (ac *arrayContainer) containerType() contype {
	return arrayContype
}

func (ac *arrayContainer) addOffset(x uint16) []container {
	low := &arrayContainer{}
	high := &arrayContainer{}
	for _, val := range ac.content {
		y := uint32(val) + uint32(x)
		if highbits(y) > 0 {
			high.content = append(high.content, lowbits(y))
		} else {
			low.content = append(low.content, lowbits(y))
		}
	}
	return []container{low, high}
}
