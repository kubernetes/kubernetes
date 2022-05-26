package match

import (
	"sync"
)

type SomePool interface {
	Get() []int
	Put([]int)
}

var segmentsPools [1024]sync.Pool

func toPowerOfTwo(v int) int {
	v--
	v |= v >> 1
	v |= v >> 2
	v |= v >> 4
	v |= v >> 8
	v |= v >> 16
	v++

	return v
}

const (
	cacheFrom             = 16
	cacheToAndHigher      = 1024
	cacheFromIndex        = 15
	cacheToAndHigherIndex = 1023
)

var (
	segments0 = []int{0}
	segments1 = []int{1}
	segments2 = []int{2}
	segments3 = []int{3}
	segments4 = []int{4}
)

var segmentsByRuneLength [5][]int = [5][]int{
	0: segments0,
	1: segments1,
	2: segments2,
	3: segments3,
	4: segments4,
}

func init() {
	for i := cacheToAndHigher; i >= cacheFrom; i >>= 1 {
		func(i int) {
			segmentsPools[i-1] = sync.Pool{New: func() interface{} {
				return make([]int, 0, i)
			}}
		}(i)
	}
}

func getTableIndex(c int) int {
	p := toPowerOfTwo(c)
	switch {
	case p >= cacheToAndHigher:
		return cacheToAndHigherIndex
	case p <= cacheFrom:
		return cacheFromIndex
	default:
		return p - 1
	}
}

func acquireSegments(c int) []int {
	// make []int with less capacity than cacheFrom
	// is faster than acquiring it from pool
	if c < cacheFrom {
		return make([]int, 0, c)
	}

	return segmentsPools[getTableIndex(c)].Get().([]int)[:0]
}

func releaseSegments(s []int) {
	c := cap(s)

	// make []int with less capacity than cacheFrom
	// is faster than acquiring it from pool
	if c < cacheFrom {
		return
	}

	segmentsPools[getTableIndex(c)].Put(s)
}
