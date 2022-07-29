package antlr

import "math"

const (
	_initalCapacity       = 16
	_initalBucketCapacity = 8
	_loadFactor           = 0.75
)

var _ Set = (*array2DHashSet)(nil)

type Set interface {
	Add(value interface{}) (added interface{})
	Len() int
	Get(value interface{}) (found interface{})
	Contains(value interface{}) bool
	Values() []interface{}
	Each(f func(interface{}) bool)
}

type array2DHashSet struct {
	buckets          [][]interface{}
	hashcodeFunction func(interface{}) int
	equalsFunction   func(interface{}, interface{}) bool

	n         int // How many elements in set
	threshold int // when to expand

	currentPrime          int // jump by 4 primes each expand or whatever
	initialBucketCapacity int
}

func (as *array2DHashSet) Each(f func(interface{}) bool) {
	if as.Len() < 1 {
		return
	}

	for _, bucket := range as.buckets {
		for _, o := range bucket {
			if o == nil {
				break
			}
			if !f(o) {
				return
			}
		}
	}
}

func (as *array2DHashSet) Values() []interface{} {
	if as.Len() < 1 {
		return nil
	}

	values := make([]interface{}, 0, as.Len())
	as.Each(func(i interface{}) bool {
		values = append(values, i)
		return true
	})
	return values
}

func (as *array2DHashSet) Contains(value interface{}) bool {
	return as.Get(value) != nil
}

func (as *array2DHashSet) Add(value interface{}) interface{} {
	if as.n > as.threshold {
		as.expand()
	}
	return as.innerAdd(value)
}

func (as *array2DHashSet) expand() {
	old := as.buckets

	as.currentPrime += 4

	var (
		newCapacity      = len(as.buckets) << 1
		newTable         = as.createBuckets(newCapacity)
		newBucketLengths = make([]int, len(newTable))
	)

	as.buckets = newTable
	as.threshold = int(float64(newCapacity) * _loadFactor)

	for _, bucket := range old {
		if bucket == nil {
			continue
		}

		for _, o := range bucket {
			if o == nil {
				break
			}

			b := as.getBuckets(o)
			bucketLength := newBucketLengths[b]
			var newBucket []interface{}
			if bucketLength == 0 {
				// new bucket
				newBucket = as.createBucket(as.initialBucketCapacity)
				newTable[b] = newBucket
			} else {
				newBucket = newTable[b]
				if bucketLength == len(newBucket) {
					// expand
					newBucketCopy := make([]interface{}, len(newBucket)<<1)
					copy(newBucketCopy[:bucketLength], newBucket)
					newBucket = newBucketCopy
					newTable[b] = newBucket
				}
			}

			newBucket[bucketLength] = o
			newBucketLengths[b]++
		}
	}
}

func (as *array2DHashSet) Len() int {
	return as.n
}

func (as *array2DHashSet) Get(o interface{}) interface{} {
	if o == nil {
		return nil
	}

	b := as.getBuckets(o)
	bucket := as.buckets[b]
	if bucket == nil { // no bucket
		return nil
	}

	for _, e := range bucket {
		if e == nil {
			return nil // empty slot; not there
		}
		if as.equalsFunction(e, o) {
			return e
		}
	}

	return nil
}

func (as *array2DHashSet) innerAdd(o interface{}) interface{} {
	b := as.getBuckets(o)

	bucket := as.buckets[b]

	// new bucket
	if bucket == nil {
		bucket = as.createBucket(as.initialBucketCapacity)
		bucket[0] = o

		as.buckets[b] = bucket
		as.n++
		return o
	}

	// look for it in bucket
	for i := 0; i < len(bucket); i++ {
		existing := bucket[i]
		if existing == nil { // empty slot; not there, add.
			bucket[i] = o
			as.n++
			return o
		}

		if as.equalsFunction(existing, o) { // found existing, quit
			return existing
		}
	}

	// full bucket, expand and add to end
	oldLength := len(bucket)
	bucketCopy := make([]interface{}, oldLength<<1)
	copy(bucketCopy[:oldLength], bucket)
	bucket = bucketCopy
	as.buckets[b] = bucket
	bucket[oldLength] = o
	as.n++
	return o
}

func (as *array2DHashSet) getBuckets(value interface{}) int {
	hash := as.hashcodeFunction(value)
	return hash & (len(as.buckets) - 1)
}

func (as *array2DHashSet) createBuckets(cap int) [][]interface{} {
	return make([][]interface{}, cap)
}

func (as *array2DHashSet) createBucket(cap int) []interface{} {
	return make([]interface{}, cap)
}

func newArray2DHashSetWithCap(
	hashcodeFunction func(interface{}) int,
	equalsFunction func(interface{}, interface{}) bool,
	initCap int,
	initBucketCap int,
) *array2DHashSet {
	if hashcodeFunction == nil {
		hashcodeFunction = standardHashFunction
	}

	if equalsFunction == nil {
		equalsFunction = standardEqualsFunction
	}

	ret := &array2DHashSet{
		hashcodeFunction: hashcodeFunction,
		equalsFunction:   equalsFunction,

		n:         0,
		threshold: int(math.Floor(_initalCapacity * _loadFactor)),

		currentPrime:          1,
		initialBucketCapacity: initBucketCap,
	}

	ret.buckets = ret.createBuckets(initCap)
	return ret
}

func newArray2DHashSet(
	hashcodeFunction func(interface{}) int,
	equalsFunction func(interface{}, interface{}) bool,
) *array2DHashSet {
	return newArray2DHashSetWithCap(hashcodeFunction, equalsFunction, _initalCapacity, _initalBucketCapacity)
}
