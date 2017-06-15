// Package hdrhistogram provides an implementation of Gil Tene's HDR Histogram
// data structure. The HDR Histogram allows for fast and accurate analysis of
// the extreme ranges of data with non-normal distributions, like latency.
package hdrhistogram

import (
	"fmt"
	"math"
)

// A Bracket is a part of a cumulative distribution.
type Bracket struct {
	Quantile       float64
	Count, ValueAt int64
}

// A Snapshot is an exported view of a Histogram, useful for serializing them.
// A Histogram can be constructed from it by passing it to Import.
type Snapshot struct {
	LowestTrackableValue  int64
	HighestTrackableValue int64
	SignificantFigures    int64
	Counts                []int64
}

// A Histogram is a lossy data structure used to record the distribution of
// non-normally distributed data (like latency) with a high degree of accuracy
// and a bounded degree of precision.
type Histogram struct {
	lowestTrackableValue        int64
	highestTrackableValue       int64
	unitMagnitude               int64
	significantFigures          int64
	subBucketHalfCountMagnitude int32
	subBucketHalfCount          int32
	subBucketMask               int64
	subBucketCount              int32
	bucketCount                 int32
	countsLen                   int32
	totalCount                  int64
	counts                      []int64
}

// New returns a new Histogram instance capable of tracking values in the given
// range and with the given amount of precision.
func New(minValue, maxValue int64, sigfigs int) *Histogram {
	if sigfigs < 1 || 5 < sigfigs {
		panic(fmt.Errorf("sigfigs must be [1,5] (was %d)", sigfigs))
	}

	largestValueWithSingleUnitResolution := 2 * math.Pow10(sigfigs)
	subBucketCountMagnitude := int32(math.Ceil(math.Log2(float64(largestValueWithSingleUnitResolution))))

	subBucketHalfCountMagnitude := subBucketCountMagnitude
	if subBucketHalfCountMagnitude < 1 {
		subBucketHalfCountMagnitude = 1
	}
	subBucketHalfCountMagnitude--

	unitMagnitude := int32(math.Floor(math.Log2(float64(minValue))))
	if unitMagnitude < 0 {
		unitMagnitude = 0
	}

	subBucketCount := int32(math.Pow(2, float64(subBucketHalfCountMagnitude)+1))

	subBucketHalfCount := subBucketCount / 2
	subBucketMask := int64(subBucketCount-1) << uint(unitMagnitude)

	// determine exponent range needed to support the trackable value with no
	// overflow:
	smallestUntrackableValue := int64(subBucketCount) << uint(unitMagnitude)
	bucketsNeeded := int32(1)
	for smallestUntrackableValue < maxValue {
		smallestUntrackableValue <<= 1
		bucketsNeeded++
	}

	bucketCount := bucketsNeeded
	countsLen := (bucketCount + 1) * (subBucketCount / 2)

	return &Histogram{
		lowestTrackableValue:        minValue,
		highestTrackableValue:       maxValue,
		unitMagnitude:               int64(unitMagnitude),
		significantFigures:          int64(sigfigs),
		subBucketHalfCountMagnitude: subBucketHalfCountMagnitude,
		subBucketHalfCount:          subBucketHalfCount,
		subBucketMask:               subBucketMask,
		subBucketCount:              subBucketCount,
		bucketCount:                 bucketCount,
		countsLen:                   countsLen,
		totalCount:                  0,
		counts:                      make([]int64, countsLen),
	}
}

// ByteSize returns an estimate of the amount of memory allocated to the
// histogram in bytes.
//
// N.B.: This does not take into account the overhead for slices, which are
// small, constant, and specific to the compiler version.
func (h *Histogram) ByteSize() int {
	return 6*8 + 5*4 + len(h.counts)*8
}

// Merge merges the data stored in the given histogram with the receiver,
// returning the number of recorded values which had to be dropped.
func (h *Histogram) Merge(from *Histogram) (dropped int64) {
	i := from.rIterator()
	for i.next() {
		v := i.valueFromIdx
		c := i.countAtIdx

		if h.RecordValues(v, c) != nil {
			dropped += c
		}
	}

	return
}

// TotalCount returns total number of values recorded.
func (h *Histogram) TotalCount() int64 {
	return h.totalCount
}

// Max returns the approximate maximum recorded value.
func (h *Histogram) Max() int64 {
	var max int64
	i := h.iterator()
	for i.next() {
		if i.countAtIdx != 0 {
			max = i.highestEquivalentValue
		}
	}
	return h.highestEquivalentValue(max)
}

// Min returns the approximate minimum recorded value.
func (h *Histogram) Min() int64 {
	var min int64
	i := h.iterator()
	for i.next() {
		if i.countAtIdx != 0 && min == 0 {
			min = i.highestEquivalentValue
			break
		}
	}
	return h.lowestEquivalentValue(min)
}

// Mean returns the approximate arithmetic mean of the recorded values.
func (h *Histogram) Mean() float64 {
	if h.totalCount == 0 {
		return 0
	}
	var total int64
	i := h.iterator()
	for i.next() {
		if i.countAtIdx != 0 {
			total += i.countAtIdx * h.medianEquivalentValue(i.valueFromIdx)
		}
	}
	return float64(total) / float64(h.totalCount)
}

// StdDev returns the approximate standard deviation of the recorded values.
func (h *Histogram) StdDev() float64 {
	if h.totalCount == 0 {
		return 0
	}

	mean := h.Mean()
	geometricDevTotal := 0.0

	i := h.iterator()
	for i.next() {
		if i.countAtIdx != 0 {
			dev := float64(h.medianEquivalentValue(i.valueFromIdx)) - mean
			geometricDevTotal += (dev * dev) * float64(i.countAtIdx)
		}
	}

	return math.Sqrt(geometricDevTotal / float64(h.totalCount))
}

// Reset deletes all recorded values and restores the histogram to its original
// state.
func (h *Histogram) Reset() {
	h.totalCount = 0
	for i := range h.counts {
		h.counts[i] = 0
	}
}

// RecordValue records the given value, returning an error if the value is out
// of range.
func (h *Histogram) RecordValue(v int64) error {
	return h.RecordValues(v, 1)
}

// RecordCorrectedValue records the given value, correcting for stalls in the
// recording process. This only works for processes which are recording values
// at an expected interval (e.g., doing jitter analysis). Processes which are
// recording ad-hoc values (e.g., latency for incoming requests) can't take
// advantage of this.
func (h *Histogram) RecordCorrectedValue(v, expectedInterval int64) error {
	if err := h.RecordValue(v); err != nil {
		return err
	}

	if expectedInterval <= 0 || v <= expectedInterval {
		return nil
	}

	missingValue := v - expectedInterval
	for missingValue >= expectedInterval {
		if err := h.RecordValue(missingValue); err != nil {
			return err
		}
		missingValue -= expectedInterval
	}

	return nil
}

// RecordValues records n occurrences of the given value, returning an error if
// the value is out of range.
func (h *Histogram) RecordValues(v, n int64) error {
	idx := h.countsIndexFor(v)
	if idx < 0 || int(h.countsLen) <= idx {
		return fmt.Errorf("value %d is too large to be recorded", v)
	}
	h.counts[idx] += n
	h.totalCount += n

	return nil
}

// ValueAtQuantile returns the recorded value at the given quantile (0..100).
func (h *Histogram) ValueAtQuantile(q float64) int64 {
	if q > 100 {
		q = 100
	}

	total := int64(0)
	countAtPercentile := int64(((q / 100) * float64(h.totalCount)) + 0.5)

	i := h.iterator()
	for i.next() {
		total += i.countAtIdx
		if total >= countAtPercentile {
			return h.highestEquivalentValue(i.valueFromIdx)
		}
	}

	return 0
}

// CumulativeDistribution returns an ordered list of brackets of the
// distribution of recorded values.
func (h *Histogram) CumulativeDistribution() []Bracket {
	var result []Bracket

	i := h.pIterator(1)
	for i.next() {
		result = append(result, Bracket{
			Quantile: i.percentile,
			Count:    i.countToIdx,
			ValueAt:  i.highestEquivalentValue,
		})
	}

	return result
}

// SignificantFigures returns the significant figures used to create the
// histogram
func (h *Histogram) SignificantFigures() int64 {
	return h.significantFigures
}

// LowestTrackableValue returns the lower bound on values that will be added
// to the histogram
func (h *Histogram) LowestTrackableValue() int64 {
	return h.lowestTrackableValue
}

// HighestTrackableValue returns the upper bound on values that will be added
// to the histogram
func (h *Histogram) HighestTrackableValue() int64 {
	return h.highestTrackableValue
}

// Histogram bar for plotting
type Bar struct {
	From, To, Count int64
}

// Pretty print as csv for easy plotting
func (b Bar) String() string {
	return fmt.Sprintf("%v, %v, %v\n", b.From, b.To, b.Count)
}

// Distribution returns an ordered list of bars of the
// distribution of recorded values, counts can be normalized to a probability
func (h *Histogram) Distribution() (result []Bar) {
	i := h.iterator()
	for i.next() {
		result = append(result, Bar{
			Count: i.countAtIdx,
			From:  h.lowestEquivalentValue(i.valueFromIdx),
			To:    i.highestEquivalentValue,
		})
	}

	return result
}

// Equals returns true if the two Histograms are equivalent, false if not.
func (h *Histogram) Equals(other *Histogram) bool {
	switch {
	case
		h.lowestTrackableValue != other.lowestTrackableValue,
		h.highestTrackableValue != other.highestTrackableValue,
		h.unitMagnitude != other.unitMagnitude,
		h.significantFigures != other.significantFigures,
		h.subBucketHalfCountMagnitude != other.subBucketHalfCountMagnitude,
		h.subBucketHalfCount != other.subBucketHalfCount,
		h.subBucketMask != other.subBucketMask,
		h.subBucketCount != other.subBucketCount,
		h.bucketCount != other.bucketCount,
		h.countsLen != other.countsLen,
		h.totalCount != other.totalCount:
		return false
	default:
		for i, c := range h.counts {
			if c != other.counts[i] {
				return false
			}
		}
	}
	return true
}

// Export returns a snapshot view of the Histogram. This can be later passed to
// Import to construct a new Histogram with the same state.
func (h *Histogram) Export() *Snapshot {
	return &Snapshot{
		LowestTrackableValue:  h.lowestTrackableValue,
		HighestTrackableValue: h.highestTrackableValue,
		SignificantFigures:    h.significantFigures,
		Counts:                append([]int64(nil), h.counts...), // copy
	}
}

// Import returns a new Histogram populated from the Snapshot data (which the
// caller must stop accessing).
func Import(s *Snapshot) *Histogram {
	h := New(s.LowestTrackableValue, s.HighestTrackableValue, int(s.SignificantFigures))
	h.counts = s.Counts
	totalCount := int64(0)
	for i := int32(0); i < h.countsLen; i++ {
		countAtIndex := h.counts[i]
		if countAtIndex > 0 {
			totalCount += countAtIndex
		}
	}
	h.totalCount = totalCount
	return h
}

func (h *Histogram) iterator() *iterator {
	return &iterator{
		h:            h,
		subBucketIdx: -1,
	}
}

func (h *Histogram) rIterator() *rIterator {
	return &rIterator{
		iterator: iterator{
			h:            h,
			subBucketIdx: -1,
		},
	}
}

func (h *Histogram) pIterator(ticksPerHalfDistance int32) *pIterator {
	return &pIterator{
		iterator: iterator{
			h:            h,
			subBucketIdx: -1,
		},
		ticksPerHalfDistance: ticksPerHalfDistance,
	}
}

func (h *Histogram) sizeOfEquivalentValueRange(v int64) int64 {
	bucketIdx := h.getBucketIndex(v)
	subBucketIdx := h.getSubBucketIdx(v, bucketIdx)
	adjustedBucket := bucketIdx
	if subBucketIdx >= h.subBucketCount {
		adjustedBucket++
	}
	return int64(1) << uint(h.unitMagnitude+int64(adjustedBucket))
}

func (h *Histogram) valueFromIndex(bucketIdx, subBucketIdx int32) int64 {
	return int64(subBucketIdx) << uint(int64(bucketIdx)+h.unitMagnitude)
}

func (h *Histogram) lowestEquivalentValue(v int64) int64 {
	bucketIdx := h.getBucketIndex(v)
	subBucketIdx := h.getSubBucketIdx(v, bucketIdx)
	return h.valueFromIndex(bucketIdx, subBucketIdx)
}

func (h *Histogram) nextNonEquivalentValue(v int64) int64 {
	return h.lowestEquivalentValue(v) + h.sizeOfEquivalentValueRange(v)
}

func (h *Histogram) highestEquivalentValue(v int64) int64 {
	return h.nextNonEquivalentValue(v) - 1
}

func (h *Histogram) medianEquivalentValue(v int64) int64 {
	return h.lowestEquivalentValue(v) + (h.sizeOfEquivalentValueRange(v) >> 1)
}

func (h *Histogram) getCountAtIndex(bucketIdx, subBucketIdx int32) int64 {
	return h.counts[h.countsIndex(bucketIdx, subBucketIdx)]
}

func (h *Histogram) countsIndex(bucketIdx, subBucketIdx int32) int32 {
	bucketBaseIdx := (bucketIdx + 1) << uint(h.subBucketHalfCountMagnitude)
	offsetInBucket := subBucketIdx - h.subBucketHalfCount
	return bucketBaseIdx + offsetInBucket
}

func (h *Histogram) getBucketIndex(v int64) int32 {
	pow2Ceiling := bitLen(v | h.subBucketMask)
	return int32(pow2Ceiling - int64(h.unitMagnitude) -
		int64(h.subBucketHalfCountMagnitude+1))
}

func (h *Histogram) getSubBucketIdx(v int64, idx int32) int32 {
	return int32(v >> uint(int64(idx)+int64(h.unitMagnitude)))
}

func (h *Histogram) countsIndexFor(v int64) int {
	bucketIdx := h.getBucketIndex(v)
	subBucketIdx := h.getSubBucketIdx(v, bucketIdx)
	return int(h.countsIndex(bucketIdx, subBucketIdx))
}

type iterator struct {
	h                                    *Histogram
	bucketIdx, subBucketIdx              int32
	countAtIdx, countToIdx, valueFromIdx int64
	highestEquivalentValue               int64
}

func (i *iterator) next() bool {
	if i.countToIdx >= i.h.totalCount {
		return false
	}

	// increment bucket
	i.subBucketIdx++
	if i.subBucketIdx >= i.h.subBucketCount {
		i.subBucketIdx = i.h.subBucketHalfCount
		i.bucketIdx++
	}

	if i.bucketIdx >= i.h.bucketCount {
		return false
	}

	i.countAtIdx = i.h.getCountAtIndex(i.bucketIdx, i.subBucketIdx)
	i.countToIdx += i.countAtIdx
	i.valueFromIdx = i.h.valueFromIndex(i.bucketIdx, i.subBucketIdx)
	i.highestEquivalentValue = i.h.highestEquivalentValue(i.valueFromIdx)

	return true
}

type rIterator struct {
	iterator
	countAddedThisStep int64
}

func (r *rIterator) next() bool {
	for r.iterator.next() {
		if r.countAtIdx != 0 {
			r.countAddedThisStep = r.countAtIdx
			return true
		}
	}
	return false
}

type pIterator struct {
	iterator
	seenLastValue          bool
	ticksPerHalfDistance   int32
	percentileToIteratorTo float64
	percentile             float64
}

func (p *pIterator) next() bool {
	if !(p.countToIdx < p.h.totalCount) {
		if p.seenLastValue {
			return false
		}

		p.seenLastValue = true
		p.percentile = 100

		return true
	}

	if p.subBucketIdx == -1 && !p.iterator.next() {
		return false
	}

	var done = false
	for !done {
		currentPercentile := (100.0 * float64(p.countToIdx)) / float64(p.h.totalCount)
		if p.countAtIdx != 0 && p.percentileToIteratorTo <= currentPercentile {
			p.percentile = p.percentileToIteratorTo
			halfDistance := math.Trunc(math.Pow(2, math.Trunc(math.Log2(100.0/(100.0-p.percentileToIteratorTo)))+1))
			percentileReportingTicks := float64(p.ticksPerHalfDistance) * halfDistance
			p.percentileToIteratorTo += 100.0 / percentileReportingTicks
			return true
		}
		done = !p.iterator.next()
	}

	return true
}

func bitLen(x int64) (n int64) {
	for ; x >= 0x8000; x >>= 16 {
		n += 16
	}
	if x >= 0x80 {
		x >>= 8
		n += 8
	}
	if x >= 0x8 {
		x >>= 4
		n += 4
	}
	if x >= 0x2 {
		x >>= 2
		n += 2
	}
	if x >= 0x1 {
		n++
	}
	return
}
