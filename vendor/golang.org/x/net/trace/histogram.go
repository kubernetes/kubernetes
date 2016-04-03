// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

// This file implements histogramming for RPC statistics collection.

import (
	"bytes"
	"fmt"
	"html/template"
	"log"
	"math"

	"golang.org/x/net/internal/timeseries"
)

const (
	bucketCount = 38
)

// histogram keeps counts of values in buckets that are spaced
// out in powers of 2: 0-1, 2-3, 4-7...
// histogram implements timeseries.Observable
type histogram struct {
	sum          int64   // running total of measurements
	sumOfSquares float64 // square of running total
	buckets      []int64 // bucketed values for histogram
	value        int     // holds a single value as an optimization
	valueCount   int64   // number of values recorded for single value
}

// AddMeasurement records a value measurement observation to the histogram.
func (h *histogram) addMeasurement(value int64) {
	// TODO: assert invariant
	h.sum += value
	h.sumOfSquares += float64(value) * float64(value)

	bucketIndex := getBucket(value)

	if h.valueCount == 0 || (h.valueCount > 0 && h.value == bucketIndex) {
		h.value = bucketIndex
		h.valueCount++
	} else {
		h.allocateBuckets()
		h.buckets[bucketIndex]++
	}
}

func (h *histogram) allocateBuckets() {
	if h.buckets == nil {
		h.buckets = make([]int64, bucketCount)
		h.buckets[h.value] = h.valueCount
		h.value = 0
		h.valueCount = -1
	}
}

func log2(i int64) int {
	n := 0
	for ; i >= 0x100; i >>= 8 {
		n += 8
	}
	for ; i > 0; i >>= 1 {
		n += 1
	}
	return n
}

func getBucket(i int64) (index int) {
	index = log2(i) - 1
	if index < 0 {
		index = 0
	}
	if index >= bucketCount {
		index = bucketCount - 1
	}
	return
}

// Total returns the number of recorded observations.
func (h *histogram) total() (total int64) {
	if h.valueCount >= 0 {
		total = h.valueCount
	}
	for _, val := range h.buckets {
		total += int64(val)
	}
	return
}

// Average returns the average value of recorded observations.
func (h *histogram) average() float64 {
	t := h.total()
	if t == 0 {
		return 0
	}
	return float64(h.sum) / float64(t)
}

// Variance returns the variance of recorded observations.
func (h *histogram) variance() float64 {
	t := float64(h.total())
	if t == 0 {
		return 0
	}
	s := float64(h.sum) / t
	return h.sumOfSquares/t - s*s
}

// StandardDeviation returns the standard deviation of recorded observations.
func (h *histogram) standardDeviation() float64 {
	return math.Sqrt(h.variance())
}

// PercentileBoundary estimates the value that the given fraction of recorded
// observations are less than.
func (h *histogram) percentileBoundary(percentile float64) int64 {
	total := h.total()

	// Corner cases (make sure result is strictly less than Total())
	if total == 0 {
		return 0
	} else if total == 1 {
		return int64(h.average())
	}

	percentOfTotal := round(float64(total) * percentile)
	var runningTotal int64

	for i := range h.buckets {
		value := h.buckets[i]
		runningTotal += value
		if runningTotal == percentOfTotal {
			// We hit an exact bucket boundary. If the next bucket has data, it is a
			// good estimate of the value. If the bucket is empty, we interpolate the
			// midpoint between the next bucket's boundary and the next non-zero
			// bucket. If the remaining buckets are all empty, then we use the
			// boundary for the next bucket as the estimate.
			j := uint8(i + 1)
			min := bucketBoundary(j)
			if runningTotal < total {
				for h.buckets[j] == 0 {
					j++
				}
			}
			max := bucketBoundary(j)
			return min + round(float64(max-min)/2)
		} else if runningTotal > percentOfTotal {
			// The value is in this bucket. Interpolate the value.
			delta := runningTotal - percentOfTotal
			percentBucket := float64(value-delta) / float64(value)
			bucketMin := bucketBoundary(uint8(i))
			nextBucketMin := bucketBoundary(uint8(i + 1))
			bucketSize := nextBucketMin - bucketMin
			return bucketMin + round(percentBucket*float64(bucketSize))
		}
	}
	return bucketBoundary(bucketCount - 1)
}

// Median returns the estimated median of the observed values.
func (h *histogram) median() int64 {
	return h.percentileBoundary(0.5)
}

// Add adds other to h.
func (h *histogram) Add(other timeseries.Observable) {
	o := other.(*histogram)
	if o.valueCount == 0 {
		// Other histogram is empty
	} else if h.valueCount >= 0 && o.valueCount > 0 && h.value == o.value {
		// Both have a single bucketed value, aggregate them
		h.valueCount += o.valueCount
	} else {
		// Two different values necessitate buckets in this histogram
		h.allocateBuckets()
		if o.valueCount >= 0 {
			h.buckets[o.value] += o.valueCount
		} else {
			for i := range h.buckets {
				h.buckets[i] += o.buckets[i]
			}
		}
	}
	h.sumOfSquares += o.sumOfSquares
	h.sum += o.sum
}

// Clear resets the histogram to an empty state, removing all observed values.
func (h *histogram) Clear() {
	h.buckets = nil
	h.value = 0
	h.valueCount = 0
	h.sum = 0
	h.sumOfSquares = 0
}

// CopyFrom copies from other, which must be a *histogram, into h.
func (h *histogram) CopyFrom(other timeseries.Observable) {
	o := other.(*histogram)
	if o.valueCount == -1 {
		h.allocateBuckets()
		copy(h.buckets, o.buckets)
	}
	h.sum = o.sum
	h.sumOfSquares = o.sumOfSquares
	h.value = o.value
	h.valueCount = o.valueCount
}

// Multiply scales the histogram by the specified ratio.
func (h *histogram) Multiply(ratio float64) {
	if h.valueCount == -1 {
		for i := range h.buckets {
			h.buckets[i] = int64(float64(h.buckets[i]) * ratio)
		}
	} else {
		h.valueCount = int64(float64(h.valueCount) * ratio)
	}
	h.sum = int64(float64(h.sum) * ratio)
	h.sumOfSquares = h.sumOfSquares * ratio
}

// New creates a new histogram.
func (h *histogram) New() timeseries.Observable {
	r := new(histogram)
	r.Clear()
	return r
}

func (h *histogram) String() string {
	return fmt.Sprintf("%d, %f, %d, %d, %v",
		h.sum, h.sumOfSquares, h.value, h.valueCount, h.buckets)
}

// round returns the closest int64 to the argument
func round(in float64) int64 {
	return int64(math.Floor(in + 0.5))
}

// bucketBoundary returns the first value in the bucket.
func bucketBoundary(bucket uint8) int64 {
	if bucket == 0 {
		return 0
	}
	return 1 << bucket
}

// bucketData holds data about a specific bucket for use in distTmpl.
type bucketData struct {
	Lower, Upper       int64
	N                  int64
	Pct, CumulativePct float64
	GraphWidth         int
}

// data holds data about a Distribution for use in distTmpl.
type data struct {
	Buckets                 []*bucketData
	Count, Median           int64
	Mean, StandardDeviation float64
}

// maxHTMLBarWidth is the maximum width of the HTML bar for visualizing buckets.
const maxHTMLBarWidth = 350.0

// newData returns data representing h for use in distTmpl.
func (h *histogram) newData() *data {
	// Force the allocation of buckets to simplify the rendering implementation
	h.allocateBuckets()
	// We scale the bars on the right so that the largest bar is
	// maxHTMLBarWidth pixels in width.
	maxBucket := int64(0)
	for _, n := range h.buckets {
		if n > maxBucket {
			maxBucket = n
		}
	}
	total := h.total()
	barsizeMult := maxHTMLBarWidth / float64(maxBucket)
	var pctMult float64
	if total == 0 {
		pctMult = 1.0
	} else {
		pctMult = 100.0 / float64(total)
	}

	buckets := make([]*bucketData, len(h.buckets))
	runningTotal := int64(0)
	for i, n := range h.buckets {
		if n == 0 {
			continue
		}
		runningTotal += n
		var upperBound int64
		if i < bucketCount-1 {
			upperBound = bucketBoundary(uint8(i + 1))
		} else {
			upperBound = math.MaxInt64
		}
		buckets[i] = &bucketData{
			Lower:         bucketBoundary(uint8(i)),
			Upper:         upperBound,
			N:             n,
			Pct:           float64(n) * pctMult,
			CumulativePct: float64(runningTotal) * pctMult,
			GraphWidth:    int(float64(n) * barsizeMult),
		}
	}
	return &data{
		Buckets:           buckets,
		Count:             total,
		Median:            h.median(),
		Mean:              h.average(),
		StandardDeviation: h.standardDeviation(),
	}
}

func (h *histogram) html() template.HTML {
	buf := new(bytes.Buffer)
	if err := distTmpl.Execute(buf, h.newData()); err != nil {
		buf.Reset()
		log.Printf("net/trace: couldn't execute template: %v", err)
	}
	return template.HTML(buf.String())
}

// Input: data
var distTmpl = template.Must(template.New("distTmpl").Parse(`
<table>
<tr>
    <td style="padding:0.25em">Count: {{.Count}}</td>
    <td style="padding:0.25em">Mean: {{printf "%.0f" .Mean}}</td>
    <td style="padding:0.25em">StdDev: {{printf "%.0f" .StandardDeviation}}</td>
    <td style="padding:0.25em">Median: {{.Median}}</td>
</tr>
</table>
<hr>
<table>
{{range $b := .Buckets}}
{{if $b}}
  <tr>
    <td style="padding:0 0 0 0.25em">[</td>
    <td style="text-align:right;padding:0 0.25em">{{.Lower}},</td>
    <td style="text-align:right;padding:0 0.25em">{{.Upper}})</td>
    <td style="text-align:right;padding:0 0.25em">{{.N}}</td>
    <td style="text-align:right;padding:0 0.25em">{{printf "%#.3f" .Pct}}%</td>
    <td style="text-align:right;padding:0 0.25em">{{printf "%#.3f" .CumulativePct}}%</td>
    <td><div style="background-color: blue; height: 1em; width: {{.GraphWidth}};"></div></td>
  </tr>
{{end}}
{{end}}
</table>
`))
