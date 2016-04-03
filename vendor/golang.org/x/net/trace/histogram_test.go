// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"math"
	"testing"
)

type sumTest struct {
	value        int64
	sum          int64
	sumOfSquares float64
	total        int64
}

var sumTests = []sumTest{
	{100, 100, 10000, 1},
	{50, 150, 12500, 2},
	{50, 200, 15000, 3},
	{50, 250, 17500, 4},
}

type bucketingTest struct {
	in     int64
	log    int
	bucket int
}

var bucketingTests = []bucketingTest{
	{0, 0, 0},
	{1, 1, 0},
	{2, 2, 1},
	{3, 2, 1},
	{4, 3, 2},
	{1000, 10, 9},
	{1023, 10, 9},
	{1024, 11, 10},
	{1000000, 20, 19},
}

type multiplyTest struct {
	in                   int64
	ratio                float64
	expectedSum          int64
	expectedTotal        int64
	expectedSumOfSquares float64
}

var multiplyTests = []multiplyTest{
	{15, 2.5, 37, 2, 562.5},
	{128, 4.6, 758, 13, 77953.9},
}

type percentileTest struct {
	fraction float64
	expected int64
}

var percentileTests = []percentileTest{
	{0.25, 48},
	{0.5, 96},
	{0.6, 109},
	{0.75, 128},
	{0.90, 205},
	{0.95, 230},
	{0.99, 256},
}

func TestSum(t *testing.T) {
	var h histogram

	for _, test := range sumTests {
		h.addMeasurement(test.value)
		sum := h.sum
		if sum != test.sum {
			t.Errorf("h.Sum = %v WANT: %v", sum, test.sum)
		}

		sumOfSquares := h.sumOfSquares
		if sumOfSquares != test.sumOfSquares {
			t.Errorf("h.SumOfSquares = %v WANT: %v", sumOfSquares, test.sumOfSquares)
		}

		total := h.total()
		if total != test.total {
			t.Errorf("h.Total = %v WANT: %v", total, test.total)
		}
	}
}

func TestMultiply(t *testing.T) {
	var h histogram
	for i, test := range multiplyTests {
		h.addMeasurement(test.in)
		h.Multiply(test.ratio)
		if h.sum != test.expectedSum {
			t.Errorf("#%v: h.sum = %v WANT: %v", i, h.sum, test.expectedSum)
		}
		if h.total() != test.expectedTotal {
			t.Errorf("#%v: h.total = %v WANT: %v", i, h.total(), test.expectedTotal)
		}
		if h.sumOfSquares != test.expectedSumOfSquares {
			t.Errorf("#%v: h.SumOfSquares = %v WANT: %v", i, test.expectedSumOfSquares, h.sumOfSquares)
		}
	}
}

func TestBucketingFunctions(t *testing.T) {
	for _, test := range bucketingTests {
		log := log2(test.in)
		if log != test.log {
			t.Errorf("log2 = %v WANT: %v", log, test.log)
		}

		bucket := getBucket(test.in)
		if bucket != test.bucket {
			t.Errorf("getBucket = %v WANT: %v", bucket, test.bucket)
		}
	}
}

func TestAverage(t *testing.T) {
	a := new(histogram)
	average := a.average()
	if average != 0 {
		t.Errorf("Average of empty histogram was %v WANT: 0", average)
	}

	a.addMeasurement(1)
	a.addMeasurement(1)
	a.addMeasurement(3)
	const expected = float64(5) / float64(3)
	average = a.average()

	if !isApproximate(average, expected) {
		t.Errorf("Average = %g WANT: %v", average, expected)
	}
}

func TestStandardDeviation(t *testing.T) {
	a := new(histogram)
	add(a, 10, 1<<4)
	add(a, 10, 1<<5)
	add(a, 10, 1<<6)
	stdDev := a.standardDeviation()
	const expected = 19.95

	if !isApproximate(stdDev, expected) {
		t.Errorf("StandardDeviation = %v WANT: %v", stdDev, expected)
	}

	// No values
	a = new(histogram)
	stdDev = a.standardDeviation()

	if !isApproximate(stdDev, 0) {
		t.Errorf("StandardDeviation = %v WANT: 0", stdDev)
	}

	add(a, 1, 1<<4)
	if !isApproximate(stdDev, 0) {
		t.Errorf("StandardDeviation = %v WANT: 0", stdDev)
	}

	add(a, 10, 1<<4)
	if !isApproximate(stdDev, 0) {
		t.Errorf("StandardDeviation = %v WANT: 0", stdDev)
	}
}

func TestPercentileBoundary(t *testing.T) {
	a := new(histogram)
	add(a, 5, 1<<4)
	add(a, 10, 1<<6)
	add(a, 5, 1<<7)

	for _, test := range percentileTests {
		percentile := a.percentileBoundary(test.fraction)
		if percentile != test.expected {
			t.Errorf("h.PercentileBoundary (fraction=%v) = %v WANT: %v", test.fraction, percentile, test.expected)
		}
	}
}

func TestCopyFrom(t *testing.T) {
	a := histogram{5, 25, []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38}, 4, -1}
	b := histogram{6, 36, []int64{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
		20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39}, 5, -1}

	a.CopyFrom(&b)

	if a.String() != b.String() {
		t.Errorf("a.String = %s WANT: %s", a.String(), b.String())
	}
}

func TestClear(t *testing.T) {
	a := histogram{5, 25, []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38}, 4, -1}

	a.Clear()

	expected := "0, 0.000000, 0, 0, []"
	if a.String() != expected {
		t.Errorf("a.String = %s WANT %s", a.String(), expected)
	}
}

func TestNew(t *testing.T) {
	a := histogram{5, 25, []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38}, 4, -1}
	b := a.New()

	expected := "0, 0.000000, 0, 0, []"
	if b.(*histogram).String() != expected {
		t.Errorf("b.(*histogram).String = %s WANT: %s", b.(*histogram).String(), expected)
	}
}

func TestAdd(t *testing.T) {
	// The tests here depend on the associativity of addMeasurement and Add.
	// Add empty observation
	a := histogram{5, 25, []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38}, 4, -1}
	b := a.New()

	expected := a.String()
	a.Add(b)
	if a.String() != expected {
		t.Errorf("a.String = %s WANT: %s", a.String(), expected)
	}

	// Add same bucketed value, no new buckets
	c := new(histogram)
	d := new(histogram)
	e := new(histogram)
	c.addMeasurement(12)
	d.addMeasurement(11)
	e.addMeasurement(12)
	e.addMeasurement(11)
	c.Add(d)
	if c.String() != e.String() {
		t.Errorf("c.String = %s WANT: %s", c.String(), e.String())
	}

	// Add bucketed values
	f := new(histogram)
	g := new(histogram)
	h := new(histogram)
	f.addMeasurement(4)
	f.addMeasurement(12)
	f.addMeasurement(100)
	g.addMeasurement(18)
	g.addMeasurement(36)
	g.addMeasurement(255)
	h.addMeasurement(4)
	h.addMeasurement(12)
	h.addMeasurement(100)
	h.addMeasurement(18)
	h.addMeasurement(36)
	h.addMeasurement(255)
	f.Add(g)
	if f.String() != h.String() {
		t.Errorf("f.String = %q WANT: %q", f.String(), h.String())
	}

	// add buckets to no buckets
	i := new(histogram)
	j := new(histogram)
	k := new(histogram)
	j.addMeasurement(18)
	j.addMeasurement(36)
	j.addMeasurement(255)
	k.addMeasurement(18)
	k.addMeasurement(36)
	k.addMeasurement(255)
	i.Add(j)
	if i.String() != k.String() {
		t.Errorf("i.String = %q WANT: %q", i.String(), k.String())
	}

	// add buckets to single value (no overlap)
	l := new(histogram)
	m := new(histogram)
	n := new(histogram)
	l.addMeasurement(0)
	m.addMeasurement(18)
	m.addMeasurement(36)
	m.addMeasurement(255)
	n.addMeasurement(0)
	n.addMeasurement(18)
	n.addMeasurement(36)
	n.addMeasurement(255)
	l.Add(m)
	if l.String() != n.String() {
		t.Errorf("l.String = %q WANT: %q", l.String(), n.String())
	}

	// mixed order
	o := new(histogram)
	p := new(histogram)
	o.addMeasurement(0)
	o.addMeasurement(2)
	o.addMeasurement(0)
	p.addMeasurement(0)
	p.addMeasurement(0)
	p.addMeasurement(2)
	if o.String() != p.String() {
		t.Errorf("o.String = %q WANT: %q", o.String(), p.String())
	}
}

func add(h *histogram, times int, val int64) {
	for i := 0; i < times; i++ {
		h.addMeasurement(val)
	}
}

func isApproximate(x, y float64) bool {
	return math.Abs(x-y) < 1e-2
}
