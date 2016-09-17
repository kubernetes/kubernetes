// Copyright 2016, Circonus, Inc. All rights reserved.
// See the LICENSE file.

// Package circllhist provides an implementation of Circonus' fixed log-linear
// histogram data structure.  This allows tracking of histograms in a
// composable way such that accurate error can be reasoned about.
package circonusllhist

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"sync"
)

const (
	DEFAULT_HIST_SIZE = int16(100)
)

var power_of_ten = [...]float64{
	1, 10, 100, 1000, 10000, 100000, 1e+06, 1e+07, 1e+08, 1e+09, 1e+10,
	1e+11, 1e+12, 1e+13, 1e+14, 1e+15, 1e+16, 1e+17, 1e+18, 1e+19, 1e+20,
	1e+21, 1e+22, 1e+23, 1e+24, 1e+25, 1e+26, 1e+27, 1e+28, 1e+29, 1e+30,
	1e+31, 1e+32, 1e+33, 1e+34, 1e+35, 1e+36, 1e+37, 1e+38, 1e+39, 1e+40,
	1e+41, 1e+42, 1e+43, 1e+44, 1e+45, 1e+46, 1e+47, 1e+48, 1e+49, 1e+50,
	1e+51, 1e+52, 1e+53, 1e+54, 1e+55, 1e+56, 1e+57, 1e+58, 1e+59, 1e+60,
	1e+61, 1e+62, 1e+63, 1e+64, 1e+65, 1e+66, 1e+67, 1e+68, 1e+69, 1e+70,
	1e+71, 1e+72, 1e+73, 1e+74, 1e+75, 1e+76, 1e+77, 1e+78, 1e+79, 1e+80,
	1e+81, 1e+82, 1e+83, 1e+84, 1e+85, 1e+86, 1e+87, 1e+88, 1e+89, 1e+90,
	1e+91, 1e+92, 1e+93, 1e+94, 1e+95, 1e+96, 1e+97, 1e+98, 1e+99, 1e+100,
	1e+101, 1e+102, 1e+103, 1e+104, 1e+105, 1e+106, 1e+107, 1e+108, 1e+109,
	1e+110, 1e+111, 1e+112, 1e+113, 1e+114, 1e+115, 1e+116, 1e+117, 1e+118,
	1e+119, 1e+120, 1e+121, 1e+122, 1e+123, 1e+124, 1e+125, 1e+126, 1e+127,
	1e-128, 1e-127, 1e-126, 1e-125, 1e-124, 1e-123, 1e-122, 1e-121, 1e-120,
	1e-119, 1e-118, 1e-117, 1e-116, 1e-115, 1e-114, 1e-113, 1e-112, 1e-111,
	1e-110, 1e-109, 1e-108, 1e-107, 1e-106, 1e-105, 1e-104, 1e-103, 1e-102,
	1e-101, 1e-100, 1e-99, 1e-98, 1e-97, 1e-96,
	1e-95, 1e-94, 1e-93, 1e-92, 1e-91, 1e-90, 1e-89, 1e-88, 1e-87, 1e-86,
	1e-85, 1e-84, 1e-83, 1e-82, 1e-81, 1e-80, 1e-79, 1e-78, 1e-77, 1e-76,
	1e-75, 1e-74, 1e-73, 1e-72, 1e-71, 1e-70, 1e-69, 1e-68, 1e-67, 1e-66,
	1e-65, 1e-64, 1e-63, 1e-62, 1e-61, 1e-60, 1e-59, 1e-58, 1e-57, 1e-56,
	1e-55, 1e-54, 1e-53, 1e-52, 1e-51, 1e-50, 1e-49, 1e-48, 1e-47, 1e-46,
	1e-45, 1e-44, 1e-43, 1e-42, 1e-41, 1e-40, 1e-39, 1e-38, 1e-37, 1e-36,
	1e-35, 1e-34, 1e-33, 1e-32, 1e-31, 1e-30, 1e-29, 1e-28, 1e-27, 1e-26,
	1e-25, 1e-24, 1e-23, 1e-22, 1e-21, 1e-20, 1e-19, 1e-18, 1e-17, 1e-16,
	1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-09, 1e-08, 1e-07, 1e-06,
	1e-05, 0.0001, 0.001, 0.01, 0.1,
}

// A Bracket is a part of a cumulative distribution.
type Bin struct {
	val   int8
	exp   int8
	count uint64
}

func NewBinRaw(val int8, exp int8, count uint64) *Bin {
	return &Bin{
		val:   val,
		exp:   exp,
		count: count,
	}
}
func NewBin() *Bin {
	return NewBinRaw(0, 0, 0)
}
func NewBinFromFloat64(d float64) *Bin {
	hb := NewBinRaw(0, 0, 0)
	hb.SetFromFloat64(d)
	return hb
}
func (hb *Bin) SetFromFloat64(d float64) *Bin {
	hb.val = -1
	if math.IsInf(d, 0) || math.IsNaN(d) {
		return hb
	}
	if d == 0.0 {
		hb.val = 0
		return hb
	}
	sign := 1
	if math.Signbit(d) {
		sign = -1
	}
	d = math.Abs(d)
	big_exp := int(math.Floor(math.Log10(d)))
	hb.exp = int8(big_exp)
	if int(hb.exp) != big_exp { //rolled
		hb.exp = 0
		if big_exp < 0 {
			hb.val = 0
		}
		return hb
	}
	d = d / hb.PowerOfTen()
	d = d * 10
	hb.val = int8(sign * int(math.Floor(d+1e-13)))
	if hb.val == 100 || hb.val == -100 {
		if hb.exp < 127 {
			hb.val = hb.val / 10
			hb.exp++
		} else {
			hb.val = 0
			hb.exp = 0
		}
	}
	if hb.val == 0 {
		hb.exp = 0
		return hb
	}
	if !((hb.val >= 10 && hb.val < 100) ||
		(hb.val <= -10 && hb.val > -100)) {
		hb.val = -1
		hb.exp = 0
	}
	return hb
}
func (hb *Bin) PowerOfTen() float64 {
	idx := int(hb.exp)
	if idx < 0 {
		idx = 256 + idx
	}
	return power_of_ten[idx]
}

func (hb *Bin) IsNaN() bool {
	if hb.val > 99 || hb.val < -99 {
		return true
	}
	return false
}
func (hb *Bin) Val() int8 {
	return hb.val
}
func (hb *Bin) Exp() int8 {
	return hb.exp
}
func (hb *Bin) Count() uint64 {
	return hb.count
}
func (hb *Bin) Value() float64 {
	if hb.IsNaN() {
		return math.NaN()
	}
	if hb.val < 10 && hb.val > -10 {
		return 0.0
	}
	return (float64(hb.val) / 10.0) * hb.PowerOfTen()
}
func (hb *Bin) BinWidth() float64 {
	if hb.IsNaN() {
		return math.NaN()
	}
	if hb.val < 10 && hb.val > -10 {
		return 0.0
	}
	return hb.PowerOfTen() / 10.0
}
func (hb *Bin) Midpoint() float64 {
	if hb.IsNaN() {
		return math.NaN()
	}
	out := hb.Value()
	if out == 0 {
		return 0
	}
	interval := hb.BinWidth()
	if out < 0 {
		interval = interval * -1
	}
	return out + interval/2.0
}
func (hb *Bin) Left() float64 {
	if hb.IsNaN() {
		return math.NaN()
	}
	out := hb.Value()
	if out >= 0 {
		return out
	}
	return out - hb.BinWidth()
}

func (h1 *Bin) Compare(h2 *Bin) int {
	if h1.val == h2.val && h1.exp == h2.exp {
		return 0
	}
	if h1.val == -1 {
		return 1
	}
	if h2.val == -1 {
		return -1
	}
	if h1.val == 0 {
		if h2.val > 0 {
			return 1
		}
		return -1
	}
	if h2.val == 0 {
		if h1.val < 0 {
			return 1
		}
		return -1
	}
	if h1.val < 0 && h2.val > 0 {
		return 1
	}
	if h1.val > 0 && h2.val < 0 {
		return -1
	}
	if h1.exp == h2.exp {
		if h1.val < h2.val {
			return 1
		}
		return -1
	}
	if h1.exp > h2.exp {
		if h1.val < 0 {
			return 1
		}
		return -1
	}
	if h1.exp < h2.exp {
		if h1.val < 0 {
			return -1
		}
		return 1
	}
	return 0
}

// This histogram structure tracks values are two decimal digits of precision
// with a bounded error that remains bounded upon composition
type Histogram struct {
	mutex  sync.Mutex
	bvs    []Bin
	used   int16
	allocd int16
}

// New returns a new Histogram
func New() *Histogram {
	return &Histogram{
		allocd: DEFAULT_HIST_SIZE,
		used:   0,
		bvs:    make([]Bin, DEFAULT_HIST_SIZE),
	}
}

// Max returns the approximate maximum recorded value.
func (h *Histogram) Max() float64 {
	return h.ValueAtQuantile(1.0)
}

// Min returns the approximate minimum recorded value.
func (h *Histogram) Min() float64 {
	return h.ValueAtQuantile(0.0)
}

// Mean returns the approximate arithmetic mean of the recorded values.
func (h *Histogram) Mean() float64 {
	return h.ApproxMean()
}

// Reset forgets all bins in the histogram (they remain allocated)
func (h *Histogram) Reset() {
	h.mutex.Lock()
	h.used = 0
	h.mutex.Unlock()
}

// RecordValue records the given value, returning an error if the value is out
// of range.
func (h *Histogram) RecordValue(v float64) error {
	return h.RecordValues(v, 1)
}

// RecordCorrectedValue records the given value, correcting for stalls in the
// recording process. This only works for processes which are recording values
// at an expected interval (e.g., doing jitter analysis). Processes which are
// recording ad-hoc values (e.g., latency for incoming requests) can't take
// advantage of this.
// CH Compat
func (h *Histogram) RecordCorrectedValue(v, expectedInterval int64) error {
	if err := h.RecordValue(float64(v)); err != nil {
		return err
	}

	if expectedInterval <= 0 || v <= expectedInterval {
		return nil
	}

	missingValue := v - expectedInterval
	for missingValue >= expectedInterval {
		if err := h.RecordValue(float64(missingValue)); err != nil {
			return err
		}
		missingValue -= expectedInterval
	}

	return nil
}

// find where a new bin should go
func (h *Histogram) InternalFind(hb *Bin) (bool, int16) {
	if h.used == 0 {
		return false, 0
	}
	rv := -1
	idx := int16(0)
	l := int16(0)
	r := h.used - 1
	for l < r {
		check := (r + l) / 2
		rv = h.bvs[check].Compare(hb)
		if rv == 0 {
			l = check
			r = check
		} else if rv > 0 {
			l = check + 1
		} else {
			r = check - 1
		}
	}
	if rv != 0 {
		rv = h.bvs[l].Compare(hb)
	}
	idx = l
	if rv == 0 {
		return true, idx
	}
	if rv < 0 {
		return false, idx
	}
	idx++
	return false, idx
}

func (h *Histogram) InsertBin(hb *Bin, count int64) uint64 {
	h.mutex.Lock()
	defer h.mutex.Unlock()
	if count == 0 {
		return 0
	}
	found, idx := h.InternalFind(hb)
	if !found {
		if h.used == h.allocd {
			new_bvs := make([]Bin, h.allocd+DEFAULT_HIST_SIZE)
			if idx > 0 {
				copy(new_bvs[0:], h.bvs[0:idx])
			}
			if idx < h.used {
				copy(new_bvs[idx+1:], h.bvs[idx:])
			}
			h.allocd = h.allocd + DEFAULT_HIST_SIZE
			h.bvs = new_bvs
		} else {
			copy(h.bvs[idx+1:], h.bvs[idx:h.used])
		}
		h.bvs[idx].val = hb.val
		h.bvs[idx].exp = hb.exp
		h.bvs[idx].count = uint64(count)
		h.used++
		return h.bvs[idx].count
	}
	var newval uint64
	if count < 0 {
		newval = h.bvs[idx].count - uint64(-count)
	} else {
		newval = h.bvs[idx].count + uint64(count)
	}
	if newval < h.bvs[idx].count { //rolled
		newval = ^uint64(0)
	}
	h.bvs[idx].count = newval
	return newval - h.bvs[idx].count
}

// RecordValues records n occurrences of the given value, returning an error if
// the value is out of range.
func (h *Histogram) RecordValues(v float64, n int64) error {
	var hb Bin
	hb.SetFromFloat64(v)
	h.InsertBin(&hb, n)
	return nil
}

// Approximate mean
func (h *Histogram) ApproxMean() float64 {
	h.mutex.Lock()
	defer h.mutex.Unlock()
	divisor := 0.0
	sum := 0.0
	for i := int16(0); i < h.used; i++ {
		midpoint := h.bvs[i].Midpoint()
		cardinality := float64(h.bvs[i].count)
		divisor += cardinality
		sum += midpoint * cardinality
	}
	if divisor == 0.0 {
		return math.NaN()
	}
	return sum / divisor
}

// Approximate sum
func (h *Histogram) ApproxSum() float64 {
	h.mutex.Lock()
	defer h.mutex.Unlock()
	sum := 0.0
	for i := int16(0); i < h.used; i++ {
		midpoint := h.bvs[i].Midpoint()
		cardinality := float64(h.bvs[i].count)
		sum += midpoint * cardinality
	}
	return sum
}

func (h *Histogram) ApproxQuantile(q_in []float64) ([]float64, error) {
	h.mutex.Lock()
	defer h.mutex.Unlock()
	q_out := make([]float64, len(q_in))
	i_q, i_b := 0, int16(0)
	total_cnt, bin_width, bin_left, lower_cnt, upper_cnt := 0.0, 0.0, 0.0, 0.0, 0.0
	if len(q_in) == 0 {
		return q_out, nil
	}
	// Make sure the requested quantiles are in order
	for i_q = 1; i_q < len(q_in); i_q++ {
		if q_in[i_q-1] > q_in[i_q] {
			return nil, errors.New("out of order")
		}
	}
	// Add up the bins
	for i_b = 0; i_b < h.used; i_b++ {
		if !h.bvs[i_b].IsNaN() {
			total_cnt += float64(h.bvs[i_b].count)
		}
	}
	if total_cnt == 0.0 {
		return nil, errors.New("empty_histogram")
	}

	for i_q = 0; i_q < len(q_in); i_q++ {
		if q_in[i_q] < 0.0 || q_in[i_q] > 1.0 {
			return nil, errors.New("out of bound quantile")
		}
		q_out[i_q] = total_cnt * q_in[i_q]
	}

	for i_b = 0; i_b < h.used; i_b++ {
		if h.bvs[i_b].IsNaN() {
			continue
		}
		bin_width = h.bvs[i_b].BinWidth()
		bin_left = h.bvs[i_b].Left()
		lower_cnt = upper_cnt
		upper_cnt = lower_cnt + float64(h.bvs[i_b].count)
		break
	}
	for i_q = 0; i_q < len(q_in); i_q++ {
		for i_b < (h.used-1) && upper_cnt < q_out[i_q] {
			i_b++
			bin_width = h.bvs[i_b].BinWidth()
			bin_left = h.bvs[i_b].Left()
			lower_cnt = upper_cnt
			upper_cnt = lower_cnt + float64(h.bvs[i_b].count)
		}
		if lower_cnt == q_out[i_q] {
			q_out[i_q] = bin_left
		} else if upper_cnt == q_out[i_q] {
			q_out[i_q] = bin_left + bin_width
		} else {
			if bin_width == 0 {
				q_out[i_q] = bin_left
			} else {
				q_out[i_q] = bin_left + (q_out[i_q]-lower_cnt)/(upper_cnt-lower_cnt)*bin_width
			}
		}
	}
	return q_out, nil
}

// ValueAtQuantile returns the recorded value at the given quantile (0..1).
func (h *Histogram) ValueAtQuantile(q float64) float64 {
	h.mutex.Lock()
	defer h.mutex.Unlock()
	q_in := make([]float64, 1)
	q_in[0] = q
	q_out, err := h.ApproxQuantile(q_in)
	if err == nil && len(q_out) == 1 {
		return q_out[0]
	}
	return math.NaN()
}

// SignificantFigures returns the significant figures used to create the
// histogram
// CH Compat
func (h *Histogram) SignificantFigures() int64 {
	return 2
}

// Equals returns true if the two Histograms are equivalent, false if not.
func (h *Histogram) Equals(other *Histogram) bool {
	h.mutex.Lock()
	other.mutex.Lock()
	defer h.mutex.Unlock()
	defer other.mutex.Unlock()
	switch {
	case
		h.used != other.used:
		return false
	default:
		for i := int16(0); i < h.used; i++ {
			if h.bvs[i].Compare(&other.bvs[i]) != 0 {
				return false
			}
			if h.bvs[i].count != other.bvs[i].count {
				return false
			}
		}
	}
	return true
}

func (h *Histogram) CopyAndReset() *Histogram {
	h.mutex.Lock()
	defer h.mutex.Unlock()
	newhist := &Histogram{
		allocd: h.allocd,
		used:   h.used,
		bvs:    h.bvs,
	}
	h.allocd = DEFAULT_HIST_SIZE
	h.bvs = make([]Bin, DEFAULT_HIST_SIZE)
	h.used = 0
	return newhist
}
func (h *Histogram) DecStrings() []string {
	h.mutex.Lock()
	defer h.mutex.Unlock()
	out := make([]string, h.used)
	for i, bin := range h.bvs[0:h.used] {
		var buffer bytes.Buffer
		buffer.WriteString("H[")
		buffer.WriteString(fmt.Sprintf("%3.1e", bin.Value()))
		buffer.WriteString("]=")
		buffer.WriteString(fmt.Sprintf("%v", bin.count))
		out[i] = buffer.String()
	}
	return out
}
