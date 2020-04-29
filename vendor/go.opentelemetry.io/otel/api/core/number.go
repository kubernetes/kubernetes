// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package core

//go:generate stringer -type=NumberKind

import (
	"fmt"
	"math"
	"sync/atomic"
)

// NumberKind describes the data type of the Number.
type NumberKind int8

const (
	// Int64NumberKind means that the Number stores int64.
	Int64NumberKind NumberKind = iota
	// Float64NumberKind means that the Number stores float64.
	Float64NumberKind
	// Uint64NumberKind means that the Number stores uint64.
	Uint64NumberKind
)

// Zero returns a zero value for a given NumberKind
func (k NumberKind) Zero() Number {
	switch k {
	case Int64NumberKind:
		return NewInt64Number(0)
	case Float64NumberKind:
		return NewFloat64Number(0.)
	case Uint64NumberKind:
		return NewUint64Number(0)
	default:
		return Number(0)
	}
}

// Minimum returns the minimum representable value
// for a given NumberKind
func (k NumberKind) Minimum() Number {
	switch k {
	case Int64NumberKind:
		return NewInt64Number(math.MinInt64)
	case Float64NumberKind:
		return NewFloat64Number(-1. * math.MaxFloat64)
	case Uint64NumberKind:
		return NewUint64Number(0)
	default:
		return Number(0)
	}
}

// Maximum returns the maximum representable value
// for a given NumberKind
func (k NumberKind) Maximum() Number {
	switch k {
	case Int64NumberKind:
		return NewInt64Number(math.MaxInt64)
	case Float64NumberKind:
		return NewFloat64Number(math.MaxFloat64)
	case Uint64NumberKind:
		return NewUint64Number(math.MaxUint64)
	default:
		return Number(0)
	}
}

// Number represents either an integral or a floating point value. It
// needs to be accompanied with a source of NumberKind that describes
// the actual type of the value stored within Number.
type Number uint64

// - constructors

// NewNumberFromRaw creates a new Number from a raw value.
func NewNumberFromRaw(r uint64) Number {
	return Number(r)
}

// NewInt64Number creates an integral Number.
func NewInt64Number(i int64) Number {
	return NewNumberFromRaw(int64ToRaw(i))
}

// NewFloat64Number creates a floating point Number.
func NewFloat64Number(f float64) Number {
	return NewNumberFromRaw(float64ToRaw(f))
}

// NewInt64Number creates an integral Number.
func NewUint64Number(u uint64) Number {
	return NewNumberFromRaw(uint64ToRaw(u))
}

// - as x

// AsNumber gets the Number.
func (n *Number) AsNumber() Number {
	return *n
}

// AsRaw gets the uninterpreted raw value. Might be useful for some
// atomic operations.
func (n *Number) AsRaw() uint64 {
	return uint64(*n)
}

// AsInt64 assumes that the value contains an int64 and returns it as
// such.
func (n *Number) AsInt64() int64 {
	return rawToInt64(n.AsRaw())
}

// AsFloat64 assumes that the measurement value contains a float64 and
// returns it as such.
func (n *Number) AsFloat64() float64 {
	return rawToFloat64(n.AsRaw())
}

// AsUint64 assumes that the value contains an uint64 and returns it
// as such.
func (n *Number) AsUint64() uint64 {
	return rawToUint64(n.AsRaw())
}

// - as x atomic

// AsNumberAtomic gets the Number atomically.
func (n *Number) AsNumberAtomic() Number {
	return NewNumberFromRaw(n.AsRawAtomic())
}

// AsRawAtomic gets the uninterpreted raw value atomically. Might be
// useful for some atomic operations.
func (n *Number) AsRawAtomic() uint64 {
	return atomic.LoadUint64(n.AsRawPtr())
}

// AsInt64Atomic assumes that the number contains an int64 and returns
// it as such atomically.
func (n *Number) AsInt64Atomic() int64 {
	return atomic.LoadInt64(n.AsInt64Ptr())
}

// AsFloat64Atomic assumes that the measurement value contains a
// float64 and returns it as such atomically.
func (n *Number) AsFloat64Atomic() float64 {
	return rawToFloat64(n.AsRawAtomic())
}

// AsUint64Atomic assumes that the number contains a uint64 and
// returns it as such atomically.
func (n *Number) AsUint64Atomic() uint64 {
	return atomic.LoadUint64(n.AsUint64Ptr())
}

// - as x ptr

// AsRawPtr gets the pointer to the raw, uninterpreted raw
// value. Might be useful for some atomic operations.
func (n *Number) AsRawPtr() *uint64 {
	return (*uint64)(n)
}

// AsInt64Ptr assumes that the number contains an int64 and returns a
// pointer to it.
func (n *Number) AsInt64Ptr() *int64 {
	return rawPtrToInt64Ptr(n.AsRawPtr())
}

// AsFloat64Ptr assumes that the number contains a float64 and returns a
// pointer to it.
func (n *Number) AsFloat64Ptr() *float64 {
	return rawPtrToFloat64Ptr(n.AsRawPtr())
}

// AsUint64Ptr assumes that the number contains a uint64 and returns a
// pointer to it.
func (n *Number) AsUint64Ptr() *uint64 {
	return rawPtrToUint64Ptr(n.AsRawPtr())
}

// - coerce

// CoerceToInt64 casts the number to int64. May result in
// data/precision loss.
func (n *Number) CoerceToInt64(kind NumberKind) int64 {
	switch kind {
	case Int64NumberKind:
		return n.AsInt64()
	case Float64NumberKind:
		return int64(n.AsFloat64())
	case Uint64NumberKind:
		return int64(n.AsUint64())
	default:
		// you get what you deserve
		return 0
	}
}

// CoerceToFloat64 casts the number to float64. May result in
// data/precision loss.
func (n *Number) CoerceToFloat64(kind NumberKind) float64 {
	switch kind {
	case Int64NumberKind:
		return float64(n.AsInt64())
	case Float64NumberKind:
		return n.AsFloat64()
	case Uint64NumberKind:
		return float64(n.AsUint64())
	default:
		// you get what you deserve
		return 0
	}
}

// CoerceToUint64 casts the number to uint64. May result in
// data/precision loss.
func (n *Number) CoerceToUint64(kind NumberKind) uint64 {
	switch kind {
	case Int64NumberKind:
		return uint64(n.AsInt64())
	case Float64NumberKind:
		return uint64(n.AsFloat64())
	case Uint64NumberKind:
		return n.AsUint64()
	default:
		// you get what you deserve
		return 0
	}
}

// - set

// SetNumber sets the number to the passed number. Both should be of
// the same kind.
func (n *Number) SetNumber(nn Number) {
	*n.AsRawPtr() = nn.AsRaw()
}

// SetRaw sets the number to the passed raw value. Both number and the
// raw number should represent the same kind.
func (n *Number) SetRaw(r uint64) {
	*n.AsRawPtr() = r
}

// SetInt64 assumes that the number contains an int64 and sets it to
// the passed value.
func (n *Number) SetInt64(i int64) {
	*n.AsInt64Ptr() = i
}

// SetFloat64 assumes that the number contains a float64 and sets it
// to the passed value.
func (n *Number) SetFloat64(f float64) {
	*n.AsFloat64Ptr() = f
}

// SetUint64 assumes that the number contains a uint64 and sets it to
// the passed value.
func (n *Number) SetUint64(u uint64) {
	*n.AsUint64Ptr() = u
}

// - set atomic

// SetNumberAtomic sets the number to the passed number
// atomically. Both should be of the same kind.
func (n *Number) SetNumberAtomic(nn Number) {
	atomic.StoreUint64(n.AsRawPtr(), nn.AsRaw())
}

// SetRawAtomic sets the number to the passed raw value
// atomically. Both number and the raw number should represent the
// same kind.
func (n *Number) SetRawAtomic(r uint64) {
	atomic.StoreUint64(n.AsRawPtr(), r)
}

// SetInt64Atomic assumes that the number contains an int64 and sets
// it to the passed value atomically.
func (n *Number) SetInt64Atomic(i int64) {
	atomic.StoreInt64(n.AsInt64Ptr(), i)
}

// SetFloat64Atomic assumes that the number contains a float64 and
// sets it to the passed value atomically.
func (n *Number) SetFloat64Atomic(f float64) {
	atomic.StoreUint64(n.AsRawPtr(), float64ToRaw(f))
}

// SetUint64Atomic assumes that the number contains a uint64 and sets
// it to the passed value atomically.
func (n *Number) SetUint64Atomic(u uint64) {
	atomic.StoreUint64(n.AsUint64Ptr(), u)
}

// - swap

// SwapNumber sets the number to the passed number and returns the old
// number. Both this number and the passed number should be of the
// same kind.
func (n *Number) SwapNumber(nn Number) Number {
	old := *n
	n.SetNumber(nn)
	return old
}

// SwapRaw sets the number to the passed raw value and returns the old
// raw value. Both number and the raw number should represent the same
// kind.
func (n *Number) SwapRaw(r uint64) uint64 {
	old := n.AsRaw()
	n.SetRaw(r)
	return old
}

// SwapInt64 assumes that the number contains an int64, sets it to the
// passed value and returns the old int64 value.
func (n *Number) SwapInt64(i int64) int64 {
	old := n.AsInt64()
	n.SetInt64(i)
	return old
}

// SwapFloat64 assumes that the number contains an float64, sets it to
// the passed value and returns the old float64 value.
func (n *Number) SwapFloat64(f float64) float64 {
	old := n.AsFloat64()
	n.SetFloat64(f)
	return old
}

// SwapUint64 assumes that the number contains an uint64, sets it to
// the passed value and returns the old uint64 value.
func (n *Number) SwapUint64(u uint64) uint64 {
	old := n.AsUint64()
	n.SetUint64(u)
	return old
}

// - swap atomic

// SwapNumberAtomic sets the number to the passed number and returns
// the old number atomically. Both this number and the passed number
// should be of the same kind.
func (n *Number) SwapNumberAtomic(nn Number) Number {
	return NewNumberFromRaw(atomic.SwapUint64(n.AsRawPtr(), nn.AsRaw()))
}

// SwapRawAtomic sets the number to the passed raw value and returns
// the old raw value atomically. Both number and the raw number should
// represent the same kind.
func (n *Number) SwapRawAtomic(r uint64) uint64 {
	return atomic.SwapUint64(n.AsRawPtr(), r)
}

// SwapInt64Atomic assumes that the number contains an int64, sets it
// to the passed value and returns the old int64 value atomically.
func (n *Number) SwapInt64Atomic(i int64) int64 {
	return atomic.SwapInt64(n.AsInt64Ptr(), i)
}

// SwapFloat64Atomic assumes that the number contains an float64, sets
// it to the passed value and returns the old float64 value
// atomically.
func (n *Number) SwapFloat64Atomic(f float64) float64 {
	return rawToFloat64(atomic.SwapUint64(n.AsRawPtr(), float64ToRaw(f)))
}

// SwapUint64Atomic assumes that the number contains an uint64, sets
// it to the passed value and returns the old uint64 value atomically.
func (n *Number) SwapUint64Atomic(u uint64) uint64 {
	return atomic.SwapUint64(n.AsUint64Ptr(), u)
}

// - add

// AddNumber assumes that this and the passed number are of the passed
// kind and adds the passed number to this number.
func (n *Number) AddNumber(kind NumberKind, nn Number) {
	switch kind {
	case Int64NumberKind:
		n.AddInt64(nn.AsInt64())
	case Float64NumberKind:
		n.AddFloat64(nn.AsFloat64())
	case Uint64NumberKind:
		n.AddUint64(nn.AsUint64())
	}
}

// AddRaw assumes that this number and the passed raw value are of the
// passed kind and adds the passed raw value to this number.
func (n *Number) AddRaw(kind NumberKind, r uint64) {
	n.AddNumber(kind, NewNumberFromRaw(r))
}

// AddInt64 assumes that the number contains an int64 and adds the
// passed int64 to it.
func (n *Number) AddInt64(i int64) {
	*n.AsInt64Ptr() += i
}

// AddFloat64 assumes that the number contains a float64 and adds the
// passed float64 to it.
func (n *Number) AddFloat64(f float64) {
	*n.AsFloat64Ptr() += f
}

// AddUint64 assumes that the number contains a uint64 and adds the
// passed uint64 to it.
func (n *Number) AddUint64(u uint64) {
	*n.AsUint64Ptr() += u
}

// - add atomic

// AddNumberAtomic assumes that this and the passed number are of the
// passed kind and adds the passed number to this number atomically.
func (n *Number) AddNumberAtomic(kind NumberKind, nn Number) {
	switch kind {
	case Int64NumberKind:
		n.AddInt64Atomic(nn.AsInt64())
	case Float64NumberKind:
		n.AddFloat64Atomic(nn.AsFloat64())
	case Uint64NumberKind:
		n.AddUint64Atomic(nn.AsUint64())
	}
}

// AddRawAtomic assumes that this number and the passed raw value are
// of the passed kind and adds the passed raw value to this number
// atomically.
func (n *Number) AddRawAtomic(kind NumberKind, r uint64) {
	n.AddNumberAtomic(kind, NewNumberFromRaw(r))
}

// AddInt64Atomic assumes that the number contains an int64 and adds
// the passed int64 to it atomically.
func (n *Number) AddInt64Atomic(i int64) {
	atomic.AddInt64(n.AsInt64Ptr(), i)
}

// AddFloat64Atomic assumes that the number contains a float64 and
// adds the passed float64 to it atomically.
func (n *Number) AddFloat64Atomic(f float64) {
	for {
		o := n.AsFloat64Atomic()
		if n.CompareAndSwapFloat64(o, o+f) {
			break
		}
	}
}

// AddUint64Atomic assumes that the number contains a uint64 and
// atomically adds the passed uint64 to it.
func (n *Number) AddUint64Atomic(u uint64) {
	atomic.AddUint64(n.AsUint64Ptr(), u)
}

// - compare and swap (atomic only)

// CompareAndSwapNumber does the atomic CAS operation on this
// number. This number and passed old and new numbers should be of the
// same kind.
func (n *Number) CompareAndSwapNumber(on, nn Number) bool {
	return atomic.CompareAndSwapUint64(n.AsRawPtr(), on.AsRaw(), nn.AsRaw())
}

// CompareAndSwapRaw does the atomic CAS operation on this
// number. This number and passed old and new raw values should be of
// the same kind.
func (n *Number) CompareAndSwapRaw(or, nr uint64) bool {
	return atomic.CompareAndSwapUint64(n.AsRawPtr(), or, nr)
}

// CompareAndSwapInt64 assumes that this number contains an int64 and
// does the atomic CAS operation on it.
func (n *Number) CompareAndSwapInt64(oi, ni int64) bool {
	return atomic.CompareAndSwapInt64(n.AsInt64Ptr(), oi, ni)
}

// CompareAndSwapFloat64 assumes that this number contains a float64 and
// does the atomic CAS operation on it.
func (n *Number) CompareAndSwapFloat64(of, nf float64) bool {
	return atomic.CompareAndSwapUint64(n.AsRawPtr(), float64ToRaw(of), float64ToRaw(nf))
}

// CompareAndSwapUint64 assumes that this number contains a uint64 and
// does the atomic CAS operation on it.
func (n *Number) CompareAndSwapUint64(ou, nu uint64) bool {
	return atomic.CompareAndSwapUint64(n.AsUint64Ptr(), ou, nu)
}

// - compare

// CompareNumber compares two Numbers given their kind.  Both numbers
// should have the same kind.  This returns:
//    0 if the numbers are equal
//    -1 if the subject `n` is less than the argument `nn`
//    +1 if the subject `n` is greater than the argument `nn`
func (n *Number) CompareNumber(kind NumberKind, nn Number) int {
	switch kind {
	case Int64NumberKind:
		return n.CompareInt64(nn.AsInt64())
	case Float64NumberKind:
		return n.CompareFloat64(nn.AsFloat64())
	case Uint64NumberKind:
		return n.CompareUint64(nn.AsUint64())
	default:
		// you get what you deserve
		return 0
	}
}

// CompareRaw compares two numbers, where one is input as a raw
// uint64, interpreting both values as a `kind` of number.
func (n *Number) CompareRaw(kind NumberKind, r uint64) int {
	return n.CompareNumber(kind, NewNumberFromRaw(r))
}

// CompareInt64 assumes that the Number contains an int64 and performs
// a comparison between the value and the other value. It returns the
// typical result of the compare function: -1 if the value is less
// than the other, 0 if both are equal, 1 if the value is greater than
// the other.
func (n *Number) CompareInt64(i int64) int {
	this := n.AsInt64()
	if this < i {
		return -1
	} else if this > i {
		return 1
	}
	return 0
}

// CompareFloat64 assumes that the Number contains a float64 and
// performs a comparison between the value and the other value. It
// returns the typical result of the compare function: -1 if the value
// is less than the other, 0 if both are equal, 1 if the value is
// greater than the other.
//
// Do not compare NaN values.
func (n *Number) CompareFloat64(f float64) int {
	this := n.AsFloat64()
	if this < f {
		return -1
	} else if this > f {
		return 1
	}
	return 0
}

// CompareUint64 assumes that the Number contains an uint64 and performs
// a comparison between the value and the other value. It returns the
// typical result of the compare function: -1 if the value is less
// than the other, 0 if both are equal, 1 if the value is greater than
// the other.
func (n *Number) CompareUint64(u uint64) int {
	this := n.AsUint64()
	if this < u {
		return -1
	} else if this > u {
		return 1
	}
	return 0
}

// - relations to zero

// IsPositive returns true if the actual value is greater than zero.
func (n *Number) IsPositive(kind NumberKind) bool {
	return n.compareWithZero(kind) > 0
}

// IsNegative returns true if the actual value is less than zero.
func (n *Number) IsNegative(kind NumberKind) bool {
	return n.compareWithZero(kind) < 0
}

// IsZero returns true if the actual value is equal to zero.
func (n *Number) IsZero(kind NumberKind) bool {
	return n.compareWithZero(kind) == 0
}

// - misc

// Emit returns a string representation of the raw value of the
// Number. A %d is used for integral values, %f for floating point
// values.
func (n *Number) Emit(kind NumberKind) string {
	switch kind {
	case Int64NumberKind:
		return fmt.Sprintf("%d", n.AsInt64())
	case Float64NumberKind:
		return fmt.Sprintf("%f", n.AsFloat64())
	case Uint64NumberKind:
		return fmt.Sprintf("%d", n.AsUint64())
	default:
		return ""
	}
}

// AsInterface returns the number as an interface{}, typically used
// for NumberKind-correct JSON conversion.
func (n *Number) AsInterface(kind NumberKind) interface{} {
	switch kind {
	case Int64NumberKind:
		return n.AsInt64()
	case Float64NumberKind:
		return n.AsFloat64()
	case Uint64NumberKind:
		return n.AsUint64()
	default:
		return math.NaN()
	}
}

// - private stuff

func (n *Number) compareWithZero(kind NumberKind) int {
	switch kind {
	case Int64NumberKind:
		return n.CompareInt64(0)
	case Float64NumberKind:
		return n.CompareFloat64(0.)
	case Uint64NumberKind:
		return n.CompareUint64(0)
	default:
		// you get what you deserve
		return 0
	}
}
