// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package attribute

import (
	"fmt"
	"reflect"

	"go.opentelemetry.io/otel/attribute/internal/xxhash"
)

// Type identifiers. These identifiers are hashed before the value of the
// corresponding type. This is done to distinguish values that are hashed with
// the same value representation (e.g. `int64(1)` and `true`, []int64{0} and
// int64(0)).
//
// These are all 8 byte length strings converted to a uint64 representation. A
// uint64 is used instead of the string directly as an optimization, it avoids
// the for loop in [xxhash] which adds minor overhead.
const (
	boolID         uint64 = 7953749933313450591 // "_boolean" (little endian)
	int64ID        uint64 = 7592915492740740150 // "64_bit_i" (little endian)
	float64ID      uint64 = 7376742710626956342 // "64_bit_f" (little endian)
	stringID       uint64 = 6874584755375207263 // "_string_" (little endian)
	boolSliceID    uint64 = 6875993255270243167 // "_[]bool_" (little endian)
	int64SliceID   uint64 = 3762322556277578591 // "_[]int64" (little endian)
	float64SliceID uint64 = 7308324551835016539 // "[]double" (little endian)
	stringSliceID  uint64 = 7453010373645655387 // "[]string" (little endian)
	byteSliceID    uint64 = 6874028470941080415 // "_[]byte_" (little endian)
	sliceID        uint64 = 7883494272577650031 // "__slice_" (little endian)
	mapID          uint64 = 6872316492666199903 // "__map___" (little endian)
	emptyID        uint64 = 7305809155345288421 // "__empty_" (little endian)
)

// Hasher computes a Distinct value from KeyValue attributes supplied with
// Write.
//
// A Hasher must be obtained from [NewHasher]. The zero value is not usable and
// its methods will panic.
type Hasher struct {
	h xxhash.Hash
}

// NewHasher returns a new Hasher.
func NewHasher() *Hasher {
	return &Hasher{h: xxhash.New()}
}

// Reset resets h to its initial state so it can be reused.
func (h *Hasher) Reset() {
	h.h.Reset()
}

// Write adds kv to the hash.
//
// Write requires attributes to be supplied in ascending key order with no
// duplicate keys. To produce the same Distinct as Set.Equivalent, write
// attributes in ascending key order, with no more than one value for each key.
// If the source contains duplicate keys, retain the last value for each key
// before calling Write.
func (h *Hasher) Write(kv KeyValue) {
	// hashKV mutates the digest h.h refers to in place and returns the same
	// Hash value it was passed. Discarding the result keeps the digest pointer
	// from flowing back into h, which would force the digest to be heap
	// allocated for every Hasher. Keeping Write this small also keeps it within
	// the inlining budget, which matters because hashKVs calls it per attribute.
	_ = hashKV(h.h, kv)
}

// Distinct returns the identifier for the attributes written to h. When Write
// is called as described above, it returns the same value as [Set.Equivalent].
func (h *Hasher) Distinct() Distinct {
	// No count of written attributes is needed to detect the empty case. The
	// sum of a digest with nothing written to it is emptyHash, which is
	// non-zero (0xef46db3751d8e999), so it passes through remapZeroHash
	// unchanged and matches emptySet.Equivalent.
	return Distinct{hash: remapZeroHash(h.h.Sum64())}
}

// remapZeroHash remaps a 0 sum to a non-zero value, because hash == 0 is a
// reserved sentinel (treated as empty/invalid).
func remapZeroHash(sum uint64) uint64 {
	if sum == 0 {
		return 1
	}
	return sum
}

// hashKVs returns a new xxHash64 hash of kvs.
//
// This routes through [Hasher] so that Set hashing and Hasher cannot disagree:
// there is exactly one implementation of how attributes are mixed and how the
// final sum is framed.
func hashKVs(kvs []KeyValue) uint64 {
	h := NewHasher()
	for _, kv := range kvs {
		h.Write(kv)
	}
	return h.Distinct().hash
}

// hashKV returns the xxHash64 hash of kv with h as the base.
func hashKV(h xxhash.Hash, kv KeyValue) xxhash.Hash {
	h = h.String(string(kv.Key))
	return hashValue(h, kv.Value)
}

func hashValue(h xxhash.Hash, v Value) xxhash.Hash {
	switch v.Type() {
	case BOOL:
		h = h.Uint64(boolID)
		h = h.Uint64(v.numeric)
	case INT64:
		h = h.Uint64(int64ID)
		h = h.Uint64(v.numeric)
	case FLOAT64:
		h = h.Uint64(float64ID)
		// Assumes numeric stored with math.Float64bits.
		h = h.Uint64(v.numeric)
	case STRING:
		h = h.Uint64(stringID)
		h = h.String(v.stringly)
	case BOOLSLICE:
		h = h.Uint64(boolSliceID)
		switch vals := v.slice.(type) {
		case [0]bool:
		case [1]bool:
			h = h.Bool(vals[0])
		case [2]bool:
			h = h.Bool(vals[0])
			h = h.Bool(vals[1])
		case [3]bool:
			h = h.Bool(vals[0])
			h = h.Bool(vals[1])
			h = h.Bool(vals[2])
		default:
			rv := reflect.ValueOf(v.slice)
			for i := 0; i < rv.Len(); i++ {
				h = h.Bool(rv.Index(i).Bool())
			}
		}
	case INT64SLICE:
		h = h.Uint64(int64SliceID)
		switch vals := v.slice.(type) {
		case [0]int64:
		case [1]int64:
			h = h.Int64(vals[0])
		case [2]int64:
			h = h.Int64(vals[0])
			h = h.Int64(vals[1])
		case [3]int64:
			h = h.Int64(vals[0])
			h = h.Int64(vals[1])
			h = h.Int64(vals[2])
		default:
			rv := reflect.ValueOf(v.slice)
			for i := 0; i < rv.Len(); i++ {
				h = h.Int64(rv.Index(i).Int())
			}
		}
	case FLOAT64SLICE:
		h = h.Uint64(float64SliceID)
		switch vals := v.slice.(type) {
		case [0]float64:
		case [1]float64:
			h = h.Float64(vals[0])
		case [2]float64:
			h = h.Float64(vals[0])
			h = h.Float64(vals[1])
		case [3]float64:
			h = h.Float64(vals[0])
			h = h.Float64(vals[1])
			h = h.Float64(vals[2])
		default:
			rv := reflect.ValueOf(v.slice)
			for i := 0; i < rv.Len(); i++ {
				h = h.Float64(rv.Index(i).Float())
			}
		}
	case STRINGSLICE:
		h = h.Uint64(stringSliceID)
		switch vals := v.slice.(type) {
		case [0]string:
		case [1]string:
			h = h.String(vals[0])
		case [2]string:
			h = h.String(vals[0])
			h = h.String(vals[1])
		case [3]string:
			h = h.String(vals[0])
			h = h.String(vals[1])
			h = h.String(vals[2])
		default:
			rv := reflect.ValueOf(v.slice)
			for i := 0; i < rv.Len(); i++ {
				h = h.String(rv.Index(i).String())
			}
		}
	case BYTESLICE:
		h = h.Uint64(byteSliceID)
		h = h.String(v.stringly)
	case SLICE:
		h = h.Uint64(sliceID)
		switch vals := v.slice.(type) {
		case [0]Value:
			// No values to hash, but the type identifier is still hashed above.
		case [1]Value:
			h = hashValue(h, vals[0])
		case [2]Value:
			h = hashValue(h, vals[0])
			h = hashValue(h, vals[1])
		case [3]Value:
			h = hashValue(h, vals[0])
			h = hashValue(h, vals[1])
			h = hashValue(h, vals[2])
		case [4]Value:
			h = hashValue(h, vals[0])
			h = hashValue(h, vals[1])
			h = hashValue(h, vals[2])
			h = hashValue(h, vals[3])
		case [5]Value:
			h = hashValue(h, vals[0])
			h = hashValue(h, vals[1])
			h = hashValue(h, vals[2])
			h = hashValue(h, vals[3])
			h = hashValue(h, vals[4])
		default:
			rv := reflect.ValueOf(v.slice)
			for i := 0; i < rv.Len(); i++ {
				h = hashValue(h, rv.Index(i).Interface().(Value))
			}
		}
	case MAP:
		h = h.Uint64(mapID)
		switch vals := v.slice.(type) {
		case [0]KeyValue:
			// No values to hash, but the type identifier is still hashed above.
		case [1]KeyValue:
			h = h.String(string(vals[0].Key))
			h = hashValue(h, vals[0].Value)
		case [2]KeyValue:
			h = h.String(string(vals[0].Key))
			h = hashValue(h, vals[0].Value)
			h = h.String(string(vals[1].Key))
			h = hashValue(h, vals[1].Value)
		case [3]KeyValue:
			h = h.String(string(vals[0].Key))
			h = hashValue(h, vals[0].Value)
			h = h.String(string(vals[1].Key))
			h = hashValue(h, vals[1].Value)
			h = h.String(string(vals[2].Key))
			h = hashValue(h, vals[2].Value)
		case [4]KeyValue:
			h = h.String(string(vals[0].Key))
			h = hashValue(h, vals[0].Value)
			h = h.String(string(vals[1].Key))
			h = hashValue(h, vals[1].Value)
			h = h.String(string(vals[2].Key))
			h = hashValue(h, vals[2].Value)
			h = h.String(string(vals[3].Key))
			h = hashValue(h, vals[3].Value)
		case [5]KeyValue:
			h = h.String(string(vals[0].Key))
			h = hashValue(h, vals[0].Value)
			h = h.String(string(vals[1].Key))
			h = hashValue(h, vals[1].Value)
			h = h.String(string(vals[2].Key))
			h = hashValue(h, vals[2].Value)
			h = h.String(string(vals[3].Key))
			h = hashValue(h, vals[3].Value)
			h = h.String(string(vals[4].Key))
			h = hashValue(h, vals[4].Value)
		default:
			rv := reflect.ValueOf(v.slice)
			for i := 0; i < rv.Len(); i++ {
				kv := rv.Index(i).Interface().(KeyValue)
				h = h.String(string(kv.Key))
				h = hashValue(h, kv.Value)
			}
		}
	case EMPTY:
		h = h.Uint64(emptyID)
	default:
		// Logging is an alternative, but using the internal logger here
		// causes an import cycle so it is not done.
		val := v.AsInterface()
		msg := fmt.Sprintf("unknown value type: %[1]v (%[1]T)", val)
		panic(msg)
	}
	return h
}
