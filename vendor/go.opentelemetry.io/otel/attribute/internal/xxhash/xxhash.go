// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package xxhash provides a wrapper around the xxhash library for attribute hashing.
package xxhash // import "go.opentelemetry.io/otel/attribute/internal/xxhash"

import (
	"encoding/binary"
	"math"

	"github.com/cespare/xxhash/v2"
)

// Hash wraps xxhash.Digest to provide an API friendly for hashing attribute values.
type Hash struct {
	d *xxhash.Digest
}

// New returns a new initialized xxHash64 hasher.
func New() Hash {
	return Hash{d: xxhash.New()}
}

func (h Hash) Uint64(val uint64) Hash {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], val)
	// errors from Write are always nil for xxhash
	// if it returns an err then panic
	_, err := h.d.Write(buf[:])
	if err != nil {
		panic("xxhash write of uint64 failed: " + err.Error())
	}
	return h
}

func (h Hash) Bool(val bool) Hash { // nolint:revive // This is a hashing function.
	if val {
		return h.Uint64(1)
	}
	return h.Uint64(0)
}

func (h Hash) Float64(val float64) Hash {
	return h.Uint64(math.Float64bits(val))
}

func (h Hash) Int64(val int64) Hash {
	return h.Uint64(uint64(val)) // nolint:gosec // Overflow doesn't matter since we are hashing.
}

func (h Hash) String(val string) Hash {
	// errors from WriteString are always nil for xxhash
	// if it returns an err then panic
	_, err := h.d.WriteString(val)
	if err != nil {
		panic("xxhash write of string failed: " + err.Error())
	}
	return h
}

// Sum64 returns the current hash value.
func (h Hash) Sum64() uint64 {
	return h.d.Sum64()
}
