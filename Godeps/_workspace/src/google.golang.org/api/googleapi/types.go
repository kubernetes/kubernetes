// Copyright 2013 Google Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package googleapi

import (
	"encoding/json"
	"strconv"
)

// Int64s is a slice of int64s that marshal as quoted strings in JSON.
type Int64s []int64

func (q *Int64s) UnmarshalJSON(raw []byte) error {
	*q = (*q)[:0]
	var ss []string
	if err := json.Unmarshal(raw, &ss); err != nil {
		return err
	}
	for _, s := range ss {
		v, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			return err
		}
		*q = append(*q, int64(v))
	}
	return nil
}

// Int32s is a slice of int32s that marshal as quoted strings in JSON.
type Int32s []int32

func (q *Int32s) UnmarshalJSON(raw []byte) error {
	*q = (*q)[:0]
	var ss []string
	if err := json.Unmarshal(raw, &ss); err != nil {
		return err
	}
	for _, s := range ss {
		v, err := strconv.ParseInt(s, 10, 32)
		if err != nil {
			return err
		}
		*q = append(*q, int32(v))
	}
	return nil
}

// Uint64s is a slice of uint64s that marshal as quoted strings in JSON.
type Uint64s []uint64

func (q *Uint64s) UnmarshalJSON(raw []byte) error {
	*q = (*q)[:0]
	var ss []string
	if err := json.Unmarshal(raw, &ss); err != nil {
		return err
	}
	for _, s := range ss {
		v, err := strconv.ParseUint(s, 10, 64)
		if err != nil {
			return err
		}
		*q = append(*q, uint64(v))
	}
	return nil
}

// Uint32s is a slice of uint32s that marshal as quoted strings in JSON.
type Uint32s []uint32

func (q *Uint32s) UnmarshalJSON(raw []byte) error {
	*q = (*q)[:0]
	var ss []string
	if err := json.Unmarshal(raw, &ss); err != nil {
		return err
	}
	for _, s := range ss {
		v, err := strconv.ParseUint(s, 10, 32)
		if err != nil {
			return err
		}
		*q = append(*q, uint32(v))
	}
	return nil
}

// Float64s is a slice of float64s that marshal as quoted strings in JSON.
type Float64s []float64

func (q *Float64s) UnmarshalJSON(raw []byte) error {
	*q = (*q)[:0]
	var ss []string
	if err := json.Unmarshal(raw, &ss); err != nil {
		return err
	}
	for _, s := range ss {
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return err
		}
		*q = append(*q, float64(v))
	}
	return nil
}

func quotedList(n int, fn func(dst []byte, i int) []byte) ([]byte, error) {
	dst := make([]byte, 0, 2+n*10) // somewhat arbitrary
	dst = append(dst, '[')
	for i := 0; i < n; i++ {
		if i > 0 {
			dst = append(dst, ',')
		}
		dst = append(dst, '"')
		dst = fn(dst, i)
		dst = append(dst, '"')
	}
	dst = append(dst, ']')
	return dst, nil
}

func (s Int64s) MarshalJSON() ([]byte, error) {
	return quotedList(len(s), func(dst []byte, i int) []byte {
		return strconv.AppendInt(dst, s[i], 10)
	})
}

func (s Int32s) MarshalJSON() ([]byte, error) {
	return quotedList(len(s), func(dst []byte, i int) []byte {
		return strconv.AppendInt(dst, int64(s[i]), 10)
	})
}

func (s Uint64s) MarshalJSON() ([]byte, error) {
	return quotedList(len(s), func(dst []byte, i int) []byte {
		return strconv.AppendUint(dst, s[i], 10)
	})
}

func (s Uint32s) MarshalJSON() ([]byte, error) {
	return quotedList(len(s), func(dst []byte, i int) []byte {
		return strconv.AppendUint(dst, uint64(s[i]), 10)
	})
}

func (s Float64s) MarshalJSON() ([]byte, error) {
	return quotedList(len(s), func(dst []byte, i int) []byte {
		return strconv.AppendFloat(dst, s[i], 'g', -1, 64)
	})
}

/*
 * Helper routines for simplifying the creation of optional fields of basic type.
 */

// Bool is a helper routine that allocates a new bool value
// to store v and returns a pointer to it.
func Bool(v bool) *bool { return &v }

// Int32 is a helper routine that allocates a new int32 value
// to store v and returns a pointer to it.
func Int32(v int32) *int32 { return &v }

// Int64 is a helper routine that allocates a new int64 value
// to store v and returns a pointer to it.
func Int64(v int64) *int64 { return &v }

// Float64 is a helper routine that allocates a new float64 value
// to store v and returns a pointer to it.
func Float64(v float64) *float64 { return &v }

// Uint32 is a helper routine that allocates a new uint32 value
// to store v and returns a pointer to it.
func Uint32(v uint32) *uint32 { return &v }

// Uint64 is a helper routine that allocates a new uint64 value
// to store v and returns a pointer to it.
func Uint64(v uint64) *uint64 { return &v }

// String is a helper routine that allocates a new string value
// to store v and returns a pointer to it.
func String(v string) *string { return &v }
