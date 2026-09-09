// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package keys

import (
	"fmt"
	"math"
	"strconv"

	"golang.org/x/tools/internal/event/label"
)

// Value is a [label.Key] for untyped values.
type Value struct {
	name        string
	description string
}

// New creates a new Key for untyped values.
func New(name, description string) *Value {
	return &Value{name: name, description: description}
}

func (k *Value) Name() string        { return k.name }
func (k *Value) Description() string { return k.description }

func (k *Value) Append(buf []byte, l label.Label) []byte {
	return fmt.Append(buf, k.From(l))
}

// Get returns the label for the key of a label.Map.
func (k *Value) Get(lm label.Map) any {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return nil
}

// From returns the value of a Label.
func (k *Value) From(t label.Label) any { return t.UnpackValue() }

// Of creates a new Label with this key and the supplied value.
func (k *Value) Of(value any) label.Label { return label.OfValue(k, value) }

// Tag represents a key for tagging labels that have no value.
// These are used when the existence of the label is the entire information it
// carries, such as marking events to be of a specific kind, or from a specific
// package.
type Tag struct {
	name        string
	description string
}

// NewTag creates a new [label.Key] for tagging labels.
func NewTag(name, description string) *Tag {
	return &Tag{name: name, description: description}
}

func (k *Tag) Name() string        { return k.name }
func (k *Tag) Description() string { return k.description }

func (k *Tag) Append(buf []byte, l label.Label) []byte { return buf }

// New creates a new Label with this key.
func (k *Tag) New() label.Label { return label.OfValue(k, nil) }

// Int is a [label.Key] for signed integers.
type Int struct {
	name        string
	description string
}

// NewInt returns a new [label.Key] for int64 values.
func NewInt(name, description string) *Int {
	return &Int{name: name, description: description}
}

func (k *Int) Name() string        { return k.name }
func (k *Int) Description() string { return k.description }

func (k *Int) Append(buf []byte, l label.Label) []byte {
	return strconv.AppendInt(buf, k.From(l), 10)
}

// Of creates a new Label with this key and the supplied value.
func (k *Int) Of(v int) label.Label { return k.Of64(int64(v)) }

// Of64 creates a new Label with this key and the supplied value.
func (k *Int) Of64(v int64) label.Label { return label.Of64(k, uint64(v)) }

// Get returns the label for the key of a label.Map.
func (k *Int) Get(lm label.Map) int64 {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return 0
}

// From returns the value of a Label.
func (k *Int) From(t label.Label) int64 { return int64(t.Unpack64()) }

// Uint is a [label.Key] for unsigned integers.
type Uint struct {
	name        string
	description string
}

// NewUint creates a new [label.Key] for unsigned values.
func NewUint(name, description string) *Uint {
	return &Uint{name: name, description: description}
}

func (k *Uint) Name() string        { return k.name }
func (k *Uint) Description() string { return k.description }

func (k *Uint) Append(buf []byte, l label.Label) []byte {
	return strconv.AppendUint(buf, k.From(l), 10)
}

// Of creates a new Label with this key and the supplied value.
func (k *Uint) Of(v uint64) label.Label { return label.Of64(k, v) }

// Get returns the label for the key of a label.Map.
func (k *Uint) Get(lm label.Map) uint64 {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return 0
}

// From returns the value of a Label.
func (k *Uint) From(t label.Label) uint64 { return t.Unpack64() }

// Float is a label.Key for floating-point values.
type Float struct {
	name        string
	description string
}

// NewFloat creates a new [label.Key] for floating-point values.
func NewFloat(name, description string) *Float {
	return &Float{name: name, description: description}
}

func (k *Float) Name() string        { return k.name }
func (k *Float) Description() string { return k.description }

func (k *Float) Append(buf []byte, l label.Label) []byte {
	return strconv.AppendFloat(buf, k.From(l), 'E', -1, 64)
}

// Of creates a new Label with this key and the supplied value.
func (k *Float) Of(v float64) label.Label {
	return label.Of64(k, math.Float64bits(v))
}

// Get returns the label for the key of a label.Map.
func (k *Float) Get(lm label.Map) float64 {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return 0
}

// From returns the value of a Label.
func (k *Float) From(t label.Label) float64 {
	return math.Float64frombits(t.Unpack64())
}

// String represents a key
type String struct {
	name        string
	description string
}

// NewString creates a new Key for int64 values.
func NewString(name, description string) *String {
	return &String{name: name, description: description}
}

func (k *String) Name() string        { return k.name }
func (k *String) Description() string { return k.description }

func (k *String) Append(buf []byte, l label.Label) []byte {
	return strconv.AppendQuote(buf, k.From(l))
}

// Of creates a new Label with this key and the supplied value.
func (k *String) Of(v string) label.Label { return label.OfString(k, v) }

// Get returns the label for the key of a label.Map.
func (k *String) Get(lm label.Map) string {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return ""
}

// From returns the value of a Label.
func (k *String) From(t label.Label) string { return t.UnpackString() }

// Error represents a key
type Error struct {
	name        string
	description string
}

// NewError returns a new [label.Key] for error values.
func NewError(name, description string) *Error {
	return &Error{name: name, description: description}
}

func (k *Error) Name() string        { return k.name }
func (k *Error) Description() string { return k.description }

func (k *Error) Append(buf []byte, l label.Label) []byte {
	return append(buf, k.From(l).Error()...)
}

// Of returns a new Label with this key and the supplied value.
func (k *Error) Of(v error) label.Label { return label.OfValue(k, v) }

// Get returns the label for the key of a label.Map.
func (k *Error) Get(lm label.Map) error {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return nil
}

// From returns the value of a Label.
func (k *Error) From(t label.Label) error {
	err, _ := t.UnpackValue().(error)
	return err
}
