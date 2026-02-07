// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package json

import (
	stdjson "encoding/json"

	"github.com/go-openapi/swag/jsonutils/adapters/ifaces"
	"github.com/go-openapi/swag/typeutils"
)

const sensibleBufferSize = 8192

type jsonError string

func (e jsonError) Error() string {
	return string(e)
}

// ErrStdlib indicates that an error comes from the stdlib JSON adapter
var ErrStdlib jsonError = "error from the JSON adapter stdlib"

var _ ifaces.Adapter = &Adapter{}

type Adapter struct {
}

// NewAdapter yields an [ifaces.Adapter] using the standard library.
func NewAdapter() *Adapter {
	return &Adapter{}
}

func (a *Adapter) Marshal(value any) ([]byte, error) {
	return stdjson.Marshal(value)
}

func (a *Adapter) Unmarshal(data []byte, value any) error {
	return stdjson.Unmarshal(data, value)
}

func (a *Adapter) OrderedMarshal(value ifaces.Ordered) ([]byte, error) {
	w := poolOfWriters.Borrow()
	defer func() {
		poolOfWriters.Redeem(w)
	}()

	if typeutils.IsNil(value) {
		w.RawString("null")

		return w.BuildBytes()
	}

	w.RawByte('{')
	first := true
	for k, v := range value.OrderedItems() {
		if first {
			first = false
		} else {
			w.RawByte(',')
		}

		w.String(k)
		w.RawByte(':')

		switch val := v.(type) {
		case ifaces.Ordered:
			w.Raw(a.OrderedMarshal(val))
		default:
			w.Raw(stdjson.Marshal(v))
		}
	}

	w.RawByte('}')

	return w.BuildBytes()
}

func (a *Adapter) OrderedUnmarshal(data []byte, value ifaces.SetOrdered) error {
	var m MapSlice
	if err := m.OrderedUnmarshalJSON(data); err != nil {
		return err
	}

	if typeutils.IsNil(m) {
		// force input value to nil
		value.SetOrderedItems(nil)

		return nil
	}

	value.SetOrderedItems(m.OrderedItems())

	return nil
}

func (a *Adapter) NewOrderedMap(capacity int) ifaces.OrderedMap {
	m := make(MapSlice, 0, capacity)

	return &m
}

// Redeem the [Adapter] when it comes from a pool.
//
// The adapter becomes immediately unusable once redeemed.
func (a *Adapter) Redeem() {
	if a == nil {
		return
	}

	RedeemAdapter(a)
}

func (a *Adapter) Reset() {
}
