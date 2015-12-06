// +build proto

/*
Copyright 2015 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package resource

import (
	"math/big"

	"speter.net/go/exp/math/dec/inf"
)

// QuantityProto is a struct that is equivalent to Quantity, but intended for
// protobuf marshalling/unmarshalling. It is generated into a serialization
// that matches Quantity. Do not use in Go structs.
//
// +protobuf=true
type QuantityProto struct {
	// The format of the quantity
	Format Format
	// The scale dimension of the value
	Scale int32
	// Bigint is serialized as a raw bytes array
	Bigint []byte
}

// ProtoTime returns the Time as a new ProtoTime value.
func (q *Quantity) QuantityProto() *QuantityProto {
	if q == nil {
		return &QuantityProto{}
	}
	p := &QuantityProto{
		Format: q.Format,
	}
	if q.Amount != nil {
		p.Scale = int32(q.Amount.Scale())
		p.Bigint = q.Amount.UnscaledBig().Bytes()
	}
	return p
}

// Size implements the protobuf marshalling interface.
func (q *Quantity) Size() (n int) { return q.QuantityProto().Size() }

// Reset implements the protobuf marshalling interface.
func (q *Quantity) Unmarshal(data []byte) error {
	p := QuantityProto{}
	if err := p.Unmarshal(data); err != nil {
		return err
	}
	q.Format = p.Format
	b := big.NewInt(0)
	b.SetBytes(p.Bigint)
	q.Amount = inf.NewDecBig(b, inf.Scale(p.Scale))
	return nil
}

// Marshal implements the protobuf marshalling interface.
func (q *Quantity) Marshal() (data []byte, err error) {
	return q.QuantityProto().Marshal()
}

// MarshalTo implements the protobuf marshalling interface.
func (q *Quantity) MarshalTo(data []byte) (int, error) {
	return q.QuantityProto().MarshalTo(data)
}
