// Copyright 2013 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"encoding/json"
	"fmt"
)

// Value is a generic interface for values resulting from a query evaluation.
type Value interface {
	Type() ValueType
	String() string
}

func (Matrix) Type() ValueType  { return ValMatrix }
func (Vector) Type() ValueType  { return ValVector }
func (*Scalar) Type() ValueType { return ValScalar }
func (*String) Type() ValueType { return ValString }

type ValueType int

const (
	ValNone ValueType = iota
	ValScalar
	ValVector
	ValMatrix
	ValString
)

// MarshalJSON implements json.Marshaler.
func (et ValueType) MarshalJSON() ([]byte, error) {
	return json.Marshal(et.String())
}

func (et *ValueType) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	switch s {
	case "<ValNone>":
		*et = ValNone
	case "scalar":
		*et = ValScalar
	case "vector":
		*et = ValVector
	case "matrix":
		*et = ValMatrix
	case "string":
		*et = ValString
	default:
		return fmt.Errorf("unknown value type %q", s)
	}
	return nil
}

func (et ValueType) String() string {
	switch et {
	case ValNone:
		return "<ValNone>"
	case ValScalar:
		return "scalar"
	case ValVector:
		return "vector"
	case ValMatrix:
		return "matrix"
	case ValString:
		return "string"
	}
	panic("ValueType.String: unhandled value type")
}
