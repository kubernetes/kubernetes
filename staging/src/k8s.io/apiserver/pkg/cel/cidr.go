/*
Copyright 2023 The Kubernetes Authors.

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

package cel

import (
	"fmt"
	"math"
	"net/netip"
	"reflect"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

// CIDR provides a CEL representation of an network address.
type CIDR struct {
	netip.Prefix
}

var (
	CIDRType = cel.ObjectType("net.CIDR")
)

// ConvertToNative implements ref.Val.ConvertToNative.
func (d CIDR) ConvertToNative(typeDesc reflect.Type) (any, error) {
	if reflect.TypeOf(d.Prefix).AssignableTo(typeDesc) {
		return d.Prefix, nil
	}
	if reflect.TypeOf("").AssignableTo(typeDesc) {
		return d.Prefix.String(), nil
	}
	return nil, fmt.Errorf("type conversion error from 'CIDR' to '%v'", typeDesc)
}

// ConvertToType implements ref.Val.ConvertToType.
func (d CIDR) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case CIDRType:
		return d
	case types.TypeType:
		return CIDRType
	case types.StringType:
		return types.String(d.Prefix.String())
	}
	return types.NewErr("type conversion error from '%s' to '%s'", CIDRType, typeVal)
}

// Equal implements ref.Val.Equal.
func (d CIDR) Equal(other ref.Val) ref.Val {
	otherD, ok := other.(CIDR)
	if !ok {
		return types.ValOrErr(other, "no such overload")
	}

	return types.Bool(d.Prefix == otherD.Prefix)
}

// Type implements ref.Val.Type.
func (d CIDR) Type() ref.Type {
	return CIDRType
}

// Value implements ref.Val.Value.
func (d CIDR) Value() any {
	return d.Prefix
}

// Size returns the size of the CIDR prefix address in bytes.
// Used in the size estimation of the runtime cost.
func (d CIDR) Size() ref.Val {
	return types.Int(int(math.Ceil(float64(d.Prefix.Bits()) / 8)))
}
