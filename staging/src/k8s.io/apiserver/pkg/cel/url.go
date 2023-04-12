/*
Copyright 2022 The Kubernetes Authors.

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
	"net/url"
	"reflect"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker/decls"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

// URL provides a CEL representation of a URL.
type URL struct {
	*url.URL
}

var (
	URLObject = decls.NewObjectType("kubernetes.URL")
	typeValue = types.NewTypeValue("kubernetes.URL")
	URLType   = cel.ObjectType("kubernetes.URL")
)

// ConvertToNative implements ref.Val.ConvertToNative.
func (d URL) ConvertToNative(typeDesc reflect.Type) (interface{}, error) {
	if reflect.TypeOf(d.URL).AssignableTo(typeDesc) {
		return d.URL, nil
	}
	if reflect.TypeOf("").AssignableTo(typeDesc) {
		return d.URL.String(), nil
	}
	return nil, fmt.Errorf("type conversion error from 'URL' to '%v'", typeDesc)
}

// ConvertToType implements ref.Val.ConvertToType.
func (d URL) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case typeValue:
		return d
	case types.TypeType:
		return typeValue
	}
	return types.NewErr("type conversion error from '%s' to '%s'", typeValue, typeVal)
}

// Equal implements ref.Val.Equal.
func (d URL) Equal(other ref.Val) ref.Val {
	otherDur, ok := other.(URL)
	if !ok {
		return types.MaybeNoSuchOverloadErr(other)
	}
	return types.Bool(d.URL.String() == otherDur.URL.String())
}

// Type implements ref.Val.Type.
func (d URL) Type() ref.Type {
	return typeValue
}

// Value implements ref.Val.Value.
func (d URL) Value() interface{} {
	return d.URL
}
