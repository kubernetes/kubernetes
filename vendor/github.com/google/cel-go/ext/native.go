// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package ext

import (
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
)

// NativeTypesOption is a functional interface for configuring handling of native types.
type NativeTypesOption = types.NativeTypeOption

// NativeTypesFieldNameHandler is a handler for mapping a reflect.StructField to a CEL field name.
// This can be used to override the default Go struct field to CEL field name mapping.
type NativeTypesFieldNameHandler = types.NativeTypesFieldNameHandler

var (
	// ParseStructTags configures if native types field names should be overridable by CEL struct tags.
	// This is equivalent to ParseStructTag("cel")
	ParseStructTags = types.ParseStructTags

	// ParseStructTag configures the struct tag to parse. The 0th item in the tag is used as the name of the CEL field.
	ParseStructTag = types.ParseStructTag

	// ParseStructField configures how to parse Go struct fields. It can be used to customize struct field parsing.
	ParseStructField = types.ParseStructField
)

// NativeTypesVersion sets the native types version support for native extensions functions.
//
// Deprecated: NativeTypesVersion is a no-op and will be removed in a future release.
func NativeTypesVersion(version uint32) NativeTypesOption {
	return func(*types.NativeTypeOptions) error {
		return nil
	}
}

// NativeTypes creates a type provider which uses reflect.Type and reflect.Value instances
// to produce type definitions that can be used within CEL.
//
// All struct types in Go are exposed to CEL via their simple package name and struct type name:
//
// ```go
// package identity
//
//	type Account struct {
//	  ID int
//	}
//
// ```
//
// The type `identity.Account` would be exported to CEL using the same qualified name, e.g.
// `identity.Account{ID: 1234}` would create a new `Account` instance with the `ID` field
// populated.
//
// Only exported fields are exposed via NativeTypes, and the type-mapping between Go and CEL
// is as follows:
//
// | Go type                             | CEL type  |
// |-------------------------------------|-----------|
// | bool                                | bool      |
// | []byte                              | bytes     |
// | float32, float64                    | double    |
// | int, int8, int16, int32, int64      | int       |
// | string                              | string    |
// | uint, uint8, uint16, uint32, uint64 | uint      |
// | time.Duration                       | duration  |
// | time.Time                           | timestamp |
// | array, slice                        | list      |
// | map                                 | map       |
//
// Please note, if you intend to configure support for proto messages in addition to native
// types, you will need to provide the protobuf types before the golang native types. The
// same advice holds if you are using custom type adapters and type providers. The native type
// provider composes over whichever type adapter and provider is configured in the cel.Env at
// the time that it is invoked.
//
// There is also the possibility to rename the fields of native structs by setting the `cel` tag
// for fields you want to override. In order to enable this feature, pass in the `ParseStructTags(true)`
// option. Here is an example to see it in action:
//
// ```go
// package identity
//
//	type Account struct {
//	  ID int
//	  OwnerName string `cel:"owner"`
//	}
//
// ```
//
// The `OwnerName` field is now accessible in CEL via `owner`, e.g. `identity.Account{owner: 'bob'}`.
// In case there are duplicated field names in the struct, an error will be returned.
func NativeTypes(args ...any) cel.EnvOption {
	return func(env *cel.Env) (*cel.Env, error) {
		p, a, err := types.ComposeTypes(env.CELTypeProvider(), env.CELTypeAdapter(), args...)
		if err != nil {
			return nil, err
		}
		env, err = cel.CustomTypeAdapter(a)(env)
		if err != nil {
			return nil, err
		}
		return cel.CustomTypeProvider(p)(env)
	}
}
