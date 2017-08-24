// Copyright 2015 CNI authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package types

import (
	"encoding"
	"fmt"
	"reflect"
	"strings"
)

// UnmarshallableBool typedef for builtin bool
// because builtin type's methods can't be declared
type UnmarshallableBool bool

// UnmarshalText implements the encoding.TextUnmarshaler interface.
// Returns boolean true if the string is "1" or "[Tt]rue"
// Returns boolean false if the string is "0" or "[Ff]alse"
func (b *UnmarshallableBool) UnmarshalText(data []byte) error {
	s := strings.ToLower(string(data))
	switch s {
	case "1", "true":
		*b = true
	case "0", "false":
		*b = false
	default:
		return fmt.Errorf("Boolean unmarshal error: invalid input %s", s)
	}
	return nil
}

// UnmarshallableString typedef for builtin string
type UnmarshallableString string

// UnmarshalText implements the encoding.TextUnmarshaler interface.
// Returns the string
func (s *UnmarshallableString) UnmarshalText(data []byte) error {
	*s = UnmarshallableString(data)
	return nil
}

// CommonArgs contains the IgnoreUnknown argument
// and must be embedded by all Arg structs
type CommonArgs struct {
	IgnoreUnknown UnmarshallableBool `json:"ignoreunknown,omitempty"`
}

// GetKeyField is a helper function to receive Values
// Values that represent a pointer to a struct
func GetKeyField(keyString string, v reflect.Value) reflect.Value {
	return v.Elem().FieldByName(keyString)
}

// UnmarshalableArgsError is used to indicate error unmarshalling args
// from the args-string in the form "K=V;K2=V2;..."
type UnmarshalableArgsError struct {
	error
}

// LoadArgs parses args from a string in the form "K=V;K2=V2;..."
func LoadArgs(args string, container interface{}) error {
	if args == "" {
		return nil
	}

	containerValue := reflect.ValueOf(container)

	pairs := strings.Split(args, ";")
	unknownArgs := []string{}
	for _, pair := range pairs {
		kv := strings.Split(pair, "=")
		if len(kv) != 2 {
			return fmt.Errorf("ARGS: invalid pair %q", pair)
		}
		keyString := kv[0]
		valueString := kv[1]
		keyField := GetKeyField(keyString, containerValue)
		if !keyField.IsValid() {
			unknownArgs = append(unknownArgs, pair)
			continue
		}
		keyFieldIface := keyField.Addr().Interface()
		u, ok := keyFieldIface.(encoding.TextUnmarshaler)
		if !ok {
			return UnmarshalableArgsError{fmt.Errorf(
				"ARGS: cannot unmarshal into field '%s' - type '%s' does not implement encoding.TextUnmarshaler",
				keyString, reflect.TypeOf(keyFieldIface))}
		}
		err := u.UnmarshalText([]byte(valueString))
		if err != nil {
			return fmt.Errorf("ARGS: error parsing value of pair %q: %v)", pair, err)
		}
	}

	isIgnoreUnknown := GetKeyField("IgnoreUnknown", containerValue).Bool()
	if len(unknownArgs) > 0 && !isIgnoreUnknown {
		return fmt.Errorf("ARGS: unknown args %q", unknownArgs)
	}
	return nil
}
