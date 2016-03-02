// Copyright 2015 CoreOS, Inc.
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

func LoadArgs(args string, container interface{}) error {
	if args == "" {
		return nil
	}

	containerValue := reflect.ValueOf(container)

	pairs := strings.Split(args, ";")
	for _, pair := range pairs {
		kv := strings.Split(pair, "=")
		if len(kv) != 2 {
			return fmt.Errorf("ARGS: invalid pair %q", pair)
		}
		keyString := kv[0]
		valueString := kv[1]
		keyField := containerValue.Elem().FieldByName(keyString)
		if !keyField.IsValid() {
			return fmt.Errorf("ARGS: invalid key %q", keyString)
		}
		u := keyField.Addr().Interface().(encoding.TextUnmarshaler)
		err := u.UnmarshalText([]byte(valueString))
		if err != nil {
			return fmt.Errorf("ARGS: error parsing value of pair %q: %v)", pair, err)
		}
	}
	return nil
}
