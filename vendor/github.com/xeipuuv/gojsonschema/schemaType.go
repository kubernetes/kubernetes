// Copyright 2015 xeipuuv ( https://github.com/xeipuuv )
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// author           xeipuuv
// author-github    https://github.com/xeipuuv
// author-mail      xeipuuv@gmail.com
//
// repository-name  gojsonschema
// repository-desc  An implementation of JSON Schema, based on IETF's draft v4 - Go language.
//
// description      Helper structure to handle schema types, and the combination of them.
//
// created          28-02-2013

package gojsonschema

import (
	"errors"
	"fmt"
	"strings"
)

type jsonSchemaType struct {
	types []string
}

// Is the schema typed ? that is containing at least one type
// When not typed, the schema does not need any type validation
func (t *jsonSchemaType) IsTyped() bool {
	return len(t.types) > 0
}

func (t *jsonSchemaType) Add(etype string) error {

	if !isStringInSlice(JSON_TYPES, etype) {
		return errors.New(formatErrorDescription(Locale.NotAValidType(), ErrorDetails{"given": "/" + etype + "/", "expected": JSON_TYPES}))
	}

	if t.Contains(etype) {
		return errors.New(formatErrorDescription(Locale.Duplicated(), ErrorDetails{"type": etype}))
	}

	t.types = append(t.types, etype)

	return nil
}

func (t *jsonSchemaType) Contains(etype string) bool {

	for _, v := range t.types {
		if v == etype {
			return true
		}
	}

	return false
}

func (t *jsonSchemaType) String() string {

	if len(t.types) == 0 {
		return STRING_UNDEFINED // should never happen
	}

	// Displayed as a list [type1,type2,...]
	if len(t.types) > 1 {
		return fmt.Sprintf("[%s]", strings.Join(t.types, ","))
	}

	// Only one type: name only
	return t.types[0]
}
