// Copyright 2018 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package post

import (
	"github.com/go-openapi/validate"
)

// ApplyDefaults applies defaults to the underlying data of the result. The data must be a JSON
// struct as returned by json.Unmarshal.
func ApplyDefaults(r *validate.Result) {
	fieldSchemata := r.FieldSchemata()
	for key, schemata := range fieldSchemata {
	LookForDefaultingScheme:
		for _, s := range schemata {
			if s.Default != nil {
				if _, found := key.Object()[key.Field()]; !found {
					key.Object()[key.Field()] = s.Default
					break LookForDefaultingScheme
				}
			}
		}
	}
}
