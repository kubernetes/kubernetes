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

// Prune recursively removes all non-specified fields from the underlying data of the result.
// The data must be a JSON struct as returned by json.Unmarshal.
func Prune(r *validate.Result) {
	prune(r.Data(), r)
}

func prune(data interface{}, result *validate.Result) {
	switch obj := data.(type) {
	case map[string]interface{}:
		pruneObject(obj, result)
		for _, val := range obj {
			prune(val, result)
		}
	case []interface{}:
		for _, item := range obj {
			prune(item, result)
		}
	}
}

func pruneObject(obj map[string]interface{}, result *validate.Result) {
	fieldSchemata := result.FieldSchemata()
	for field := range obj {
		if schemata, ok := fieldSchemata[validate.NewFieldKey(obj, field)]; !ok || len(schemata) == 0 {
			delete(obj, field)
		}
	}
}
