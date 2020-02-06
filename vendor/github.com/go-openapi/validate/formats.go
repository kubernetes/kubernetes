// Copyright 2015 go-swagger maintainers
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

package validate

import (
	"reflect"

	"github.com/go-openapi/spec"
	"github.com/go-openapi/strfmt"
)

type formatValidator struct {
	Format       string
	Path         string
	In           string
	KnownFormats strfmt.Registry
}

func (f *formatValidator) SetPath(path string) {
	f.Path = path
}

func (f *formatValidator) Applies(source interface{}, kind reflect.Kind) bool {
	doit := func() bool {
		if source == nil {
			return false
		}
		switch source := source.(type) {
		case *spec.Items:
			return kind == reflect.String && f.KnownFormats.ContainsName(source.Format)
		case *spec.Parameter:
			return kind == reflect.String && f.KnownFormats.ContainsName(source.Format)
		case *spec.Schema:
			return kind == reflect.String && f.KnownFormats.ContainsName(source.Format)
		case *spec.Header:
			return kind == reflect.String && f.KnownFormats.ContainsName(source.Format)
		}
		return false
	}
	r := doit()
	debugLog("format validator for %q applies %t for %T (kind: %v)\n", f.Path, r, source, kind)
	return r
}

func (f *formatValidator) Validate(val interface{}) *Result {
	result := new(Result)
	debugLog("validating \"%v\" against format: %s", val, f.Format)

	if err := FormatOf(f.Path, f.In, f.Format, val.(string), f.KnownFormats); err != nil {
		result.AddErrors(err)
	}

	if result.HasErrors() {
		return result
	}
	return nil
}
