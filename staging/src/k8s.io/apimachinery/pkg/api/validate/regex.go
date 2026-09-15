/*
Copyright The Kubernetes Authors.

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

package validate

import (
	"context"
	"regexp"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/api/validate/content"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// Matches verifies that the value matches re. The pattern is not implicitly
// anchored: use ^ and $ to match the whole value.
func Matches[T ~string](_ context.Context, _ operation.Operation, fldPath *field.Path, value, _ *T, re *regexp.Regexp, message string, origin string) field.ErrorList {
	if value == nil {
		return nil
	}
	str := (string)(*value)
	if re.MatchString(str) {
		return nil
	}
	return field.ErrorList{
		field.Invalid(fldPath, str, content.RegexError(message, re.String())).WithOrigin(origin),
	}
}
