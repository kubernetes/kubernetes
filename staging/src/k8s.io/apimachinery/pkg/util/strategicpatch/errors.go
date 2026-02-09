/*
Copyright 2017 The Kubernetes Authors.

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

package strategicpatch

import (
	"fmt"
)

type LookupPatchMetaError struct {
	Path string
	Err  error
}

func (e LookupPatchMetaError) Error() string {
	return fmt.Sprintf("LookupPatchMetaError(%s): %v", e.Path, e.Err)
}

type FieldNotFoundError struct {
	Path  string
	Field string
}

func (e FieldNotFoundError) Error() string {
	return fmt.Sprintf("unable to find api field %q in %s", e.Field, e.Path)
}

type InvalidTypeError struct {
	Path     string
	Expected string
	Actual   string
}

func (e InvalidTypeError) Error() string {
	return fmt.Sprintf("invalid type for %s: got %q, expected %q", e.Path, e.Actual, e.Expected)
}
