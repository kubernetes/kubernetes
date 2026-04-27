/*
Copyright 2023 The Kubernetes Authors.

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

package klog

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/go-logr/logr"
)

// Format wraps a value of an arbitrary type and implement fmt.Stringer and
// logr.Marshaler for them. Stringer returns pretty-printed JSON. MarshalLog
// returns the original value with a type that has no special methods, in
// particular no MarshalLog or MarshalJSON.
//
// Wrapping values like that is useful when the value has a broken
// implementation of these special functions (for example, a type which
// inherits String from TypeMeta, but then doesn't re-implement String) or the
// implementation produces output that is less readable or unstructured (for
// example, the generated String functions for Kubernetes API types).
func Format(obj interface{}) interface{} {
	return formatAny{Object: obj}
}

type formatAny struct {
	Object interface{}
}

func (f formatAny) String() string {
	var buffer strings.Builder
	encoder := json.NewEncoder(&buffer)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(&f.Object); err != nil {
		return fmt.Sprintf("error marshaling %T to JSON: %v", f, err)
	}
	return buffer.String()
}

func (f formatAny) MarshalLog() interface{} {
	// Returning a pointer to a pointer ensures that zapr doesn't find a
	// fmt.Stringer or logr.Marshaler when it checks the type of the
	// value. It then falls back to reflection, which dumps the value being
	// pointed to (JSON doesn't have pointers).
	ptr := &f.Object
	return &ptr
}

var _ fmt.Stringer = formatAny{}
var _ logr.Marshaler = formatAny{}
