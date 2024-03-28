/*
Copyright 2014 The Kubernetes Authors.

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

package runtime

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

type notRegisteredErr struct {
	context fmt.Stringer
	gvk     schema.GroupVersionKind
	target  GroupVersioner
	t       reflect.Type
}

func NewNotRegisteredErrForKind(schemeName string, gvk schema.GroupVersionKind) error {
	return &notRegisteredErr{context: schemeContext(schemeName), gvk: gvk}
}

func NewNotRegisteredErrForType(schemeName string, t reflect.Type) error {
	return &notRegisteredErr{context: schemeContext(schemeName), t: t}
}

func NewNotRegisteredErrForTarget(schemeName string, t reflect.Type, target GroupVersioner) error {
	return &notRegisteredErr{context: schemeContext(schemeName), t: t, target: target}
}

func NewNotRegisteredGVKErrForTarget(schemeName string, gvk schema.GroupVersionKind, target GroupVersioner) error {
	return &notRegisteredErr{context: schemeContext(schemeName), gvk: gvk, target: target}
}

func NewNotRegisteredGVKErrForTargetWithContext(context string, gvk schema.GroupVersionKind, target GroupVersioner) error {
	return &notRegisteredErr{context: genericContext(context), gvk: gvk, target: target}
}

type schemeContext string

func (s schemeContext) String() string {
	if len(s) == 0 {
		return ""
	}
	return " in scheme " + strconv.Quote(string(s))
}

type genericContext string

func (s genericContext) String() string {
	if len(s) == 0 {
		return ""
	}
	return " " + string(s)
}

func (k *notRegisteredErr) Error() string {
	if k.t != nil && k.target != nil {
		return fmt.Sprintf("%v is not suitable for converting to %q%s", k.t, k.target, k.context)
	}
	nullGVK := schema.GroupVersionKind{}
	if k.gvk != nullGVK && k.target != nil {
		return fmt.Sprintf("%q is not suitable for converting to %q%s", k.gvk.GroupVersion(), k.target, k.context)
	}
	if k.t != nil {
		return fmt.Sprintf("no kind is registered for the type %v%s", k.t, k.context)
	}
	if len(k.gvk.Kind) == 0 {
		return fmt.Sprintf("no version %q has been registered%s", k.gvk.GroupVersion(), k.context)
	}
	if k.gvk.Version == APIVersionInternal {
		return fmt.Sprintf("no kind %q is registered for the internal version of group %q%s", k.gvk.Kind, k.gvk.Group, k.context)
	}

	return fmt.Sprintf("no kind %q is registered for version %q%s", k.gvk.Kind, k.gvk.GroupVersion(), k.context)
}

// IsNotRegisteredError returns true if the error indicates the provided
// object or input data is not registered.
func IsNotRegisteredError(err error) bool {
	if err == nil {
		return false
	}
	_, ok := err.(*notRegisteredErr)
	return ok
}

type missingKindErr struct {
	data string
}

func NewMissingKindErr(data string) error {
	return &missingKindErr{data}
}

func (k *missingKindErr) Error() string {
	return fmt.Sprintf("Object 'Kind' is missing in '%s'", k.data)
}

// IsMissingKind returns true if the error indicates that the provided object
// is missing a 'Kind' field.
func IsMissingKind(err error) bool {
	if err == nil {
		return false
	}
	_, ok := err.(*missingKindErr)
	return ok
}

type missingVersionErr struct {
	data string
}

func NewMissingVersionErr(data string) error {
	return &missingVersionErr{data}
}

func (k *missingVersionErr) Error() string {
	return fmt.Sprintf("Object 'apiVersion' is missing in '%s'", k.data)
}

// IsMissingVersion returns true if the error indicates that the provided object
// is missing a 'Version' field.
func IsMissingVersion(err error) bool {
	if err == nil {
		return false
	}
	_, ok := err.(*missingVersionErr)
	return ok
}

// strictDecodingError is a base error type that is returned by a strict Decoder such
// as UniversalStrictDecoder.
type strictDecodingError struct {
	errors []error
}

// NewStrictDecodingError creates a new strictDecodingError object.
func NewStrictDecodingError(errors []error) error {
	return &strictDecodingError{
		errors: errors,
	}
}

func (e *strictDecodingError) Error() string {
	var s strings.Builder
	s.WriteString("strict decoding error: ")
	for i, err := range e.errors {
		if i != 0 {
			s.WriteString(", ")
		}
		s.WriteString(err.Error())
	}
	return s.String()
}

func (e *strictDecodingError) Errors() []error {
	return e.errors
}

// IsStrictDecodingError returns true if the error indicates that the provided object
// strictness violations.
func IsStrictDecodingError(err error) bool {
	if err == nil {
		return false
	}
	_, ok := err.(*strictDecodingError)
	return ok
}

// AsStrictDecodingError returns a strict decoding error
// containing all the strictness violations.
func AsStrictDecodingError(err error) (*strictDecodingError, bool) {
	if err == nil {
		return nil, false
	}
	strictErr, ok := err.(*strictDecodingError)
	return strictErr, ok
}
