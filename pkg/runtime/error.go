/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api/unversioned"
)

type notRegisteredErr struct {
	gvk unversioned.GroupVersionKind
	t   reflect.Type
}

// NewNotRegisteredErr is exposed for testing.
func NewNotRegisteredErr(gvk unversioned.GroupVersionKind, t reflect.Type) error {
	return &notRegisteredErr{gvk: gvk, t: t}
}

func (k *notRegisteredErr) Error() string {
	if k.t != nil {
		return fmt.Sprintf("no kind is registered for the type %v", k.t)
	}
	if len(k.gvk.Kind) == 0 {
		return fmt.Sprintf("no version %q has been registered", k.gvk.GroupVersion())
	}
	if k.gvk.Version == APIVersionInternal {
		return fmt.Sprintf("no kind %q is registered for the internal version of group %q", k.gvk.Kind, k.gvk.Group)
	}

	return fmt.Sprintf("no kind %q is registered for version %q", k.gvk.Kind, k.gvk.GroupVersion())
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

// IsMissingVersion returns true if the error indicates that the provided object
// is missing a 'Versioj' field.
func NewMissingVersionErr(data string) error {
	return &missingVersionErr{data}
}

func (k *missingVersionErr) Error() string {
	return fmt.Sprintf("Object 'apiVersion' is missing in '%s'", k.data)
}

func IsMissingVersion(err error) bool {
	if err == nil {
		return false
	}
	_, ok := err.(*missingVersionErr)
	return ok
}
