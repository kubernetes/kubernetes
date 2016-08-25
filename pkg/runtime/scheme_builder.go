/*
Copyright 2015 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/api/unversioned"
)

// SchemeFunc registers something with the given scheme.
type SchemeFunc func(*Scheme) error

// SchemeFunc registers something with the given scheme, for the given version.
// This allows general scheme building functions to be written.
type VersionedSchemeFunc func(*Scheme, unversioned.GroupVersion) error

// SchemeBuilder collects functions that add things to a scheme. It's to allow
// code to compile without explicitly referencing generated types. You should
// declare one in each package that will have generated deep copy or conversion
// functions.
type SchemeBuilder struct {
	schemeAppliers          []SchemeFunc
	versionedSchemeAppliers []VersionedSchemeFunc
}

// AddToScheme applies all the stored functions to the scheme. A non-nil error
// indicates that one function failed and the attempt was abandoned.
func (sb *SchemeBuilder) AddToScheme(s *Scheme) error {
	if len(sb.versionedSchemeAppliers) > 0 {
		return fmt.Errorf("versioned scheme appliers were registered, you must call VersionedAddToScheme")
	}
	for _, f := range sb.schemeAppliers {
		if err := f(s); err != nil {
			return err
		}
	}
	return nil
}

// VersionedAddToScheme applies all the stored functions to the scheme. A
// non-nil error indicates that one function failed and the attempt was
// abandoned. You must call this function instead of AddToScheme if any
// version-dependent scheme functions were registered.
func (sb *SchemeBuilder) VersionedAddToScheme(s *Scheme, gv unversioned.GroupVersion) error {
	for _, f := range sb.schemeAppliers {
		if err := f(s); err != nil {
			return err
		}
	}
	for _, f := range sb.versionedSchemeAppliers {
		if err := f(s, gv); err != nil {
			return err
		}
	}
	return nil
}

// Register adds a scheme setup function to the list. Chainable.
func (sb *SchemeBuilder) Register(funcs ...func(*Scheme) error) *SchemeBuilder {
	for _, f := range funcs {
		sb.schemeAppliers = append(sb.schemeAppliers, f)
	}
	return sb
}

// VersionedRegister adds a versioned scheme setup function to the list. Chainable.
func (sb *SchemeBuilder) VersionedRegister(funcs ...func(*Scheme, unversioned.GroupVersion) error) *SchemeBuilder {
	for _, f := range funcs {
		sb.versionedSchemeAppliers = append(sb.versionedSchemeAppliers, f)
	}
	return sb
}

// NewSchemeBuilder calls Register for you.
func NewSchemeBuilder(funcs ...func(*Scheme) error) *SchemeBuilder {
	var sb SchemeBuilder
	return sb.Register(funcs...)
}
