// Copyright 2020 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package validate

import (
	"sigs.k8s.io/kustomize/api/ifc"
)

// FieldValidator implements ifc.Validator to check
// the values of various KRM string fields,
// e.g. labels, annotations, names, namespaces.
//
// TODO: Have this use kyaml/yaml/internal/k8sgen/pkg/labels
//  which has label and annotation validation code, but is internal
//  so this impl would need to move to kyaml (a fine idea).
type FieldValidator struct {
}

var _ ifc.Validator = (*FieldValidator)(nil)

func NewFieldValidator() *FieldValidator {
	return &FieldValidator{}
}

// TODO(#FieldValidator): implement MakeAnnotationValidator
func (f FieldValidator) MakeAnnotationValidator() func(map[string]string) error {
	return func(x map[string]string) error {
		return nil
	}
}

// TODO(#FieldValidator): implement MakeAnnotationNameValidator
func (f FieldValidator) MakeAnnotationNameValidator() func([]string) error {
	return func(x []string) error {
		return nil
	}
}

// TODO(#FieldValidator): implement MakeLabelValidator
func (f FieldValidator) MakeLabelValidator() func(map[string]string) error {
	return func(x map[string]string) error {
		return nil
	}
}

// TODO(#FieldValidator): implement MakeLabelNameValidator
func (f FieldValidator) MakeLabelNameValidator() func([]string) error {
	return func(x []string) error {
		return nil
	}
}

// TODO(#FieldValidator): implement ValidateNamespace
func (f FieldValidator) ValidateNamespace(s string) []string {
	var errs []string
	return errs
}

// TODO(#FieldValidator): implement ErrIfInvalidKey
func (f FieldValidator) ErrIfInvalidKey(s string) error {
	return nil
}

// TODO(#FieldValidator): implement IsEnvVarName
func (f FieldValidator) IsEnvVarName(k string) error {
	return nil
}
