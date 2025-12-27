/*
Copyright 2025 The Kubernetes Authors.

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

package namespace

import (
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/core"
)

func TestDeclarativeValidateForNamespaceCreate(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: "v1",
	})
	testCases := map[string]struct {
		input        core.Namespace
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: core.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: "test"},
			},
			expectedErrs: field.ErrorList{},
		},
		"invalid: empty name and generateName": {
			input: core.Namespace{},
			expectedErrs: field.ErrorList{{
				Type:  field.ErrorTypeRequired,
				Field: "metadata.name",
			}},
		},
		"invalid: non-rfc1123 name: at sign": {
			input: core.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: "@test"},
			},
			expectedErrs: field.ErrorList{{
				Type:   field.ErrorTypeInvalid,
				Field:  "metadata.name",
				Origin: "format=k8s-short-name",
			}},
		},
		"invalid: non-rfc1123 name: too long": {
			input: core.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: strings.Repeat("a", 254)},
			},
			expectedErrs: field.ErrorList{{
				Type:   field.ErrorTypeInvalid,
				Field:  "metadata.name",
				Origin: "format=k8s-short-name",
			}},
		},
		// TODO: add more test cases when increasing declarative validation coverage
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestDeclarativeValidateForNamespaceUpdate(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: "v1",
	})
	testCases := map[string]struct {
		old          core.Namespace
		update       core.Namespace
		expectedErrs field.ErrorList
	}{
		"invalid: changing name": {
			old: core.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test1",
					ResourceVersion: "1",
				},
			},
			update: core.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test2",
					ResourceVersion: "2",
				},
			},
			expectedErrs: field.ErrorList{
				{
					Type:   field.ErrorTypeInvalid,
					Field:  "metadata.name",
					Origin: "immutable",
				}},
		},
		// TODO: add more test cases when increasing declarative validation coverage
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func TestDeclarativeValidateForNamespaceStatusUpdate(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: "v1",
	})
	testCases := map[string]struct {
		old          core.Namespace
		update       core.Namespace
		expectedErrs field.ErrorList
	}{
		"invalid: changing name": {
			old: core.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test1",
					ResourceVersion: "1",
				},
				Status: core.NamespaceStatus{
					Phase: core.NamespaceActive,
				},
			},
			update: core.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test2",
					ResourceVersion: "2",
				},
				Status: core.NamespaceStatus{
					Phase: core.NamespaceActive,
				},
			},
			expectedErrs: field.ErrorList{
				{
					Type:   field.ErrorTypeInvalid,
					Field:  "metadata.name",
					Origin: "immutable",
				}},
		},
		// TODO: add more test cases when increasing declarative validation coverage
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, StatusStrategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func TestDeclarativeValidateForNamespaceFinalizeUpdate(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: "v1",
	})
	testCases := map[string]struct {
		old          core.Namespace
		update       core.Namespace
		expectedErrs field.ErrorList
	}{
		"invalid: changing name": {
			old: core.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test1",
					ResourceVersion: "1",
				},
			},
			update: core.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test2",
					ResourceVersion: "2",
				},
			},
			expectedErrs: field.ErrorList{
				{
					Type:   field.ErrorTypeInvalid,
					Field:  "metadata.name",
					Origin: "immutable",
				}},
		},
		// TODO: add more test cases when increasing declarative validation coverage
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, FinalizeStrategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}
