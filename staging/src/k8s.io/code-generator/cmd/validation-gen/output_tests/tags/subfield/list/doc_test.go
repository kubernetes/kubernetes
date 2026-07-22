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

package list

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestStructValidation(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Valid case: empty ownerReferences, distinct finalizers, empty labels
	st.Value(&Struct{
		ObjectMeta: metav1.ObjectMeta{
			Finalizers: []string{"finalizer1", "finalizer2"},
		},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{})

	// Invalid case: duplicate UIDs
	st.Value(&Struct{
		ObjectMeta: metav1.ObjectMeta{
			OwnerReferences: []metav1.OwnerReference{
				{UID: "1", Name: "ref1"},
				{UID: "1", Name: "ref2"},
			},
		},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Duplicate(field.NewPath("objectMeta", "ownerReferences").Index(1), metav1.OwnerReference{UID: "1", Name: "ref2"}),
		field.Invalid(field.NewPath("objectMeta", "ownerReferences").Index(0).Child("name"), "ref1", "").WithOrigin("validateFalse"),
		field.Invalid(field.NewPath("objectMeta", "ownerReferences").Index(1).Child("name"), "ref2", "").WithOrigin("validateFalse"),
	})

	// Invalid case: duplicate finalizers
	st.Value(&Struct{
		ObjectMeta: metav1.ObjectMeta{
			Finalizers: []string{"finalizer1", "finalizer1"},
		},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Duplicate(field.NewPath("objectMeta", "finalizers").Index(1), "finalizer1"),
	})

	// Invalid case: non-empty labels trigger fixed failure
	st.Value(&Struct{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{"key1": "val1"},
		},
	}).ExpectValidateFalseByPath(map[string][]string{
		"objectMeta.labels": {"labels key error"},
	})

	// Invalid case: ownerReference name triggers fixed failure
	st.Value(&Struct{
		ObjectMeta: metav1.ObjectMeta{
			OwnerReferences: []metav1.OwnerReference{
				{UID: "1", Name: "ref1"},
			},
		},
	}).ExpectValidateFalseByPath(map[string][]string{
		"objectMeta.ownerReferences[0].name": {"ownerReference name error"},
	})
}
