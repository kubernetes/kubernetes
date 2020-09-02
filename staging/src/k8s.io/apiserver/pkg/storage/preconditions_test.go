/*
Copyright 2020 The Kubernetes Authors.

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

package storage

import (
	"fmt"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/apis/example"
)

// Constant UID and ResourceVersion values used in tests.
const (
	uid1             = "u1111"
	uid2             = "u2222"
	resourceVersion1 = "rv1111"
	resourceVersion2 = "rv2222"
)

func TestSafeCheck(t *testing.T) {
	testCases := map[string]struct {
		preconditions *Preconditions
		objName       string
		obj           runtime.Object
		objExists     bool
		// expectedErrorMessageWords contains the words that we expect in the
		// error message returned by the preconditions check. A test passes if
		// and only if there's at least one inner slice for which all words are
		// present in the error message.
		expectedErrorMessageWords [][]string
	}{
		"nil preconditions": {
			preconditions: nil,
			objName:       "foo",
			obj:           &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: types.UID(uid1), ResourceVersion: resourceVersion1}},
			objExists:     true,
		},
		"nil preconditions, object doesn't exist": {
			preconditions: nil,
			objName:       "foo",
			obj:           &example.Pod{},
			objExists:     false,
		},
		"empty preconditions": {
			preconditions: &Preconditions{},
			objName:       "foo",
			obj:           &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: types.UID(uid1), ResourceVersion: resourceVersion1}},
			objExists:     true,
		},
		"empty preconditions, object doesn't exist": {
			preconditions: &Preconditions{},
			objName:       "foo",
			obj:           &example.Pod{},
			objExists:     false,
		},
		"UID and ResourceVersion are met": {
			preconditions: NewPreconditions(uid1, resourceVersion1),
			objName:       "foo",
			obj:           &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: types.UID(uid1), ResourceVersion: resourceVersion1}},
			objExists:     true,
		},
		"UID is met": {
			preconditions: NewUIDPreconditions(uid1),
			objName:       "foo",
			obj:           &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: types.UID(uid1)}},
			objExists:     true,
		},
		"ResourceVersion is met": {
			preconditions: NewResourceVersionPreconditions(resourceVersion1),
			objName:       "foo",
			obj:           &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: resourceVersion1}},
			objExists:     true,
		},
		"UID is met, ResourceVersion is not": {
			preconditions: NewPreconditions(uid1, resourceVersion1),
			objName:       "foo",
			obj:           &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: types.UID(uid1), ResourceVersion: resourceVersion2}},
			objExists:     true,
			expectedErrorMessageWords: [][]string{
				{"Precondition", "ResourceVersion", resourceVersion1, resourceVersion2},
			},
		},
		"ResourceVersion is met, UID is not": {
			preconditions: NewPreconditions(uid1, resourceVersion1),
			objName:       "foo",
			obj:           &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: types.UID(uid2), ResourceVersion: resourceVersion1}},
			objExists:     true,
			expectedErrorMessageWords: [][]string{
				{"Precondition", "UID", uid1, uid2},
			},
		},
		"neither UID nor ResourceVersion is met": {
			preconditions: NewPreconditions(uid1, resourceVersion1),
			objName:       "foo",
			obj:           &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: types.UID(uid2), ResourceVersion: resourceVersion2}},
			objExists:     true,
			expectedErrorMessageWords: [][]string{
				{"Precondition", "UID", uid1, uid2},
				{"Precondition", "ResourceVersion", resourceVersion1, resourceVersion2},
			},
		},
		"UID is not met because object doesn't exist": {
			preconditions: NewUIDPreconditions(uid1),
			obj:           &example.Pod{},
			expectedErrorMessageWords: [][]string{
				{"Precondition", "UID", "object", "exist"},
			},
		},
		"ResourceVersion is not met because object doesn't exist": {
			preconditions: NewResourceVersionPreconditions(resourceVersion1),
			obj:           &example.Pod{},
			expectedErrorMessageWords: [][]string{
				{"Precondition", "ResourceVersion", "object", "exist"},
			},
		},
		"neither UID nor ResourceVersion are met because object doesn't exist": {
			preconditions: NewPreconditions(uid1, resourceVersion1),
			obj:           &example.Pod{},
			expectedErrorMessageWords: [][]string{
				{"Precondition", "UID", "object", "exist"},
				{"Precondition", "ResourceVersion", "object", "exist"},
			},
		},
	}

	for description, tc := range testCases {
		t.Run(description, func(t *testing.T) {
			gotErr := tc.preconditions.SafeCheck(tc.objName, tc.obj, tc.objExists)
			if err := pass(gotErr, tc.expectedErrorMessageWords); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func pass(got error, expectedWordsSets [][]string) error {
	if got == nil {
		if len(expectedWordsSets) == 0 {
			return nil
		}
		return fmt.Errorf("got: no error, expected: an error with message containing words: %s", formatExpectedWords(expectedWordsSets))
	}
	if !IsInvalidObj(got) {
		return fmt.Errorf("expected error of type \"Invalid Object\", got error: \"%v\", of a different type", got)
	}
	errMsg := strings.ToLower(got.Error())
	for _, anExpectedWordsSet := range expectedWordsSets {
		allExpectedWordsFound := true
		for _, expectedWord := range anExpectedWordsSet {
			if !strings.Contains(errMsg, strings.ToLower(expectedWord)) {
				allExpectedWordsFound = false
				break
			}
		}
		if allExpectedWordsFound {
			return nil
		}
	}
	return fmt.Errorf("got error with message \"%v\", expected message must contain words: %s", got, formatExpectedWords(expectedWordsSets))
}

func formatExpectedWords(expectedWordsSets [][]string) string {
	var buffer strings.Builder
	for i, ews := range expectedWordsSets {
		buffer.WriteString("\n\n\t")
		if _, err := buffer.WriteString(strings.Join(ews, ", ")); err != nil {
			panic(fmt.Sprintf("failed to format expected words %v while preparing test failure output: %v", ews, err))
		}
		if i < len(expectedWordsSets)-1 {
			buffer.WriteString("\n\n\tOR")
		}
	}
	return buffer.String()
}
