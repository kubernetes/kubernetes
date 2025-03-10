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

package cacher

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/storage"
)

func TestValidateUndelegatedListOptions(t *testing.T) {
	type opts struct {
		ResourceVersion      string
		ResourceVersionMatch metav1.ResourceVersionMatch
		Recursive            bool
		Limit                int64
		Continue             string
	}
	testCases := []opts{}
	keyPrefix := "/pods/"
	continueToken, err := storage.EncodeContinue("/pods/a", keyPrefix, 1)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	for _, rv := range []string{"", "0", "1"} {
		for _, match := range []metav1.ResourceVersionMatch{"", metav1.ResourceVersionMatchExact, metav1.ResourceVersionMatchNotOlderThan} {
			for _, c := range []string{"", continueToken} {
				for _, limit := range []int64{0, 100} {
					for _, recursive := range []bool{true, false} {
						opt := opts{
							ResourceVersion:      rv,
							ResourceVersionMatch: match,
							Limit:                limit,
							Continue:             c,
							Recursive:            recursive,
						}
						testCases = append(testCases, opt)
					}

				}
			}
		}
	}
	for _, tc := range testCases {
		opt := storage.ListOptions{
			ResourceVersion:      tc.ResourceVersion,
			ResourceVersionMatch: tc.ResourceVersionMatch,
			Recursive:            tc.Recursive,
			Predicate: storage.SelectionPredicate{
				Continue: tc.Continue,
				Limit:    tc.Limit,
			}}
		if shouldDelegateList(opt) {
			continue
		}
		_, _, err := storage.ValidateListOptions(keyPrefix, storage.APIObjectVersioner{}, opt)
		if err != nil {
			t.Errorf("Expected List requests %+v to pass validation, got: %v", tc, err)
		}
	}
}
