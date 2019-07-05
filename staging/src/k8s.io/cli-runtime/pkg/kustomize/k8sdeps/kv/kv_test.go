/*
Copyright 2019 The Kubernetes Authors.

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

package kv

import (
	"reflect"
	"testing"
)

func TestKeyValuesFromLines(t *testing.T) {
	tests := []struct {
		desc          string
		content       string
		expectedPairs []Pair
		expectedErr   bool
	}{
		{
			desc: "valid kv content parse",
			content: `
		k1=v1
		k2=v2
		`,
			expectedPairs: []Pair{
				{Key: "k1", Value: "v1"},
				{Key: "k2", Value: "v2"},
			},
			expectedErr: false,
		},
		{
			desc: "content with comments",
			content: `
		k1=v1
		#k2=v2
		`,
			expectedPairs: []Pair{
				{Key: "k1", Value: "v1"},
			},
			expectedErr: false,
		},
		// TODO: add negative testcases
	}

	for _, test := range tests {
		pairs, err := KeyValuesFromLines([]byte(test.content))
		if test.expectedErr && err == nil {
			t.Fatalf("%s should not return error", test.desc)
		}

		if !reflect.DeepEqual(pairs, test.expectedPairs) {
			t.Errorf("%s should succeed, got:%v exptected:%v", test.desc, pairs, test.expectedPairs)
		}

	}
}
