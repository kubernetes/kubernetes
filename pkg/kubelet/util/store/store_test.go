/*
Copyright 2017 The Kubernetes Authors.

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

package store

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIsValidKey(t *testing.T) {
	testcases := []struct {
		key   string
		valid bool
	}{
		{
			"    ",
			false,
		},
		{
			"/foo/bar",
			false,
		},
		{
			".foo",
			false,
		},
		{
			"a78768279290d33d0b82eaea43cb8346f500057cb5bd250e88c97a5585385d66",
			true,
		},
		{
			"a7.87-6_8",
			true,
		},
		{
			"a7.87-677-",
			false,
		},
	}

	for _, tc := range testcases {
		if tc.valid {
			assert.NoError(t, ValidateKey(tc.key))
		} else {
			assert.Error(t, ValidateKey(tc.key))
		}
	}
}
