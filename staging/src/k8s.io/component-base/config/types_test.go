/*
Copyright 2021 The Kubernetes Authors.

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

package config

import (
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVModule(t *testing.T) {
	testcases := []struct {
		arg         string
		expectError string
		expectValue VModuleConfiguration
		expectParam string
	}{
		{
			arg: "gopher*=1",
			expectValue: VModuleConfiguration{
				{
					FilePattern: "gopher*",
					Verbosity:   1,
				},
			},
		},
		{
			arg: "foo=1,bar=2",
			expectValue: VModuleConfiguration{
				{
					FilePattern: "foo",
					Verbosity:   1,
				},
				{
					FilePattern: "bar",
					Verbosity:   2,
				},
			},
		},
		{
			arg: "foo=1,bar=2,",
			expectValue: VModuleConfiguration{
				{
					FilePattern: "foo",
					Verbosity:   1,
				},
				{
					FilePattern: "bar",
					Verbosity:   2,
				},
			},
			expectParam: "foo=1,bar=2",
		},
		{
			arg:         "gopher*",
			expectError: `"gopher*" does not have the pattern=N format`,
		},
		{
			arg:         "=1",
			expectError: `"=1" does not have the pattern=N format`,
		},
		{
			arg:         "foo=-1",
			expectError: `parsing verbosity in "foo=-1": strconv.ParseUint: parsing "-1": invalid syntax`,
		},
		{
			arg: fmt.Sprintf("validint32=%d", math.MaxInt32),
			expectValue: VModuleConfiguration{
				{
					FilePattern: "validint32",
					Verbosity:   math.MaxInt32,
				},
			},
		},
		{
			arg:         fmt.Sprintf("invalidint32=%d", math.MaxInt32+1),
			expectError: `parsing verbosity in "invalidint32=2147483648": strconv.ParseUint: parsing "2147483648": value out of range`,
		},
	}

	for _, test := range testcases {
		t.Run(test.arg, func(t *testing.T) {
			var actual VModuleConfiguration
			err := actual.Set(test.arg)
			if test.expectError != "" {
				if err == nil {
					t.Fatal("parsing should have failed")
				}
				assert.Equal(t, test.expectError, err.Error(), "parse error")
			} else {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				param := actual.String()
				expectParam := test.expectParam
				if expectParam == "" {
					expectParam = test.arg
				}
				assert.Equal(t, expectParam, param, "encoded parameter value not identical")
			}
		})
	}
}
