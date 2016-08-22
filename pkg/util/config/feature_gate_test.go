/*
Copyright 2016 The Kubernetes Authors.

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
	"strings"
	"testing"

	"github.com/spf13/pflag"
)

func TestFeatureGateFlag(t *testing.T) {
	// gates for testing
	const testAlpha = "testAlpha"
	const testBeta = "testBeta"

	tests := []struct {
		arg        string
		expect     map[string]bool
		parseError string
	}{
		{
			arg: "",
			expect: map[string]bool{
				allAlpha:  false,
				testAlpha: false,
				testBeta:  false,
			},
		},
		{
			arg: "fooBarBaz=maybeidk",
			expect: map[string]bool{
				allAlpha:  false,
				testAlpha: false,
				testBeta:  false,
			},
			parseError: "unrecognized key: fooBarBaz",
		},
		{
			arg: "allAlpha=false",
			expect: map[string]bool{
				allAlpha:  false,
				testAlpha: false,
				testBeta:  false,
			},
		},
		{
			arg: "allAlpha=true",
			expect: map[string]bool{
				allAlpha:  true,
				testAlpha: true,
				testBeta:  false,
			},
		},
		{
			arg: "allAlpha=banana",
			expect: map[string]bool{
				allAlpha:  false,
				testAlpha: false,
				testBeta:  false,
			},
			parseError: "invalid value of allAlpha",
		},
		{
			arg: "allAlpha=false,testAlpha=true",
			expect: map[string]bool{
				allAlpha:  false,
				testAlpha: true,
				testBeta:  false,
			},
		},
		{
			arg: "testAlpha=true,allAlpha=false",
			expect: map[string]bool{
				allAlpha:  false,
				testAlpha: true,
				testBeta:  false,
			},
		},
		{
			arg: "allAlpha=true,testAlpha=false",
			expect: map[string]bool{
				allAlpha:  true,
				testAlpha: false,
				testBeta:  false,
			},
		},
		{
			arg: "testAlpha=false,allAlpha=true",
			expect: map[string]bool{
				allAlpha:  true,
				testAlpha: false,
				testBeta:  false,
			},
		},
		{
			arg: "testBeta=true,allAlpha=false",
			expect: map[string]bool{
				allAlpha:  false,
				testAlpha: false,
				testBeta:  true,
			},
		},
	}
	for i, test := range tests {
		fs := pflag.NewFlagSet("testfeaturegateflag", pflag.ContinueOnError)
		f := DefaultFeatureGate
		f.known[testAlpha] = featureSpec{false, alpha}
		f.known[testBeta] = featureSpec{false, beta}
		f.AddFlag(fs)

		err := fs.Parse([]string{fmt.Sprintf("--%s=%s", flagName, test.arg)})
		if test.parseError != "" {
			if !strings.Contains(err.Error(), test.parseError) {
				t.Errorf("%d: Parse() Expected %v, Got %v", i, test.parseError, err)
			}
		} else if err != nil {
			t.Errorf("%d: Parse() Expected nil, Got %v", i, err)
		}
		for k, v := range test.expect {
			if f.enabled[k] != v {
				t.Errorf("%d: expected %s=%v, Got %v", i, k, v, f.enabled[k])
			}
		}
	}
}
