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
	const testAlphaGate = "TestAlpha"
	const testBetaGate = "TestBeta"

	tests := []struct {
		arg        string
		expect     map[string]bool
		parseError string
	}{
		{
			arg: "",
			expect: map[string]bool{
				allAlphaGate:  false,
				testAlphaGate: false,
				testBetaGate:  false,
			},
		},
		{
			arg: "fooBarBaz=maybeidk",
			expect: map[string]bool{
				allAlphaGate:  false,
				testAlphaGate: false,
				testBetaGate:  false,
			},
			parseError: "unrecognized key: fooBarBaz",
		},
		{
			arg: "AllAlpha=false",
			expect: map[string]bool{
				allAlphaGate:  false,
				testAlphaGate: false,
				testBetaGate:  false,
			},
		},
		{
			arg: "AllAlpha=true",
			expect: map[string]bool{
				allAlphaGate:  true,
				testAlphaGate: true,
				testBetaGate:  false,
			},
		},
		{
			arg: "AllAlpha=banana",
			expect: map[string]bool{
				allAlphaGate:  false,
				testAlphaGate: false,
				testBetaGate:  false,
			},
			parseError: "invalid value of AllAlpha",
		},
		{
			arg: "AllAlpha=false,TestAlpha=true",
			expect: map[string]bool{
				allAlphaGate:  false,
				testAlphaGate: true,
				testBetaGate:  false,
			},
		},
		{
			arg: "TestAlpha=true,AllAlpha=false",
			expect: map[string]bool{
				allAlphaGate:  false,
				testAlphaGate: true,
				testBetaGate:  false,
			},
		},
		{
			arg: "AllAlpha=true,TestAlpha=false",
			expect: map[string]bool{
				allAlphaGate:  true,
				testAlphaGate: false,
				testBetaGate:  false,
			},
		},
		{
			arg: "TestAlpha=false,AllAlpha=true",
			expect: map[string]bool{
				allAlphaGate:  true,
				testAlphaGate: false,
				testBetaGate:  false,
			},
		},
		{
			arg: "TestBeta=true,AllAlpha=false",
			expect: map[string]bool{
				allAlphaGate:  false,
				testAlphaGate: false,
				testBetaGate:  true,
			},
		},
	}
	for i, test := range tests {
		fs := pflag.NewFlagSet("testfeaturegateflag", pflag.ContinueOnError)
		f := DefaultFeatureGate
		f.known[testAlphaGate] = featureSpec{false, alpha}
		f.known[testBetaGate] = featureSpec{false, beta}
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

func TestFeatureGateFlagDefaults(t *testing.T) {
	// gates for testing
	const testAlphaGate = "TestAlpha"
	const testBetaGate = "TestBeta"

	// Don't parse the flag, assert defaults are used.
	f := DefaultFeatureGate
	f.known[testAlphaGate] = featureSpec{false, alpha}
	f.known[testBetaGate] = featureSpec{true, beta}

	if f.lookup(testAlphaGate) != false {
		t.Errorf("Expected false")
	}
	if f.lookup(testBetaGate) != true {
		t.Errorf("Expected true")
	}
}
