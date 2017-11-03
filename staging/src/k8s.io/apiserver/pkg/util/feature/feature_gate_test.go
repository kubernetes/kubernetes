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

package feature

import (
	"fmt"
	"sort"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/spf13/pflag"
)

func TestFeatureGateKnownFeatures(t *testing.T) {
	knownValue := &atomic.Value{}
	knownValue.Store(map[Feature]FeatureSpec{})

	enabled := map[Feature]bool{}
	enabledValue := &atomic.Value{}
	enabledValue.Store(enabled)

	f := &featureGate{known: knownValue, enabled: enabledValue}
	var (
		test1 Feature = "test1"
		test2 Feature = "test2"
		test3 Feature = "test3"

		hello Component = "hello"
		world Component = "world"
	)
	if err := f.Add(map[Feature]FeatureSpec{
		// test1 only available for component hello, world
		test1: {Components: []Component{"hello", "world"}},
		// test2 only available for component hello
		test2: {Components: []Component{"hello"}},
		// test3 available for all components
		test3: {},
	}); err != nil {
		t.Errorf("failed to add feature: %v", err)
	}
	known := f.KnownFeatures(hello)
	sort.Strings(known)
	if len(known) != 3 ||
		!strings.HasPrefix(known[0], "test1") ||
		!strings.HasPrefix(known[1], "test2") ||
		!strings.HasPrefix(known[2], "test3") {
		t.Errorf("unexpected features %#v", known)
	}
	known = f.KnownFeatures(world)
	sort.Strings(known)
	if len(known) != 2 ||
		!strings.HasPrefix(known[0], "test1") ||
		!strings.HasPrefix(known[1], "test3") {
		t.Errorf("unexpected features %#v", known)
	}
	known = f.KnownFeatures("")
	sort.Strings(known)
	if len(known) != 1 || !strings.HasPrefix(known[0], "test3") {
		t.Errorf("unexpected features %#v", known)
	}
}

func TestFeatureSpecEquals(t *testing.T) {
	for _, test := range []struct {
		a        FeatureSpec
		b        FeatureSpec
		expected bool
	}{
		{
			a:        FeatureSpec{Default: false, PreRelease: Alpha},
			b:        FeatureSpec{Default: false, PreRelease: Alpha},
			expected: true,
		},
		{
			a:        FeatureSpec{Default: false, PreRelease: Alpha},
			b:        FeatureSpec{Default: true, PreRelease: Alpha},
			expected: false,
		},
		{
			a:        FeatureSpec{Default: false, PreRelease: Alpha},
			b:        FeatureSpec{Default: false, PreRelease: Beta},
			expected: false,
		},
		{
			a:        FeatureSpec{Default: false, PreRelease: Alpha},
			b:        FeatureSpec{Default: false, PreRelease: Alpha, Components: []Component{"test"}},
			expected: false,
		},
		{
			a:        FeatureSpec{Default: false, PreRelease: Alpha, Components: []Component{"hello", "world"}},
			b:        FeatureSpec{Default: false, PreRelease: Alpha, Components: []Component{"world", "hello"}},
			expected: true,
		},
	} {
		if test.a.Equals(test.b) != test.expected {
			t.Errorf("comparison of %v == %v failed, expect %v", test.a, test.b, test.expected)
		}
	}
}

func TestFeatureGateFlag(t *testing.T) {
	// gates for testing
	const testAlphaGate Feature = "TestAlpha"
	const testBetaGate Feature = "TestBeta"

	tests := []struct {
		arg        string
		expect     map[Feature]bool
		parseError string
	}{
		{
			arg: "",
			expect: map[Feature]bool{
				allAlphaGate:  false,
				testAlphaGate: false,
				testBetaGate:  false,
			},
		},
		{
			arg: "fooBarBaz=maybeidk",
			expect: map[Feature]bool{
				allAlphaGate:  false,
				testAlphaGate: false,
				testBetaGate:  false,
			},
			parseError: "unrecognized key: fooBarBaz",
		},
		{
			arg: "AllAlpha=false",
			expect: map[Feature]bool{
				allAlphaGate:  false,
				testAlphaGate: false,
				testBetaGate:  false,
			},
		},
		{
			arg: "AllAlpha=true",
			expect: map[Feature]bool{
				allAlphaGate:  true,
				testAlphaGate: true,
				testBetaGate:  false,
			},
		},
		{
			arg: "AllAlpha=banana",
			expect: map[Feature]bool{
				allAlphaGate:  false,
				testAlphaGate: false,
				testBetaGate:  false,
			},
			parseError: "invalid value of AllAlpha",
		},
		{
			arg: "AllAlpha=false,TestAlpha=true",
			expect: map[Feature]bool{
				allAlphaGate:  false,
				testAlphaGate: true,
				testBetaGate:  false,
			},
		},
		{
			arg: "TestAlpha=true,AllAlpha=false",
			expect: map[Feature]bool{
				allAlphaGate:  false,
				testAlphaGate: true,
				testBetaGate:  false,
			},
		},
		{
			arg: "AllAlpha=true,TestAlpha=false",
			expect: map[Feature]bool{
				allAlphaGate:  true,
				testAlphaGate: false,
				testBetaGate:  false,
			},
		},
		{
			arg: "TestAlpha=false,AllAlpha=true",
			expect: map[Feature]bool{
				allAlphaGate:  true,
				testAlphaGate: false,
				testBetaGate:  false,
			},
		},
		{
			arg: "TestBeta=true,AllAlpha=false",
			expect: map[Feature]bool{
				allAlphaGate:  false,
				testAlphaGate: false,
				testBetaGate:  true,
			},
		},
	}
	for i, test := range tests {
		fs := pflag.NewFlagSet("testfeaturegateflag", pflag.ContinueOnError)
		f := NewFeatureGate()
		f.Add(map[Feature]FeatureSpec{
			testAlphaGate: {Default: false, PreRelease: Alpha},
			testBetaGate:  {Default: false, PreRelease: Beta},
		})
		f.AddFlag(fs, "")

		err := fs.Parse([]string{fmt.Sprintf("--%s=%s", flagName, test.arg)})
		if test.parseError != "" {
			if !strings.Contains(err.Error(), test.parseError) {
				t.Errorf("%d: Parse() Expected %v, Got %v", i, test.parseError, err)
			}
		} else if err != nil {
			t.Errorf("%d: Parse() Expected nil, Got %v", i, err)
		}
		for k, v := range test.expect {
			if actual := f.enabled.Load().(map[Feature]bool)[k]; actual != v {
				t.Errorf("%d: expected %s=%v, Got %v", i, k, v, actual)
			}
		}
	}
}

func TestFeatureGateOverride(t *testing.T) {
	const testAlphaGate Feature = "TestAlpha"
	const testBetaGate Feature = "TestBeta"

	// Don't parse the flag, assert defaults are used.
	var f FeatureGate = NewFeatureGate()
	f.Add(map[Feature]FeatureSpec{
		testAlphaGate: {Default: false, PreRelease: Alpha},
		testBetaGate:  {Default: false, PreRelease: Beta},
	})

	f.Set("TestAlpha=true,TestBeta=true")
	if f.Enabled(testAlphaGate) != true {
		t.Errorf("Expected true")
	}
	if f.Enabled(testBetaGate) != true {
		t.Errorf("Expected true")
	}

	f.Set("TestAlpha=false")
	if f.Enabled(testAlphaGate) != false {
		t.Errorf("Expected false")
	}
	if f.Enabled(testBetaGate) != true {
		t.Errorf("Expected true")
	}
}

func TestFeatureGateFlagDefaults(t *testing.T) {
	// gates for testing
	const testAlphaGate Feature = "TestAlpha"
	const testBetaGate Feature = "TestBeta"

	// Don't parse the flag, assert defaults are used.
	var f FeatureGate = NewFeatureGate()
	f.Add(map[Feature]FeatureSpec{
		testAlphaGate: {Default: false, PreRelease: Alpha},
		testBetaGate:  {Default: true, PreRelease: Beta},
	})

	if f.Enabled(testAlphaGate) != false {
		t.Errorf("Expected false")
	}
	if f.Enabled(testBetaGate) != true {
		t.Errorf("Expected true")
	}
}
