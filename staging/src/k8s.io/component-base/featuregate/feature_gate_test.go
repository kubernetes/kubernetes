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

package featuregate

import (
	"fmt"
	"strings"
	"testing"

	"github.com/spf13/pflag"
	"github.com/stretchr/testify/assert"

	"k8s.io/component-base/metrics/legacyregistry"
	featuremetrics "k8s.io/component-base/metrics/prometheus/feature"
	"k8s.io/component-base/metrics/testutil"
)

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
				allBetaGate:   false,
				testAlphaGate: false,
				testBetaGate:  false,
			},
		},
		{
			arg: "fooBarBaz=true",
			expect: map[Feature]bool{
				allAlphaGate:  false,
				allBetaGate:   false,
				testAlphaGate: false,
				testBetaGate:  false,
			},
			//parseError: "unrecognized feature gate: fooBarBaz",
		},
		{
			arg: "AllAlpha=false",
			expect: map[Feature]bool{
				allAlphaGate:  false,
				allBetaGate:   false,
				testAlphaGate: false,
				testBetaGate:  false,
			},
		},
		{
			arg: "AllAlpha=true",
			expect: map[Feature]bool{
				allAlphaGate:  true,
				allBetaGate:   false,
				testAlphaGate: true,
				testBetaGate:  false,
			},
		},
		{
			arg: "AllAlpha=banana",
			expect: map[Feature]bool{
				allAlphaGate:  false,
				allBetaGate:   false,
				testAlphaGate: false,
				testBetaGate:  false,
			},
			parseError: "invalid value of AllAlpha",
		},
		{
			arg: "AllAlpha=false,TestAlpha=true",
			expect: map[Feature]bool{
				allAlphaGate:  false,
				allBetaGate:   false,
				testAlphaGate: true,
				testBetaGate:  false,
			},
		},
		{
			arg: "TestAlpha=true,AllAlpha=false",
			expect: map[Feature]bool{
				allAlphaGate:  false,
				allBetaGate:   false,
				testAlphaGate: true,
				testBetaGate:  false,
			},
		},
		{
			arg: "AllAlpha=true,TestAlpha=false",
			expect: map[Feature]bool{
				allAlphaGate:  true,
				allBetaGate:   false,
				testAlphaGate: false,
				testBetaGate:  false,
			},
		},
		{
			arg: "TestAlpha=false,AllAlpha=true",
			expect: map[Feature]bool{
				allAlphaGate:  true,
				allBetaGate:   false,
				testAlphaGate: false,
				testBetaGate:  false,
			},
		},
		{
			arg: "TestBeta=true,AllAlpha=false",
			expect: map[Feature]bool{
				allAlphaGate:  false,
				allBetaGate:   false,
				testAlphaGate: false,
				testBetaGate:  true,
			},
		},

		{
			arg: "AllBeta=false",
			expect: map[Feature]bool{
				allAlphaGate:  false,
				allBetaGate:   false,
				testAlphaGate: false,
				testBetaGate:  false,
			},
		},
		{
			arg: "AllBeta=true",
			expect: map[Feature]bool{
				allAlphaGate:  false,
				allBetaGate:   true,
				testAlphaGate: false,
				testBetaGate:  true,
			},
		},
		{
			arg: "AllBeta=banana",
			expect: map[Feature]bool{
				allAlphaGate:  false,
				allBetaGate:   false,
				testAlphaGate: false,
				testBetaGate:  false,
			},
			parseError: "invalid value of AllBeta",
		},
		{
			arg: "AllBeta=false,TestBeta=true",
			expect: map[Feature]bool{
				allAlphaGate:  false,
				allBetaGate:   false,
				testAlphaGate: false,
				testBetaGate:  true,
			},
		},
		{
			arg: "TestBeta=true,AllBeta=false",
			expect: map[Feature]bool{
				allAlphaGate:  false,
				allBetaGate:   false,
				testAlphaGate: false,
				testBetaGate:  true,
			},
		},
		{
			arg: "AllBeta=true,TestBeta=false",
			expect: map[Feature]bool{
				allAlphaGate:  false,
				allBetaGate:   true,
				testAlphaGate: false,
				testBetaGate:  false,
			},
		},
		{
			arg: "TestBeta=false,AllBeta=true",
			expect: map[Feature]bool{
				allAlphaGate:  false,
				allBetaGate:   true,
				testAlphaGate: false,
				testBetaGate:  false,
			},
		},
		{
			arg: "TestAlpha=true,AllBeta=false",
			expect: map[Feature]bool{
				allAlphaGate:  false,
				allBetaGate:   false,
				testAlphaGate: true,
				testBetaGate:  false,
			},
		},
	}
	for i, test := range tests {
		t.Run(test.arg, func(t *testing.T) {
			fs := pflag.NewFlagSet("testfeaturegateflag", pflag.ContinueOnError)
			f := NewFeatureGate()
			f.Add(map[Feature]FeatureSpec{
				testAlphaGate: {Default: false, PreRelease: Alpha},
				testBetaGate:  {Default: false, PreRelease: Beta},
			})
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
				if actual := f.enabled.Load().(map[Feature]bool)[k]; actual != v {
					t.Errorf("%d: expected %s=%v, Got %v", i, k, v, actual)
				}
			}
		})
	}
}

func TestFeatureGateOverride(t *testing.T) {
	const testAlphaGate Feature = "TestAlpha"
	const testBetaGate Feature = "TestBeta"

	// Don't parse the flag, assert defaults are used.
	var f *featureGate = NewFeatureGate()
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
	var f *featureGate = NewFeatureGate()
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

func TestFeatureGateKnownFeatures(t *testing.T) {
	// gates for testing
	const (
		testAlphaGate      Feature = "TestAlpha"
		testBetaGate       Feature = "TestBeta"
		testGAGate         Feature = "TestGA"
		testDeprecatedGate Feature = "TestDeprecated"
	)

	// Don't parse the flag, assert defaults are used.
	var f *featureGate = NewFeatureGate()
	f.Add(map[Feature]FeatureSpec{
		testAlphaGate:      {Default: false, PreRelease: Alpha},
		testBetaGate:       {Default: true, PreRelease: Beta},
		testGAGate:         {Default: true, PreRelease: GA},
		testDeprecatedGate: {Default: false, PreRelease: Deprecated},
	})

	known := strings.Join(f.KnownFeatures(), " ")

	assert.Contains(t, known, testAlphaGate)
	assert.Contains(t, known, testBetaGate)
	assert.NotContains(t, known, testGAGate)
	assert.NotContains(t, known, testDeprecatedGate)
}

func TestFeatureGateSetFromMap(t *testing.T) {
	// gates for testing
	const testAlphaGate Feature = "TestAlpha"
	const testBetaGate Feature = "TestBeta"
	const testLockedTrueGate Feature = "TestLockedTrue"
	const testLockedFalseGate Feature = "TestLockedFalse"

	tests := []struct {
		name        string
		setmap      map[string]bool
		expect      map[Feature]bool
		setmapError string
	}{
		{
			name: "set TestAlpha and TestBeta true",
			setmap: map[string]bool{
				"TestAlpha": true,
				"TestBeta":  true,
			},
			expect: map[Feature]bool{
				testAlphaGate: true,
				testBetaGate:  true,
			},
		},
		{
			name: "set TestBeta true",
			setmap: map[string]bool{
				"TestBeta": true,
			},
			expect: map[Feature]bool{
				testAlphaGate: false,
				testBetaGate:  true,
			},
		},
		{
			name: "set TestAlpha false",
			setmap: map[string]bool{
				"TestAlpha": false,
			},
			expect: map[Feature]bool{
				testAlphaGate: false,
				testBetaGate:  false,
			},
		},
		{
			name: "set TestInvaild true",
			setmap: map[string]bool{
				"TestInvaild": true,
			},
			expect: map[Feature]bool{
				testAlphaGate: false,
				testBetaGate:  false,
			},
			//setmapError: "unrecognized feature gate:",
		},
		{
			name: "set locked gates",
			setmap: map[string]bool{
				"TestLockedTrue":  true,
				"TestLockedFalse": false,
			},
			expect: map[Feature]bool{
				testAlphaGate: false,
				testBetaGate:  false,
			},
		},
		{
			name: "set locked gates",
			setmap: map[string]bool{
				"TestLockedTrue": false,
			},
			expect: map[Feature]bool{
				testAlphaGate: false,
				testBetaGate:  false,
			},
			setmapError: "cannot set feature gate TestLockedTrue to false, feature is locked to true",
		},
		{
			name: "set locked gates",
			setmap: map[string]bool{
				"TestLockedFalse": true,
			},
			expect: map[Feature]bool{
				testAlphaGate: false,
				testBetaGate:  false,
			},
			setmapError: "cannot set feature gate TestLockedFalse to true, feature is locked to false",
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("SetFromMap %s", test.name), func(t *testing.T) {
			f := NewFeatureGate()
			f.Add(map[Feature]FeatureSpec{
				testAlphaGate:       {Default: false, PreRelease: Alpha},
				testBetaGate:        {Default: false, PreRelease: Beta},
				testLockedTrueGate:  {Default: true, PreRelease: GA, LockToDefault: true},
				testLockedFalseGate: {Default: false, PreRelease: GA, LockToDefault: true},
			})
			err := f.SetFromMap(test.setmap)
			if test.setmapError != "" {
				if err == nil {
					t.Errorf("expected error, got none")
				} else if !strings.Contains(err.Error(), test.setmapError) {
					t.Errorf("%d: SetFromMap(%#v) Expected err:%v, Got err:%v", i, test.setmap, test.setmapError, err)
				}
			} else if err != nil {
				t.Errorf("%d: SetFromMap(%#v) Expected success, Got err:%v", i, test.setmap, err)
			}
			for k, v := range test.expect {
				if actual := f.Enabled(k); actual != v {
					t.Errorf("%d: SetFromMap(%#v) Expected %s=%v, Got %s=%v", i, test.setmap, k, v, k, actual)
				}
			}
		})
	}
}

func TestFeatureGateMetrics(t *testing.T) {
	// gates for testing
	featuremetrics.ResetFeatureInfoMetric()
	const testAlphaGate Feature = "TestAlpha"
	const testBetaGate Feature = "TestBeta"
	const testAlphaEnabled Feature = "TestAlphaEnabled"
	const testBetaDisabled Feature = "TestBetaDisabled"
	testedMetrics := []string{"kubernetes_feature_enabled"}
	expectedOutput := `
		# HELP kubernetes_feature_enabled [BETA] This metric records the data about the stage and enablement of a k8s feature.
        # TYPE kubernetes_feature_enabled gauge
        kubernetes_feature_enabled{name="TestAlpha",stage="ALPHA"} 0
        kubernetes_feature_enabled{name="TestBeta",stage="BETA"} 1
		kubernetes_feature_enabled{name="TestAlphaEnabled",stage="ALPHA"} 1
        kubernetes_feature_enabled{name="AllAlpha",stage="ALPHA"} 0
        kubernetes_feature_enabled{name="AllBeta",stage="BETA"} 0
		kubernetes_feature_enabled{name="TestBetaDisabled",stage="ALPHA"} 0
`

	f := NewFeatureGate()
	fMap := map[Feature]FeatureSpec{
		testAlphaGate:    {Default: false, PreRelease: Alpha},
		testAlphaEnabled: {Default: false, PreRelease: Alpha},
		testBetaGate:     {Default: true, PreRelease: Beta},
		testBetaDisabled: {Default: true, PreRelease: Alpha},
	}
	f.Add(fMap)
	f.SetFromMap(map[string]bool{"TestAlphaEnabled": true, "TestBetaDisabled": false})
	f.AddMetrics()
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedOutput), testedMetrics...); err != nil {
		t.Fatal(err)
	}
}

func TestFeatureGateString(t *testing.T) {
	// gates for testing
	const testAlphaGate Feature = "TestAlpha"
	const testBetaGate Feature = "TestBeta"
	const testGAGate Feature = "TestGA"

	featuremap := map[Feature]FeatureSpec{
		testGAGate:    {Default: true, PreRelease: GA},
		testAlphaGate: {Default: false, PreRelease: Alpha},
		testBetaGate:  {Default: true, PreRelease: Beta},
	}

	tests := []struct {
		setmap map[string]bool
		expect string
	}{
		{
			setmap: map[string]bool{
				"TestAlpha": false,
			},
			expect: "TestAlpha=false",
		},
		{
			setmap: map[string]bool{
				"TestAlpha": false,
				"TestBeta":  true,
			},
			expect: "TestAlpha=false,TestBeta=true",
		},
		{
			setmap: map[string]bool{
				"TestGA":    true,
				"TestAlpha": false,
				"TestBeta":  true,
			},
			expect: "TestAlpha=false,TestBeta=true,TestGA=true",
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("SetFromMap %s", test.expect), func(t *testing.T) {
			f := NewFeatureGate()
			f.Add(featuremap)
			f.SetFromMap(test.setmap)
			result := f.String()
			if result != test.expect {
				t.Errorf("%d: SetFromMap(%#v) Expected %s, Got %s", i, test.setmap, test.expect, result)
			}
		})
	}
}

func TestFeatureGateOverrideDefault(t *testing.T) {
	t.Run("overrides take effect", func(t *testing.T) {
		f := NewFeatureGate()
		if err := f.Add(map[Feature]FeatureSpec{
			"TestFeature1": {Default: true},
			"TestFeature2": {Default: false},
		}); err != nil {
			t.Fatal(err)
		}
		if err := f.OverrideDefault("TestFeature1", false); err != nil {
			t.Fatal(err)
		}
		if err := f.OverrideDefault("TestFeature2", true); err != nil {
			t.Fatal(err)
		}
		if f.Enabled("TestFeature1") {
			t.Error("expected TestFeature1 to have effective default of false")
		}
		if !f.Enabled("TestFeature2") {
			t.Error("expected TestFeature2 to have effective default of true")
		}
	})

	t.Run("overrides are preserved across deep copies", func(t *testing.T) {
		f := NewFeatureGate()
		if err := f.Add(map[Feature]FeatureSpec{"TestFeature": {Default: false}}); err != nil {
			t.Fatal(err)
		}
		if err := f.OverrideDefault("TestFeature", true); err != nil {
			t.Fatal(err)
		}
		fcopy := f.DeepCopy()
		if !fcopy.Enabled("TestFeature") {
			t.Error("default override was not preserved by deep copy")
		}
	})

	t.Run("reflected in known features", func(t *testing.T) {
		f := NewFeatureGate()
		if err := f.Add(map[Feature]FeatureSpec{"TestFeature": {
			Default:    false,
			PreRelease: Alpha,
		}}); err != nil {
			t.Fatal(err)
		}
		if err := f.OverrideDefault("TestFeature", true); err != nil {
			t.Fatal(err)
		}
		var found bool
		for _, s := range f.KnownFeatures() {
			if !strings.Contains(s, "TestFeature") {
				continue
			}
			found = true
			if !strings.Contains(s, "default=true") {
				t.Errorf("expected override of default to be reflected in known feature description %q", s)
			}
		}
		if !found {
			t.Error("found no entry for TestFeature in known features")
		}
	})

	t.Run("may not change default for specs with locked defaults", func(t *testing.T) {
		f := NewFeatureGate()
		if err := f.Add(map[Feature]FeatureSpec{
			"LockedFeature": {
				Default:       true,
				LockToDefault: true,
			},
		}); err != nil {
			t.Fatal(err)
		}
		if f.OverrideDefault("LockedFeature", false) == nil {
			t.Error("expected error when attempting to override the default for a feature with a locked default")
		}
		if f.OverrideDefault("LockedFeature", true) == nil {
			t.Error("expected error when attempting to override the default for a feature with a locked default")
		}
	})

	t.Run("does not supersede explicitly-set value", func(t *testing.T) {
		f := NewFeatureGate()
		if err := f.Add(map[Feature]FeatureSpec{"TestFeature": {Default: true}}); err != nil {
			t.Fatal(err)
		}
		if err := f.OverrideDefault("TestFeature", false); err != nil {
			t.Fatal(err)
		}
		if err := f.SetFromMap(map[string]bool{"TestFeature": true}); err != nil {
			t.Fatal(err)
		}
		if !f.Enabled("TestFeature") {
			t.Error("expected feature to be effectively enabled despite default override")
		}
	})

	t.Run("prevents re-registration of feature spec after overriding default", func(t *testing.T) {
		f := NewFeatureGate()
		if err := f.Add(map[Feature]FeatureSpec{
			"TestFeature": {
				Default:    true,
				PreRelease: Alpha,
			},
		}); err != nil {
			t.Fatal(err)
		}
		if err := f.OverrideDefault("TestFeature", false); err != nil {
			t.Fatal(err)
		}
		if err := f.Add(map[Feature]FeatureSpec{
			"TestFeature": {
				Default:    true,
				PreRelease: Alpha,
			},
		}); err == nil {
			t.Error("expected re-registration to return a non-nil error after overriding its default")
		}
	})

	t.Run("does not allow override for an unknown feature", func(t *testing.T) {
		f := NewFeatureGate()
		if err := f.OverrideDefault("TestFeature", true); err == nil {
			t.Error("expected an error to be returned in attempt to override default for unregistered feature")
		}
	})

	t.Run("returns error if already added to flag set", func(t *testing.T) {
		f := NewFeatureGate()
		fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
		f.AddFlag(fs)

		if err := f.OverrideDefault("TestFeature", true); err == nil {
			t.Error("expected a non-nil error to be returned")
		}
	})
}
