/*
Copyright 2024 The Kubernetes Authors.

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

package version

import (
	"fmt"
	"strings"
	"testing"

	"github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/util/version"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/featuregate"
)

const (
	testComponent = "test"
)

func TestEffectiveVersionRegistry(t *testing.T) {
	r := NewComponentGlobalsRegistry()
	ver1 := NewEffectiveVersion("1.31")
	ver2 := NewEffectiveVersion("1.28")

	if r.EffectiveVersionFor(testComponent) != nil {
		t.Fatalf("expected nil EffectiveVersion initially")
	}
	if err := r.Register(testComponent, ver1, nil, false); err != nil {
		t.Fatalf("expected no error to register new component, but got err: %v", err)
	}
	if !r.EffectiveVersionFor(testComponent).EqualTo(ver1) {
		t.Fatalf("expected EffectiveVersionFor to return the version registered")
	}
	// overwrite
	if err := r.Register(testComponent, ver2, nil, false); err == nil {
		t.Fatalf("expected error to register existing component when override is false")
	}
	if err := r.Register(testComponent, ver2, nil, true); err != nil {
		t.Fatalf("expected no error to overriding existing component, but got err: %v", err)
	}
	if !r.EffectiveVersionFor(testComponent).EqualTo(ver2) {
		t.Fatalf("expected EffectiveVersionFor to return the version overridden")
	}
}

func testRegistry(t *testing.T) *componentGlobalsRegistry {
	r := componentGlobalsRegistry{
		componentGlobals:       map[string]ComponentGlobals{},
		emulationVersionConfig: make(cliflag.ConfigurationMap),
		featureGatesConfig:     make(map[string][]string),
	}
	verKube := NewEffectiveVersion("1.31")
	fgKube := featuregate.NewVersionedFeatureGate(version.MustParse("0.0"))
	err := fgKube.AddVersioned(map[featuregate.Feature]featuregate.VersionedSpecs{
		"kubeA": {
			{Version: version.MustParse("1.31"), Default: true, LockToDefault: true, PreRelease: featuregate.GA},
			{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Beta},
			{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Alpha},
		},
		"kubeB": {
			{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
		},
		"commonC": {
			{Version: version.MustParse("1.29"), Default: true, PreRelease: featuregate.Beta},
			{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Alpha},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	verTest := NewEffectiveVersion("2.8")
	fgTest := featuregate.NewVersionedFeatureGate(version.MustParse("0.0"))
	err = fgTest.AddVersioned(map[featuregate.Feature]featuregate.VersionedSpecs{
		"testA": {
			{Version: version.MustParse("2.10"), Default: true, PreRelease: featuregate.GA},
			{Version: version.MustParse("2.8"), Default: false, PreRelease: featuregate.Beta},
			{Version: version.MustParse("2.7"), Default: false, PreRelease: featuregate.Alpha},
		},
		"testB": {
			{Version: version.MustParse("2.9"), Default: false, PreRelease: featuregate.Alpha},
		},
		"commonC": {
			{Version: version.MustParse("2.9"), Default: true, PreRelease: featuregate.Beta},
			{Version: version.MustParse("2.7"), Default: false, PreRelease: featuregate.Alpha},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	_ = r.Register(DefaultKubeComponent, verKube, fgKube, true)
	_ = r.Register(testComponent, verTest, fgTest, true)
	return &r
}

func TestVersionFlagOptions(t *testing.T) {
	r := testRegistry(t)
	emuVers := strings.Join(r.versionFlagOptions(true), "\n")
	expectedEmuVers := "kube=1.31..1.31 (default=1.31)\ntest=2.8..2.8 (default=2.8)"
	if emuVers != expectedEmuVers {
		t.Errorf("wanted emulation version flag options to be: %s, got %s", expectedEmuVers, emuVers)
	}
	minCompVers := strings.Join(r.versionFlagOptions(false), "\n")
	expectedMinCompVers := "kube=1.30..1.31 (default=1.30)\ntest=2.7..2.8 (default=2.7)"
	if minCompVers != expectedMinCompVers {
		t.Errorf("wanted min compatibility version flag options to be: %s, got %s", expectedMinCompVers, minCompVers)
	}
}

func TestVersionedFeatureGateFlag(t *testing.T) {
	r := testRegistry(t)
	known := strings.Join(r.knownFeatures(), "\n")
	expectedKnown := "kube:AllAlpha=true|false (ALPHA - default=false)\n" +
		"kube:AllBeta=true|false (BETA - default=false)\n" +
		"kube:commonC=true|false (BETA - default=true)\n" +
		"kube:kubeB=true|false (ALPHA - default=false)\n" +
		"test:AllAlpha=true|false (ALPHA - default=false)\n" +
		"test:AllBeta=true|false (BETA - default=false)\n" +
		"test:commonC=true|false (ALPHA - default=false)\n" +
		"test:testA=true|false (BETA - default=false)"
	if known != expectedKnown {
		t.Errorf("wanted min compatibility version flag options to be:\n%s, got:\n%s", expectedKnown, known)
	}
}

func TestFlags(t *testing.T) {
	tests := []struct {
		name                         string
		emulationVersionFlag         string
		featureGatesFlag             string
		parseError                   string
		expectedKubeEmulationVersion *version.Version
		expectedTestEmulationVersion *version.Version
		expectedKubeFeatureValues    map[featuregate.Feature]bool
		expectedTestFeatureValues    map[featuregate.Feature]bool
	}{
		{
			name:                         "setting kube emulation version",
			emulationVersionFlag:         "kube=1.30",
			expectedKubeEmulationVersion: version.MajorMinor(1, 30),
		},
		{
			name:                         "setting kube emulation version, prefix v ok",
			emulationVersionFlag:         "kube=v1.30",
			expectedKubeEmulationVersion: version.MajorMinor(1, 30),
		},
		{
			name:                         "setting test emulation version",
			emulationVersionFlag:         "test=2.7",
			expectedKubeEmulationVersion: version.MajorMinor(1, 31),
			expectedTestEmulationVersion: version.MajorMinor(2, 7),
		},
		{
			name:                 "version missing component",
			emulationVersionFlag: "1.31",
			parseError:           "component not registered: 1.31",
		},
		{
			name:                 "version unregistered component",
			emulationVersionFlag: "test3=1.31",
			parseError:           "component not registered: test3",
		},
		{
			name:                 "invalid version",
			emulationVersionFlag: "test=1.foo",
			parseError:           "illegal version string \"1.foo\"",
		},
		{
			name:                         "setting test feature flag",
			emulationVersionFlag:         "test=2.7",
			featureGatesFlag:             "test:testA=true",
			expectedKubeEmulationVersion: version.MajorMinor(1, 31),
			expectedTestEmulationVersion: version.MajorMinor(2, 7),
			expectedKubeFeatureValues:    map[featuregate.Feature]bool{"kubeA": true, "kubeB": false, "commonC": true},
			expectedTestFeatureValues:    map[featuregate.Feature]bool{"testA": true, "testB": false, "commonC": false},
		},
		{
			name:                 "setting future test feature flag",
			emulationVersionFlag: "test=2.7",
			featureGatesFlag:     "test:testA=true,test:testB=true",
			parseError:           "cannot set feature gate testB to true, feature is PreAlpha at emulated version 2.7",
		},
		{
			name:                         "setting kube feature flag",
			emulationVersionFlag:         "test=2.7,kube=1.30",
			featureGatesFlag:             "test:commonC=true,commonC=false,kube:kubeB=true",
			expectedKubeEmulationVersion: version.MajorMinor(1, 30),
			expectedTestEmulationVersion: version.MajorMinor(2, 7),
			expectedKubeFeatureValues:    map[featuregate.Feature]bool{"kubeA": false, "kubeB": true, "commonC": false},
			expectedTestFeatureValues:    map[featuregate.Feature]bool{"testA": false, "testB": false, "commonC": true},
		},
		{
			name:                 "setting locked kube feature flag",
			emulationVersionFlag: "test=2.7",
			featureGatesFlag:     "kubeA=false",
			parseError:           "cannot set feature gate kubeA to false, feature is locked to true",
		},
		{
			name:                 "setting unknown test feature flag",
			emulationVersionFlag: "test=2.7",
			featureGatesFlag:     "test:testD=true",
			parseError:           "unrecognized feature gate: testD",
		},
		{
			name:                 "setting unknown component feature flag",
			emulationVersionFlag: "test=2.7",
			featureGatesFlag:     "test3:commonC=true",
			parseError:           "component not registered: test3",
		},
	}
	for i, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fs := pflag.NewFlagSet("testflag", pflag.ContinueOnError)
			r := testRegistry(t)
			r.AddFlags(fs)

			err := fs.Parse([]string{fmt.Sprintf("--emulated-version=%s", test.emulationVersionFlag),
				fmt.Sprintf("--feature-gates=%s", test.featureGatesFlag)})
			if err == nil {
				err = r.Set()
			}
			if test.parseError != "" {
				if err == nil || !strings.Contains(err.Error(), test.parseError) {
					t.Fatalf("%d: Parse() expected: %v, got: %v", i, test.parseError, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("%d: Parse() expected: nil, got: %v", i, err)
			}
			if test.expectedKubeEmulationVersion != nil {
				v := r.EffectiveVersionFor("kube").EmulationVersion()
				if !v.EqualTo(test.expectedKubeEmulationVersion) {
					t.Fatalf("%d: EmulationVersion expected: %s, got: %s", i, test.expectedKubeEmulationVersion.String(), v.String())
					return
				}
			}
			if test.expectedTestEmulationVersion != nil {
				v := r.EffectiveVersionFor("test").EmulationVersion()
				if !v.EqualTo(test.expectedTestEmulationVersion) {
					t.Fatalf("%d: EmulationVersion expected: %s, got: %s", i, test.expectedTestEmulationVersion.String(), v.String())
					return
				}
			}
			for f, v := range test.expectedKubeFeatureValues {
				if r.FeatureGateFor(DefaultKubeComponent).Enabled(f) != v {
					t.Errorf("%d: expected kube feature Enabled(%s)=%v", i, f, v)
				}
			}
			for f, v := range test.expectedTestFeatureValues {
				if r.FeatureGateFor(testComponent).Enabled(f) != v {
					t.Errorf("%d: expected test feature Enabled(%s)=%v", i, f, v)
				}
			}
		})
	}
}
