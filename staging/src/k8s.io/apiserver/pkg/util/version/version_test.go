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
)

func TestEffectiveVersionRegistry(t *testing.T) {
	r := NewEffectiveVersionRegistry()
	testComponent := "test"
	ver1 := NewEffectiveVersion("1.31")
	ver2 := NewEffectiveVersion("1.28")

	if r.EffectiveVersionFor(testComponent) != nil {
		t.Fatalf("expected nil EffectiveVersion initially")
	}
	if !r.EffectiveVersionForOrRegister(testComponent, ver1).EqualTo(ver1) {
		t.Fatalf("expected EffectiveVersionForOrRegister to return the version specified")
	}
	// overwrite
	r.RegisterEffectiveVersionFor(testComponent, ver2)
	if !r.EffectiveVersionFor(testComponent).EqualTo(ver2) {
		t.Fatalf("expected EffectiveVersion to be %s", ver2.String())
	}
	if !r.EffectiveVersionForOrRegister(testComponent, ver1).EqualTo(ver2) {
		t.Fatalf("expected EffectiveVersionForOrRegister to return the version already registered")
	}
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name                    string
		binaryVersion           string
		emulationVersion        string
		minCompatibilityVersion string
		expectErrors            bool
	}{
		{
			name:                    "patch version diff ok",
			binaryVersion:           "v1.32.2",
			emulationVersion:        "v1.32.1",
			minCompatibilityVersion: "v1.31.5",
		},
		{
			name:                    "emulation version one minor lower than binary ok",
			binaryVersion:           "v1.32.2",
			emulationVersion:        "v1.31.0",
			minCompatibilityVersion: "v1.31.0",
		},
		{
			name:                    "binary version 1.31, emulation version lower than 1.31",
			binaryVersion:           "v1.31.2",
			emulationVersion:        "v1.30.0",
			minCompatibilityVersion: "v1.30.0",
			expectErrors:            true,
		},
		{
			name:                    "binary version 1.31, emulation version 1.31",
			binaryVersion:           "v1.31.2",
			emulationVersion:        "v1.31.0",
			minCompatibilityVersion: "v1.30.0",
		},
		{
			name:                    "binary version lower than 1.31",
			binaryVersion:           "v1.30.2",
			emulationVersion:        "v1.29.0",
			minCompatibilityVersion: "v1.29.0",
		},
		{
			name:                    "emulation version two minor lower than binary not ok",
			binaryVersion:           "v1.33.2",
			emulationVersion:        "v1.31.0",
			minCompatibilityVersion: "v1.32.0",
			expectErrors:            true,
		},
		{
			name:                    "emulation version one minor higher than binary not ok",
			binaryVersion:           "v1.32.2",
			emulationVersion:        "v1.33.0",
			minCompatibilityVersion: "v1.31.0",
			expectErrors:            true,
		},
		{
			name:                    "emulation version two minor higher than binary not ok",
			binaryVersion:           "v1.32.2",
			emulationVersion:        "v1.34.0",
			minCompatibilityVersion: "v1.31.0",
			expectErrors:            true,
		},
		{
			name:                    "compatibility version same as binary not ok",
			binaryVersion:           "v1.32.2",
			emulationVersion:        "v1.32.0",
			minCompatibilityVersion: "v1.32.0",
			expectErrors:            true,
		},
		{
			name:                    "compatibility version two minor lower than binary not ok",
			binaryVersion:           "v1.32.2",
			emulationVersion:        "v1.32.0",
			minCompatibilityVersion: "v1.30.0",
			expectErrors:            true,
		},
		{
			name:                    "compatibility version one minor higher than binary not ok",
			binaryVersion:           "v1.32.2",
			emulationVersion:        "v1.32.0",
			minCompatibilityVersion: "v1.33.0",
			expectErrors:            true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			binaryVersion := version.MustParseGeneric(test.binaryVersion)
			effective := &effectiveVersion{}
			emulationVersion := version.MustParseGeneric(test.emulationVersion)
			minCompatibilityVersion := version.MustParseGeneric(test.minCompatibilityVersion)
			effective.Set(binaryVersion, emulationVersion, minCompatibilityVersion)

			errs := effective.Validate()
			if len(errs) > 0 && !test.expectErrors {
				t.Errorf("expected no errors, errors found %+v", errs)
			}

			if len(errs) == 0 && test.expectErrors {
				t.Errorf("expected errors, no errors found")
			}
		})
	}
}

func TestEffectiveVersionsFlag(t *testing.T) {
	tests := []struct {
		name                     string
		emulationVerson          string
		expectedEmulationVersion *version.Version
		parseError               string
	}{
		{
			name:                     "major.minor ok",
			emulationVerson:          "1.30",
			expectedEmulationVersion: version.MajorMinor(1, 30),
		},
		{
			name:                     "v prefix ok",
			emulationVerson:          "v1.30",
			expectedEmulationVersion: version.MajorMinor(1, 30),
		},
		{
			name:            "semantic version not ok",
			emulationVerson: "1.30.1",
			parseError:      "version 1.30.1 is not in the format of major.minor",
		},
		{
			name:            "invalid version",
			emulationVerson: "1.foo",
			parseError:      "illegal version string",
		},
	}
	for i, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fs := pflag.NewFlagSet("testflag", pflag.ContinueOnError)
			effective := NewEffectiveVersion("1.30")
			effective.AddFlags(fs, "test")

			err := fs.Parse([]string{fmt.Sprintf("--test-emulated-version=%s", test.emulationVerson)})
			if test.parseError != "" {
				if !strings.Contains(err.Error(), test.parseError) {
					t.Fatalf("%d: Parse() Expected %v, Got %v", i, test.parseError, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("%d: Parse() Expected nil, Got %v", i, err)
			}
			if !effective.EmulationVersion().EqualTo(test.expectedEmulationVersion) {
				t.Errorf("%d: EmulationVersion Expected %s, Got %s", i, test.expectedEmulationVersion.String(), effective.EmulationVersion().String())
			}
		})
	}
}
