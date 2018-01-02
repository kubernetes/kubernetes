// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build host coreos src kvm

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"testing"

	"github.com/coreos/rkt/pkg/aci/acitest"
	"github.com/coreos/rkt/tests/testutils"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
)

type osArchTest struct {
	image        string
	rktCmd       string
	expectedLine string
	expectError  bool
}

func osArchTestRemoveImages(tests []osArchTest) {
	for _, tt := range tests {
		os.Remove(tt.image)
	}
}

func getMissingOrInvalidTests(t *testing.T, ctx *testutils.RktRunCtx) []osArchTest {
	var tests []osArchTest

	defer osArchTestRemoveImages(tests)

	manifestOSArch := schema.ImageManifest{
		Name: "coreos.com/rkt-missing-os-arch-test",
		App: &types.App{
			Exec: types.Exec{
				"/inspect",
				"--print-msg=HelloWorld",
			},
			User: "0", Group: "0",
			WorkingDirectory: "/",
		},
		Labels: types.Labels{
			{"version", "1.25.0"},
		},
	}

	// Copy the lables of the image manifest to use the common
	// part in all test cases.
	labels := make(types.Labels, len(manifestOSArch.Labels))
	copy(labels, manifestOSArch.Labels)

	// Test a valid image as a sanity check
	manifestOSArch.Labels = append(
		labels,
		types.Label{"os", "linux"},
		types.Label{"arch", "amd64"},
	)

	goodManifestFile := "good-manifest.json"
	goodManifestStr, err := acitest.ImageManifestString(&manifestOSArch)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := ioutil.WriteFile(goodManifestFile, []byte(goodManifestStr), 0600); err != nil {
		t.Fatalf("Cannot write good manifest: %v", err)
	}
	defer os.Remove(goodManifestFile)

	goodImage := patchTestACI("rkt-good-image.aci", fmt.Sprintf("--manifest=%s", goodManifestFile))
	goodTest := osArchTest{
		image:        goodImage,
		rktCmd:       fmt.Sprintf("%s --insecure-options=image run --mds-register=false %s", ctx.Cmd(), goodImage),
		expectedLine: "HelloWorld",
		expectError:  false,
	}
	tests = append(tests, goodTest)

	// Test an image with a missing os label
	manifestOSArch.Labels = append(labels, types.Label{"arch", "amd64"})

	missingOSManifestFile := "missingOS-manifest.json"
	missingOSManifestStr, err := acitest.ImageManifestString(&manifestOSArch)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := ioutil.WriteFile(missingOSManifestFile, []byte(missingOSManifestStr), 0600); err != nil {
		t.Fatalf("Cannot write missing OS manifest: %v", err)
	}
	defer os.Remove(missingOSManifestFile)

	missingOSImage := patchTestACI("rkt-missing-os.aci", fmt.Sprintf("--manifest=%s", missingOSManifestFile))
	missingOSTest := osArchTest{
		image:        missingOSImage,
		rktCmd:       fmt.Sprintf("%s --insecure-options=image run %s", ctx.Cmd(), missingOSImage),
		expectedLine: "missing os label in the image manifest",
		expectError:  true,
	}
	tests = append(tests, missingOSTest)

	// Test an image with a missing arch label
	manifestOSArch.Labels = append(labels, types.Label{"os", "linux"})

	missingArchManifestFile := "missingArch-manifest.json"
	missingArchManifestStr, err := acitest.ImageManifestString(&manifestOSArch)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := ioutil.WriteFile(missingArchManifestFile, []byte(missingArchManifestStr), 0600); err != nil {
		t.Fatalf("Cannot write missing Arch manifest: %v", err)
	}
	defer os.Remove(missingArchManifestFile)

	missingArchImage := patchTestACI("rkt-missing-arch.aci", fmt.Sprintf("--manifest=%s", missingArchManifestFile))
	missingArchTest := osArchTest{
		image:        missingArchImage,
		rktCmd:       fmt.Sprintf("%s --insecure-options=image run %s", ctx.Cmd(), missingArchImage),
		expectedLine: "missing arch label in the image manifest",
		expectError:  true,
	}
	tests = append(tests, missingArchTest)

	// Test an image with an invalid os
	manifestOSArch.Labels = append(
		labels,
		types.Label{"os", "freebsd"},
		types.Label{"arch", "amd64"},
	)

	invalidOSManifestFile := "invalid-os-manifest.json"
	invalidOSManifestStr, err := acitest.ImageManifestString(&manifestOSArch)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := ioutil.WriteFile(invalidOSManifestFile, []byte(invalidOSManifestStr), 0600); err != nil {
		t.Fatalf("Cannot write invalid os manifest: %v", err)
	}
	defer os.Remove(invalidOSManifestFile)

	invalidOSImage := patchTestACI("rkt-invalid-os.aci", fmt.Sprintf("--manifest=%s", invalidOSManifestFile))
	invalidOSTest := osArchTest{
		image:        invalidOSImage,
		rktCmd:       fmt.Sprintf("%s --insecure-options=image run %s", ctx.Cmd(), invalidOSImage),
		expectedLine: `bad os "freebsd"`,
		expectError:  true,
	}
	tests = append(tests, invalidOSTest)

	// Test an image with an invalid arch
	manifestOSArch.Labels = append(
		labels,
		types.Label{"os", "linux"},
		types.Label{"arch", "armv5l"},
	)

	invalidArchManifestFile := "invalid-arch-manifest.json"
	invalidArchManifestStr, err := acitest.ImageManifestString(&manifestOSArch)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := ioutil.WriteFile(invalidArchManifestFile, []byte(invalidArchManifestStr), 0600); err != nil {
		t.Fatalf("Cannot write invalid arch manifest: %v", err)
	}
	defer os.Remove(invalidArchManifestFile)

	retTests := tests
	tests = nil
	return retTests
}

// TestMissingOrInvalidOSArchRun tests that rkt errors out when it tries to run
// an image (not present in the store) with a missing or unsupported os/arch
func TestMissingOrInvalidOSArchRun(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()
	tests := getMissingOrInvalidTests(t, ctx)
	defer osArchTestRemoveImages(tests)

	for i, tt := range tests {
		t.Logf("Running test #%v: %v", i, tt.rktCmd)
		runRktAndCheckOutput(t, tt.rktCmd, tt.expectedLine, tt.expectError)
	}
}

// TestMissingOrInvalidOSArchFetchRun tests that rkt errors out when it tries
// to run an already fetched image with a missing or unsupported os/arch
func TestMissingOrInvalidOSArchFetchRun(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()
	tests := getMissingOrInvalidTests(t, ctx)
	defer osArchTestRemoveImages(tests)

	for i, tt := range tests {
		imgHash, err := importImageAndFetchHash(t, ctx, "", tt.image)
		if err != nil {
			t.Fatalf("%v", err)
		}
		rktCmd := fmt.Sprintf("%s run --mds-register=false %s", ctx.Cmd(), imgHash)
		t.Logf("Running test #%v: %v", i, rktCmd)
		runRktAndCheckOutput(t, rktCmd, tt.expectedLine, tt.expectError)
	}
}
