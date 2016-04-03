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

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"strings"
	"testing"

	"github.com/coreos/rkt/tests/testutils"
)

const (
	manifestOSArchTemplate = `{"acKind":"ImageManifest","acVersion":"0.7.4","name":"IMG_NAME","labels":[{"name":"version","value":"1.2.1"}ARCH_OS],"app":{"exec":["/inspect", "--print-msg=HelloWorld"],"user":"0","group":"0","workingDirectory":"/"}}`
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
	testImageName := "coreos.com/rkt-missing-os-arch-test"
	manifest := strings.Replace(manifestOSArchTemplate, "IMG_NAME", testImageName, 1)

	// Test a valid image as a sanity check
	goodManifestStr := strings.Replace(manifest, "ARCH_OS", `,{"name":"os","value":"linux"},{"name":"arch","value":"amd64"}`, 1)
	goodManifestFile := "good-manifest.json"
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
	missingOSManifestStr := strings.Replace(manifest, "ARCH_OS", `,{"name":"arch","value":"amd64"}`, 1)
	missingOSManifestFile := "missingOS-manifest.json"
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
	missingArchManifestStr := strings.Replace(manifest, "ARCH_OS", `,{"name":"os","value":"linux"}`, 1)
	missingArchManifestFile := "missingArch-manifest.json"
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
	invalidOSManifestStr := strings.Replace(manifest, "ARCH_OS", `,{"name":"os","value":"freebsd"},{"name":"arch","value":"amd64"}`, 1)
	invalidOSManifestFile := "invalid-os-manifest.json"
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
	invalidArchManifestStr := strings.Replace(manifest, "ARCH_OS", `,{"name":"os","value":"linux"},{"name":"arch","value":"armv5l"}`, 1)
	invalidArchManifestFile := "invalid-arch-manifest.json"
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
		imgHash := importImageAndFetchHash(t, ctx, "", tt.image)
		rktCmd := fmt.Sprintf("%s run --mds-register=false %s", ctx.Cmd(), imgHash)
		t.Logf("Running test #%v: %v", i, rktCmd)
		runRktAndCheckOutput(t, rktCmd, tt.expectedLine, tt.expectError)
	}
}
