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

	// The expected image manifest of the 'rkt-inspect-image-cat-manifest.aci'.
	manifestTemplate = `{"acKind":"ImageManifest","acVersion":"0.7.4","name":"IMG_NAME","labels":[{"name":"version","value":"1.2.1"},{"name":"arch","value":"amd64"},{"name":"os","value":"linux"}],"app":{"exec":["/inspect"],"user":"0","group":"0","workingDirectory":"/","environment":[{"name":"VAR_FROM_MANIFEST","value":"manifest"}]}}`
)

// TestImageCatManifest tests 'rkt image cat-manifest', it will:
// Read some existing image manifest via the image name, and verify the result.
// Read some existing image manifest via the image hash, and verify the result.
// Read some non-existing image manifest via the image name, and verify nothing is found.
// Read some non-existing image manifest via the image hash, and verify nothing is found.
func TestImageCatManifest(t *testing.T) {
	testImageName := "coreos.com/rkt-image-cat-manifest-test"
	expectManifest := strings.Replace(manifestTemplate, "IMG_NAME", testImageName, -1)

	tmpManifest, err := ioutil.TempFile("", "rkt-TestImageCatManifest-")
	if err != nil {
		t.Fatalf("Cannot create temp manifest: %v", err)
	}
	defer os.Remove(tmpManifest.Name())
	if err := ioutil.WriteFile(tmpManifest.Name(), []byte(expectManifest), 0600); err != nil {
		t.Fatalf("Cannot write to temp manifest: %v", err)
	}

	testImage := patchTestACI("rkt-inspect-image-cat-manifest.aci", "--manifest", tmpManifest.Name())
	defer os.Remove(testImage)
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	testImageHash := importImageAndFetchHash(t, ctx, "", testImage)

	tests := []struct {
		image      string
		shouldFind bool
		expect     string
	}{
		{
			testImageName,
			true,
			expectManifest,
		},
		{
			testImageHash,
			true,
			expectManifest,
		},
		{
			"sha512-not-existed",
			false,
			"",
		},
		{
			"some~random~aci~name",
			false,
			"",
		},
	}

	for i, tt := range tests {
		runCmd := fmt.Sprintf("%s image cat-manifest %s", ctx.Cmd(), tt.image)
		t.Logf("Running test #%d", i)
		runRktAndCheckOutput(t, runCmd, tt.expect, !tt.shouldFind)
	}
}
