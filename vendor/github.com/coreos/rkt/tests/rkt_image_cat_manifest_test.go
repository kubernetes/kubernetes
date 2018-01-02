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

// TestImageCatManifest tests 'rkt image cat-manifest', it will:
// Read some existing image manifest via the image name, and verify the result.
// Read some existing image manifest via the image hash, and verify the result.
// Read some non-existing image manifest via the image name, and verify nothing is found.
// Read some non-existing image manifest via the image hash, and verify nothing is found.
func TestImageCatManifest(t *testing.T) {
	manifestCat := schema.ImageManifest{
		Name: "coreos.com/rkt-image-cat-manifest-test",
		App: &types.App{
			Exec: types.Exec{"/inspect"},
			User: "0", Group: "0",
			WorkingDirectory: "/",
			Environment: types.Environment{
				{"VAR_FROM_MANIFEST", "manifest"},
			},
		},
		Labels: types.Labels{
			{"version", "1.25.0"},
			{"arch", "amd64"},
			{"os", "linux"},
		},
	}

	expectManifest, err := acitest.ImageManifestString(&manifestCat)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

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

	testImageHash, err := importImageAndFetchHash(t, ctx, "", testImage)
	if err != nil {
		t.Fatalf("%v", err)
	}

	tests := []struct {
		image      string
		shouldFind bool
		expect     string
	}{
		{
			string(manifestCat.Name),
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
		runCmd := fmt.Sprintf("%s image cat-manifest --pretty-print=false %s", ctx.Cmd(), tt.image)
		t.Logf("Running test #%d", i)
		runRktAndCheckOutput(t, runCmd, tt.expect, !tt.shouldFind)
	}
}
