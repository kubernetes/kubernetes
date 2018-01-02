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
	"path/filepath"
	"testing"

	"github.com/coreos/rkt/pkg/aci/acitest"
	"github.com/coreos/rkt/tests/testutils"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
)

// TestImageRender tests 'rkt image render', it will import some existing empty
// image with a dependency on an image with the inspect binary, render it with
// rkt image render and check that the exported image has the /inspect file and
// that its hash matches the original inspect binary hash
func TestImageRender(t *testing.T) {
	baseImage := getInspectImagePath()
	emptyImage := getEmptyImagePath()

	inspectFile := testutils.GetValueFromEnvOrPanic("INSPECT_BINARY")
	inspectHash := getHashOrPanic(inspectFile)

	manifestRender := schema.ImageManifest{
		Name: "coreos.com/rkt-image-render-test",
		App: &types.App{
			Exec: types.Exec{"/inspect"},
			User: "0", Group: "0",
			WorkingDirectory: "/",
			Environment: types.Environment{
				{"VAR_FROM_MANIFEST", "manifest"},
			},
		},
		Dependencies: types.Dependencies{
			{ImageName: "coreos.com/rkt-inspect"},
		},
		Labels: types.Labels{
			{"version", "1.25.0"},
			{"arch", "amd64"},
			{"os", "linux"},
		},
	}

	expectManifest, err := acitest.ImageManifestString(&manifestRender)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tmpDir := mustTempDir("rkt-TestImageRender-")
	defer os.RemoveAll(tmpDir)

	tmpManifest, err := ioutil.TempFile(tmpDir, "manifest")
	if err != nil {
		panic(fmt.Sprintf("Cannot create temp manifest: %v", err))
	}
	if err := ioutil.WriteFile(tmpManifest.Name(), []byte(expectManifest), 0600); err != nil {
		panic(fmt.Sprintf("Cannot write to temp manifest: %v", err))
	}
	defer os.Remove(tmpManifest.Name())

	testImage := patchACI(emptyImage, "rkt-inspect-image-render.aci", "--manifest", tmpManifest.Name())
	defer os.Remove(testImage)
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	_, err = importImageAndFetchHash(t, ctx, "", baseImage)
	if err != nil {
		t.Fatalf("%v", err)
	}
	testImageShortHash, err := importImageAndFetchHash(t, ctx, "", testImage)
	if err != nil {
		t.Fatalf("%v", err)
	}

	tests := []struct {
		image        string
		shouldFind   bool
		expectedHash string
	}{
		{
			string(manifestRender.Name),
			true,
			inspectHash,
		},
		{
			testImageShortHash,
			true,
			inspectHash,
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
		expectedStatus := 254
		if tt.shouldFind {
			expectedStatus = 0
		}
		outputPath := filepath.Join(tmpDir, fmt.Sprintf("rendered-%d", i))
		runCmd := fmt.Sprintf("%s image render --rootfs-only %s %s", ctx.Cmd(), tt.image, outputPath)
		t.Logf("Running 'image render' test #%v: %v", i, runCmd)
		spawnAndWaitOrFail(t, runCmd, expectedStatus)

		if !tt.shouldFind {
			continue
		}

		renderedInspectHash, err := getHash(filepath.Join(outputPath, "inspect"))
		if err != nil {
			t.Fatalf("Cannot get rendered inspect binary's hash")
		}
		if renderedInspectHash != tt.expectedHash {
			t.Fatalf("Expected /inspect hash %q but got %s", tt.expectedHash, renderedInspectHash)
		}
	}
}
