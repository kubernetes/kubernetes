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
	"os"
	"path/filepath"
	"testing"

	"github.com/coreos/rkt/tests/testutils"
)

// TestImageExtract tests 'rkt image extract', it will import some existing
// image with the inspect binary, extract it with rkt image extract and check
// that the exported /inspect hash matches the original inspect binary hash
func TestImageExtract(t *testing.T) {
	testImage := getInspectImagePath()
	testImageName := "coreos.com/rkt-inspect"

	inspectFile := testutils.GetValueFromEnvOrPanic("INSPECT_BINARY")
	inspectHash := getHashOrPanic(inspectFile)

	tmpDir := mustTempDir("rkt-TestImageRender-")
	defer os.RemoveAll(tmpDir)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

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
			testImageName,
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
		outputPath := filepath.Join(tmpDir, fmt.Sprintf("extracted-%d", i))
		runCmd := fmt.Sprintf("%s image extract --rootfs-only %s %s", ctx.Cmd(), tt.image, outputPath)
		t.Logf("Running 'image extract' test #%v: %v", i, runCmd)
		spawnAndWaitOrFail(t, runCmd, expectedStatus)

		if !tt.shouldFind {
			continue
		}

		extractedInspectHash, err := getHash(filepath.Join(outputPath, "inspect"))
		if err != nil {
			t.Fatalf("Cannot get rendered inspect binary's hash")
		}
		if extractedInspectHash != tt.expectedHash {
			t.Fatalf("Expected /inspect hash %q but got %s", tt.expectedHash, extractedInspectHash)
		}
	}
}
