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

// TestImageExport tests 'rkt image export', it will import some existing
// image, export it with rkt image export and check that the exported ACI hash
// matches the hash of the imported ACI
func TestImageExport(t *testing.T) {
	manifest := schema.ImageManifest{
		Name: "coreos.com/rkt-image-export-test",
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

	expectManifest, err := acitest.ImageManifestString(&manifest)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tmpDir := mustTempDir("rkt-TestImageExport-")
	defer os.RemoveAll(tmpDir)

	tmpManifest, err := ioutil.TempFile(tmpDir, "manifest")
	if err != nil {
		panic(fmt.Sprintf("Cannot create temp manifest: %v", err))
	}
	defer tmpManifest.Close()
	tmpManifestName := tmpManifest.Name()
	if err := ioutil.WriteFile(tmpManifestName, []byte(expectManifest), 0600); err != nil {
		panic(fmt.Sprintf("Cannot write to temp manifest: %v", err))
	}
	defer os.Remove(tmpManifestName)

	testImage := patchTestACI("rkt-inspect-image-export.aci", "--manifest", tmpManifestName)
	defer os.Remove(testImage)
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	testImageID, err := importImageAndFetchHash(t, ctx, "", testImage)
	if err != nil {
		t.Fatalf("%v", err)
	}

	testImageHash, err := getHash(testImage)
	if err != nil {
		panic(fmt.Sprintf("Error getting image hash: %v", err))
	}

	tests := []struct {
		image        string
		shouldFind   bool
		expectedHash string
	}{
		{
			string(manifest.Name),
			true,
			testImageHash,
		},
		{
			testImageID,
			true,
			testImageHash,
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
		outputAciPath := filepath.Join(tmpDir, fmt.Sprintf("exported-%d.aci", i))
		runCmd := fmt.Sprintf("%s image export %s %s", ctx.Cmd(), tt.image, outputAciPath)
		t.Logf("Running 'image export' test #%v: %v", i, runCmd)
		spawnAndWaitOrFail(t, runCmd, expectedStatus)

		if !tt.shouldFind {
			continue
		}

		exportedHash, err := getHash(outputAciPath)
		if err != nil {
			t.Fatalf("Error getting exported image hash: %v", err)
		}

		if exportedHash != tt.expectedHash {
			t.Fatalf("Expected hash %q but got %s", tt.expectedHash, exportedHash)
		}
	}
}
