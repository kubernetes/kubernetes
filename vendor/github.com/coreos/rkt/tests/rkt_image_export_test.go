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
	"path/filepath"
	"strings"
	"testing"

	"github.com/coreos/rkt/tests/testutils"
)

const (
	manifestExportTemplate = `{"acKind":"ImageManifest","acVersion":"0.7.4","name":"IMG_NAME","labels":[{"name":"version","value":"1.2.1"},{"name":"arch","value":"amd64"},{"name":"os","value":"linux"}],"app":{"exec":["/inspect"],"user":"0","group":"0","workingDirectory":"/","environment":[{"name":"VAR_FROM_MANIFEST","value":"manifest"}]}}`
)

// TestImageExport tests 'rkt image export', it will import some existing
// image, export it with rkt image export and check that the exported ACI hash
// matches the hash of the imported ACI
func TestImageExport(t *testing.T) {
	testImageName := "coreos.com/rkt-image-export-test"
	expectManifest := strings.Replace(manifestExportTemplate, "IMG_NAME", testImageName, -1)

	tmpDir := createTempDirOrPanic("rkt-TestImageExport-")
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

	testImageID := importImageAndFetchHash(t, ctx, "", testImage)

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
			testImageName,
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
		expectedStatus := 1
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
