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

	"github.com/coreos/rkt/tests/testutils"
)

// TestCatManifest tests 'rkt cat-manifest', it will:
func TestCatManifest(t *testing.T) {
	const imgName = "rkt-cat-manifest-test"

	image := patchTestACI(fmt.Sprintf("%s.aci", imgName), fmt.Sprintf("--name=%s", imgName))
	defer os.Remove(image)

	imageHash := getHashOrPanic(image)
	imgID := ImageID{path: image, hash: imageHash}

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	// Prepare image
	cmd := fmt.Sprintf("%s --insecure-options=image prepare %s", ctx.Cmd(), imgID.path)
	podUuid := runRktAndGetUUID(t, cmd)

	tmpDir := mustTempDir(imgName)
	defer os.RemoveAll(tmpDir)

	tests := []struct {
		uuid     string
		match    string
		uuidFile bool
	}{
		{
			podUuid,
			imgName,
			false,
		},
		{
			podUuid,
			imageHash[:20],
			false,
		},
		{
			"1234567890abcdef",
			"no matches found for",
			false,
		},
		{
			"",
			imageHash[:20],
			true,
		},
	}

	for i, tt := range tests {
		if tt.uuidFile == true {
			podUUID := runRktAndGetUUID(t, cmd)
			uuidFile, err := ioutil.TempFile(tmpDir, "uuid-file")
			if err != nil {
				panic(fmt.Sprintf("Cannot create uuid-file: %v", err))
			}
			uuidFilePath := uuidFile.Name()
			if err := ioutil.WriteFile(uuidFilePath, []byte(podUUID), 0600); err != nil {
				panic(fmt.Sprintf("Cannot write pod UUID to uuid-file: %v", err))
			}
			runCmd := fmt.Sprintf("%s cat-manifest --uuid-file=%s --pretty-print=false %s", ctx.Cmd(), uuidFilePath)
			t.Logf("Running test #%d", i)
			runRktAndCheckRegexOutput(t, runCmd, tt.match)
		} else {
			runCmd := fmt.Sprintf("%s cat-manifest --pretty-print=false %s", ctx.Cmd(), tt.uuid)
			t.Logf("Running test #%d", i)
			runRktAndCheckRegexOutput(t, runCmd, tt.match)
		}
	}
}
