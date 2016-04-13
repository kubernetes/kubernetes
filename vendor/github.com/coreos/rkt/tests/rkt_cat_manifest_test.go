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

	tmpDir := createTempDirOrPanic(imgName)
	defer os.RemoveAll(tmpDir)

	tests := []struct {
		uuid  string
		match string
	}{
		{
			podUuid,
			imgName,
		},
		{
			podUuid,
			imageHash[:20],
		},
		{
			"1234567890abcdef",
			"no matches found for",
		},
	}

	for i, tt := range tests {
		runCmd := fmt.Sprintf("%s cat-manifest %s", ctx.Cmd(), tt.uuid)
		t.Logf("Running test #%d", i)
		runRktAndCheckRegexOutput(t, runCmd, tt.match)
	}
}
