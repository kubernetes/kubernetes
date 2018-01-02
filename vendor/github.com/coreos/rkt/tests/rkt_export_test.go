// Copyright 2016 The rkt Authors
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
	"path/filepath"
	"testing"

	"github.com/coreos/rkt/tests/testutils"
)

const (
	testFile    = "test.txt"
	testContent = "ThisIsATest"
)

type ExportTestCase struct {
	runArgs        string
	writeArgs      string
	readArgs       string
	exportArgs     string
	expectedResult string
	unmountOverlay bool
	multiAppPod    bool
	NeedsOverlay   bool
	NeedsUserNS    bool
}

type exportTest []ExportTestCase

var exportTestCases = map[string]ExportTestCase{
	"noOverlaySimpleTest": {
		"--no-overlay --insecure-options=image",
		"--write-file --file-name=" + testFile + " --content=" + testContent,
		"--read-file --file-name=" + testFile,
		"",
		testContent,
		false,
		false,
		false,
		false,
	},

	"specifiedAppTest": {
		"--no-overlay --insecure-options=image",
		"--write-file --file-name=" + testFile + " --content=" + testContent,
		"--read-file --file-name=" + testFile,
		"--app=rkt-inspect",
		testContent,
		false,
		false,
		false,
		false,
	},

	"multiAppPodTest": {
		"--no-overlay --insecure-options=image",
		"--write-file --file-name=" + testFile + " --content=" + testContent,
		"--read-file --file-name=" + testFile,
		"--app=rkt-inspect",
		testContent,
		false,
		true,
		false,
		false,
	},

	"userNS": {
		"--private-users --no-overlay --insecure-options=image",
		"--write-file --file-name=" + testFile + " --content=" + testContent,
		"--read-file --file-name=" + testFile,
		"",
		testContent,
		false,
		false,
		false,
		true,
	},

	"overlaySimpleTest": {
		"--insecure-options=image",
		"--write-file --file-name=" + testFile + " --content=" + testContent,
		"--read-file --file-name=" + testFile,
		"",
		testContent,
		false,
		false,
		true,
		false,
	},

	"overlaySimulateReboot": {
		"--insecure-options=image",
		"--write-file --file-name=" + testFile + " --content=" + testContent,
		"--read-file --file-name=" + testFile,
		"",
		testContent,
		true,
		false,
		true,
		false,
	},
}

func (ct ExportTestCase) Execute(t *testing.T, ctx *testutils.RktRunCtx) {
	tmpDir := mustTempDir("rkt-TestExport-tmp-")
	defer os.RemoveAll(tmpDir)

	tmpTestAci := filepath.Join(tmpDir, "test.aci")

	// Prepare the image with modifications
	var additionalRunArgs string
	if ct.multiAppPod {
		tmpAdditionalAci := patchTestACI("other.aci", "--name=other")
		defer os.Remove(tmpAdditionalAci)
		const otherArgs = "--write-file --file-name=test.txt --content=NotTheRightContent"
		additionalRunArgs = fmt.Sprintf("%s --exec=/inspect -- %s", tmpAdditionalAci, otherArgs)
	} else {
		additionalRunArgs = ""
	}
	const runInspect = "%s %s %s %s --exec=/inspect -- %s --- %s"
	prepareCmd := fmt.Sprintf(runInspect, ctx.Cmd(), "prepare", ct.runArgs, getInspectImagePath(), ct.writeArgs, additionalRunArgs)
	t.Logf("Preparing 'inspect --write-file'")
	uuid := runRktAndGetUUID(t, prepareCmd)

	runCmd := fmt.Sprintf("%s run-prepared %s", ctx.Cmd(), uuid)
	t.Logf("Running 'inspect --write-file'")
	child := spawnOrFail(t, runCmd)
	waitOrFail(t, child, 0)

	if ct.unmountOverlay {
		unmountPod(t, ctx, uuid, true)
	}

	// Export the image
	exportCmd := fmt.Sprintf("%s export %s %s %s", ctx.Cmd(), ct.exportArgs, uuid, tmpTestAci)
	t.Logf("Running 'export'")
	child = spawnOrFail(t, exportCmd)
	waitOrFail(t, child, 0)

	// Run the newly created ACI and check the output
	readCmd := fmt.Sprintf(runInspect, ctx.Cmd(), "run", ct.runArgs, tmpTestAci, ct.readArgs, "")
	t.Logf("Running 'inspect --read-file'")
	child = spawnOrFail(t, readCmd)
	if ct.expectedResult != "" {
		if _, out, err := expectRegexWithOutput(child, ct.expectedResult); err != nil {
			t.Fatalf("expected %q but not found: %v\n%s", ct.expectedResult, err, out)
		}
	}
	waitOrFail(t, child, 0)

	// run garbage collector on pods and images
	runGC(t, ctx)
	runImageGC(t, ctx)
}
