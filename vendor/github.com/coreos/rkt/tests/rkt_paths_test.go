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

// +build coreos src kvm

package main

import (
	"fmt"
	"os"
	"testing"

	"github.com/coreos/rkt/tests/testutils"
)

// TestPathsWrite checks whether access to paths like /proc/sysrq-trigger are
// restricted
func TestPathsWrite(t *testing.T) {
	if TestedFlavor.Kvm == true {
		t.Skip("Not running test for kvm flavour")
	}

	imageFile := patchTestACI("rkt-inspect-paths.aci",
		"--exec=/inspect --write-file --print-msg=testing-insecure-option")
	defer os.Remove(imageFile)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	tests := []struct {
		Path    string
		Content string
	}{
		{
			Path:    "/proc/sysrq-trigger",
			Content: "h", // Print help message in dmesg: not dangerous
		},
	}

	for _, tt := range tests {
		// Without --insecure-options=paths
		expectErr := fmt.Sprintf("open %s: read-only file system", tt.Path)
		for _, insecureOption := range []string{"image", "image,ondisk,capabilities", "all-fetch"} {
			// run
			cmd := fmt.Sprintf(`%s --debug --insecure-options=%s run --set-env=FILE=%s --set-env=CONTENT=%s %s`,
				ctx.Cmd(), insecureOption, tt.Path, tt.Content, imageFile)
			t.Logf("run: attempting to write on %q with --insecure-options=%s (expecting error)\n", tt.Path, insecureOption)
			runRktAndCheckOutput(t, cmd, expectErr, true)
			// run-prepared
			t.Logf("run-prepared: attempting to write on %q with --insecure-options=%s (expecting error)\n", tt.Path, insecureOption)
			cmd = fmt.Sprintf(`%s --insecure-options=%s prepare --set-env=FILE=%s --set-env=CONTENT=%s %s`,
				ctx.Cmd(), insecureOption, tt.Path, tt.Content, imageFile)
			uuid := runRktAndGetUUID(t, cmd)
			cmd = fmt.Sprintf("%s --debug --insecure-options=%s run-prepared --mds-register=false %s", ctx.Cmd(), insecureOption, uuid)
			runRktAndCheckOutput(t, cmd, expectErr, true)
		}

		/* TODO(lucab): re-enable once stage1 has https://github.com/systemd/systemd/pull/4395
		 *
		 * // With --insecure-options=paths
		 * expectOk := "testing-insecure-option"
		 * for _, insecureOption := range []string{"image,paths", "image,paths,ondisk,capabilities", "image,all-run", "all"} {
		 * 	// run
		 * 	t.Logf("run: attempting to write on %q with --insecure-options=%s (expecting success)\n", tt.Path, insecureOption)
		 * 	cmd := fmt.Sprintf(`%s --debug --insecure-options=%s run --set-env=FILE=%s --set-env=CONTENT=%s %s`,
		 * 		ctx.Cmd(), insecureOption, tt.Path, tt.Content, imageFile)
		 * 	runRktAndCheckOutput(t, cmd, expectOk, false)
		 * 	// run-prepared
		 * 	t.Logf("run-prepared: attempting to write on %q with --insecure-options=%s (expecting success)\n", tt.Path, insecureOption)
		 * 	cmd = fmt.Sprintf(`%s --insecure-options=%s prepare --set-env=FILE=%s --set-env=CONTENT=%s %s`,
		 * 		ctx.Cmd(), insecureOption, tt.Path, tt.Content, imageFile)
		 * 	uuid := runRktAndGetUUID(t, cmd)
		 * 	cmd = fmt.Sprintf("%s --debug --insecure-options=%s run-prepared --mds-register=false %s", ctx.Cmd(), insecureOption, uuid)
		 * 	runRktAndCheckOutput(t, cmd, expectOk, false)
		 * }
		 */
	}
}

// TestProcPathsStat checks that access to inaccessible paths under
// /proc or /sys is correctly restricted:
// https://github.com/coreos/rkt/issues/2484
func TestProcPathsStat(t *testing.T) {
	tests := []struct {
		Path         string
		ExpectedMode string
		SkipUnderKvm bool
	}{
		{
			Path:         "/sys/firmware",
			ExpectedMode: "d---------",
			SkipUnderKvm: false,
		},
		{
			Path:         "/proc/kcore",
			ExpectedMode: "----------",
			SkipUnderKvm: true,
		},
	}

	for _, tt := range tests {
		if TestedFlavor.Kvm == true && tt.SkipUnderKvm == true {
			t.Skip("Not running test for kvm flavour")
		}

		hiddenImage := patchTestACI("rkt-inspect-stat-procfs.aci", fmt.Sprintf("--exec=/inspect --stat-file --file-name %s", tt.Path))
		defer os.Remove(hiddenImage)

		hiddenCtx := testutils.NewRktRunCtx()
		defer hiddenCtx.Cleanup()

		//run
		hiddenCmd := fmt.Sprintf("%s --debug --insecure-options=image run %s", hiddenCtx.Cmd(), hiddenImage)
		hiddenExpectedLine := fmt.Sprintf("%s: mode: %s", tt.Path, tt.ExpectedMode)
		runRktAndCheckOutput(t, hiddenCmd, hiddenExpectedLine, false)
		// run-prepared
		hiddenCmd = fmt.Sprintf(`%s --insecure-options=image prepare %s`, hiddenCtx.Cmd(), hiddenImage)
		uuid := runRktAndGetUUID(t, hiddenCmd)
		hiddenCmd = fmt.Sprintf("%s --debug run-prepared --mds-register=false %s", hiddenCtx.Cmd(), uuid)
		runRktAndCheckOutput(t, hiddenCmd, hiddenExpectedLine, false)
	}
}

// TestDevPathsStat checks that particular devices have been created as
// expected.
func TestDevPathsStat(t *testing.T) {
	tests := []struct {
		Path          string
		ExpectedMode  string
		ExpectedOwner int
		ExpectedGroup int
	}{
		{
			Path:          "/dev/ptmx",
			ExpectedMode:  "Dcrw-rw-rw-",
			ExpectedOwner: 0,
			ExpectedGroup: 0,
		},
	}

	for _, tt := range tests {
		hiddenImage := patchTestACI("rkt-inspect-stat-devices.aci", fmt.Sprintf("--exec=/inspect --stat-file --file-name %s", tt.Path))
		defer os.Remove(hiddenImage)

		hiddenCtx := testutils.NewRktRunCtx()
		defer hiddenCtx.Cleanup()

		//run
		hiddenCmd := fmt.Sprintf("%s --debug --insecure-options=image run %s", hiddenCtx.Cmd(), hiddenImage)

		lines := ""
		lines += fmt.Sprintf("%s: mode: %s.*", tt.Path, tt.ExpectedMode)
		lines += fmt.Sprintf("%s: user: %d.*", tt.Path, tt.ExpectedOwner)
		lines += fmt.Sprintf("%s: group: %d.*", tt.Path, tt.ExpectedGroup)

		// enable multi-line matching and search for all substrings
		hiddenExpectedLines := fmt.Sprintf("(?s:.*%s)", lines)

		runRktAndCheckREOutput(t, hiddenCmd, hiddenExpectedLines, false)

		// run-prepared
		hiddenCmd = fmt.Sprintf(`%s --insecure-options=image prepare %s`, hiddenCtx.Cmd(), hiddenImage)
		uuid := runRktAndGetUUID(t, hiddenCmd)
		hiddenCmd = fmt.Sprintf("%s --debug run-prepared --mds-register=false %s", hiddenCtx.Cmd(), uuid)

		runRktAndCheckREOutput(t, hiddenCmd, hiddenExpectedLines, false)
	}
}
