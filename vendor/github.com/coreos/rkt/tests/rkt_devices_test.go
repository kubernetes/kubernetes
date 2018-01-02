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

// +build host coreos src

package main

import (
	"fmt"
	"testing"

	"github.com/coreos/rkt/tests/testutils"
)

func TestDevices(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	image := getInspectImagePath()

	for _, tt := range []struct {
		rktArgs        string
		rktAppArgs     string
		execArgs       string
		expectedOutput string
		expectErr      bool
	}{
		/* There should be no restriction on /dev/null */
		{
			rktArgs:        "--insecure-options=image",
			rktAppArgs:     "",
			execArgs:       "--check-mknod=c:1:3:/null",
			expectedOutput: "mknod /null: succeed",
			expectErr:      false,
		},

		/* Test the old ptmx device node, before devpts filesystem
		 * existed. It should be blocked. Containers should use the new
		 * ptmx from devpts instead. It is created with
		 * "mknod name c 5 2" according to:
		 * https://github.com/torvalds/linux/blob/master/Documentation/filesystems/devpts.txt
		 */
		{
			rktArgs:        "--insecure-options=image",
			rktAppArgs:     "",
			execArgs:       "--check-mknod=c:5:2:/ptmx",
			expectedOutput: "mknod /ptmx: fail",
			expectErr:      true,
		},

		/* /dev/loop-control has major:minor 10:237 according to:
		 * https://github.com/torvalds/linux/blob/master/Documentation/devices.txt#L424
		 */
		{
			rktArgs:        "--insecure-options=image",
			rktAppArgs:     "",
			execArgs:       "--check-mknod=c:10:237:/loop-control",
			expectedOutput: "mknod /loop-control: fail",
			expectErr:      true,
		},

		/* We should be able to create /dev/loop-control with the paths
		 * insecure option.
		 */
		{
			rktArgs:        "--insecure-options=image,paths",
			rktAppArgs:     "",
			execArgs:       "--check-mknod=c:10:237:/loop-control",
			expectedOutput: "mknod /loop-control: succeed",
			expectErr:      false,
		},

		/* Test mounting /dev/loop-control. We should be able to access it
		 * without the paths insecure option. It should return "invalid
		 * argument" instead of "operation not permitted".
		 */
		{
			rktArgs:        "--insecure-options=image --volume loopcontrol,kind=host,source=/dev/loop-control --set-env=FILE=/tmp/loop-control",
			rktAppArgs:     "--mount volume=loopcontrol,target=/tmp/loop-control",
			execArgs:       "--read-file",
			expectedOutput: `Cannot read file "/tmp/loop-control": read /tmp/loop-control: invalid argument`,
			expectErr:      true,
		},

		/* Test mounting /dev/loop-control. We should be able to create
		 * other devices with mknod with the paths insecure option.
		 * Let's try the old ptmx device again.
		 */
		{
			rktArgs:        "--insecure-options=image,paths --volume loopcontrol,kind=host,source=/dev/loop-control --set-env=FILE=/tmp/loop-control",
			rktAppArgs:     "--mount volume=loopcontrol,target=/tmp/loop-control",
			execArgs:       "--check-mknod=c:5:2:/ptmx",
			expectedOutput: "mknod /ptmx: succeed",
			expectErr:      false,
		},
	} {
		rktCmd := fmt.Sprintf(
			"%s --debug run %s %s %s --exec=/inspect -- %s",
			ctx.Cmd(), tt.rktArgs, image, tt.rktAppArgs, tt.execArgs)
		t.Logf("Running %s", rktCmd)

		runRktAndCheckOutput(t, rktCmd, tt.expectedOutput, tt.expectErr)
	}
}
