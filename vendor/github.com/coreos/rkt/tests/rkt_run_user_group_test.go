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
	"os"
	"testing"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/tests/testutils"
)

func TestAppUserGroup(t *testing.T) {
	imageDummy := patchTestACI("rkt-inspect-dummy.aci", "--name=dummy")
	defer os.Remove(imageDummy)

	supportsUserNS := common.SupportsUserNS() && checkUserNS() == nil

	for _, tt := range []struct {
		imageParams []string
		rktParams   string
		expected    string
	}{
		{
			expected: "User: uid=0 euid=0 gid=0 egid=0",
		},
		{
			rktParams: "--user=200",
			expected:  "User: uid=200 euid=200 gid=0 egid=0",
		},
		{
			rktParams: "--group=300",
			expected:  "User: uid=0 euid=0 gid=300 egid=300",
		},
		{
			rktParams: "--user=200 --group=300",
			expected:  "User: uid=200 euid=200 gid=300 egid=300",
		},
		{
			rktParams: "--user=user1 --group=300",
			expected:  "User: uid=1000 euid=1000 gid=300 egid=300",
		},
		{
			rktParams: "--user=200 --group=group1",
			expected:  "User: uid=200 euid=200 gid=100 egid=100",
		},
		{
			imageParams: []string{"--user=400", "--group=500"},
			expected:    "User: uid=400 euid=400 gid=500 egid=500",
		},
		{
			imageParams: []string{"--user=400", "--group=500"},
			rktParams:   "--user=200",
			expected:    "User: uid=200 euid=200 gid=500 egid=500",
		},
		{
			imageParams: []string{"--user=400", "--group=500"},
			rktParams:   "--group=300",
			expected:    "User: uid=400 euid=400 gid=300 egid=300",
		},
		{
			imageParams: []string{"--user=400", "--group=500"},
			rktParams:   "--user=200 --group=300",
			expected:    "User: uid=200 euid=200 gid=300 egid=300",
		},
		{
			imageParams: []string{"--user=400", "--group=500"},
			rktParams:   "--user=user1 --group=group1",
			expected:    "User: uid=1000 euid=1000 gid=100 egid=100",
		},
		{
			imageParams: []string{"--user=root", "--group=root"},
			expected:    "User: uid=0 euid=0 gid=0 egid=0",
		},
		{
			rktParams: "--user=root --group=root",
			expected:  "User: uid=0 euid=0 gid=0 egid=0",
		},
		{
			rktParams: "--user=/inspect --group=/inspect",
			expected:  "User: uid=0 euid=0 gid=0 egid=0",
		},
	} {
		func() {
			ctx := testutils.NewRktRunCtx()
			defer ctx.Cleanup()

			tt.imageParams = append(tt.imageParams, "--exec=/inspect --print-user")
			image := patchTestACI("rkt-inspect-user-group.aci", tt.imageParams...)
			defer os.Remove(image)

			userNSOpts := []string{""}
			if supportsUserNS {
				userNSOpts = append(userNSOpts, "--private-users --no-overlay")
			}

			for _, userNSOpt := range userNSOpts {
				// run the user/group overridden app first
				rktCmd := fmt.Sprintf(
					"%s --debug --insecure-options=image run %s %s %s %s",
					ctx.Cmd(),
					userNSOpt,
					image, tt.rktParams,
					imageDummy,
				)
				runRktAndCheckRegexOutput(t, rktCmd, tt.expected)

				// run the user/group overridden app last
				rktCmd = fmt.Sprintf(
					"%s --debug --insecure-options=image run %s %s %s %s",
					ctx.Cmd(),
					userNSOpt,
					imageDummy,
					image, tt.rktParams,
				)
				runRktAndCheckRegexOutput(t, rktCmd, tt.expected)
			}
		}()
	}
}
