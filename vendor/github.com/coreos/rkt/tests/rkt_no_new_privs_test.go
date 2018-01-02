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

	"github.com/coreos/rkt/tests/testutils"
)

func TestNoNewPrivileges(t *testing.T) {
	for _, tt := range []struct {
		rktParams string
		patch     []string
		expected  string
	}{
		{
			patch:    []string{"--isolators=os/linux/no-new-privileges,true"},
			expected: "no_new_privs: 1 err: errno 0",
		},
		{
			rktParams: "--user=1000 --group=100",
			patch:     []string{"--isolators=os/linux/no-new-privileges,true"},
			expected:  "no_new_privs: 1 err: errno 0",
		},
		{
			patch:    []string{"--isolators=os/linux/no-new-privileges,false"},
			expected: "no_new_privs: 0 err: errno 0",
		},
		{
			rktParams: "--user=1000 --group=100",
			patch:     []string{"--isolators=os/linux/no-new-privileges,false", "--seccomp-mode=retain", "--seccomp-set=@appc.io/all"},
			expected:  "no_new_privs: 0 err: errno 0",
		},
		{
			patch:    []string{`--isolators=os/linux/no-new-privileges,false:os/linux/no-new-privileges,true`},
			expected: "no_new_privs: 1 err: errno 0",
		},
		{
			rktParams: "--user=1000 --group=100",
			patch:     []string{`--isolators=os/linux/no-new-privileges,false:os/linux/no-new-privileges,true`},
			expected:  "no_new_privs: 1 err: errno 0",
		},
		{
			patch:    nil,
			expected: "no_new_privs: 0 err: errno 0",
		},
	} {
		func() {
			ctx := testutils.NewRktRunCtx()
			defer ctx.Cleanup()

			ps := []string{}
			if len(tt.patch) > 0 {
				ps = append(ps, tt.patch...)
			}

			image := patchTestACI("rkt-no-new-privs.aci", ps...)
			defer os.Remove(image)

			rktParams := fmt.Sprintf(
				"%s --exec=/inspect -- -print-no-new-privs",
				tt.rktParams,
			)

			rktCmd := fmt.Sprintf(
				"%s --debug --insecure-options=image,paths run %s %s",
				ctx.Cmd(),
				image,
				rktParams,
			)

			runRktAndCheckOutput(t, rktCmd, tt.expected, false)
		}()
	}
}
