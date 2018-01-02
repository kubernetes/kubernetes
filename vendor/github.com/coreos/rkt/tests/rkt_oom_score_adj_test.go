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

func TestOOMScoreAdjust(t *testing.T) {
	for _, tt := range []struct {
		patch    []string
		appArgs  string
		expected string
	}{
		{
			patch:    []string{"--isolators=os/linux/oom-score-adj,99"},
			expected: "<<<99",
		},
		{
			patch:    []string{"--isolators=os/linux/oom-score-adj,-50"},
			expected: "<<<-50",
		},
		{
			patch:    nil,
			expected: "<<<0",
		},
		{
			patch:    nil,
			appArgs:  "--oom-score-adj 42",
			expected: "<<<42",
		},
	} {
		func() {
			ctx := testutils.NewRktRunCtx()
			defer ctx.Cleanup()

			ps := []string{}
			if len(tt.patch) > 0 {
				ps = append(ps, tt.patch...)
			}

			image := patchTestACI("rkt-oom-adj.aci", ps...)
			defer os.Remove(image)

			imageParams := "--exec=/inspect -- -read-file -file-name /proc/self/oom_score_adj"

			rktCmd := fmt.Sprintf(
				"%s --debug --insecure-options=image run %s %s %s",
				ctx.Cmd(),
				image,
				tt.appArgs,
				imageParams,
			)

			runRktAndCheckOutput(t, rktCmd, tt.expected, false)
		}()
	}
}
