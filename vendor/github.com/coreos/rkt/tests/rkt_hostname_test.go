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

// +build host coreos src kvm

package main

import (
	"fmt"
	"os"
	"testing"

	"github.com/coreos/rkt/tests/testutils"
)

// TestHostname test that the --hostname option works.
func TestHostname(t *testing.T) {
	imageFile := patchTestACI("rkt-inspect-hostname.aci", "--exec=/inspect --print-hostname")
	defer os.Remove(imageFile)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	tests := []struct {
		hostname string
	}{
		{
			"test-hostname",
		},
		{
			"",
		},
	}

	for _, tt := range tests {
		rktCmd := fmt.Sprintf("%s prepare --insecure-options=image %s", ctx.Cmd(), imageFile)
		uuid := runRktAndGetUUID(t, rktCmd)

		expectedHostname := "rkt-" + uuid
		hostnameParam := ""
		if tt.hostname != "" {
			expectedHostname = tt.hostname
			hostnameParam = fmt.Sprintf("--hostname=%s", tt.hostname)
		}

		rktCmd = fmt.Sprintf("%s run-prepared %s %s", ctx.Cmd(), hostnameParam, uuid)
		expected := fmt.Sprintf("Hostname: %s", expectedHostname)
		runRktAndCheckOutput(t, rktCmd, expected, false)
	}
}
