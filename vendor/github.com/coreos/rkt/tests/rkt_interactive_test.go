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
	"os"
	"strings"
	"testing"
	"time"

	"github.com/coreos/rkt/tests/testutils"
)

var interactiveTests = []struct {
	testName     string
	aciBuildArgs []string
	rktArgs      string
	say          string
	expect       string
}{
	{
		`Check tty without interactive`,
		[]string{"--exec=/inspect --check-tty"},
		`--debug --insecure-options=image run --mds-register=false ^INTERACTIVE^`,
		``,
		`stdin is not a terminal`,
	},
	{
		`Check tty without interactive (with parameter)`,
		[]string{"--exec=/inspect"},
		`--debug --insecure-options=image run --mds-register=false ^INTERACTIVE^ -- --check-tty`,
		``,
		`stdin is not a terminal`,
	},
	{
		`Check tty with interactive`,
		[]string{"--exec=/inspect --check-tty"},
		`--debug --insecure-options=image run --mds-register=false --interactive ^INTERACTIVE^`,
		``,
		`stdin is a terminal`,
	},
	{
		`Check tty with interactive (with parameter)`,
		[]string{"--exec=/inspect"},
		`--debug --insecure-options=image run --mds-register=false --interactive ^INTERACTIVE^ -- --check-tty`,
		``,
		`stdin is a terminal`,
	},
	{
		`Reading from stdin`,
		[]string{"--exec=/inspect --read-stdin"},
		`--debug --insecure-options=image run --mds-register=false --interactive ^INTERACTIVE^`,
		`Saluton`,
		`Received text: Saluton`,
	},
	{
		`Reading from stdin (with parameter)`,
		[]string{"--exec=/inspect"},
		`--debug --insecure-options=image run --mds-register=false --interactive ^INTERACTIVE^ -- --read-stdin`,
		`Saluton`,
		`Received text: Saluton`,
	},
}

func TestInteractive(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	for i, tt := range interactiveTests {
		t.Logf("Running test #%v: %v", i, tt.testName)

		aciFileName := patchTestACI("rkt-inspect-interactive.aci", tt.aciBuildArgs...)
		defer os.Remove(aciFileName)

		rktCmd := fmt.Sprintf("%s %s", ctx.Cmd(), tt.rktArgs)
		rktCmd = strings.Replace(rktCmd, "^INTERACTIVE^", aciFileName, -1)
		child := spawnOrFail(t, rktCmd)
		if tt.say != "" {
			if err := expectTimeoutWithOutput(child, "Enter text:", time.Minute); err != nil {
				t.Fatalf("Waited for the prompt but not found #%v: %v", i, err)
			}

			if err := child.SendLine(tt.say); err != nil {
				t.Fatalf("Failed to send %q on the prompt #%v: %v", tt.say, i, err)
			}
		}

		if err := expectTimeoutWithOutput(child, tt.expect, time.Minute); err != nil {
			t.Fatalf("Expected %q but not found #%v: %v", tt.expect, i, err)
		}

		if err := child.Wait(); err != nil {
			t.Fatalf("rkt didn't terminate correctly: %v", err)
		}
	}
}
