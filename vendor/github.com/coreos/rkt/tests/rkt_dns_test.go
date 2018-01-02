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
	"crypto/sha1"
	"fmt"
	"io/ioutil"
	"os"
	"testing"
	"time"

	"github.com/coreos/rkt/tests/testutils"
)

// TestDNS is checking how rkt fills /etc/resolv.conf
func TestDNSParam(t *testing.T) {
	imageFile := patchTestACI("rkt-inspect-exit.aci", "--exec=/inspect --print-msg=Hello --read-file")
	defer os.Remove(imageFile)
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	for i, tt := range []struct {
		paramDNS      string
		expectedLine  string
		expectedError bool
	}{
		{
			paramDNS:      "",
			expectedLine:  "Cannot read file",
			expectedError: TestedFlavor.ExitStatusPreserved,
		},
		{
			paramDNS:      "--dns=8.8.4.4",
			expectedLine:  "nameserver 8.8.4.4",
			expectedError: false,
		},
		{
			paramDNS:      "--dns=8.8.8.8 --dns=8.8.4.4",
			expectedLine:  "nameserver 8.8.8.8",
			expectedError: false,
		},
		{
			paramDNS:      "--dns=8.8.8.8 --dns=8.8.4.4 --dns-search=search.com --dns-opt=debug",
			expectedLine:  "nameserver 8.8.4.4",
			expectedError: false,
		},
		{
			paramDNS:      "--dns-search=foo.com --dns-search=bar.com",
			expectedLine:  "search foo.com bar.com",
			expectedError: false,
		},
		{
			paramDNS:      "--dns-opt=debug --dns-opt=use-vc --dns-opt=rotate",
			expectedLine:  "options debug use-vc rotate",
			expectedError: false,
		},
		{
			paramDNS:      "--dns-opt=debug --dns-opt=use-vc --dns-opt=rotate --dns-domain=example.net",
			expectedLine:  "domain example.net",
			expectedError: false,
		},
	} {

		rktCmd := fmt.Sprintf(`%s --insecure-options=image run --set-env=FILE=/etc/resolv.conf %s %s`,
			ctx.Cmd(), tt.paramDNS, imageFile)
		_ = i
		// t.Logf("%d: %s\n", i, rktCmd)
		runRktAndCheckOutput(t, rktCmd, tt.expectedLine, tt.expectedError)
	}
}

// TestHostDNS checks that --dns=host reflects the host's /etc/resolv.conf
func TestDNSHost(t *testing.T) {
	dat, err := ioutil.ReadFile("/etc/resolv.conf")
	if err != nil {
		t.Fatal("Could not read host's resolv.conf", err)
	}

	sum := fmt.Sprintf("%x", sha1.Sum(dat))
	t.Log("Expecting sum", sum)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	appCmd := "--exec=/inspect -- --hash-file"
	rktCmd := fmt.Sprintf("%s --insecure-options=image run --dns=host --set-env=FILE=/etc/resolv.conf %s %s",
		ctx.Cmd(), getInspectImagePath(), appCmd)

	child := spawnOrFail(t, rktCmd)
	ctx.RegisterChild(child)
	defer waitOrFail(t, child, 0)

	expectedRegex := `sha1sum: ([0-9a-f]+)`
	result, out, err := expectRegexTimeoutWithOutput(child, expectedRegex, 30*time.Second)
	if err != nil {
		t.Fatalf("Error: %v\nOutput: %v", err, out)
	}

	if result[1] != sum {
		t.Fatalf("container's /etc/host has sha1sum %s expected %s", result[1], sum)
	}
}
