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

// +build fly

package main

import (
	"fmt"
	"os"
	"testing"

	"github.com/coreos/rkt/tests/testutils"
)

func TestFlyNetns(t *testing.T) {
	testImageArgs := []string{"--exec=/inspect --print-netns"}
	testImage := patchTestACI("rkt-inspect-stage1-fly.aci", testImageArgs...)
	defer os.Remove(testImage)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	cmd := fmt.Sprintf("%s --debug --insecure-options=image run %s", ctx.Cmd(), testImage)
	child := spawnOrFail(t, cmd)
	ctx.RegisterChild(child)
	defer waitOrFail(t, child, 0)

	expectedRegex := `NetNS: (net:\[\d+\])`
	result, out, err := expectRegexWithOutput(child, expectedRegex)
	if err != nil {
		t.Fatalf("Error: %v\nOutput: %v", err, out)
	}

	ns, err := os.Readlink("/proc/self/ns/net")
	if err != nil {
		t.Fatalf("Cannot evaluate NetNS symlink: %v", err)
	}

	if nsChanged := ns != result[1]; nsChanged {
		t.Fatalf("container left host netns")
	}
}
