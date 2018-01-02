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

func TestRktEnsureErrors(t *testing.T) {
	const imgName = "rkt-ensure-errors-test"

	image := patchTestACI(fmt.Sprintf("%s.aci", imgName), fmt.Sprintf("--name=%s", imgName))
	defer os.Remove(image)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	runCmd := fmt.Sprintf("%s run --insecure-options=image --net=notavalidnetwork %s", ctx.Cmd(), image)
	t.Logf("Running test: %s", runCmd)
	runRktAndCheckRegexOutput(t, runCmd, "stage1: failed to setup network")
}
