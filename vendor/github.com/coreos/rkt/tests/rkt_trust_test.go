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
	"testing"

	"github.com/coreos/rkt/tests/testutils"
)

func runImage(t *testing.T, ctx *testutils.RktRunCtx, imageFile string, expected string, shouldFail bool) {
	cmd := fmt.Sprintf(`%s --debug run --mds-register=false %s`, ctx.Cmd(), imageFile)
	runRktAndCheckOutput(t, cmd, expected, shouldFail)
}

func TestTrust(t *testing.T) {
	imageFile := patchTestACI("rkt-inspect-trust1.aci", "--exec=/inspect --print-msg=Hello", "--name=rkt-prefix.com/my-app")
	defer os.Remove(imageFile)

	imageFile2 := patchTestACI("rkt-inspect-trust2.aci", "--exec=/inspect --print-msg=Hello", "--name=rkt-alternative.com/my-app")
	defer os.Remove(imageFile2)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	t.Logf("Run the non-signed image: it should fail\n")
	runImage(t, ctx, imageFile, "error opening signature file", true)

	t.Logf("Sign the images\n")
	ascFile := runSignImage(t, imageFile, 1)
	defer os.Remove(ascFile)
	ascFile = runSignImage(t, imageFile2, 1)
	defer os.Remove(ascFile)

	t.Logf("Run the signed image without trusting the key: it should fail\n")
	runImage(t, ctx, imageFile, "openpgp: signature made by unknown entity", true)

	t.Logf("Trust the key with the wrong prefix\n")
	runRktTrust(t, ctx, "wrong-prefix.com/my-app", 1)

	t.Logf("Run a signed image with the key installed in the wrong prefix: it should fail\n")
	runImage(t, ctx, imageFile, "openpgp: signature made by unknown entity", true)

	t.Logf("Trust the key with the correct prefix, but wrong key\n")
	runRktTrust(t, ctx, "rkt-prefix.com/my-app", 2)

	t.Logf("Run a signed image with the wrong key installed: it should fail\n")
	runImage(t, ctx, imageFile, "openpgp: signature made by unknown entity", true)

	t.Logf("Trust the key with the correct prefix\n")
	runRktTrust(t, ctx, "rkt-prefix.com/my-app", 1)

	t.Logf("Finally, run successfully the signed image\n")
	runImage(t, ctx, imageFile, "Hello", false)
	runImage(t, ctx, imageFile2, "openpgp: signature made by unknown entity", true)

	t.Logf("Trust the key on unrelated prefixes\n")
	runRktTrust(t, ctx, "foo.com", 1)
	runRktTrust(t, ctx, "example.com/my-app", 1)

	t.Logf("But still only the first image can be executed\n")
	runImage(t, ctx, imageFile, "Hello", false)
	runImage(t, ctx, imageFile2, "openpgp: signature made by unknown entity", true)

	t.Logf("Trust the key for all images (rkt trust --root)\n")
	runRktTrust(t, ctx, "", 1)

	t.Logf("Now both images can be executed\n")
	runImage(t, ctx, imageFile, "Hello", false)
	runImage(t, ctx, imageFile2, "Hello", false)
}
