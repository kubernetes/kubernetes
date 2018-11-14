// +build !windows

/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package util

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Can't just call Chroot() because it will affect other tests in the
// same process.  Golang makes it hard to just call fork(), so instead
// we exec ourselves again, and run the test in the new subprocess.

func testChrootReal(t *testing.T) {
	testfile := filepath.FromSlash("/" + filepath.Base(os.Args[0]))
	dir := filepath.Dir(os.Args[0])
	if dir == "." {
		t.Skip("skipping: running test at root somehow")
	}
	if err := Chroot(dir); err != nil {
		if strings.Contains(err.Error(), "operation not permitted") {
			t.Skip("skipping: insufficient permissions to chroot")
		}
		t.Fatalf("chroot error: %v", err)
	}

	// All file access should now be relative to `dir`
	if _, err := os.Stat(testfile); err != nil {
		t.Errorf("Expected file %q to exist, but got %v", testfile, err)
	}
}

func TestChroot(t *testing.T) {
	if os.Getenv("GO_TEST_CHROOT_FOR_REALZ") == "1" {
		testChrootReal(t)
		return
	}

	cmd := exec.Command(os.Args[0], "-test.v", "-test.run=TestChroot")
	cmd.Env = []string{"GO_TEST_CHROOT_FOR_REALZ=1"}

	out, err := cmd.Output()
	t.Logf("subprocess output:\n%s", out)
	if err != nil {
		t.Errorf("subprocess error: %v", err)
	}
}
