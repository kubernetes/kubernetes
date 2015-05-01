/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"

	. "github.com/onsi/ginkgo"
)

var (
	root = absOrDie(filepath.Clean(filepath.Join(path.Base(os.Args[0]), "..")))
)

var _ = Describe("Shell", func() {

	defer GinkgoRecover()
	// Slurp up all the tests in hack/e2e-suite
	bashE2ERoot := filepath.Join(root, "hack/e2e-suite")
	files, err := ioutil.ReadDir(bashE2ERoot)
	if err != nil {
		Fail(fmt.Sprintf("Error reading test suites from %v %v", bashE2ERoot, err.Error()))
	}

	for _, file := range files {
		fileName := file.Name() // Make a copy
		It(fmt.Sprintf("tests that %v passes", fileName), func() {
			// A number of scripts only work on gce
			if !providerIs("gce", "gke") {
				By(fmt.Sprintf("Skipping Shell test %s, which is only supported for provider gce and gke (not %s)",
					fileName, testContext.Provider))
				return
			}
			runCmdTest(filepath.Join(bashE2ERoot, fileName))
		})
	}
})

func absOrDie(path string) string {
	out, err := filepath.Abs(path)
	if err != nil {
		panic(err)
	}
	return out
}

// Runs the given cmd test.
func runCmdTest(path string) {
	By(fmt.Sprintf("Running %v", path))
	cmd := exec.Command(path)
	cmd.Stdout = bytes.NewBuffer(nil)
	cmd.Stderr = cmd.Stdout

	if err := cmd.Run(); err != nil {
		Fail(fmt.Sprintf("Error running %v:\nCommand output:\n%v\n", cmd, cmd.Stdout))
		return
	}
	return
}
