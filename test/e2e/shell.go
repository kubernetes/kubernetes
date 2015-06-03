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
	"os/exec"
	"path/filepath"

	. "github.com/onsi/ginkgo"
)

var _ = Describe("Shell", func() {
	defer GinkgoRecover()

	It(fmt.Sprintf("should pass tests for services.sh"), func() {
		// The services script only works on gce/gke
		if !providerIs("gce", "gke") {
			By(fmt.Sprintf("Skipping Shell test services.sh, which is only supported for provider gce and gke (not %s)",
				testContext.Provider))
			return
		}
		runCmdTest(filepath.Join(testContext.RepoRoot, "hack/e2e-suite/services.sh"))
	})
})

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
