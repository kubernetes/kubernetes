/*
Copyright 2025 The Kubernetes Authors.

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

package invariants

import (
	"fmt"
	"slices"

	"github.com/onsi/ginkgo/v2"
	ginkgotypes "github.com/onsi/ginkgo/v2/types"
)

// Please speak to SIG-Testing leads before adding anything to this file.

// invariantsSelected returns true if the invariant check should be done
// because the corresponding test (identified by context and invariantsLeafText)
// ran.
//
// Note that this does not check whether the test is merely meant to run (when
// called before the test run after an internal dry-run) or really ran (when
// called after a test run). If the caller cares about that difference, it
// has to check report.SuiteConfig.DryRun itself.
func invariantsSelected(report ginkgo.Report, sig, invariantsContextText, invariantsLeafText string) bool {
	// Check if we ran the dummy test.
	// The actual hierarchy text includes the SIG.
	hierarchyText := fmt.Sprintf("[sig-%s] %s", sig, invariantsContextText)
	for _, spec := range report.SpecReports {
		if spec.LeafNodeText == invariantsLeafText &&
			slices.Index(spec.ContainerHierarchyTexts, hierarchyText) >= 0 {
			return spec.State.Is(ginkgotypes.SpecStatePassed)
		}
	}

	return false
}
