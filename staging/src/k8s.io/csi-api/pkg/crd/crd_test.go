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

// These tests verify the manifest files in this package and the
// addons directory are in sync.
package crd_test

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"k8s.io/apimachinery/pkg/util/diff"
)

func TestBootstrapCRDs(t *testing.T) {
	verifyCopiesAreInSync(
		t,
		"csidriver.yaml", /* filename */
		"manifests",      /* sourceOfTruthDir */
		[]string{"../../../../../../cluster/addons/storage-crds"}, /* copyDirs */
	)
	verifyCopiesAreInSync(
		t,
		"csinodeinfo.yaml", /* filename */
		"manifests",        /* sourceOfTruthDir */
		[]string{"../../../../../../cluster/addons/storage-crds"}, /* copyDirs */
	)
}

// verifyCopiesAreInSync fails if any copies are different from source of truth.
func verifyCopiesAreInSync(t *testing.T, filename string, sourceOfTruthDir string, copyDirs []string) {
	sourceOfTruthFilename := filepath.Join(sourceOfTruthDir, filename)

	if len(copyDirs) <= 0 {
		t.Fatalf("copyDirs is empty. There are no copies to validate.")
	}

	expectedYAML, err := ioutil.ReadFile(sourceOfTruthFilename)
	if err != nil {
		t.Fatal(err)
	}

	for _, copyDir := range copyDirs {
		copyFilename := filepath.Join(copyDir, filename)
		actualYAML, err := ioutil.ReadFile(copyFilename)
		if err != nil {
			t.Fatal(err)
		}

		if string(actualYAML) != string(expectedYAML) {
			t.Errorf("Data in %q does not match source of truth in %q.", copyFilename, sourceOfTruthFilename)

			const updateEnvVar = "UPDATE_CSI_CRD_FIXTURE_DATA"
			if os.Getenv(updateEnvVar) == "true" {
				if err := ioutil.WriteFile(copyFilename, []byte(expectedYAML), os.FileMode(0755)); err == nil {
					t.Logf("Updated data in %s", copyFilename)
					t.Logf("Verify the diff, commit changes, and rerun the tests")
				} else {
					t.Logf("Could not update data in %s: %v", copyFilename, err)
				}
			} else {
				t.Logf("Diff between source of truth data and copy data in %s:\n-------------\n%s", copyFilename, diff.StringDiff(string(actualYAML), string(expectedYAML)))
				t.Logf("If the change is expected, re-run with %s=true to update the copy data", updateEnvVar)
			}
		}
	}
}
