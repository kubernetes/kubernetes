/*
Copyright 2021 The Kubernetes Authors.

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

package test

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/pod-security-admission/api"
	"k8s.io/pod-security-admission/policy"
	"sigs.k8s.io/yaml"
)

const updateEnvVar = "UPDATE_POD_SECURITY_FIXTURE_DATA"

// TestFixtures ensures fixtures are registered for every check,
// and that in-memory fixtures match serialized fixtures in testdata.
// When adding new versions or checks, serialized fixtures can be updated by running:
//
//     UPDATE_POD_SECURITY_FIXTURE_DATA=true go test k8s.io/pod-security-admission/test
func TestFixtures(t *testing.T) {
	expectedFiles := sets.NewString("testdata/README.md")

	defaultChecks := policy.DefaultChecks()

	const newestMinorVersionToTest = 23

	policyVersions := computeVersionsToTest(t, defaultChecks)
	newestMinorVersionWithPolicyChanges := policyVersions[len(policyVersions)-1].Minor()

	if newestMinorVersionToTest < newestMinorVersionWithPolicyChanges {
		t.Fatalf("fixtures only tested up to %d, but policy changes exist up to %d", newestMinorVersionToTest, newestMinorVersionWithPolicyChanges)
	}

	for _, level := range []api.Level{api.LevelBaseline, api.LevelRestricted} {
		for version := 0; version <= newestMinorVersionToTest; version++ {
			passDir := filepath.Join("testdata", string(level), fmt.Sprintf("v1.%d", version), "pass")
			failDir := filepath.Join("testdata", string(level), fmt.Sprintf("v1.%d", version), "fail")

			// render the minimal valid pod fixture
			validPod, err := GetMinimalValidPod(level, api.MajorMinorVersion(1, version))
			if err != nil {
				t.Fatal(err)
			}
			expectedFiles.Insert(testFixtureFile(t, passDir, "base", validPod))

			// render check-specific fixtures
			checkIDs, err := checksForLevelAndVersion(defaultChecks, level, api.MajorMinorVersion(1, version))
			if err != nil {
				t.Fatal(err)
			}
			if len(checkIDs) == 0 {
				t.Fatal(fmt.Errorf("no checks registered for %s/1.%d", level, version))
			}
			for _, checkID := range checkIDs {
				checkData, err := getFixtures(fixtureKey{level: level, version: api.MajorMinorVersion(1, version), check: checkID})
				if err != nil {
					t.Fatal(err)
				}

				for i, pod := range checkData.pass {
					expectedFiles.Insert(testFixtureFile(t, passDir, fmt.Sprintf("%s%d", strings.ToLower(checkID), i), pod))
				}
				for i, pod := range checkData.fail {
					expectedFiles.Insert(testFixtureFile(t, failDir, fmt.Sprintf("%s%d", strings.ToLower(checkID), i), pod))
				}
			}
		}
	}

	actualFileList := []string{}
	err := filepath.Walk("testdata", func(path string, f os.FileInfo, err error) error {
		if !f.IsDir() {
			actualFileList = append(actualFileList, path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	actualFiles := sets.NewString(actualFileList...)
	if missingFiles := expectedFiles.Difference(actualFiles); len(missingFiles) > 0 {
		t.Errorf("unexpected missing fixtures:\n%s", strings.Join(missingFiles.List(), "\n"))
	}
	if extraFiles := actualFiles.Difference(expectedFiles); len(extraFiles) > 0 {
		t.Errorf("unexpected extra fixtures:\n%s", strings.Join(extraFiles.List(), "\n"))
		if os.Getenv(updateEnvVar) == "true" {
			for extra := range extraFiles {
				os.Remove(extra)
			}
			t.Logf("Removed extra fixture files")
			t.Logf("Verify the diff, commit changes, and rerun the tests")
		} else {
			t.Logf("If the files are expected to be removed, re-run with %s=true to drop extra fixture files", updateEnvVar)
		}
	}

}

func testFixtureFile(t *testing.T, dir, name string, pod *corev1.Pod) string {
	filename := filepath.Join(dir, name+".yaml")
	pod = pod.DeepCopy()
	pod.Name = name

	expectedYAML, _ := ioutil.ReadFile(filename)

	jsonData, err := runtime.Encode(scheme.Codecs.LegacyCodec(corev1.SchemeGroupVersion), pod)
	if err != nil {
		t.Fatal(err)
	}
	yamlData, err := yaml.JSONToYAML(jsonData)
	if err != nil {
		t.Fatal(err)
	}

	// clean up noise in fixtures
	yamlData = []byte(strings.ReplaceAll(string(yamlData), "  creationTimestamp: null\n", ""))
	yamlData = []byte(strings.ReplaceAll(string(yamlData), "    resources: {}\n", ""))
	yamlData = []byte(strings.ReplaceAll(string(yamlData), "status: {}\n", ""))

	if string(yamlData) != string(expectedYAML) {
		t.Errorf("fixture data does not match the test fixture in %s", filename)

		if os.Getenv(updateEnvVar) == "true" {
			if err := os.MkdirAll(dir, os.FileMode(0755)); err != nil {
				t.Fatal(err)
			}
			if err := ioutil.WriteFile(filename, []byte(yamlData), os.FileMode(0755)); err == nil {
				t.Logf("Updated data in %s", filename)
				t.Logf("Verify the diff, commit changes, and rerun the tests")
			} else {
				t.Logf("Could not update data in %s: %v", filename, err)
			}
		} else {
			t.Logf("Diff between generated data and fixture data in %s:\n-------------\n%s", filename, diff.StringDiff(string(yamlData), string(expectedYAML)))
			t.Logf("If the change is expected, re-run with %s=true to update the fixtures", updateEnvVar)
		}
	}
	return filename
}
