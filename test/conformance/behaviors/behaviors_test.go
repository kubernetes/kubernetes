/*
Copyright 2020 The Kubernetes Authors.

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

package behaviors

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"testing"

	"gopkg.in/yaml.v2"
)

func TestValidate(t *testing.T) {
	var behaviorFiles []string

	err := filepath.Walk(".",
		func(path string, info os.FileInfo, err error) error {
			if err != nil {
				t.Errorf("%q", err.Error())
			}

			r, _ := regexp.Compile(".+.yaml$")
			if r.MatchString(path) {
				behaviorFiles = append(behaviorFiles, path)
			}
			return nil
		})
	if err != nil {
		t.Errorf("%q", err.Error())
	}

	for _, file := range behaviorFiles {
		validateSuite(file, t)
	}
}

func validateSuite(path string, t *testing.T) {
	var suite Suite
	yamlFile, err := ioutil.ReadFile(path)
	if err != nil {
		t.Errorf("%q", err.Error())
	}
	err = yaml.Unmarshal(yamlFile, &suite)

	if err != nil {
		t.Errorf("%q", err.Error())
	}

	behaviorIDList := make(map[string]bool)

	for _, behavior := range suite.Behaviors {

		// Ensure no behavior IDs are duplicated
		if _, ok := behaviorIDList[behavior.ID]; ok {
			t.Errorf("Duplicate behavior ID: %s", behavior.ID)
		}
		behaviorIDList[behavior.ID] = true
	}
}
