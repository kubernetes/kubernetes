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
	"testing"
)

func TestValidate(t *testing.T) {
	behaviorFiles, err := BehaviorFileList(".")
	if err != nil {
		t.Errorf("%q", err.Error())
	}

	for _, file := range behaviorFiles {
		validateSuite(file, t)
	}
}

func validateSuite(path string, t *testing.T) {
	suite, err := LoadSuite(path)
	if err != nil {
		t.Errorf("%q", err.Error())
	}
	err = ValidateSuite(suite)
	if err != nil {
		t.Errorf("error validating %s: %q", path, err.Error())
	}
}
