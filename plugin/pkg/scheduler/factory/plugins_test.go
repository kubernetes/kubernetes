/*
Copyright 2015 The Kubernetes Authors.

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

package factory

import "testing"

func TestAlgorithmNameValidation(t *testing.T) {
	algorithmNamesShouldValidate := []string{
		"1SomeAlgo1rithm",
		"someAlgor-ithm1",
	}
	algorithmNamesShouldNotValidate := []string{
		"-SomeAlgorithm",
		"SomeAlgorithm-",
		"Some,Alg:orithm",
	}
	for _, name := range algorithmNamesShouldValidate {
		if !validName.MatchString(name) {
			t.Errorf("%v should be a valid algorithm name but is not valid.", name)
		}
	}
	for _, name := range algorithmNamesShouldNotValidate {
		if validName.MatchString(name) {
			t.Errorf("%v should be an invalid algorithm name but is valid.", name)
		}
	}
}
