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

// Area is a conformance area composed of a list of test suites
type Area struct {
	Area   string  `json:"area,omitempty"`
	Suites []Suite `json:"suites,omitempty"`
}

// Suite is a conformance test suite composed of a list of behaviors
type Suite struct {
	Suite       string     `json:"suite,omitempty"`
	Description string     `json:"description,omitempty"`
	Behaviors   []Behavior `json:"behaviors,omitempty"`
}

// Behavior describes the set of properties for a conformance behavior
type Behavior struct {
	ID          string `json:"id,omitempty"`
	APIObject   string `json:"apiObject,omitempty"`
	APIField    string `json:"apiField,omitempty"`
	APIType     string `json:"apiType,omitempty"`
	Description string `json:"description,omitempty"`
}

// ConformanceData describes the structure of the conformance.yaml file
type ConformanceData struct {
	// A URL to the line of code in the kube src repo for the test. Omitted from the YAML to avoid exposing line number.
	URL string `yaml:"-"`
	// Extracted from the "Testname:" comment before the test
	TestName string
	// CodeName is taken from the actual ginkgo descriptions, e.g. `[sig-apps] Foo should bar [Conformance]`
	CodeName string
	// Extracted from the "Description:" comment before the test
	Description string
	// Version when this test is added or modified ex: v1.12, v1.13
	Release string
	// File is the filename where the test is defined. We intentionally don't save the line here to avoid meaningless changes.
	File string
	// Behaviors is the list of conformance behaviors tested by a particular e2e test
	Behaviors []string `yaml:"behaviors,omitempty"`
}
