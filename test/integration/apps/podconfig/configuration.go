/*
Copyright 2017 The Kubernetes Authors.

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

package podconfig

import (
	"testing"
)

//Name of the package
var Name = "Configurations"

//Tests List of internal tests.
var Tests = []testing.InternalTest{
	{Name: "Affinity", F: testAffinity},
	{Name: "Disruption", F: testDisruption},
	{Name: "Services", F: testServices},
}

// RunTests Starting point of tests in this package
func RunTests(t *testing.T) {

	setup(t)
	defer teardown(t)

	//Run one test case after other
	for _, tst := range Tests {
		t.Run(tst.Name, tst.F)
	}

	//Tear-down

}

func setup(t *testing.T) {
	return
}

func testAffinity(t *testing.T) {
	//Tests Related to Inter-Affinity and Node-Affinity
	t.SkipNow()
}

func testServices(t *testing.T) {
	//Tests Related to Basic Services
	t.SkipNow()
}

func testDisruption(t *testing.T) {
	//Tests Related to Pod Disruption budget
	t.SkipNow()
}

func teardown(t *testing.T) {
	return
}
