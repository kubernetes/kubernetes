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

package podcontrollers

import (
	"testing"
)

//Name of the package
var Name = "PodControllers"

//Tests List of internal tests.
var Tests = []testing.InternalTest{
	{Name: "Jobs", F: testJob},
	{Name: "ReplicaSet", F: testReplicaSet},
	{Name: "Statefulsets", F: testStatefulsets},
	{Name: "Daemonsets", F: testDaemonsets},
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

func testJob(t *testing.T) {
	//Tests Related to jobs
	t.SkipNow()
}

func testReplicaSet(t *testing.T) {
	//Tests Related to ReplicaSet
	t.SkipNow()
}

func testStatefulsets(t *testing.T) {
	//Tests Related to Statefuslet
	t.SkipNow()
}

func testDaemonsets(t *testing.T) {
	//Tests Related to Daemonset
	t.SkipNow()
}

func teardown(t *testing.T) {
	return
}
