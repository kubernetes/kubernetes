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

package upgrades

import (
	"testing"
)

//Name of the package
var Name = "Upgrades"

//Tests List of internal tests.
var Tests = []testing.InternalTest{
	{Name: "Statefulset", F: statefulsetUpgrades},
	{Name: "Daemon", F: daemonsetUpgrades},
	{Name: "Deployment", F: deploymentUpgrades},
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

func statefulsetUpgrades(t *testing.T) {
	//Tests Related SS upgrades
	t.SkipNow()
}

func deploymentUpgrades(t *testing.T) {
	//Tests Related Deployment upgrades
	t.SkipNow()
}

func daemonsetUpgrades(t *testing.T) {
	//Tests Related to DaemonSet upgrades
	t.SkipNow()
}

func teardown(t *testing.T) {
	return
}
