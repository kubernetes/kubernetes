/*
Copyright 2014 Google Inc. All rights reserved.

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

package e2e

import (
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

type testSpec struct {
	// The test to run
	test func(c *client.Client) bool
	// The human readable name of this test
	name string
	// The id for this test.  It should be constant for the life of the test.
	id int
}

type testInfo struct {
	passed bool
	spec   testSpec
}

// Output a summary in the TAP (test anything protocol) format for automated processing.
// See http://testanything.org/ for more info
func outputTAPSummary(infoList []testInfo) {
	glog.Infof("1..%d", len(infoList))
	for _, info := range infoList {
		if info.passed {
			glog.Infof("ok %d - %s", info.spec.id, info.spec.name)
		} else {
			glog.Infof("not ok %d - %s", info.spec.id, info.spec.name)
		}
	}
}

func RunE2ETests(authConfig, certDir, host, repoRoot string) {
	testContext = testContextType{authConfig, certDir, host, repoRoot}
	util.ReallyCrash = true
	util.InitLogs()
	defer util.FlushLogs()

	go func() {
		defer util.FlushLogs()
		time.Sleep(5 * time.Minute)
		glog.Fatalf("This test has timed out. Cleanup not guaranteed.")
	}()

	c := loadClientOrDie()

	// Define the tests.  Important: for a clean test grid, please keep ids for a test constant.
	tests := []testSpec{
		{TestKubernetesROService, "TestKubernetesROService", 1},
		{TestKubeletSendsEvent, "TestKubeletSendsEvent", 2},
		{TestImportantURLs, "TestImportantURLs", 3},
		{TestPodUpdate, "TestPodUpdate", 4},
		{TestNetwork, "TestNetwork", 5},
		{TestClusterDNS, "TestClusterDNS", 6},
		{TestPodHasServiceEnvVars, "TestPodHasServiceEnvVars", 7},
		{TestBasic, "TestBasic", 8},
	}

	info := []testInfo{}
	passed := true
	for i, test := range tests {
		glog.Infof("Running test %d %s", i+1, test.name)
		testPassed := test.test(c)
		if !testPassed {
			glog.Infof("        test %d failed", i+1)
			passed = false
		} else {
			glog.Infof("        test %d passed", i+1)
		}
		// TODO: clean up objects created during a test after the test, so cases
		// are independent.
		info = append(info, testInfo{testPassed, test})
	}
	outputTAPSummary(info)
	if !passed {
		glog.Fatalf("At least one test failed")
	} else {
		glog.Infof("All tests pass")
	}
}
