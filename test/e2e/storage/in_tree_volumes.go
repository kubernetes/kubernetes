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

package storage

import (
	. "github.com/onsi/ginkgo"
	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

// List of testDrivers to be executed in below loop
var testDrivers = []func() drivers.TestDriver{
	drivers.InitNFSDriver,
	drivers.InitGlusterFSDriver,
	drivers.InitISCSIDriver,
	drivers.InitRbdDriver,
	drivers.InitCephFSDriver,
	drivers.InitHostPathDriver,
	drivers.InitHostPathSymlinkDriver,
	drivers.InitEmptydirDriver,
	drivers.InitCinderDriver,
	drivers.InitGcePdDriver,
	drivers.InitVSphereDriver,
	drivers.InitAzureDriver,
	drivers.InitAwsDriver,
}

// List of testSuites to be executed in below loop
var testSuites = []func() testsuites.TestSuite{
	testsuites.InitVolumesTestSuite,
	testsuites.InitVolumeIOTestSuite,
	testsuites.InitVolumeModeTestSuite,
	testsuites.InitSubPathTestSuite,
	testsuites.InitProvisioningTestSuite,
}

func intreeTunePattern(patterns []testpatterns.TestPattern) []testpatterns.TestPattern {
	return patterns
}

// This executes testSuites for in-tree volumes.
var _ = utils.SIGDescribe("In-tree Volumes", func() {
	f := framework.NewDefaultFramework("volumes")

	var (
		ns     *v1.Namespace
		config framework.VolumeTestConfig
	)

	BeforeEach(func() {
		ns = f.Namespace
		config = framework.VolumeTestConfig{
			Namespace: ns.Name,
			Prefix:    "volume",
		}
	})

	for _, initDriver := range testDrivers {
		curDriver := initDriver()
		Context(drivers.GetDriverNameWithFeatureTags(curDriver), func() {
			driver := curDriver

			BeforeEach(func() {
				// setupDriver
				drivers.SetCommonDriverParameters(driver, f, config)
				driver.CreateDriver()
			})

			AfterEach(func() {
				// Cleanup driver
				driver.CleanupDriver()
			})

			testsuites.RunTestSuite(f, config, driver, testSuites, intreeTunePattern)
		})
	}
})
