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
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"

	"github.com/onsi/ginkgo"
)

// List of testDrivers to be executed in below loop
var csiTestDrivers = []func() testsuites.TestDriver{
	drivers.InitHostPathCSIDriver,
	drivers.InitGcePDCSIDriver,
	// Don't run tests with mock driver (drivers.InitMockCSIDriver), it does not provide persistent storage.
}

// List of testSuites to be executed in below loop
var csiTestSuites = []func() testsuites.TestSuite{
	testsuites.InitEphemeralTestSuite,
	testsuites.InitVolumesTestSuite,
	testsuites.InitVolumeIOTestSuite,
	testsuites.InitVolumeModeTestSuite,
	testsuites.InitSubPathTestSuite,
	testsuites.InitProvisioningTestSuite,
	testsuites.InitSnapshottableTestSuite,
	testsuites.InitMultiVolumeTestSuite,
	testsuites.InitDisruptiveTestSuite,
	testsuites.InitVolumeExpandTestSuite,
	testsuites.InitVolumeLimitsTestSuite,
	testsuites.InitTopologyTestSuite,
}

// This executes testSuites for csi volumes.
var _ = utils.SIGDescribe("CSI Volumes", func() {
	for _, initDriver := range csiTestDrivers {
		curDriver := initDriver()

		ginkgo.Context(testsuites.GetDriverNameWithFeatureTags(curDriver), func() {
			testsuites.DefineTestSuite(curDriver, csiTestSuites)
		})
	}
})
