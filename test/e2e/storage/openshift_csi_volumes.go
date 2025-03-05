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

// This is a copy of csi_volumes.go with OpenShift specific test driver.
// Used a copy of the file to avoid conflicts when editing the existing file.
package storage

import (
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

// List of testDrivers to be executed in below loop
var ocpCSITestDrivers = []func() storageframework.TestDriver{
	drivers.InitGroupSnapshotHostpathCSIDriver,
}

// This executes testSuites for csi volumes.
var _ = utils.SIGDescribe("OCP CSI Volumes", func() {
	for _, initDriver := range ocpCSITestDrivers {
		curDriver := initDriver()

		args := storageframework.GetDriverNameWithFeatureTags(curDriver)
		args = append(args, func() {
			storageframework.DefineTestSuites(curDriver, testsuites.CSISuites)
		})
		framework.Context(args...)
	}
})
