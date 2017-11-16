/*
Copyright 2016 The Kubernetes Authors.

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

package remote

import (
	"time"
)

// TestSuite is the interface of a test suite, such as node e2e, node conformance,
// node soaking, cri validation etc.
type TestSuite interface {
	// SetupTestPackage setup the test package in the given directory. TestSuite
	// should put all necessary binaries and dependencies into the path. The caller
	// will:
	// * create a tarball with the directory.
	// * deploy the tarball to the testing host.
	// * untar the tarball to the testing workspace on the testing host.
	SetupTestPackage(path, systemSpecName string) error
	// RunTest runs test on the node in the given workspace and returns test output
	// and test error if there is any.
	// * host is the target node to run the test.
	// * workspace is the directory on the testing host the test is running in. Note
	//   that the test package is unpacked in the workspace before running the test.
	// * results is the directory the test should write result into. All logs should be
	//   saved as *.log, all junit file should start with junit*.
	// * imageDesc is the description of the image the test is running on.
	//   It will be used for logging purpose only.
	// * junitFilePrefix is the prefix of output junit file.
	// * testArgs is the arguments passed to test.
	// * ginkgoArgs is the arguments passed to ginkgo.
	// * systemSpecName is the name of the system spec used for validating the
	//   image on which the test runs.
	// * timeout is the test timeout.
	RunTest(host, workspace, results, imageDesc, junitFilePrefix, testArgs, ginkgoArgs, systemSpecName string, timeout time.Duration) (string, error)
}
