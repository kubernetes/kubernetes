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

// To run the node e2e tests remotely against one or more hosts on gce:
// $ go run run_remote.go --v 2 --ssh-env gce --hosts <comma separated hosts>
// To run the node e2e tests remotely against one or more images on gce and provision them:
// $ go run run_remote.go --v 2 --project <project> --zone <zone> --ssh-env gce --images <comma separated images>
package main

import (
	"flag"

	"k8s.io/klog/v2"

	"k8s.io/kubernetes/test/e2e_node/remote"
)

var testSuite = flag.String("test-suite", "default", "Test suite the runner initializes with. Currently support default|cadvisor|conformance")

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	suite, err := remote.GetTestSuite(*testSuite)
	if err != nil {
		klog.Fatalf("error looking up testsuite [%v] - registered test suites [%v]",
			err,
			remote.GetTestSuiteKeys())
	}
	remote.RunRemoteTestSuite(suite)
}
