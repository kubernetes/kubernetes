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
	"fmt"

	"k8s.io/klog/v2"

	"k8s.io/kubernetes/test/e2e_node/remote"
	_ "k8s.io/kubernetes/test/e2e_node/remote/gce"
	"k8s.io/kubernetes/test/e2e_node/system"
)

var testSuite = flag.String("test-suite", "default", "Test suite the runner initializes with. Currently support default|cadvisor|conformance")
var _ = flag.String("system-spec-name", "", fmt.Sprintf("The name of the system spec used for validating the image in the node conformance test. The specs are at %s. If unspecified, the default built-in spec (system.DefaultSpec) will be used.", system.SystemSpecPath))
var _ = flag.String("extra-envs", "", "The extra environment variables needed for node e2e tests. Format: a list of key=value pairs, e.g., env1=val1,env2=val2")
var _ = flag.String("runtime-config", "", "The runtime configuration for the API server on the node e2e tests.. Format: a list of key=value pairs, e.g., env1=val1,env2=val2")
var _ = flag.String("kubelet-config-file", "", "The KubeletConfiguration file that should be applied to the kubelet")

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
