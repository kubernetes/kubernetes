/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

// To run tests in this suite
// `$ ginkgo -- --node-name node-e2e-test-1  --api-server-address <serveraddress> --logtostderr`
package e2e_node

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"flag"
	"testing"
)

var kubeletAddress = flag.String("kubelet-address", "localhost:10250", "Host and port of the kubelet")
var apiServerAddress = flag.String("api-server-address", "localhost:8080", "Host and port of the api server")
var nodeName = flag.String("node-name", "", "Name of the node")

func TestE2eNode(t *testing.T) {
	flag.Parse()
	RegisterFailHandler(Fail)
	RunSpecs(t, "E2eNode Suite")
}

// Setup the kubelet on the node
var _ = BeforeSuite(func() {
})

// Tear down the kubelet on the node
var _ = AfterSuite(func() {
})
