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
// Local: `$ ginkgo -- --logtostderr -v 2`
// Remote: `$ ginkgo -- --node-name <hostname> --api-server-address=<hostname:api_port> --kubelet-address=<hostname=kubelet_port> --logtostderr -v 2`
package e2e_node

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"flag"
	"testing"
)

var kubeletAddress = flag.String("kubelet-address", "http://127.0.0.1:10255", "Host and port of the kubelet")
var apiServerAddress = flag.String("api-server-address", "http://127.0.0.1:8080", "Host and port of the api server")
var nodeName = flag.String("node-name", "127.0.0.1", "Name of the node")

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
