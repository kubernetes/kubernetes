/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
// NOTE: This test suite requires sudo capabilities to run the kubelet and kube-apiserver.
// $ sudo -v && ginkgo test/e2e_node/ -- --logtostderr --v 2 --node-name `hostname` --start-services
package e2e_node

import (
	"bytes"
	"flag"
	"fmt"
	"os/exec"
	"strings"
	"testing"

	"github.com/golang/glog"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var kubeletAddress = flag.String("kubelet-address", "http://127.0.0.1:10255", "Host and port of the kubelet")
var apiServerAddress = flag.String("api-server-address", "http://127.0.0.1:8080", "Host and port of the api server")
var nodeName = flag.String("node-name", "", "Name of the node")
var buildServices = flag.Bool("build-services", true, "If true, build local executables")
var startServices = flag.Bool("start-services", true, "If true, start local node services")
var stopServices = flag.Bool("stop-services", true, "If true, stop local node services after running tets")

var e2es *e2eService

func TestE2eNode(t *testing.T) {
	flag.Parse()
	RegisterFailHandler(Fail)
	RunSpecs(t, "E2eNode Suite")
}

// Setup the kubelet on the node
var _ = BeforeSuite(func() {
	if *buildServices {
		buildGo()
	}
	if *nodeName == "" {
		output, err := exec.Command("hostname").CombinedOutput()
		if err != nil {
			glog.Fatal("Could not get node name from hostname %v.  Output:\n%s", err, output)
		}
		*nodeName = strings.TrimSpace(fmt.Sprintf("%s", output))
	}

	if *startServices {
		e2es = newE2eService(*nodeName)
		if err := e2es.start(); err != nil {
			Fail(fmt.Sprintf("Unable to start node services.\n%v", err))
		}
		glog.Infof("Node services started.  Running tests...")
	} else {
		glog.Infof("Running tests without starting services.")
	}
})

// Tear down the kubelet on the node
var _ = AfterSuite(func() {
	if e2es != nil && *startServices && *stopServices {
		glog.Infof("Stopping node services...")
		e2es.stop()
		b := &bytes.Buffer{}
		b.WriteString("-------------------------------------------------------------\n")
		b.WriteString(fmt.Sprintf("kubelet output:\n%s\n", e2es.kubeletCombinedOut.String()))
		b.WriteString("-------------------------------------------------------------\n")
		b.WriteString(fmt.Sprintf("apiserver output:\n%s", e2es.apiServerCombinedOut.String()))
		b.WriteString("-------------------------------------------------------------\n")
		b.WriteString(fmt.Sprintf("etcd output:\n%s", e2es.etcdCombinedOut.String()))
		b.WriteString("-------------------------------------------------------------\n")
		glog.V(2).Infof(b.String())

	}
})
