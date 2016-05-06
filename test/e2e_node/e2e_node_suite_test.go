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
// NOTE: This test suite requires password-less sudo capabilities to run the kubelet and kube-apiserver.
package e2e_node

import (
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/golang/glog"
	. "github.com/onsi/ginkgo"
	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/types"
	. "github.com/onsi/gomega"
)

var e2es *e2eService

func TestE2eNode(t *testing.T) {
	flag.Parse()
	rand.Seed(time.Now().UTC().UnixNano())
	RegisterFailHandler(Fail)
	reporters := []Reporter{&LogReporter{}}
	RunSpecsWithDefaultAndCustomReporters(t, "E2eNode Suite", reporters)
}

// Setup the kubelet on the node
var _ = BeforeSuite(func() {
	if *buildServices {
		buildGo()
	}
	if *nodeName == "" {
		output, err := exec.Command("hostname").CombinedOutput()
		if err != nil {
			glog.Fatalf("Could not get node name from hostname %v.  Output:\n%s", err, output)
		}
		*nodeName = strings.TrimSpace(fmt.Sprintf("%s", output))
	}

	// TODO(yifan): Temporary workaround to disable coreos from auto restart
	// by masking the locksmithd.
	// We should mask locksmithd when provisioning the machine.
	maskLocksmithdOnCoreos()

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
	}
	glog.Infof("Tests Finished")
})

var _ Reporter = &LogReporter{}

type LogReporter struct{}

func (lr *LogReporter) SpecSuiteWillBegin(config config.GinkgoConfigType, summary *types.SuiteSummary) {
	b := &bytes.Buffer{}
	b.WriteString("******************************************************\n")
	glog.Infof(b.String())
}

func (lr *LogReporter) BeforeSuiteDidRun(setupSummary *types.SetupSummary) {}

func (lr *LogReporter) SpecWillRun(specSummary *types.SpecSummary) {}

func (lr *LogReporter) SpecDidComplete(specSummary *types.SpecSummary) {}

func (lr *LogReporter) AfterSuiteDidRun(setupSummary *types.SetupSummary) {}

func (lr *LogReporter) SpecSuiteDidEnd(summary *types.SuiteSummary) {
	// Only log the binary output if the suite failed.
	b := &bytes.Buffer{}
	if e2es != nil && !summary.SuiteSucceeded {
		b.WriteString(fmt.Sprintf("Process Log For Failed Suite On %s\n", *nodeName))
		b.WriteString("-------------------------------------------------------------\n")
		b.WriteString(fmt.Sprintf("kubelet output:\n%s\n", e2es.kubeletCombinedOut.String()))
		b.WriteString("-------------------------------------------------------------\n")
		b.WriteString(fmt.Sprintf("apiserver output:\n%s\n", e2es.apiServerCombinedOut.String()))
		b.WriteString("-------------------------------------------------------------\n")
		b.WriteString(fmt.Sprintf("etcd output:\n%s\n", e2es.etcdCombinedOut.String()))
	}
	b.WriteString("******************************************************\n")
	glog.Infof(b.String())
}

func maskLocksmithdOnCoreos() {
	data, err := ioutil.ReadFile("/etc/os-release")
	if err != nil {
		glog.Fatalf("Could not read /etc/os-release: %v", err)
	}
	if bytes.Contains(data, []byte("ID=coreos")) {
		if output, err := exec.Command("sudo", "systemctl", "mask", "--now", "locksmithd").CombinedOutput(); err != nil {
			glog.Fatalf("Could not mask locksmithd: %v, output: %q", err, string(output))
		}
	}
	glog.Infof("Locksmithd is masked successfully")
}
