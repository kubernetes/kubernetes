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

// To run tests in this suite
// NOTE: This test suite requires password-less sudo capabilities to run the kubelet and kube-apiserver.
package e2e_node

import (
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"os/exec"
	"path"
	"testing"
	"time"

	commontest "k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/golang/glog"
	. "github.com/onsi/ginkgo"
	"github.com/onsi/ginkgo/config"
	more_reporters "github.com/onsi/ginkgo/reporters"
	. "github.com/onsi/gomega"
)

var e2es *E2EServices

var prePullImages = flag.Bool("prepull-images", true, "If true, prepull images so image pull failures do not cause test failures.")

// TODO(random-liu): Should we allow user to specify this flag? Maybe add a warning in the description.
var startServicesOnly = flag.Bool("start-services-only", false, "If true, only start services (etcd, apiserver), and not run test. (default false)")

func init() {
	framework.RegisterCommonFlags()
	framework.RegisterNodeFlags()
}

func TestE2eNode(t *testing.T) {
	flag.Parse()
	if *startServicesOnly {
		// If start-services-only is specified, only run all services without running real test.
		RunE2EServices()
		return
	}

	rand.Seed(time.Now().UTC().UnixNano())
	RegisterFailHandler(Fail)
	reporters := []Reporter{}
	reportDir := framework.TestContext.ReportDir
	if reportDir != "" {
		// Create the directory if it doesn't already exists
		if err := os.MkdirAll(reportDir, 0755); err != nil {
			glog.Errorf("Failed creating report directory: %v", err)
		} else {
			// Configure a junit reporter to write to the directory
			junitFile := fmt.Sprintf("junit_%s%02d.xml", framework.TestContext.ReportPrefix, config.GinkgoConfig.ParallelNode)
			junitPath := path.Join(reportDir, junitFile)
			reporters = append(reporters, more_reporters.NewJUnitReporter(junitPath))
		}
	}
	RunSpecsWithDefaultAndCustomReporters(t, "E2eNode Suite", reporters)
}

// Setup the kubelet on the node
var _ = SynchronizedBeforeSuite(func() []byte {
	if *buildServices {
		buildGo()
	}
	if framework.TestContext.NodeName == "" {
		hostname, err := os.Hostname()
		if err != nil {
			glog.Fatalf("Could not get node name: %v", err)
		}
		framework.TestContext.NodeName = hostname
	}

	// Initialize node name here, so that the following code can get right node name.
	if framework.TestContext.NodeName == "" {
		hostname, err := os.Hostname()
		if err != nil {
			glog.Fatalf("Could not get node name: %v", err)
		}
		framework.TestContext.NodeName = hostname
	}
	// Pre-pull the images tests depend on so we can fail immediately if there is an image pull issue
	// This helps with debugging test flakes since it is hard to tell when a test failure is due to image pulling.
	if *prePullImages {
		glog.Infof("Pre-pulling images so that they are cached for the tests.")
		err := PrePullAllImages()
		Expect(err).ShouldNot(HaveOccurred())
	}

	// TODO(yifan): Temporary workaround to disable coreos from auto restart
	// by masking the locksmithd.
	// We should mask locksmithd when provisioning the machine.
	maskLocksmithdOnCoreos()

	var data []byte
	if *startServices {
		e2es = NewE2eServices()
		manifestPath, err := e2es.StartE2EServices()
		if err != nil {
			Fail(fmt.Sprintf("Unable to start node services.\n%v", err))
		}
		data = []byte(manifestPath)
		glog.Infof("Node services started.  Running tests...")
	} else {
		glog.Infof("Running tests without starting services.")
	}

	// Reference common test to make the import valid.
	commontest.CurrentSuite = commontest.NodeE2E

	return data
}, func(data []byte) {
	// TODO(random-liu): Get manifest path from kubelet flags when we move kubelet start logic out of the test.
	framework.TestContext.ManifestPath = string(data)

	// Initialize node name here to make sure every test node will get right node name.
	if framework.TestContext.NodeName == "" {
		hostname, err := os.Hostname()
		if err != nil {
			glog.Fatalf("Could not get node name: %v", err)
		}
		framework.TestContext.NodeName = hostname
	}
})

// Tear down the kubelet on the node
var _ = SynchronizedAfterSuite(func() {}, func() {
	if e2es != nil {
		if *startServices && *stopServices {
			glog.Infof("Stopping node services...")
			e2es.StopE2EServices()
		}
	}

	glog.Infof("Tests Finished")
})

func maskLocksmithdOnCoreos() {
	data, err := ioutil.ReadFile("/etc/os-release")
	if err != nil {
		// Not all distros contain this file.
		glog.Infof("Could not read /etc/os-release: %v", err)
		return
	}
	if bytes.Contains(data, []byte("ID=coreos")) {
		if output, err := exec.Command("sudo", "systemctl", "mask", "--now", "locksmithd").CombinedOutput(); err != nil {
			glog.Fatalf("Could not mask locksmithd: %v, output: %q", err, string(output))
		}
		glog.Infof("Locksmithd is masked successfully")
	}
}
