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
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"os/exec"
	"path"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	commontest "k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e_node/services"

	"github.com/golang/glog"
	. "github.com/onsi/ginkgo"
	"github.com/onsi/ginkgo/config"
	more_reporters "github.com/onsi/ginkgo/reporters"
	. "github.com/onsi/gomega"
	"github.com/spf13/pflag"
)

var e2es *services.E2EServices

var prePullImages = flag.Bool("prepull-images", true, "If true, prepull images so image pull failures do not cause test failures.")
var runServicesMode = flag.Bool("run-services-mode", false, "If true, only run services (etcd, apiserver) in current process, and not run test.")

func init() {
	framework.RegisterCommonFlags()
	framework.RegisterNodeFlags()
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)
	// Mark the run-services-mode flag as hidden to prevent user from using it.
	pflag.CommandLine.MarkHidden("run-services-mode")
}

func TestE2eNode(t *testing.T) {
	pflag.Parse()
	if *runServicesMode {
		// If run-services-mode is specified, only run services in current process.
		services.RunE2EServices()
		return
	}
	// If run-services-mode is not specified, run test.
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
	// Initialize node name here, so that the following code can get right node name.
	if framework.TestContext.NodeName == "" {
		hostname, err := os.Hostname()
		Expect(err).NotTo(HaveOccurred(), "should be able to get node name")
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

	if *startServices {
		e2es = services.NewE2EServices()
		Expect(e2es.Start()).To(Succeed(), "should be able to start node services.")
		glog.Infof("Node services started.  Running tests...")
	} else {
		glog.Infof("Running tests without starting services.")
	}

	glog.Infof("Wait for the node to be ready")
	waitForNodeReady()

	// Reference common test to make the import valid.
	commontest.CurrentSuite = commontest.NodeE2E

	data, err := json.Marshal(&framework.TestContext.NodeTestContextType)
	Expect(err).NotTo(HaveOccurred(), "should be able to serialize node test context.")

	return data
}, func(data []byte) {
	// The node test context is updated in the first function, update it on every test node.
	err := json.Unmarshal(data, &framework.TestContext.NodeTestContextType)
	Expect(err).NotTo(HaveOccurred(), "should be able to deserialize node test context.")
})

// Tear down the kubelet on the node
var _ = SynchronizedAfterSuite(func() {}, func() {
	if e2es != nil {
		if *startServices && *stopServices {
			glog.Infof("Stopping node services...")
			e2es.Stop()
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
		output, err := exec.Command("sudo", "systemctl", "mask", "--now", "locksmithd").CombinedOutput()
		Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("should be able to mask locksmithd - output: %q", string(output)))
		glog.Infof("Locksmithd is masked successfully")
	}
}

func waitForNodeReady() {
	const (
		// nodeReadyTimeout is the time to wait for node to become ready.
		nodeReadyTimeout = 2 * time.Minute
		// nodeReadyPollInterval is the interval to check node ready.
		nodeReadyPollInterval = 1 * time.Second
	)
	config, err := framework.LoadConfig()
	Expect(err).NotTo(HaveOccurred())
	client, err := clientset.NewForConfig(config)
	Expect(err).NotTo(HaveOccurred())
	Eventually(func() error {
		nodes, err := client.Nodes().List(api.ListOptions{})
		Expect(err).NotTo(HaveOccurred())
		if nodes == nil {
			return fmt.Errorf("the node list is nil.")
		}
		Expect(len(nodes.Items) > 1).NotTo(BeTrue())
		if len(nodes.Items) == 0 {
			return fmt.Errorf("empty node list: %+v", nodes)
		}
		node := nodes.Items[0]
		if !api.IsNodeReady(&node) {
			return fmt.Errorf("node is not ready: %+v", node)
		}
		return nil
	}, nodeReadyTimeout, nodeReadyPollInterval).Should(Succeed())
}
