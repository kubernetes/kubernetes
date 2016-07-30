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
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	namespacecontroller "k8s.io/kubernetes/pkg/controller/namespace"
	"k8s.io/kubernetes/pkg/util/wait"
	commontest "k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/golang/glog"
	. "github.com/onsi/ginkgo"
	more_reporters "github.com/onsi/ginkgo/reporters"
	. "github.com/onsi/gomega"
)

var e2es *e2eService

// context is the test context shared by all parallel nodes.
// Originally we setup the test environment and initialize global variables
// in BeforeSuite, and then used the global variables in the test.
// However, after we make the test parallel, ginkgo will run all tests
// in several parallel test nodes. And for each test node, the BeforeSuite
// and AfterSuite will be run.
// We don't want to start services (kubelet, apiserver and etcd) for all
// parallel nodes, but we do want to set some globally shared variable which
// could be used in test.
// We have to use SynchronizedBeforeSuite to achieve that. The first
// function of SynchronizedBeforeSuite is only called once, and the second
// function is called in each parallel test node. The result returned by
// the first function will be the parameter of the second function.
// So we'll start all services and initialize the shared context in the first
// function, and propagate the context to all parallel test nodes in the
// second function.
// Notice no lock is needed for shared context, because context should only be
// initialized in the first function in SynchronizedBeforeSuite. After that
// it should never be modified.
var context SharedContext

var prePullImages = flag.Bool("prepull-images", true, "If true, prepull images so image pull failures do not cause test failures.")
var junitFileNumber = flag.Int("junit-file-number", 1, "Used to create junit filename - e.g. junit_01.xml.")

func init() {
	framework.RegisterCommonFlags()
	framework.RegisterNodeFlags()
}

func TestE2eNode(t *testing.T) {
	flag.Parse()

	rand.Seed(time.Now().UTC().UnixNano())
	RegisterFailHandler(Fail)
	reporters := []Reporter{}
	if *reportDir != "" {
		// Create the directory if it doesn't already exists
		if err := os.MkdirAll(*reportDir, 0755); err != nil {
			glog.Errorf("Failed creating report directory: %v", err)
		} else {
			// Configure a junit reporter to write to the directory
			junitFile := fmt.Sprintf("junit_%02d.xml", *junitFileNumber)
			junitPath := path.Join(*reportDir, junitFile)
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

	shared := &SharedContext{}
	if *startServices {
		e2es = newE2eService(framework.TestContext.NodeName, framework.TestContext.CgroupsPerQOS, shared)
		if err := e2es.start(); err != nil {
			Fail(fmt.Sprintf("Unable to start node services.\n%v", err))
		}
		glog.Infof("Node services started.  Running tests...")
	} else {
		glog.Infof("Running tests without starting services.")
	}

	glog.Infof("Starting namespace controller")
	startNamespaceController()

	// Reference common test to make the import valid.
	commontest.CurrentSuite = commontest.NodeE2E

	data, err := json.Marshal(shared)
	Expect(err).NotTo(HaveOccurred())

	return data
}, func(data []byte) {
	// Set the shared context got from the synchronized initialize function
	shared := &SharedContext{}
	Expect(json.Unmarshal(data, shared)).To(Succeed())
	context = *shared

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
		e2es.getLogFiles()
		if *startServices && *stopServices {
			glog.Infof("Stopping node services...")
			e2es.stop()
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

const (
	// ncResyncPeriod is resync period of the namespace controller
	ncResyncPeriod = 5 * time.Minute
	// ncConcurrency is concurrency of the namespace controller
	ncConcurrency = 2
)

func startNamespaceController() {
	// Use the default QPS
	config := restclient.AddUserAgent(&restclient.Config{Host: framework.TestContext.Host}, "node-e2e-namespace-controller")
	client, err := clientset.NewForConfig(config)
	Expect(err).NotTo(HaveOccurred())
	clientPool := dynamic.NewClientPool(config, dynamic.LegacyAPIPathResolverFunc)
	resources, err := client.Discovery().ServerPreferredNamespacedResources()
	Expect(err).NotTo(HaveOccurred())
	nc := namespacecontroller.NewNamespaceController(client, clientPool, resources, ncResyncPeriod, api.FinalizerKubernetes)
	go nc.Run(ncConcurrency, wait.NeverStop)
}
