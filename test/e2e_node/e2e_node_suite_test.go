// +build linux

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
	"syscall"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/system"
	nodeutil "k8s.io/kubernetes/pkg/api/v1/node"
	commontest "k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e_node/services"

	"github.com/kardianos/osext"
	. "github.com/onsi/ginkgo"
	"github.com/onsi/ginkgo/config"
	morereporters "github.com/onsi/ginkgo/reporters"
	. "github.com/onsi/gomega"
	"github.com/spf13/pflag"
	"k8s.io/klog"
)

var e2es *services.E2EServices

// TODO(random-liu): Change the following modes to sub-command.
var runServicesMode = flag.Bool("run-services-mode", false, "If true, only run services (etcd, apiserver) in current process, and not run test.")
var runKubeletMode = flag.Bool("run-kubelet-mode", false, "If true, only start kubelet, and not run test.")
var systemValidateMode = flag.Bool("system-validate-mode", false, "If true, only run system validation in current process, and not run test.")
var systemSpecFile = flag.String("system-spec-file", "", "The name of the system spec file that will be used for node conformance test. If it's unspecified or empty, the default system spec (system.DefaultSysSpec) will be used.")

func init() {
	framework.RegisterCommonFlags()
	framework.RegisterNodeFlags()
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)
	// Mark the run-services-mode flag as hidden to prevent user from using it.
	pflag.CommandLine.MarkHidden("run-services-mode")
	// It's weird that if I directly use pflag in TestContext, it will report error.
	// It seems that someone is using flag.Parse() after init() and TestMain().
	// TODO(random-liu): Find who is using flag.Parse() and cause errors and move the following logic
	// into TestContext.
	// TODO(pohly): remove RegisterNodeFlags from test_context.go enable Viper config support here?
}

func TestMain(m *testing.M) {
	rand.Seed(time.Now().UnixNano())
	pflag.Parse()
	framework.AfterReadingAllFlags(&framework.TestContext)
	os.Exit(m.Run())
}

// When running the containerized conformance test, we'll mount the
// host root filesystem as readonly to /rootfs.
const rootfs = "/rootfs"

func TestE2eNode(t *testing.T) {
	if *runServicesMode {
		// If run-services-mode is specified, only run services in current process.
		services.RunE2EServices(t)
		return
	}
	if *runKubeletMode {
		// If run-kubelet-mode is specified, only start kubelet.
		services.RunKubelet()
		return
	}
	if *systemValidateMode {
		// If system-validate-mode is specified, only run system validation in current process.
		spec := &system.DefaultSysSpec
		if *systemSpecFile != "" {
			var err error
			spec, err = loadSystemSpecFromFile(*systemSpecFile)
			if err != nil {
				klog.Exitf("Failed to load system spec: %v", err)
			}
		}
		if framework.TestContext.NodeConformance {
			// Chroot to /rootfs to make system validation can check system
			// as in the root filesystem.
			// TODO(random-liu): Consider to chroot the whole test process to make writing
			// test easier.
			if err := syscall.Chroot(rootfs); err != nil {
				klog.Exitf("chroot %q failed: %v", rootfs, err)
			}
		}
		if _, err := system.ValidateSpec(*spec, framework.TestContext.ContainerRuntime); err != nil {
			klog.Exitf("system validation failed: %v", err)
		}
		return
	}
	// If run-services-mode is not specified, run test.
	RegisterFailHandler(Fail)
	reporters := []Reporter{}
	reportDir := framework.TestContext.ReportDir
	if reportDir != "" {
		// Create the directory if it doesn't already exists
		if err := os.MkdirAll(reportDir, 0755); err != nil {
			klog.Errorf("Failed creating report directory: %v", err)
		} else {
			// Configure a junit reporter to write to the directory
			junitFile := fmt.Sprintf("junit_%s_%02d.xml", framework.TestContext.ReportPrefix, config.GinkgoConfig.ParallelNode)
			junitPath := path.Join(reportDir, junitFile)
			reporters = append(reporters, morereporters.NewJUnitReporter(junitPath))
		}
	}
	RunSpecsWithDefaultAndCustomReporters(t, "E2eNode Suite", reporters)
}

// Setup the kubelet on the node
var _ = SynchronizedBeforeSuite(func() []byte {
	// Run system validation test.
	Expect(validateSystem()).To(Succeed(), "system validation")

	// Pre-pull the images tests depend on so we can fail immediately if there is an image pull issue
	// This helps with debugging test flakes since it is hard to tell when a test failure is due to image pulling.
	if framework.TestContext.PrepullImages {
		klog.Infof("Pre-pulling images so that they are cached for the tests.")
		err := PrePullAllImages()
		Expect(err).ShouldNot(HaveOccurred())
	}

	// TODO(yifan): Temporary workaround to disable coreos from auto restart
	// by masking the locksmithd.
	// We should mask locksmithd when provisioning the machine.
	maskLocksmithdOnCoreos()

	if *startServices {
		// If the services are expected to stop after test, they should monitor the test process.
		// If the services are expected to keep running after test, they should not monitor the test process.
		e2es = services.NewE2EServices(*stopServices)
		Expect(e2es.Start()).To(Succeed(), "should be able to start node services.")
		klog.Infof("Node services started.  Running tests...")
	} else {
		klog.Infof("Running tests without starting services.")
	}

	klog.Infof("Wait for the node to be ready")
	waitForNodeReady()

	// Reference common test to make the import valid.
	commontest.CurrentSuite = commontest.NodeE2E

	return nil
}, func([]byte) {
	// update test context with node configuration.
	Expect(updateTestContext()).To(Succeed(), "update test context with node config.")
})

// Tear down the kubelet on the node
var _ = SynchronizedAfterSuite(func() {}, func() {
	if e2es != nil {
		if *startServices && *stopServices {
			klog.Infof("Stopping node services...")
			e2es.Stop()
		}
	}

	klog.Infof("Tests Finished")
})

// validateSystem runs system validation in a separate process and returns error if validation fails.
func validateSystem() error {
	testBin, err := osext.Executable()
	if err != nil {
		return fmt.Errorf("can't get current binary: %v", err)
	}
	// Pass all flags into the child process, so that it will see the same flag set.
	output, err := exec.Command(testBin, append([]string{"--system-validate-mode"}, os.Args[1:]...)...).CombinedOutput()
	// The output of system validation should have been formatted, directly print here.
	fmt.Print(string(output))
	if err != nil {
		return fmt.Errorf("system validation failed: %v", err)
	}
	return nil
}

func maskLocksmithdOnCoreos() {
	data, err := ioutil.ReadFile("/etc/os-release")
	if err != nil {
		// Not all distros contain this file.
		klog.Infof("Could not read /etc/os-release: %v", err)
		return
	}
	if bytes.Contains(data, []byte("ID=coreos")) {
		output, err := exec.Command("systemctl", "mask", "--now", "locksmithd").CombinedOutput()
		Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("should be able to mask locksmithd - output: %q", string(output)))
		klog.Infof("Locksmithd is masked successfully")
	}
}

func waitForNodeReady() {
	const (
		// nodeReadyTimeout is the time to wait for node to become ready.
		nodeReadyTimeout = 2 * time.Minute
		// nodeReadyPollInterval is the interval to check node ready.
		nodeReadyPollInterval = 1 * time.Second
	)
	client, err := getAPIServerClient()
	Expect(err).NotTo(HaveOccurred(), "should be able to get apiserver client.")
	Eventually(func() error {
		node, err := getNode(client)
		if err != nil {
			return fmt.Errorf("failed to get node: %v", err)
		}
		if !nodeutil.IsNodeReady(node) {
			return fmt.Errorf("node is not ready: %+v", node)
		}
		return nil
	}, nodeReadyTimeout, nodeReadyPollInterval).Should(Succeed())
}

// updateTestContext updates the test context with the node name.
// TODO(random-liu): Using dynamic kubelet configuration feature to
// update test context with node configuration.
func updateTestContext() error {
	client, err := getAPIServerClient()
	if err != nil {
		return fmt.Errorf("failed to get apiserver client: %v", err)
	}
	// Update test context with current node object.
	node, err := getNode(client)
	if err != nil {
		return fmt.Errorf("failed to get node: %v", err)
	}
	framework.TestContext.NodeName = node.Name // Set node name.
	// Update test context with current kubelet configuration.
	// This assumes all tests which dynamically change kubelet configuration
	// must: 1) run in serial; 2) restore kubelet configuration after test.
	kubeletCfg, err := getCurrentKubeletConfig()
	if err != nil {
		return fmt.Errorf("failed to get kubelet configuration: %v", err)
	}
	framework.TestContext.KubeletConfig = *kubeletCfg // Set kubelet config.
	return nil
}

// getNode gets node object from the apiserver.
func getNode(c *clientset.Clientset) (*v1.Node, error) {
	nodes, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "should be able to list nodes.")
	if nodes == nil {
		return nil, fmt.Errorf("the node list is nil.")
	}
	Expect(len(nodes.Items) > 1).NotTo(BeTrue(), "should not be more than 1 nodes.")
	if len(nodes.Items) == 0 {
		return nil, fmt.Errorf("empty node list: %+v", nodes)
	}
	return &nodes.Items[0], nil
}

// getAPIServerClient gets a apiserver client.
func getAPIServerClient() (*clientset.Clientset, error) {
	config, err := framework.LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %v", err)
	}
	client, err := clientset.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create client: %v", err)
	}
	return client, nil
}

// loadSystemSpecFromFile returns the system spec from the file with the
// filename.
func loadSystemSpecFromFile(filename string) (*system.SysSpec, error) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	data, err := utilyaml.ToJSON(b)
	if err != nil {
		return nil, err
	}
	spec := new(system.SysSpec)
	if err := json.Unmarshal(data, spec); err != nil {
		return nil, err
	}
	return spec, nil
}
