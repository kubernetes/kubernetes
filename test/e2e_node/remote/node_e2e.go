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

package remote

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/test/e2e_node/builder"
)

// NodeE2ERemote contains the specific functions in the node e2e test suite.
type NodeE2ERemote struct{}

func InitNodeE2ERemote() TestSuite {
	// TODO: Register flags.
	return &NodeE2ERemote{}
}

const (
	localGCIMounterPath = "cluster/gce/gci/mounter/mounter"
	CNIRelease          = "07a8a28637e97b22eb8dfe710eeae1344f69d16e"
	CNIDirectory        = "cni"
	CNIURL              = "https://storage.googleapis.com/kubernetes-release/network-plugins/cni-" + CNIRelease + ".tar.gz"
)

// SetupTestPackage sets up the test package with binaries k8s required for node e2e tests
func (n *NodeE2ERemote) SetupTestPackage(tardir string) error {
	// Build the executables
	if err := builder.BuildGo(); err != nil {
		return fmt.Errorf("failed to build the depedencies: %v", err)
	}

	// Make sure we can find the newly built binaries
	buildOutputDir, err := builder.GetK8sBuildOutputDir()
	if err != nil {
		return fmt.Errorf("failed to locate kubernetes build output directory %v", err)
	}

	// Copy binaries
	requiredBins := []string{"kubelet", "e2e_node.test", "ginkgo"}
	for _, bin := range requiredBins {
		source := filepath.Join(buildOutputDir, bin)
		if _, err := os.Stat(source); err != nil {
			return fmt.Errorf("failed to locate test binary %s: %v", bin, err)
		}
		out, err := exec.Command("cp", source, filepath.Join(tardir, bin)).CombinedOutput()
		if err != nil {
			return fmt.Errorf("failed to copy %q: %v Output: %q", bin, err, out)
		}
	}

	// Include the GCI mounter artifacts in the deployed tarball
	k8sDir, err := builder.GetK8sRootDir()
	if err != nil {
		return fmt.Errorf("Could not find K8s root dir! Err: %v", err)
	}
	source := filepath.Join(k8sDir, localGCIMounterPath)

	// Require the GCI mounter script, we want to make sure the remote test runner stays up to date if the mounter file moves
	if _, err := os.Stat(source); err != nil {
		return fmt.Errorf("Could not find GCI mounter script at %q! If this script has been (re)moved, please update the e2e node remote test runner accordingly! Err: %v", source, err)
	}

	bindir := "cluster/gce/gci/mounter"
	bin := "mounter"
	destdir := filepath.Join(tardir, bindir)
	dest := filepath.Join(destdir, bin)
	out, err := exec.Command("mkdir", "-p", filepath.Join(tardir, bindir)).CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to create directory %q for GCI mounter script. Err: %v. Output:\n%s", destdir, err, out)
	}
	out, err = exec.Command("cp", source, dest).CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to copy GCI mounter script to the archive bin. Err: %v. Output:\n%s", err, out)
	}
	return nil
}

// Install the cni plugin.
func installCNI(host, workspace string) error {
	glog.Infof("Install CNI on %q", host)
	cniPath := filepath.Join(workspace, CNIDirectory)
	cmd := getSSHCommand(" ; ",
		fmt.Sprintf("mkdir -p %s", cniPath),
		fmt.Sprintf("wget -O - %s | tar -xz -C %s", CNIURL, cniPath),
	)
	if output, err := SSH(host, "sh", "-c", cmd); err != nil {
		return fmt.Errorf("failed to install cni plugin on %q: %v output: %q", host, err, output)
	}
	return nil
}

// configureFirewall configures iptable firewall rules.
func configureFirewall(host string) error {
	glog.Infof("Configure iptables firewall rules on %q", host)
	// TODO: consider calling bootstrap script to configure host based on OS
	output, err := SSH(host, "iptables", "-L", "INPUT")
	if err != nil {
		return fmt.Errorf("failed to get iptables INPUT on %q: %v output: %q", host, err, output)
	}
	if strings.Contains(output, "Chain INPUT (policy DROP)") {
		cmd := getSSHCommand("&&",
			"(iptables -C INPUT -w -p TCP -j ACCEPT || iptables -A INPUT -w -p TCP -j ACCEPT)",
			"(iptables -C INPUT -w -p UDP -j ACCEPT || iptables -A INPUT -w -p UDP -j ACCEPT)",
			"(iptables -C INPUT -w -p ICMP -j ACCEPT || iptables -A INPUT -w -p ICMP -j ACCEPT)")
		output, err := SSH(host, "sh", "-c", cmd)
		if err != nil {
			return fmt.Errorf("failed to configured firewall on %q: %v output: %v", host, err, output)
		}
	}
	output, err = SSH(host, "iptables", "-L", "FORWARD")
	if err != nil {
		return fmt.Errorf("failed to get iptables FORWARD on %q: %v output: %q", host, err, output)
	}
	if strings.Contains(output, "Chain FORWARD (policy DROP)") {
		cmd := getSSHCommand("&&",
			"(iptables -C FORWARD -w -p TCP -j ACCEPT || iptables -A FORWARD -w -p TCP -j ACCEPT)",
			"(iptables -C FORWARD -w -p UDP -j ACCEPT || iptables -A FORWARD -w -p UDP -j ACCEPT)",
			"(iptables -C FORWARD -w -p ICMP -j ACCEPT || iptables -A FORWARD -w -p ICMP -j ACCEPT)")
		output, err = SSH(host, "sh", "-c", cmd)
		if err != nil {
			return fmt.Errorf("failed to configured firewall on %q: %v output: %v", host, err, output)
		}
	}
	return nil
}

// cleanupNodeProcesses kills all running node processes may conflict with the test.
func cleanupNodeProcesses(host string) {
	glog.Infof("Killing any existing node processes on %q", host)
	cmd := getSSHCommand(" ; ",
		"pkill kubelet",
		"pkill kube-apiserver",
		"pkill etcd",
		"pkill e2e_node.test",
	)
	// No need to log an error if pkill fails since pkill will fail if the commands are not running.
	// If we are unable to stop existing running k8s processes, we should see messages in the kubelet/apiserver/etcd
	// logs about failing to bind the required ports.
	SSH(host, "sh", "-c", cmd)
}

// updateGCIMounterPath updates kubelet flags to set gci mounter path. This will only take effect for
// GCI image.
func updateGCIMounterPath(args, host, workspace string) (string, error) {
	// Determine if tests will run on a GCI node.
	output, err := SSH(host, "cat", "/etc/os-release")
	if err != nil {
		return args, fmt.Errorf("issue detecting node's OS via node's /etc/os-release. Err: %v, Output:\n%s", err, output)
	}
	if !strings.Contains(output, "ID=gci") {
		// This is not a GCI image
		return args, nil
	}

	// If we are testing on a GCI node, we chmod 544 the mounter and specify a different mounter path in the test args.
	// We do this here because the local var `workspace` tells us which /tmp/node-e2e-%d is relevant to the current test run.

	// Determine if the GCI mounter script exists locally.
	k8sDir, err := builder.GetK8sRootDir()
	if err != nil {
		return args, fmt.Errorf("could not find K8s root dir! Err: %v", err)
	}
	source := filepath.Join(k8sDir, localGCIMounterPath)

	// Require the GCI mounter script, we want to make sure the remote test runner stays up to date if the mounter file moves
	if _, err = os.Stat(source); err != nil {
		return args, fmt.Errorf("could not find GCI mounter script at %q! If this script has been (re)moved, please update the e2e node remote test runner accordingly! Err: %v", source, err)
	}

	glog.Infof("GCI node and GCI mounter both detected, modifying --experimental-mounter-path accordingly")
	// Note this implicitly requires the script to be where we expect in the tarball, so if that location changes the error
	// here will tell us to update the remote test runner.
	mounterPath := filepath.Join(workspace, localGCIMounterPath)
	output, err = SSH(host, "sh", "-c", fmt.Sprintf("'chmod 544 %s'", mounterPath))
	if err != nil {
		return args, fmt.Errorf("unabled to chmod 544 GCI mounter script. Err: %v, Output:\n%s", err, output)
	}
	// Insert args at beginning of test args, so any values from command line take precedence
	args = fmt.Sprintf("--kubelet-flags=--experimental-mounter-path=%s ", mounterPath) + args
	return args, nil
}

// RunTest runs test on the node.
func (n *NodeE2ERemote) RunTest(host, workspace, results, junitFilePrefix, testArgs, ginkgoFlags string, timeout time.Duration) (string, error) {
	// Install the cni plugin.
	if err := installCNI(host, workspace); err != nil {
		return "", err
	}

	// Configure iptables firewall rules
	if err := configureFirewall(host); err != nil {
		return "", err
	}

	// Kill any running node processes
	cleanupNodeProcesses(host)

	testArgs, err := updateGCIMounterPath(testArgs, host, workspace)
	if err != nil {
		return "", err
	}

	// Run the tests
	glog.Infof("Starting tests on %q", host)
	cmd := getSSHCommand(" && ",
		fmt.Sprintf("cd %s", workspace),
		fmt.Sprintf("timeout -k 30s %fs ./ginkgo %s ./e2e_node.test -- --logtostderr --v 4 --node-name=%s --report-dir=%s --report-prefix=%s %s",
			timeout.Seconds(), ginkgoFlags, host, results, junitFilePrefix, testArgs),
	)
	return SSH(host, "sh", "-c", cmd)
}
