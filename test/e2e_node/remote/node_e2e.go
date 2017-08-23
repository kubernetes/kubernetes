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

const (
	localCOSMounterPath = "cluster/gce/gci/mounter/mounter"
	systemSpecPath      = "test/e2e_node/system/specs"
)

// NodeE2ERemote contains the specific functions in the node e2e test suite.
type NodeE2ERemote struct{}

func InitNodeE2ERemote() TestSuite {
	// TODO: Register flags.
	return &NodeE2ERemote{}
}

// SetupTestPackage sets up the test package with binaries k8s required for node e2e tests
func (n *NodeE2ERemote) SetupTestPackage(tardir, systemSpecName string) error {
	// Build the executables
	if err := builder.BuildGo(); err != nil {
		return fmt.Errorf("failed to build the depedencies: %v", err)
	}

	// Make sure we can find the newly built binaries
	buildOutputDir, err := builder.GetK8sBuildOutputDir()
	if err != nil {
		return fmt.Errorf("failed to locate kubernetes build output directory: %v", err)
	}

	rootDir, err := builder.GetK8sRootDir()
	if err != nil {
		return fmt.Errorf("failed to locate kubernetes root directory: %v", err)
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

	if systemSpecName != "" {
		// Copy system spec file
		source := filepath.Join(rootDir, systemSpecPath, systemSpecName+".yaml")
		if _, err := os.Stat(source); err != nil {
			return fmt.Errorf("failed to locate system spec %q: %v", source, err)
		}
		out, err := exec.Command("cp", source, tardir).CombinedOutput()
		if err != nil {
			return fmt.Errorf("failed to copy system spec %q: %v, output: %q", source, err, out)
		}
	}

	// Include the GCI/COS mounter artifacts in the deployed tarball
	err = tarAddCOSMounter(tardir)
	if err != nil {
		return err
	}
	return nil
}

// dest is relative to the root of the tar
func tarAddFile(tar, source, dest string) error {
	dir := filepath.Dir(dest)
	tardir := filepath.Join(tar, dir)
	tardest := filepath.Join(tar, dest)

	out, err := exec.Command("mkdir", "-p", tardir).CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to create archive bin subdir %q, was dest for file %q. Err: %v. Output:\n%s", tardir, source, err, out)
	}
	out, err = exec.Command("cp", source, tardest).CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to copy file %q to the archive bin subdir %q. Err: %v. Output:\n%s", source, tardir, err, out)
	}
	return nil
}

// Includes the GCI/COS mounter artifacts in the deployed tarball
func tarAddCOSMounter(tar string) error {
	k8sDir, err := builder.GetK8sRootDir()
	if err != nil {
		return fmt.Errorf("Could not find K8s root dir! Err: %v", err)
	}

	source := filepath.Join(k8sDir, localCOSMounterPath)

	// Require the GCI/COS mounter script, we want to make sure the remote test runner stays up to date if the mounter file moves
	if _, err := os.Stat(source); err != nil {
		return fmt.Errorf("Could not find GCI/COS mounter script at %q! If this script has been (re)moved, please update the e2e node remote test runner accordingly! Err: %v", source, err)
	}

	tarAddFile(tar, source, localCOSMounterPath)
	return nil
}

// prependCOSMounterFlag prepends the flag for setting the GCI mounter path to
// args and returns the result.
func prependCOSMounterFlag(args, host, workspace string) (string, error) {
	// If we are testing on a GCI/COS node, we chmod 544 the mounter and specify a different mounter path in the test args.
	// We do this here because the local var `workspace` tells us which /tmp/node-e2e-%d is relevant to the current test run.

	// Determine if the GCI/COS mounter script exists locally.
	k8sDir, err := builder.GetK8sRootDir()
	if err != nil {
		return args, fmt.Errorf("could not find K8s root dir! Err: %v", err)
	}
	source := filepath.Join(k8sDir, localCOSMounterPath)

	// Require the GCI/COS mounter script, we want to make sure the remote test runner stays up to date if the mounter file moves
	if _, err = os.Stat(source); err != nil {
		return args, fmt.Errorf("could not find GCI/COS mounter script at %q! If this script has been (re)moved, please update the e2e node remote test runner accordingly! Err: %v", source, err)
	}

	glog.V(2).Infof("GCI/COS node and GCI/COS mounter both detected, modifying --experimental-mounter-path accordingly")
	// Note this implicitly requires the script to be where we expect in the tarball, so if that location changes the error
	// here will tell us to update the remote test runner.
	mounterPath := filepath.Join(workspace, localCOSMounterPath)
	output, err := SSH(host, "sh", "-c", fmt.Sprintf("'chmod 544 %s'", mounterPath))
	if err != nil {
		return args, fmt.Errorf("unabled to chmod 544 GCI/COS mounter script. Err: %v, Output:\n%s", err, output)
	}
	// Insert args at beginning of test args, so any values from command line take precedence
	args = fmt.Sprintf("--kubelet-flags=--experimental-mounter-path=%s ", mounterPath) + args
	return args, nil
}

// prependMemcgNotificationFlag prepends the flag for enabling memcg
// notification to args and returns the result.
func prependMemcgNotificationFlag(args string) string {
	return "--kubelet-flags=--experimental-kernel-memcg-notification=true " + args
}

// updateOSSpecificKubeletFlags updates the Kubelet args with OS specific
// settings.
func updateOSSpecificKubeletFlags(args, host, workspace string) (string, error) {
	output, err := SSH(host, "cat", "/etc/os-release")
	if err != nil {
		return "", fmt.Errorf("issue detecting node's OS via node's /etc/os-release. Err: %v, Output:\n%s", err, output)
	}
	switch {
	case strings.Contains(output, "ID=gci"), strings.Contains(output, "ID=cos"):
		args = prependMemcgNotificationFlag(args)
		return prependCOSMounterFlag(args, host, workspace)
	case strings.Contains(output, "ID=ubuntu"):
		return prependMemcgNotificationFlag(args), nil
	}
	return args, nil
}

// RunTest runs test on the node.
func (n *NodeE2ERemote) RunTest(host, workspace, results, imageDesc, junitFilePrefix, testArgs, ginkgoArgs, systemSpecName string, timeout time.Duration) (string, error) {
	// Install the cni plugins and add a basic CNI configuration.
	if err := setupCNI(host, workspace); err != nil {
		return "", err
	}

	// Configure iptables firewall rules
	if err := configureFirewall(host); err != nil {
		return "", err
	}

	// Kill any running node processes
	cleanupNodeProcesses(host)

	testArgs, err := updateOSSpecificKubeletFlags(testArgs, host, workspace)
	if err != nil {
		return "", err
	}

	systemSpecFile := ""
	if systemSpecName != "" {
		systemSpecFile = systemSpecName + ".yaml"
	}

	// Run the tests
	glog.V(2).Infof("Starting tests on %q", host)
	cmd := getSSHCommand(" && ",
		fmt.Sprintf("cd %s", workspace),
		fmt.Sprintf("timeout -k 30s %fs ./ginkgo %s ./e2e_node.test -- --system-spec-name=%s --system-spec-file=%s --logtostderr --v 4 --node-name=%s --report-dir=%s --report-prefix=%s --image-description=\"%s\" %s",
			timeout.Seconds(), ginkgoArgs, systemSpecName, systemSpecFile, host, results, junitFilePrefix, imageDesc, testArgs),
	)
	return SSH(host, "sh", "-c", cmd)
}
