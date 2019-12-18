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
	"runtime"
	"strings"
	"time"

	"k8s.io/klog"

	"k8s.io/kubernetes/test/e2e_node/builder"
	"k8s.io/kubernetes/test/utils"
)

// ConformanceRemote contains the specific functions in the node conformance test suite.
type ConformanceRemote struct{}

// InitConformanceRemote initializes the node conformance test suite.
func InitConformanceRemote() TestSuite {
	return &ConformanceRemote{}
}

// getConformanceDirectory gets node conformance test build directory.
func getConformanceDirectory() (string, error) {
	k8sRoot, err := utils.GetK8sRootDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(k8sRoot, "test", "e2e_node", "conformance", "build"), nil
}

// commandToString is a helper function which formats command to string.
func commandToString(c *exec.Cmd) string {
	return strings.Join(append([]string{c.Path}, c.Args[1:]...), " ")
}

// Image path constants.
const (
	conformanceRegistry         = "k8s.gcr.io"
	conformanceArch             = runtime.GOARCH
	conformanceTarfile          = "node_conformance.tar"
	conformanceTestBinary       = "e2e_node.test"
	conformanceImageLoadTimeout = time.Duration(30) * time.Second
)

// timestamp is used as an unique id of current test.
var timestamp = getTimestamp()

// getConformanceTestImageName returns name of the conformance test image given the system spec name.
func getConformanceTestImageName(systemSpecName string) string {
	if systemSpecName == "" {
		return fmt.Sprintf("%s/node-test-%s:%s", conformanceRegistry, conformanceArch, timestamp)
	}
	return fmt.Sprintf("%s/node-test-%s-%s:%s", conformanceRegistry, systemSpecName, conformanceArch, timestamp)
}

// buildConformanceTest builds node conformance test image tarball into binDir.
func buildConformanceTest(binDir, systemSpecName string) error {
	// Get node conformance directory.
	conformancePath, err := getConformanceDirectory()
	if err != nil {
		return fmt.Errorf("failed to get node conformance directory: %v", err)
	}
	// Build docker image.
	cmd := exec.Command("make", "-C", conformancePath, "BIN_DIR="+binDir,
		"REGISTRY="+conformanceRegistry,
		"ARCH="+conformanceArch,
		"VERSION="+timestamp,
		"SYSTEM_SPEC_NAME="+systemSpecName)
	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("failed to build node conformance docker image: command - %q, error - %v, output - %q",
			commandToString(cmd), err, output)
	}
	// Save docker image into tar file.
	cmd = exec.Command("docker", "save", "-o", filepath.Join(binDir, conformanceTarfile), getConformanceTestImageName(systemSpecName))
	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("failed to save node conformance docker image into tar file: command - %q, error - %v, output - %q",
			commandToString(cmd), err, output)
	}
	return nil
}

// SetupTestPackage sets up the test package with binaries k8s required for node conformance test
func (c *ConformanceRemote) SetupTestPackage(tardir, systemSpecName string) error {
	// Build the executables
	if err := builder.BuildGo(); err != nil {
		return fmt.Errorf("failed to build the dependencies: %v", err)
	}

	// Make sure we can find the newly built binaries
	buildOutputDir, err := utils.GetK8sBuildOutputDir()
	if err != nil {
		return fmt.Errorf("failed to locate kubernetes build output directory %v", err)
	}

	// Build node conformance tarball.
	if err := buildConformanceTest(buildOutputDir, systemSpecName); err != nil {
		return fmt.Errorf("failed to build node conformance test: %v", err)
	}

	// Copy files
	requiredFiles := []string{"kubelet", conformanceTestBinary, conformanceTarfile}
	for _, file := range requiredFiles {
		source := filepath.Join(buildOutputDir, file)
		if _, err := os.Stat(source); err != nil {
			return fmt.Errorf("failed to locate test file %s: %v", file, err)
		}
		output, err := exec.Command("cp", source, filepath.Join(tardir, file)).CombinedOutput()
		if err != nil {
			return fmt.Errorf("failed to copy %q: error - %v output - %q", file, err, output)
		}
	}

	return nil
}

// loadConformanceImage loads node conformance image from tar file.
func loadConformanceImage(host, workspace string) error {
	tarfile := filepath.Join(workspace, conformanceTarfile)
	if output, err := SSH(host, "timeout", conformanceImageLoadTimeout.String(),
		"docker", "load", "-i", tarfile); err != nil {
		return fmt.Errorf("failed to load node conformance image from tar file %q: error - %v output - %q",
			tarfile, err, output)
	}
	return nil
}

// kubeletLauncherLog is the log of kubelet launcher.
const kubeletLauncherLog = "kubelet-launcher.log"

// kubeletPodPath is a fixed known pod specification path. We can not use the random pod
// manifest directory generated in e2e_node.test because we need to mount the directory into
// the conformance test container, it's easier if it's a known directory.
// TODO(random-liu): Get rid of this once we switch to cluster e2e node bootstrap script.
var kubeletPodPath = "conformance-pod-manifest-" + timestamp

// getPodPath returns pod manifest full path.
func getPodPath(workspace string) string {
	return filepath.Join(workspace, kubeletPodPath)
}

// isSystemd returns whether the node is a systemd node.
func isSystemd(host string) (bool, error) {
	// Returns "systemd" if /run/systemd/system is found, empty string otherwise.
	output, err := SSH(host, "test", "-e", "/run/systemd/system", "&&", "echo", "systemd", "||", "true")
	if err != nil {
		return false, fmt.Errorf("failed to check systemd: error - %v output - %q", err, output)
	}
	return strings.TrimSpace(output) != "", nil
}

// launchKubelet launches kubelet by running e2e_node.test binary in run-kubelet-mode.
// This is a temporary solution, we should change node e2e to use the same node bootstrap
// with cluster e2e and launch kubelet outside of the test for both regular node e2e and
// node conformance test.
// TODO(random-liu): Switch to use standard node bootstrap script.
func launchKubelet(host, workspace, results, testArgs string) error {
	podManifestPath := getPodPath(workspace)
	if output, err := SSH(host, "mkdir", podManifestPath); err != nil {
		return fmt.Errorf("failed to create kubelet pod manifest path %q: error - %v output - %q",
			podManifestPath, err, output)
	}
	startKubeletCmd := fmt.Sprintf("./%s --run-kubelet-mode --logtostderr --node-name=%s"+
		" --report-dir=%s %s --kubelet-flags=--pod-manifest-path=%s > %s 2>&1",
		conformanceTestBinary, host, results, testArgs, podManifestPath, filepath.Join(results, kubeletLauncherLog))
	var cmd []string
	systemd, err := isSystemd(host)
	if err != nil {
		return fmt.Errorf("failed to check systemd: %v", err)
	}
	if systemd {
		cmd = []string{
			"systemd-run", "sh", "-c", getSSHCommand(" && ",
				// Switch to workspace.
				fmt.Sprintf("cd %s", workspace),
				// Launch kubelet by running e2e_node.test in run-kubelet-mode.
				startKubeletCmd,
			),
		}
	} else {
		cmd = []string{
			"sh", "-c", getSSHCommand(" && ",
				// Switch to workspace.
				fmt.Sprintf("cd %s", workspace),
				// Launch kubelet by running e2e_node.test in run-kubelet-mode with nohup.
				fmt.Sprintf("(nohup %s &)", startKubeletCmd),
			),
		}
	}
	klog.V(2).Infof("Launch kubelet with command: %v", cmd)
	output, err := SSH(host, cmd...)
	if err != nil {
		return fmt.Errorf("failed to launch kubelet with command %v: error - %v output - %q",
			cmd, err, output)
	}
	klog.Info("Successfully launch kubelet")
	return nil
}

// kubeletStopGracePeriod is the grace period to wait before forcibly killing kubelet.
const kubeletStopGracePeriod = 10 * time.Second

// stopKubelet stops kubelet launcher and kubelet gracefully.
func stopKubelet(host, workspace string) error {
	klog.Info("Gracefully stop kubelet launcher")
	if output, err := SSH(host, "pkill", conformanceTestBinary); err != nil {
		return fmt.Errorf("failed to gracefully stop kubelet launcher: error - %v output - %q",
			err, output)
	}
	klog.Info("Wait for kubelet launcher to stop")
	stopped := false
	for start := time.Now(); time.Since(start) < kubeletStopGracePeriod; time.Sleep(time.Second) {
		// Check whether the process is still running.
		output, err := SSH(host, "pidof", conformanceTestBinary, "||", "true")
		if err != nil {
			return fmt.Errorf("failed to check kubelet stopping: error - %v output -%q",
				err, output)
		}
		// Kubelet is stopped
		if strings.TrimSpace(output) == "" {
			stopped = true
			break
		}
	}
	if !stopped {
		klog.Info("Forcibly stop kubelet")
		if output, err := SSH(host, "pkill", "-SIGKILL", conformanceTestBinary); err != nil {
			return fmt.Errorf("failed to forcibly stop kubelet: error - %v output - %q",
				err, output)
		}
	}
	klog.Info("Successfully stop kubelet")
	// Clean up the pod manifest path
	podManifestPath := getPodPath(workspace)
	if output, err := SSH(host, "rm", "-f", filepath.Join(workspace, podManifestPath)); err != nil {
		return fmt.Errorf("failed to cleanup pod manifest directory %q: error - %v, output - %q",
			podManifestPath, err, output)
	}
	return nil
}

// RunTest runs test on the node.
func (c *ConformanceRemote) RunTest(host, workspace, results, imageDesc, junitFilePrefix, testArgs, _, systemSpecName, extraEnvs string, timeout time.Duration) (string, error) {
	// Install the cni plugins and add a basic CNI configuration.
	if err := setupCNI(host, workspace); err != nil {
		return "", err
	}

	// Configure iptables firewall rules.
	if err := configureFirewall(host); err != nil {
		return "", err
	}

	// Kill any running node processes.
	cleanupNodeProcesses(host)

	// Load node conformance image.
	if err := loadConformanceImage(host, workspace); err != nil {
		return "", err
	}

	// Launch kubelet.
	if err := launchKubelet(host, workspace, results, testArgs); err != nil {
		return "", err
	}
	// Stop kubelet.
	defer func() {
		if err := stopKubelet(host, workspace); err != nil {
			// Only log an error if failed to stop kubelet because it is not critical.
			klog.Errorf("failed to stop kubelet: %v", err)
		}
	}()

	// Run the tests
	klog.V(2).Infof("Starting tests on %q", host)
	podManifestPath := getPodPath(workspace)
	cmd := fmt.Sprintf("'timeout -k 30s %fs docker run --rm --privileged=true --net=host -v /:/rootfs -v %s:%s -v %s:/var/result -e TEST_ARGS=--report-prefix=%s -e EXTRA_ENVS=%s %s'",
		timeout.Seconds(), podManifestPath, podManifestPath, results, junitFilePrefix, extraEnvs, getConformanceTestImageName(systemSpecName))
	testOutput, err := SSH(host, "sh", "-c", cmd)
	if err != nil {
		return testOutput, err
	}

	return testOutput, nil
}
