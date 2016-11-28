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
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/golang/glog"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/test/e2e_node/builder"
)

var testTimeoutSeconds = flag.Duration("test-timeout", 45*time.Minute, "How long (in golang duration format) to wait for ginkgo tests to complete.")
var resultsDir = flag.String("results-dir", "/tmp/", "Directory to scp test results to.")

const (
	archiveName  = "e2e_node_test.tar.gz"
	CNIRelease   = "07a8a28637e97b22eb8dfe710eeae1344f69d16e"
	CNIDirectory = "cni"
)

var CNIURL = fmt.Sprintf("https://storage.googleapis.com/kubernetes-release/network-plugins/cni-%s.tar.gz", CNIRelease)

// CreateTestArchive builds the local source and creates a tar archive e2e_node_test.tar.gz containing
// the binaries k8s required for node e2e tests
func CreateTestArchive() (string, error) {
	// Build the executables
	if err := builder.BuildGo(); err != nil {
		return "", fmt.Errorf("failed to build the depedencies: %v", err)
	}

	// Make sure we can find the newly built binaries
	buildOutputDir, err := builder.GetK8sBuildOutputDir()
	if err != nil {
		return "", fmt.Errorf("failed to locate kubernetes build output directory %v", err)
	}

	glog.Infof("Building archive...")
	tardir, err := ioutil.TempDir("", "node-e2e-archive")
	if err != nil {
		return "", fmt.Errorf("failed to create temporary directory %v.", err)
	}
	defer os.RemoveAll(tardir)

	// Copy binaries
	requiredBins := []string{"kubelet", "e2e_node.test", "ginkgo"}
	for _, bin := range requiredBins {
		source := filepath.Join(buildOutputDir, bin)
		if _, err := os.Stat(source); err != nil {
			return "", fmt.Errorf("failed to locate test binary %s: %v", bin, err)
		}
		out, err := exec.Command("cp", source, filepath.Join(tardir, bin)).CombinedOutput()
		if err != nil {
			return "", fmt.Errorf("failed to copy %q: %v Output: %q", bin, err, out)
		}
	}

	// Include the GCI mounter artifacts in the deployed tarball
	k8sDir, err := builder.GetK8sRootDir()
	if err != nil {
		return "", fmt.Errorf("Could not find K8s root dir! Err: %v", err)
	}
	localSource := "cluster/gce/gci/mounter/mounter"
	source := filepath.Join(k8sDir, localSource)

	// Require the GCI mounter script, we want to make sure the remote test runner stays up to date if the mounter file moves
	if _, err := os.Stat(source); err != nil {
		return "", fmt.Errorf("Could not find GCI mounter script at %q! If this script has been (re)moved, please update the e2e node remote test runner accordingly! Err: %v", source, err)
	}

	bindir := "cluster/gce/gci/mounter"
	bin := "mounter"
	destdir := filepath.Join(tardir, bindir)
	dest := filepath.Join(destdir, bin)
	out, err := exec.Command("mkdir", "-p", filepath.Join(tardir, bindir)).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to create directory %q for GCI mounter script. Err: %v. Output:\n%s", destdir, err, out)
	}
	out, err = exec.Command("cp", source, dest).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to copy GCI mounter script to the archive bin. Err: %v. Output:\n%s", err, out)
	}

	// Build the tar
	out, err = exec.Command("tar", "-zcvf", archiveName, "-C", tardir, ".").CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to build tar %v.  Output:\n%s", err, out)
	}

	dir, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("failed to get working directory %v.", err)
	}
	return filepath.Join(dir, archiveName), nil
}

// Returns the command output, whether the exit was ok, and any errors
func RunRemote(archive string, host string, cleanup bool, junitFilePrefix string, testArgs string, ginkgoFlags string) (string, bool, error) {
	// Create the temp staging directory
	glog.Infof("Staging test binaries on %s", host)
	workspace := fmt.Sprintf("/tmp/node-e2e-%s", getTimestamp())
	// Do not sudo here, so that we can use scp to copy test archive to the directdory.
	if output, err := SSHNoSudo(host, "mkdir", workspace); err != nil {
		// Exit failure with the error
		return "", false, fmt.Errorf("failed to create workspace directory: %v output: %q", err, output)
	}
	if cleanup {
		defer func() {
			output, err := SSH(host, "rm", "-rf", workspace)
			if err != nil {
				glog.Errorf("failed to cleanup workspace %s on host %v.  Output:\n%s", workspace, err, output)
			}
		}()
	}

	// Install the cni plugin.
	cniPath := filepath.Join(workspace, CNIDirectory)
	cmd := getSSHCommand(" ; ",
		fmt.Sprintf("mkdir -p %s", cniPath),
		fmt.Sprintf("wget -O - %s | tar -xz -C %s", CNIURL, cniPath),
	)
	if output, err := SSH(host, "sh", "-c", cmd); err != nil {
		// Exit failure with the error
		return "", false, fmt.Errorf("failed to install cni plugin: %v output: %q", err, output)
	}

	// Configure iptables firewall rules
	// TODO: consider calling bootstrap script to configure host based on OS
	output, err := SSH(host, "iptables", "-L", "INPUT")
	if err != nil {
		return "", false, fmt.Errorf("failed to get iptables INPUT: %v output: %q", err, output)
	}
	if strings.Contains(output, "Chain INPUT (policy DROP)") {
		cmd = getSSHCommand("&&",
			"(iptables -C INPUT -w -p TCP -j ACCEPT || iptables -A INPUT -w -p TCP -j ACCEPT)",
			"(iptables -C INPUT -w -p UDP -j ACCEPT || iptables -A INPUT -w -p UDP -j ACCEPT)",
			"(iptables -C INPUT -w -p ICMP -j ACCEPT || iptables -A INPUT -w -p ICMP -j ACCEPT)")
		output, err := SSH(host, "sh", "-c", cmd)
		if err != nil {
			return "", false, fmt.Errorf("failed to configured firewall: %v output: %v", err, output)
		}
	}
	output, err = SSH(host, "iptables", "-L", "FORWARD")
	if err != nil {
		return "", false, fmt.Errorf("failed to get iptables FORWARD: %v output: %q", err, output)
	}
	if strings.Contains(output, "Chain FORWARD (policy DROP)") {
		cmd = getSSHCommand("&&",
			"(iptables -C FORWARD -w -p TCP -j ACCEPT || iptables -A FORWARD -w -p TCP -j ACCEPT)",
			"(iptables -C FORWARD -w -p UDP -j ACCEPT || iptables -A FORWARD -w -p UDP -j ACCEPT)",
			"(iptables -C FORWARD -w -p ICMP -j ACCEPT || iptables -A FORWARD -w -p ICMP -j ACCEPT)")
		output, err = SSH(host, "sh", "-c", cmd)
		if err != nil {
			return "", false, fmt.Errorf("failed to configured firewall: %v output: %v", err, output)
		}
	}

	// Copy the archive to the staging directory
	if output, err = runSSHCommand("scp", archive, fmt.Sprintf("%s:%s/", GetHostnameOrIp(host), workspace)); err != nil {
		// Exit failure with the error
		return "", false, fmt.Errorf("failed to copy test archive: %v, output: %q", err, output)
	}

	// Kill any running node processes
	cmd = getSSHCommand(" ; ",
		"pkill kubelet",
		"pkill kube-apiserver",
		"pkill etcd",
	)
	// No need to log an error if pkill fails since pkill will fail if the commands are not running.
	// If we are unable to stop existing running k8s processes, we should see messages in the kubelet/apiserver/etcd
	// logs about failing to bind the required ports.
	glog.Infof("Killing any existing node processes on %s", host)
	SSH(host, "sh", "-c", cmd)

	// Extract the archive
	cmd = getSSHCommand(" && ",
		fmt.Sprintf("cd %s", workspace),
		fmt.Sprintf("tar -xzvf ./%s", archiveName),
	)
	glog.Infof("Extracting tar on %s", host)
	if output, err = SSH(host, "sh", "-c", cmd); err != nil {
		// Exit failure with the error
		return "", false, fmt.Errorf("failed to extract test archive: %v, output: %q", err, output)
	}

	// If we are testing on a GCI node, we chmod 544 the mounter and specify a different mounter path in the test args.
	// We do this here because the local var `workspace` tells us which /tmp/node-e2e-%d is relevant to the current test run.

	// Determine if the GCI mounter script exists locally.
	k8sDir, err := builder.GetK8sRootDir()
	if err != nil {
		return "", false, fmt.Errorf("Could not find K8s root dir! Err: %v", err)
	}
	localSource := "cluster/gce/gci/mounter/mounter"
	source := filepath.Join(k8sDir, localSource)

	// Require the GCI mounter script, we want to make sure the remote test runner stays up to date if the mounter file moves
	if _, err = os.Stat(source); err != nil {
		return "", false, fmt.Errorf("Could not find GCI mounter script at %q! If this script has been (re)moved, please update the e2e node remote test runner accordingly! Err: %v", source, err)
	}

	// Determine if tests will run on a GCI node.
	output, err = SSH(host, "sh", "-c", "'cat /etc/os-release'")
	if err != nil {
		glog.Errorf("Issue detecting node's OS via node's /etc/os-release. Err: %v, Output:\n%s", err, output)
		return "", false, fmt.Errorf("Issue detecting node's OS via node's /etc/os-release. Err: %v, Output:\n%s", err, output)
	}
	if strings.Contains(output, "ID=gci") {
		glog.Infof("GCI node and GCI mounter both detected, modifying --experimental-mounter-path accordingly")
		// Note this implicitly requires the script to be where we expect in the tarball, so if that location changes the error
		// here will tell us to update the remote test runner.
		mounterPath := filepath.Join(workspace, "cluster/gce/gci/mounter/mounter")
		output, err = SSH(host, "sh", "-c", fmt.Sprintf("'chmod 544 %s'", mounterPath))
		if err != nil {
			glog.Errorf("Unable to chmod 544 GCI mounter script. Err: %v, Output:\n%s", err, output)
			return "", false, err
		}
		// Insert args at beginning of testArgs, so any values from command line take precedence
		testArgs = fmt.Sprintf("--kubelet-flags=--experimental-mounter-path=%s ", mounterPath) + testArgs
	}

	// Run the tests
	cmd = getSSHCommand(" && ",
		fmt.Sprintf("cd %s", workspace),
		fmt.Sprintf("timeout -k 30s %fs ./ginkgo %s ./e2e_node.test -- --logtostderr --v 4 --node-name=%s --report-dir=%s/results --report-prefix=%s %s",
			testTimeoutSeconds.Seconds(), ginkgoFlags, host, workspace, junitFilePrefix, testArgs),
	)
	aggErrs := []error{}

	glog.Infof("Starting tests on %s", host)
	output, err = SSH(host, "sh", "-c", cmd)
	// Do not log the output here, let the caller deal with the test output.
	if err != nil {
		aggErrs = append(aggErrs, err)

		// Encountered an unexpected error. The remote test harness may not
		// have finished retrieved and stored all the logs in this case. Try
		// to get some logs for debugging purposes.
		// TODO: This is a best-effort, temporary hack that only works for
		// journald nodes. We should have a more robust way to collect logs.
		var (
			logName  = "system.log"
			logPath  = fmt.Sprintf("/tmp/%s-%s", getTimestamp(), logName)
			destPath = fmt.Sprintf("%s/%s-%s", *resultsDir, host, logName)
		)
		glog.Infof("Test failed unexpectedly. Attempting to retreiving system logs (only works for nodes with journald)")
		// Try getting the system logs from journald and store it to a file.
		// Don't reuse the original test directory on the remote host because
		// it could've be been removed if the node was rebooted.
		if output, err := SSH(host, "sh", "-c", fmt.Sprintf("'journalctl --system --all > %s'", logPath)); err == nil {
			glog.Infof("Got the system logs from journald; copying it back...")
			if output, err := runSSHCommand("scp", fmt.Sprintf("%s:%s", GetHostnameOrIp(host), logPath), destPath); err != nil {
				glog.Infof("Failed to copy the log: err: %v, output: %q", err, output)
			}
		} else {
			glog.Infof("Failed to run journactl (normal if it doesn't exist on the node): %v, output: %q", err, output)
		}
	}

	glog.Infof("Copying test artifacts from %s", host)
	scpErr := getTestArtifacts(host, workspace)
	if scpErr != nil {
		aggErrs = append(aggErrs, scpErr)
	}

	return output, len(aggErrs) == 0, utilerrors.NewAggregate(aggErrs)
}

// timestampFormat is the timestamp format used in the node e2e directory name.
const timestampFormat = "20060102T150405"

func getTimestamp() string {
	return fmt.Sprintf(time.Now().Format(timestampFormat))
}

func getTestArtifacts(host, testDir string) error {
	logPath := filepath.Join(*resultsDir, host)
	if err := os.MkdirAll(logPath, 0755); err != nil {
		return fmt.Errorf("failed to create log directory %q: %v", logPath, err)
	}
	// Copy logs to artifacts/hostname
	_, err := runSSHCommand("scp", "-r", fmt.Sprintf("%s:%s/results/*.log", GetHostnameOrIp(host), testDir), logPath)
	if err != nil {
		return err
	}
	// Copy junit to the top of artifacts
	_, err = runSSHCommand("scp", fmt.Sprintf("%s:%s/results/junit*", GetHostnameOrIp(host), testDir), *resultsDir)
	if err != nil {
		return err
	}
	return nil
}

// WriteLog is a temporary function to make it possible to write log
// in the runner. This is used to collect serial console log.
// TODO(random-liu): Use the log-dump script in cluster e2e.
func WriteLog(host, filename, content string) error {
	f, err := os.Create(filepath.Join(*resultsDir, host, filename))
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString(content)
	return err
}
