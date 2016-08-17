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

package e2e_node

import (
	"flag"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"strings"
	"sync"

	"github.com/golang/glog"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
)

var sshOptions = flag.String("ssh-options", "", "Commandline options passed to ssh.")
var sshEnv = flag.String("ssh-env", "", "Use predefined ssh options for environment.  Options: gce")
var testTimeoutSeconds = flag.Int("test-timeout", 45*60, "How long (in seconds) to wait for ginkgo tests to complete.")
var resultsDir = flag.String("results-dir", "/tmp/", "Directory to scp test results to.")
var ginkgoFlags = flag.String("ginkgo-flags", "", "Passed to ginkgo to specify additional flags such as --skip=.")

var sshOptionsMap map[string]string

const (
	archiveName  = "e2e_node_test.tar.gz"
	CNIRelease   = "8a936732094c0941e1543ef5d292a1f4fffa1ac5"
	CNIDirectory = "cni"
)

var CNIURL = fmt.Sprintf("https://storage.googleapis.com/kubernetes-release/network-plugins/cni-%s.tar.gz", CNIRelease)

var hostnameIpOverrides = struct {
	sync.RWMutex
	m map[string]string
}{m: make(map[string]string)}

func init() {
	usr, err := user.Current()
	if err != nil {
		glog.Fatal(err)
	}
	sshOptionsMap = map[string]string{
		"gce": fmt.Sprintf("-i %s/.ssh/google_compute_engine -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o CheckHostIP=no -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o LogLevel=ERROR", usr.HomeDir),
	}
}

func AddHostnameIp(hostname, ip string) {
	hostnameIpOverrides.Lock()
	defer hostnameIpOverrides.Unlock()
	hostnameIpOverrides.m[hostname] = ip
}

func GetHostnameOrIp(hostname string) string {
	hostnameIpOverrides.RLock()
	defer hostnameIpOverrides.RUnlock()
	if ip, found := hostnameIpOverrides.m[hostname]; found {
		return ip
	}
	return hostname
}

// CreateTestArchive builds the local source and creates a tar archive e2e_node_test.tar.gz containing
// the binaries k8s required for node e2e tests
func CreateTestArchive() (string, error) {
	// Build the executables
	if err := buildGo(); err != nil {
		return "", fmt.Errorf("failed to build the depedencies: %v", err)
	}

	// Make sure we can find the newly built binaries
	buildOutputDir, err := getK8sBuildOutputDir()
	if err != nil {
		return "", fmt.Errorf("failed to locate kubernetes build output directory %v", err)
	}

	ginkgoTest := filepath.Join(buildOutputDir, "e2e_node.test")
	if _, err := os.Stat(ginkgoTest); err != nil {
		return "", fmt.Errorf("failed to locate test binary %s", ginkgoTest)
	}
	kubelet := filepath.Join(buildOutputDir, "kubelet")
	if _, err := os.Stat(kubelet); err != nil {
		return "", fmt.Errorf("failed to locate binary %s", kubelet)
	}
	apiserver := filepath.Join(buildOutputDir, "kube-apiserver")
	if _, err := os.Stat(apiserver); err != nil {
		return "", fmt.Errorf("failed to locate binary %s", apiserver)
	}
	ginkgo := filepath.Join(buildOutputDir, "ginkgo")
	if _, err := os.Stat(apiserver); err != nil {
		return "", fmt.Errorf("failed to locate binary %s", ginkgo)
	}

	glog.Infof("Building archive...")
	tardir, err := ioutil.TempDir("", "node-e2e-archive")
	if err != nil {
		return "", fmt.Errorf("failed to create temporary directory %v.", err)
	}
	defer os.RemoveAll(tardir)

	// Copy binaries
	out, err := exec.Command("cp", ginkgoTest, filepath.Join(tardir, "e2e_node.test")).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to copy e2e_node.test %v.", err)
	}
	out, err = exec.Command("cp", kubelet, filepath.Join(tardir, "kubelet")).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to copy kubelet %v.", err)
	}
	out, err = exec.Command("cp", apiserver, filepath.Join(tardir, "kube-apiserver")).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to copy kube-apiserver %v.", err)
	}
	out, err = exec.Command("cp", ginkgo, filepath.Join(tardir, "ginkgo")).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to copy ginkgo %v.", err)
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
func RunRemote(archive string, host string, cleanup bool, junitFilePrefix string, setupNode bool, testArgs string) (string, bool, error) {
	if setupNode {
		uname, err := user.Current()
		if err != nil {
			return "", false, fmt.Errorf("could not find username: %v", err)
		}
		output, err := RunSshCommand("ssh", GetHostnameOrIp(host), "--", "sudo", "usermod", "-a", "-G", "docker", uname.Username)
		if err != nil {
			return "", false, fmt.Errorf("instance %s not running docker daemon - Command failed: %s", host, output)
		}
	}

	// Create the temp staging directory
	glog.Infof("Staging test binaries on %s", host)
	tmp := fmt.Sprintf("/tmp/gcloud-e2e-%d", rand.Int31())
	_, err := RunSshCommand("ssh", GetHostnameOrIp(host), "--", "mkdir", tmp)
	if err != nil {
		// Exit failure with the error
		return "", false, err
	}
	if cleanup {
		defer func() {
			output, err := RunSshCommand("ssh", GetHostnameOrIp(host), "--", "rm", "-rf", tmp)
			if err != nil {
				glog.Errorf("failed to cleanup tmp directory %s on host %v.  Output:\n%s", tmp, err, output)
			}
		}()
	}

	// Install the cni plugin.
	cniPath := filepath.Join(tmp, CNIDirectory)
	if _, err := RunSshCommand("ssh", GetHostnameOrIp(host), "--", "sh", "-c",
		getSshCommand(" ; ", fmt.Sprintf("sudo mkdir -p %s", cniPath),
			fmt.Sprintf("sudo wget -O - %s | sudo tar -xz -C %s", CNIURL, cniPath))); err != nil {
		// Exit failure with the error
		return "", false, err
	}

	// Copy the archive to the staging directory
	_, err = RunSshCommand("scp", archive, fmt.Sprintf("%s:%s/", GetHostnameOrIp(host), tmp))
	if err != nil {
		// Exit failure with the error
		return "", false, err
	}

	// Kill any running node processes
	cmd := getSshCommand(" ; ",
		"sudo pkill kubelet",
		"sudo pkill kube-apiserver",
		"sudo pkill etcd",
	)
	// No need to log an error if pkill fails since pkill will fail if the commands are not running.
	// If we are unable to stop existing running k8s processes, we should see messages in the kubelet/apiserver/etcd
	// logs about failing to bind the required ports.
	glog.Infof("Killing any existing node processes on %s", host)
	RunSshCommand("ssh", GetHostnameOrIp(host), "--", "sh", "-c", cmd)

	// Extract the archive
	cmd = getSshCommand(" && ", fmt.Sprintf("cd %s", tmp), fmt.Sprintf("tar -xzvf ./%s", archiveName))
	glog.Infof("Extracting tar on %s", host)
	output, err := RunSshCommand("ssh", GetHostnameOrIp(host), "--", "sh", "-c", cmd)
	if err != nil {
		// Exit failure with the error
		return "", false, err
	}
	// Run the tests
	cmd = getSshCommand(" && ",
		fmt.Sprintf("cd %s", tmp),
		fmt.Sprintf("timeout -k 30s %ds ./ginkgo %s ./e2e_node.test -- --logtostderr --v 2 --build-services=false --stop-services=%t --node-name=%s --report-dir=%s/results --report-prefix=%s %s", *testTimeoutSeconds, *ginkgoFlags, cleanup, host, tmp, junitFilePrefix, testArgs),
	)
	aggErrs := []error{}

	glog.Infof("Starting tests on %s", host)
	output, err = RunSshCommand("ssh", GetHostnameOrIp(host), "--", "sh", "-c", cmd)

	if err != nil {
		aggErrs = append(aggErrs, err)
	}

	glog.Infof("Copying test artifacts from %s", host)
	scpErr := getTestArtifacts(host, tmp)
	if scpErr != nil {
		aggErrs = append(aggErrs, scpErr)
	}

	return output, len(aggErrs) == 0, utilerrors.NewAggregate(aggErrs)
}

func getTestArtifacts(host, testDir string) error {
	_, err := RunSshCommand("scp", "-r", fmt.Sprintf("%s:%s/results/", GetHostnameOrIp(host), testDir), fmt.Sprintf("%s/%s", *resultsDir, host))
	if err != nil {
		return err
	}

	// Copy junit to the top of artifacts
	_, err = RunSshCommand("scp", fmt.Sprintf("%s:%s/results/junit*", GetHostnameOrIp(host), testDir), fmt.Sprintf("%s/", *resultsDir))
	if err != nil {
		return err
	}
	return nil
}

// getSshCommand handles proper quoting so that multiple commands are executed in the same shell over ssh
func getSshCommand(sep string, args ...string) string {
	return fmt.Sprintf("'%s'", strings.Join(args, sep))
}

// runSshCommand executes the ssh or scp command, adding the flag provided --ssh-options
func RunSshCommand(cmd string, args ...string) (string, error) {
	if env, found := sshOptionsMap[*sshEnv]; found {
		args = append(strings.Split(env, " "), args...)
	}
	if *sshOptions != "" {
		args = append(strings.Split(*sshOptions, " "), args...)
	}
	output, err := exec.Command(cmd, args...).CombinedOutput()
	if err != nil {
		return fmt.Sprintf("%s", output), fmt.Errorf("command [%s %s] failed with error: %v and output:\n%s", cmd, strings.Join(args, " "), err, output)
	}
	return fmt.Sprintf("%s", output), nil
}
