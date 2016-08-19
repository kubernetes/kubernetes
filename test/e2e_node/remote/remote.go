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
	"math/rand"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/golang/glog"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/test/e2e_node/build"
)

var sshOptions = flag.String("ssh-options", "", "Commandline options passed to ssh.")
var sshEnv = flag.String("ssh-env", "", "Use predefined ssh options for environment.  Options: gce")
var testTimeoutSeconds = flag.Duration("test-timeout", 45*time.Minute, "How long (in golang duration format) to wait for ginkgo tests to complete.")
var resultsDir = flag.String("results-dir", "/tmp/", "Directory to scp test results to.")

var sshOptionsMap map[string]string

const (
	archiveName  = "e2e_node_test.tar.gz"
	CNIRelease   = "07a8a28637e97b22eb8dfe710eeae1344f69d16e"
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
	if err := build.BuildGo(); err != nil {
		return "", fmt.Errorf("failed to build the depedencies: %v", err)
	}

	// Make sure we can find the newly built binaries
	buildOutputDir, err := build.GetK8sBuildOutputDir()
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

	// Build the tar
	out, err := exec.Command("tar", "-zcvf", archiveName, "-C", tardir, ".").CombinedOutput()
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
func RunRemote(archive string, host string, cleanup bool, junitFilePrefix string, setupNode bool, testArgs string, ginkgoFlags string) (string, bool, error) {
	if setupNode {
		uname, err := user.Current()
		if err != nil {
			return "", false, fmt.Errorf("could not find username: %v", err)
		}
		output, err := Ssh(host, "usermod", "-a", "-G", "docker", uname.Username)
		if err != nil {
			return "", false, fmt.Errorf("instance %s not running docker daemon - Command failed: %s", host, output)
		}
	}

	// Create the temp staging directory
	glog.Infof("Staging test binaries on %s", host)
	tmp := fmt.Sprintf("/tmp/gcloud-e2e-%d", rand.Int31())
	// Do not sudo here, so that we can use scp to copy test archive to the directdory.
	_, err := SshNoSudo(host, "mkdir", tmp)
	if err != nil {
		// Exit failure with the error
		return "", false, err
	}
	if cleanup {
		defer func() {
			output, err := Ssh(host, "rm", "-rf", tmp)
			if err != nil {
				glog.Errorf("failed to cleanup tmp directory %s on host %v.  Output:\n%s", tmp, err, output)
			}
		}()
	}

	// Install the cni plugin.
	cniPath := filepath.Join(tmp, CNIDirectory)
	cmd := getSshCommand(" ; ",
		fmt.Sprintf("mkdir -p %s", cniPath),
		fmt.Sprintf("wget -O - %s | tar -xz -C %s", CNIURL, cniPath),
	)
	if _, err := Ssh(host, "sh", "-c", cmd); err != nil {
		// Exit failure with the error
		return "", false, err
	}

	// Copy the archive to the staging directory
	_, err = runSshCommand("scp", archive, fmt.Sprintf("%s:%s/", GetHostnameOrIp(host), tmp))
	if err != nil {
		// Exit failure with the error
		return "", false, err
	}

	// Kill any running node processes
	cmd = getSshCommand(" ; ",
		"pkill kubelet",
		"pkill kube-apiserver",
		"pkill etcd",
	)
	// No need to log an error if pkill fails since pkill will fail if the commands are not running.
	// If we are unable to stop existing running k8s processes, we should see messages in the kubelet/apiserver/etcd
	// logs about failing to bind the required ports.
	glog.Infof("Killing any existing node processes on %s", host)
	Ssh(host, "sh", "-c", cmd)

	// Extract the archive
	cmd = getSshCommand(" && ",
		fmt.Sprintf("cd %s", tmp),
		fmt.Sprintf("tar -xzvf ./%s", archiveName),
	)
	glog.Infof("Extracting tar on %s", host)
	output, err := Ssh(host, "sh", "-c", cmd)
	if err != nil {
		// Exit failure with the error
		return "", false, err
	}
	// Run the tests
	cmd = getSshCommand(" && ",
		fmt.Sprintf("cd %s", tmp),
		fmt.Sprintf("timeout -k 30s %fs ./ginkgo %s ./e2e_node.test -- --logtostderr --v 4 --node-name=%s --report-dir=%s/results --report-prefix=%s %s",
			testTimeoutSeconds.Seconds(), ginkgoFlags, host, tmp, junitFilePrefix, testArgs),
	)
	aggErrs := []error{}

	glog.Infof("Starting tests on %s", host)
	output, err = Ssh(host, "sh", "-c", cmd)

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
	_, err := runSshCommand("scp", "-r", fmt.Sprintf("%s:%s/results/", GetHostnameOrIp(host), testDir), fmt.Sprintf("%s/%s", *resultsDir, host))
	if err != nil {
		return err
	}

	// Copy junit to the top of artifacts
	_, err = runSshCommand("scp", fmt.Sprintf("%s:%s/results/junit*", GetHostnameOrIp(host), testDir), fmt.Sprintf("%s/", *resultsDir))
	if err != nil {
		return err
	}
	return nil
}

// getSshCommand handles proper quoting so that multiple commands are executed in the same shell over ssh
func getSshCommand(sep string, args ...string) string {
	return fmt.Sprintf("'%s'", strings.Join(args, sep))
}

// Ssh executes ssh command with runSshCommand as root. The `sudo` makes sure that all commands
// are executed by root, so that there won't be permission mismatch between different commands.
func Ssh(host string, cmd ...string) (string, error) {
	return runSshCommand("ssh", append([]string{GetHostnameOrIp(host), "--", "sudo"}, cmd...)...)
}

// SsshNoSudo executes ssh command with runSshCommand as normal user. Sometimes we need this,
// for example creating a directory that we'll copy files there with scp.
func SshNoSudo(host string, cmd ...string) (string, error) {
	return runSshCommand("ssh", append([]string{GetHostnameOrIp(host), "--"}, cmd...)...)
}

// runSshCommand executes the ssh or scp command, adding the flag provided --ssh-options
func runSshCommand(cmd string, args ...string) (string, error) {
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
