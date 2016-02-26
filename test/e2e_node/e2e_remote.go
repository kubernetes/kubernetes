/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

	"github.com/golang/glog"
)

var sshOptions = flag.String("ssh-options", "", "Commandline options passed to ssh.")
var sshEnv = flag.String("ssh-env", "", "Use predefined ssh options for environment.  Options: gce")

var sshOptionsMap map[string]string

const archiveName = "e2e_node_test.tar.gz"

func init() {
	usr, err := user.Current()
	if err != nil {
		glog.Fatal(err)
	}
	sshOptionsMap = map[string]string{
		"gce": fmt.Sprintf("-i %s/.ssh/google_compute_engine -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o CheckHostIP=no -o StrictHostKeyChecking=no", usr.HomeDir),
	}
}

// CreateTestArchive builds the local source and creates a tar archive e2e_node_test.tar.gz containing
// the binaries k8s required for node e2e tests
func CreateTestArchive() string {
	// Build the executables
	buildGo()

	// Build the e2e tests into an executable
	glog.Infof("Building ginkgo k8s test binaries...")
	testDir, err := getK8sNodeTestDir()
	if err != nil {
		glog.Fatalf("Failed to locate test/e2e_node directory %v.", err)
	}
	out, err := exec.Command("ginkgo", "build", testDir).CombinedOutput()
	if err != nil {
		glog.Fatalf("Failed to build e2e tests under %s %v.  Output:\n%s", testDir, err, out)
	}
	ginkgoTest := filepath.Join(testDir, "e2e_node.test")
	if _, err := os.Stat(ginkgoTest); err != nil {
		glog.Fatalf("Failed to locate test binary %s", ginkgoTest)
	}
	defer os.Remove(ginkgoTest)

	// Make sure we can find the newly built binaries
	buildOutputDir, err := getK8sBuildOutputDir()
	if err != nil {
		glog.Fatalf("Failed to locate kubernetes build output directory %v", err)
	}
	kubelet := filepath.Join(buildOutputDir, "kubelet")
	if _, err := os.Stat(kubelet); err != nil {
		glog.Fatalf("Failed to locate binary %s", kubelet)
	}
	apiserver := filepath.Join(buildOutputDir, "kube-apiserver")
	if _, err := os.Stat(apiserver); err != nil {
		glog.Fatalf("Failed to locate binary %s", apiserver)
	}

	glog.Infof("Building archive...")
	tardir, err := ioutil.TempDir("", "node-e2e-archive")
	if err != nil {
		glog.Fatalf("Failed to create temporary directory %v.", err)
	}
	defer os.RemoveAll(tardir)

	// Copy binaries
	out, err = exec.Command("cp", ginkgoTest, filepath.Join(tardir, "e2e_node.test")).CombinedOutput()
	if err != nil {
		glog.Fatalf("Failed to copy e2e_node.test %v.", err)
	}
	out, err = exec.Command("cp", kubelet, filepath.Join(tardir, "kubelet")).CombinedOutput()
	if err != nil {
		glog.Fatalf("Failed to copy kubelet %v.", err)
	}
	out, err = exec.Command("cp", apiserver, filepath.Join(tardir, "kube-apiserver")).CombinedOutput()
	if err != nil {
		glog.Fatalf("Failed to copy kube-apiserver %v.", err)
	}

	// Build the tar
	out, err = exec.Command("tar", "-zcvf", archiveName, "-C", tardir, ".").CombinedOutput()
	if err != nil {
		glog.Fatalf("Failed to build tar %v.  Output:\n%s", err, out)
	}

	dir, err := os.Getwd()
	if err != nil {
		glog.Fatalf("Failed to get working directory %v.", err)
	}
	return filepath.Join(dir, archiveName)
}

// RunRemote copies the archive file to a /tmp file on host, unpacks it, and runs the e2e_node.test
func RunRemote(archive string, host string) (string, error) {
	// Create the temp staging directory
	tmp := fmt.Sprintf("/tmp/gcloud-e2e-%d", rand.Int31())
	_, err := RunSshCommand("ssh", host, "--", "mkdir", tmp)
	if err != nil {
		return "", err
	}
	defer func() {
		output, err := RunSshCommand("ssh", host, "--", "rm", "-rf", tmp)
		if err != nil {
			glog.Errorf("Failed to cleanup tmp directory %s on host %v.  Output:\n%s", tmp, err, output)
		}
	}()

	// Copy the archive to the staging directory
	_, err = RunSshCommand("scp", archive, fmt.Sprintf("%s:%s/", host, tmp))
	if err != nil {
		return "", err
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
	RunSshCommand("ssh", host, "--", "sh", "-c", cmd)

	// Extract the archive and run the tests
	cmd = getSshCommand(" && ",
		fmt.Sprintf("cd %s", tmp),
		fmt.Sprintf("tar -xzvf ./%s", archiveName),
		fmt.Sprintf("./e2e_node.test --logtostderr --v 2 --build-services=false --node-name=%s", host),
	)
	output, err := RunSshCommand("ssh", host, "--", "sh", "-c", cmd)
	if err != nil {
		return "", err
	}

	return output, nil
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
		return fmt.Sprintf("%s", output), fmt.Errorf("Command [%s %s] failed with error: %v and output:\n%s", cmd, strings.Join(args, " "), err, output)
	}
	return fmt.Sprintf("%s", output), nil
}
