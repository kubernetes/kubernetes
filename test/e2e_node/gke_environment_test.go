/*
Copyright 2017 The Kubernetes Authors.

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
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"

	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

// checkProcess checks whether there's a process whose command line contains
// the specified pattern and whose parent process id is ppid using the
// pre-built information in cmdToProcessMap.
func checkProcess(pattern string, ppid int, cmdToProcessMap map[string][]process) error {
	for cmd, processes := range cmdToProcessMap {
		if !strings.Contains(cmd, pattern) {
			continue
		}
		for _, p := range processes {
			if p.ppid == ppid {
				return nil
			}
		}
	}
	return fmt.Errorf("failed to find the process whose cmdline contains %q with ppid = %d", pattern, ppid)
}

// checkIPTables checks whether the functionality required by kube-proxy works
// in iptables.
func checkIPTables() (err error) {
	cmds := [][]string{
		{"iptables", "-N", "KUBE-PORTALS-HOST", "-t", "nat"},
		{"iptables", "-I", "OUTPUT", "-t", "nat", "-m", "comment", "--comment", "ClusterIPs", "-j", "KUBE-PORTALS-HOST"},
		{"iptables", "-A", "KUBE-PORTALS-HOST", "-t", "nat", "-m", "comment", "--comment", "test-1:", "-p", "tcp", "-m", "tcp", "--dport", "443", "-d", "10.0.0.1/32", "-j", "DNAT", "--to-destination", "10.240.0.1:11111"},
		{"iptables", "-C", "KUBE-PORTALS-HOST", "-t", "nat", "-m", "comment", "--comment", "test-1:", "-p", "tcp", "-m", "tcp", "--dport", "443", "-d", "10.0.0.1/32", "-j", "DNAT", "--to-destination", "10.240.0.1:11111"},
		{"iptables", "-A", "KUBE-PORTALS-HOST", "-t", "nat", "-m", "comment", "--comment", "test-2:", "-p", "tcp", "-m", "tcp", "--dport", "80", "-d", "10.0.0.1/32", "-j", "REDIRECT", "--to-ports", "22222"},
		{"iptables", "-C", "KUBE-PORTALS-HOST", "-t", "nat", "-m", "comment", "--comment", "test-2:", "-p", "tcp", "-m", "tcp", "--dport", "80", "-d", "10.0.0.1/32", "-j", "REDIRECT", "--to-ports", "22222"},
	}
	cleanupCmds := [][]string{
		{"iptables", "-F", "KUBE-PORTALS-HOST", "-t", "nat"},
		{"iptables", "-D", "OUTPUT", "-t", "nat", "-m", "comment", "--comment", "ClusterIPs", "-j", "KUBE-PORTALS-HOST"},
		{"iptables", "-X", "KUBE-PORTALS-HOST", "-t", "nat"},
	}
	defer func() {
		for _, cmd := range cleanupCmds {
			if _, cleanupErr := runCommand(cmd...); cleanupErr != nil && err == nil {
				err = cleanupErr
				return
			}
		}
	}()
	for _, cmd := range cmds {
		if _, err := runCommand(cmd...); err != nil {
			return err
		}
	}
	return
}

// checkPublicGCR checks the access to the public Google Container Registry by
// pulling the busybox image.
func checkPublicGCR() error {
	const image = "gcr.io/google-containers/busybox"
	output, err := runCommand("docker", "images", "-q", image)
	if len(output) != 0 {
		if _, err := runCommand("docker", "rmi", "-f", image); err != nil {
			return err
		}
	}
	output, err = runCommand("docker", "pull", image)
	if len(output) == 0 {
		return fmt.Errorf("failed to pull %s", image)
	}
	if _, err = runCommand("docker", "rmi", "-f", image); err != nil {
		return err
	}
	return nil
}

// checkDockerConfig runs docker's check-config.sh script and ensures that all
// expected kernel configs are enabled.
func checkDockerConfig() error {
	var (
		re   = regexp.MustCompile("\x1B\\[([0-9]{1,2}(;[0-9]{1,2})?)?[mGK]")
		bins = []string{
			"/usr/share/docker.io/contrib/check-config.sh",
			"/usr/share/docker/contrib/check-config.sh",
		}
		whitelist = map[string]bool{
			"CONFIG_MEMCG_SWAP_ENABLED": true,
			"CONFIG_RT_GROUP_SCHED":     true,
			"CONFIG_EXT3_FS":            true,
			"CONFIG_EXT3_FS_XATTR":      true,
			"CONFIG_EXT3_FS_POSIX_ACL":  true,
			"CONFIG_EXT3_FS_SECURITY":   true,
			"/dev/zfs":                  true,
			"zfs command":               true,
			"zpool command":             true,
		}
		missing = map[string]bool{}
	)
	for _, bin := range bins {
		if _, err := os.Stat(bin); os.IsNotExist(err) {
			continue
		}
		output, err := runCommand(bin)
		if err != nil {
			return err
		}
		for _, line := range strings.Split(output, "\n") {
			if !strings.Contains(line, "missing") {
				continue
			}
			line = re.ReplaceAllString(line, "")
			fields := strings.Split(line, ":")
			if len(fields) != 2 {
				continue
			}
			key := strings.TrimFunc(fields[0], func(c rune) bool {
				return c == ' ' || c == '-'
			})
			if _, found := whitelist[key]; !found {
				missing[key] = true
			}
		}
		if len(missing) != 0 {
			return fmt.Errorf("missing docker config: %v", missing)
		}
		break
	}
	return nil
}

// checkDockerNetworkClient checks client networking by pinging an external IP
// address from a container.
func checkDockerNetworkClient() error {
	const imageName = "gcr.io/google-containers/busybox"
	output, err := runCommand("docker", "run", "--rm", imageName, "sh", "-c", "ping -w 5 -q google.com")
	if err != nil {
		return err
	}
	if !strings.Contains(output, `0% packet loss`) {
		return fmt.Errorf("failed to ping from container: %s", output)
	}
	return nil
}

// checkDockerNetworkServer checks server networking by running an echo server
// within a container and accessing it from outside.
func checkDockerNetworkServer() error {
	const (
		imageName     = "gcr.io/google-containers/nginx:1.7.9"
		hostAddr      = "127.0.0.1"
		hostPort      = "8088"
		containerPort = "80"
		containerID   = "nginx"
		message       = "Welcome to nginx!"
	)
	var (
		portMapping = fmt.Sprintf("%s:%s", hostPort, containerPort)
		host        = fmt.Sprintf("http://%s:%s", hostAddr, hostPort)
	)
	runCommand("docker", "rm", "-f", containerID)
	if _, err := runCommand("docker", "run", "-d", "--name", containerID, "-p", portMapping, imageName); err != nil {
		return err
	}
	output, err := runCommand("curl", host)
	if err != nil {
		return err
	}
	if !strings.Contains(output, message) {
		return fmt.Errorf("failed to connect to container")
	}
	// Clean up
	if _, err = runCommand("docker", "rm", "-f", containerID); err != nil {
		return err
	}
	if _, err = runCommand("docker", "rmi", imageName); err != nil {
		return err
	}
	return nil
}

// checkDockerAppArmor checks whether AppArmor is enabled and has the
// "docker-default" profile.
func checkDockerAppArmor() error {
	buf, err := ioutil.ReadFile("/sys/module/apparmor/parameters/enabled")
	if err != nil {
		return err
	}
	if string(buf) != "Y\n" {
		return fmt.Errorf("apparmor module is not loaded")
	}

	// Checks that the "docker-default" profile is loaded and enforced.
	buf, err = ioutil.ReadFile("/sys/kernel/security/apparmor/profiles")
	if err != nil {
		return err
	}
	if !strings.Contains(string(buf), "docker-default (enforce)") {
		return fmt.Errorf("'docker-default' profile is not loaded and enforced")
	}

	// Checks that the `apparmor_parser` binary is present.
	_, err = exec.LookPath("apparmor_parser")
	if err != nil {
		return fmt.Errorf("'apparmor_parser' is not in directories named by the PATH env")
	}
	return nil
}

// checkDockerSeccomp checks whether the Docker supports seccomp.
func checkDockerSeccomp() error {
	const (
		seccompProfileFileName = "/tmp/no_mkdir.json"
		seccompProfile         = `{
        "defaultAction": "SCMP_ACT_ALLOW",
        "syscalls": [
            {
                "name": "mkdir",
                "action": "SCMP_ACT_ERRNO"
            }
        ]}`
		image = "gcr.io/google-appengine/debian8:2017-06-07-171918"
	)
	if err := ioutil.WriteFile(seccompProfileFileName, []byte(seccompProfile), 0644); err != nil {
		return err
	}
	// Starts a container with no seccomp profile and ensures that unshare
	// succeeds.
	_, err := runCommand("docker", "run", "--rm", "-i", "--security-opt", "seccomp=unconfined", image, "unshare", "-r", "whoami")
	if err != nil {
		return err
	}
	// Starts a container with the default seccomp profile and ensures that
	// unshare (a blacklisted system call in the default profile) fails.
	cmd := []string{"docker", "run", "--rm", "-i", image, "unshare", "-r", "whoami"}
	_, err = runCommand(cmd...)
	if err == nil {
		return fmt.Errorf("%q did not fail as expected", strings.Join(cmd, " "))
	}
	// Starts a container with a custom seccomp profile that blacklists mkdir
	// and ensures that unshare succeeds.
	_, err = runCommand("docker", "run", "--rm", "-i", "--security-opt", fmt.Sprintf("seccomp=%s", seccompProfileFileName), image, "unshare", "-r", "whoami")
	if err != nil {
		return err
	}
	// Starts a container with a custom seccomp profile that blacklists mkdir
	// and ensures that mkdir fails.
	cmd = []string{"docker", "run", "--rm", "-i", "--security-opt", fmt.Sprintf("seccomp=%s", seccompProfileFileName), image, "mkdir", "-p", "/tmp/foo"}
	_, err = runCommand(cmd...)
	if err == nil {
		return fmt.Errorf("%q did not fail as expected", strings.Join(cmd, " "))
	}
	return nil
}

// checkDockerStorageDriver checks whether the current storage driver used by
// Docker is overlay.
func checkDockerStorageDriver() error {
	output, err := runCommand("docker", "info")
	if err != nil {
		return err
	}
	for _, line := range strings.Split(string(output), "\n") {
		if !strings.Contains(line, "Storage Driver:") {
			continue
		}
		if !strings.Contains(line, "overlay") {
			return fmt.Errorf("storage driver is not 'overlay': %s", line)
		}
		return nil
	}
	return fmt.Errorf("failed to find storage driver")
}

var _ = framework.KubeDescribe("GKE system requirements [Conformance] [Feature:GKEEnv]", func() {
	BeforeEach(func() {
		framework.RunIfSystemSpecNameIs("gke")
	})

	It("The required processes should be running", func() {
		cmdToProcessMap, err := getCmdToProcessMap()
		framework.ExpectNoError(err)
		for _, p := range []struct {
			cmd  string
			ppid int
		}{
			{"google_accounts_daemon", 1},
			{"google_clock_skew_daemon", 1},
			{"google_ip_forwarding_daemon", 1},
		} {
			framework.ExpectNoError(checkProcess(p.cmd, p.ppid, cmdToProcessMap))
		}
	})
	It("The iptable rules should work (required by kube-proxy)", func() {
		framework.ExpectNoError(checkIPTables())
	})
	It("The GCR is accessible", func() {
		framework.ExpectNoError(checkPublicGCR())
	})
	It("The docker configuration validation should pass", func() {
		framework.RunIfContainerRuntimeIs("docker")
		framework.ExpectNoError(checkDockerConfig())
	})
	It("The docker container network should work", func() {
		framework.RunIfContainerRuntimeIs("docker")
		framework.ExpectNoError(checkDockerNetworkServer())
		framework.ExpectNoError(checkDockerNetworkClient())
	})
	It("The docker daemon should support AppArmor and seccomp", func() {
		framework.RunIfContainerRuntimeIs("docker")
		framework.ExpectNoError(checkDockerAppArmor())
		framework.ExpectNoError(checkDockerSeccomp())
	})
	It("The docker storage driver should work", func() {
		framework.Skipf("GKE does not currently require overlay")
		framework.ExpectNoError(checkDockerStorageDriver())
	})
})

// runCommand runs the cmd and returns the combined stdout and stderr, or an
// error if the command failed.
func runCommand(cmd ...string) (string, error) {
	output, err := exec.Command(cmd[0], cmd[1:]...).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to run %q: %s (%s)", strings.Join(cmd, " "), err, output)
	}
	return string(output), nil
}

// getPPID returns the PPID for the pid.
func getPPID(pid int) (int, error) {
	statusFile := "/proc/" + strconv.Itoa(pid) + "/status"
	content, err := ioutil.ReadFile(statusFile)
	if err != nil {
		return 0, err
	}
	for _, line := range strings.Split(string(content), "\n") {
		if !strings.HasPrefix(line, "PPid:") {
			continue
		}
		s := strings.TrimSpace(strings.TrimPrefix(line, "PPid:"))
		ppid, err := strconv.Atoi(s)
		if err != nil {
			return 0, err
		}
		return ppid, nil
	}
	return 0, fmt.Errorf("no PPid in %s", statusFile)
}

// process contains a process ID and its parent's process ID.
type process struct {
	pid  int
	ppid int
}

// getCmdToProcessMap returns a mapping from the process command line to its
// process ids.
func getCmdToProcessMap() (map[string][]process, error) {
	root, err := os.Open("/proc")
	if err != nil {
		return nil, err
	}
	defer root.Close()
	dirs, err := root.Readdirnames(0)
	if err != nil {
		return nil, err
	}
	result := make(map[string][]process)
	for _, dir := range dirs {
		pid, err := strconv.Atoi(dir)
		if err != nil {
			continue
		}
		ppid, err := getPPID(pid)
		if err != nil {
			continue
		}
		content, err := ioutil.ReadFile("/proc/" + dir + "/cmdline")
		if err != nil || len(content) == 0 {
			continue
		}
		cmd := string(bytes.Replace(content, []byte("\x00"), []byte(" "), -1))
		result[cmd] = append(result[cmd], process{pid, ppid})
	}
	return result, nil
}
