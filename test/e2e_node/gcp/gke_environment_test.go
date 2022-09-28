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

package gcp

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"

	"github.com/onsi/ginkgo/v2"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
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
	const image = "registry.k8s.io/busybox"
	output, err := runCommand("docker", "images", "-q", image)
	if err != nil {
		return err
	}

	if len(output) != 0 {
		if _, err := runCommand("docker", "rmi", "-f", image); err != nil {
			return err
		}
	}
	output, err = runCommand("docker", "pull", image)
	if err != nil {
		return err
	}

	if len(output) == 0 {
		return fmt.Errorf("failed to pull %s", image)
	}
	if _, err := runCommand("docker", "rmi", "-f", image); err != nil {
		return err
	}
	return nil
}

var _ = SIGDescribe("GKE system requirements [NodeConformance][Feature:GKEEnv][NodeFeature:GKEEnv]", func() {
	ginkgo.BeforeEach(func() {
		e2eskipper.RunIfSystemSpecNameIs("gke")
	})

	ginkgo.It("The required processes should be running", func() {
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
	ginkgo.It("The iptable rules should work (required by kube-proxy)", func() {
		framework.ExpectNoError(checkIPTables())
	})
	ginkgo.It("The GCR is accessible", func() {
		framework.ExpectNoError(checkPublicGCR())
	})
})

// getPPID returns the PPID for the pid.
func getPPID(pid int) (int, error) {
	statusFile := "/proc/" + strconv.Itoa(pid) + "/status"
	content, err := os.ReadFile(statusFile)
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
		content, err := os.ReadFile("/proc/" + dir + "/cmdline")
		if err != nil || len(content) == 0 {
			continue
		}
		cmd := string(bytes.Replace(content, []byte("\x00"), []byte(" "), -1))
		result[cmd] = append(result[cmd], process{pid, ppid})
	}
	return result, nil
}

// runCommand runs the cmd and returns the combined stdout and stderr, or an
// error if the command failed.
func runCommand(cmd ...string) (string, error) {
	output, err := exec.Command(cmd[0], cmd[1:]...).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to run %q: %s (%s)", strings.Join(cmd, " "), err, output)
	}
	return string(output), nil
}
