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
	"net/http"
	"os"
	"os/exec"
	"path"
	"reflect"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/golang/glog"
)

var serverStartTimeout = flag.Duration("server-start-timeout", time.Second*120, "Time to wait for each server to become healthy.")
var reportDir = flag.String("report-dir", "", "Path to the directory where the JUnit XML reports should be saved. Default is empty, which doesn't generate these reports.")

type e2eService struct {
	etcdCmd             *killCmd
	etcdDataDir         string
	apiServerCmd        *killCmd
	kubeletCmd          *killCmd
	kubeletStaticPodDir string
	nodeName            string
	logFiles            map[string]logFileData
}

type logFileData struct {
	files             []string
	journalctlCommand []string
}

const (
	// This is consistent with the level used in a cluster e2e test.
	LOG_VERBOSITY_LEVEL = "4"
)

func newE2eService(nodeName string) *e2eService {
	// Special log files that need to be collected for additional debugging.
	var logFiles = map[string]logFileData{
		"kern.log":   {[]string{"/var/log/kern.log"}, []string{"-k"}},
		"docker.log": {[]string{"/var/log/docker.log", "/var/log/upstart/docker.log"}, []string{"-u", "docker"}},
	}

	return &e2eService{nodeName: nodeName, logFiles: logFiles}
}

func (es *e2eService) start() error {
	if _, err := getK8sBin("kubelet"); err != nil {
		return err
	}
	if _, err := getK8sBin("kube-apiserver"); err != nil {
		return err
	}

	cmd, err := es.startEtcd()
	if err != nil {
		return err
	}
	es.etcdCmd = cmd

	cmd, err = es.startApiServer()
	if err != nil {
		return err
	}
	es.apiServerCmd = cmd

	cmd, err = es.startKubeletServer()
	if err != nil {
		return err
	}
	es.kubeletCmd = cmd

	return nil
}

// Get logs of interest either via journalctl or by creating sym links.
// Since we scp files from the remote directory, symlinks will be treated as normal files and file contents will be copied over.
func (es *e2eService) getLogFiles() {
	// Nothing to do if report dir is not specified.
	if *reportDir == "" {
		return
	}
	journaldFound := isJournaldAvailable()
	for targetFileName, logFileData := range es.logFiles {
		targetLink := path.Join(*reportDir, targetFileName)
		if journaldFound {
			// Skip log files that do not have an equivalent in journald based machines.
			if len(logFileData.journalctlCommand) == 0 {
				continue
			}
			out, err := exec.Command("sudo", append([]string{"journalctl"}, logFileData.journalctlCommand...)...).CombinedOutput()
			if err != nil {
				glog.Errorf("failed to get %q from journald: %v, %v", targetFileName, string(out), err)
			} else {
				if err = ioutil.WriteFile(targetLink, out, 0755); err != nil {
					glog.Errorf("failed to write logs to %q: %v", targetLink, err)
				}
			}
			continue
		}
		for _, file := range logFileData.files {
			if _, err := os.Stat(file); err != nil {
				// Expected file not found on this distro.
				continue
			}
			if err := copyLogFile(file, targetLink); err != nil {
				glog.Error(err)
			} else {
				break
			}
		}
	}
}

func copyLogFile(src, target string) error {
	// If not a journald based distro, then just symlink files.
	if out, err := exec.Command("sudo", "cp", src, target).CombinedOutput(); err != nil {
		return fmt.Errorf("failed to copy %q to %q: %v, %v", src, target, out, err)
	}
	if out, err := exec.Command("sudo", "chmod", "a+r", target).CombinedOutput(); err != nil {
		return fmt.Errorf("failed to make log file %q world readable: %v, %v", target, out, err)
	}
	return nil
}

func isJournaldAvailable() bool {
	_, err := exec.LookPath("journalctl")
	return err == nil
}

func (es *e2eService) stop() {
	if err := es.stopService(es.kubeletCmd); err != nil {
		glog.Errorf("Failed to stop kubelet: %v", err)
	}
	if es.kubeletStaticPodDir != "" {
		err := os.RemoveAll(es.kubeletStaticPodDir)
		if err != nil {
			glog.Errorf("Failed to delete kubelet static pod directory %s.\n%v", es.kubeletStaticPodDir, err)
		}
	}
	if err := es.stopService(es.apiServerCmd); err != nil {
		glog.Errorf("Failed to stop kube-apiserver: %v", err)
	}
	if err := es.stopService(es.etcdCmd); err != nil {
		glog.Errorf("Failed to stop etcd: %v", err)
	}
	if es.etcdDataDir != "" {
		err := os.RemoveAll(es.etcdDataDir)
		if err != nil {
			glog.Errorf("Failed to delete etcd data directory %s.\n%v", es.etcdDataDir, err)
		}
	}
}

func (es *e2eService) startEtcd() (*killCmd, error) {
	dataDir, err := ioutil.TempDir("", "node-e2e")
	if err != nil {
		return nil, err
	}
	es.etcdDataDir = dataDir
	cmd := exec.Command("etcd")
	// Execute etcd in the data directory instead of using --data-dir because the flag sometimes requires additional
	// configuration (e.g. --name in version 0.4.9)
	cmd.Dir = es.etcdDataDir
	hcc := newHealthCheckCommand(
		"http://127.0.0.1:4001/v2/keys/", // Trailing slash is required,
		cmd,
		"etcd.log")
	return &killCmd{name: "etcd", cmd: cmd}, es.startServer(hcc)
}

func (es *e2eService) startApiServer() (*killCmd, error) {
	cmd := exec.Command("sudo", getApiServerBin(),
		"--etcd-servers", "http://127.0.0.1:4001",
		"--insecure-bind-address", "0.0.0.0",
		"--service-cluster-ip-range", "10.0.0.1/24",
		"--kubelet-port", "10250",
		"--allow-privileged", "true",
		"--v", LOG_VERBOSITY_LEVEL, "--logtostderr",
	)
	hcc := newHealthCheckCommand(
		"http://127.0.0.1:8080/healthz",
		cmd,
		"kube-apiserver.log")
	return &killCmd{name: "kube-apiserver", cmd: cmd}, es.startServer(hcc)
}

func (es *e2eService) startKubeletServer() (*killCmd, error) {
	dataDir, err := ioutil.TempDir("", "node-e2e-pod")
	if err != nil {
		return nil, err
	}
	es.kubeletStaticPodDir = dataDir
	var killOverride *exec.Cmd
	cmdArgs := []string{}
	if systemdRun, err := exec.LookPath("systemd-run"); err == nil {
		// On systemd services, detection of a service / unit works reliably while
		// detection of a process started from an ssh session does not work.
		// Since kubelet will typically be run as a service it also makes more
		// sense to test it that way
		unitName := fmt.Sprintf("kubelet-%d.service", rand.Int31())
		cmdArgs = append(cmdArgs, systemdRun, "--unit="+unitName, getKubeletServerBin())
		killOverride = exec.Command("sudo", "systemctl", "kill", unitName)
		es.logFiles["kubelet.log"] = logFileData{
			journalctlCommand: []string{"-u", unitName},
		}
	} else {
		cmdArgs = append(cmdArgs, getKubeletServerBin())
	}
	cmdArgs = append(cmdArgs,
		"--api-servers", "http://127.0.0.1:8080",
		"--address", "0.0.0.0",
		"--port", "10250",
		"--hostname-override", es.nodeName, // Required because hostname is inconsistent across hosts
		"--volume-stats-agg-period", "10s", // Aggregate volumes frequently so tests don't need to wait as long
		"--allow-privileged", "true",
		"--serialize-image-pulls", "false",
		"--config", es.kubeletStaticPodDir,
		"--file-check-frequency", "10s", // Check file frequently so tests won't wait too long
		"--v", LOG_VERBOSITY_LEVEL, "--logtostderr",
	)
	cmd := exec.Command("sudo", cmdArgs...)
	hcc := newHealthCheckCommand(
		"http://127.0.0.1:10255/healthz",
		cmd,
		"kubelet.log")
	return &killCmd{name: "kubelet", cmd: cmd, override: killOverride}, es.startServer(hcc)
}

func (es *e2eService) startServer(cmd *healthCheckCommand) error {
	cmdErrorChan := make(chan error)
	go func() {
		defer close(cmdErrorChan)

		// Create the output filename
		outPath := path.Join(*reportDir, cmd.outputFilename)
		outfile, err := os.Create(outPath)
		if err != nil {
			cmdErrorChan <- fmt.Errorf("Failed to create file %s for `%s` %v.", outPath, cmd, err)
			return
		}
		defer outfile.Close()
		defer outfile.Sync()

		// Set the command to write the output file
		cmd.Cmd.Stdout = outfile
		cmd.Cmd.Stderr = outfile

		// Killing the sudo command should kill the server as well.
		attrs := &syscall.SysProcAttr{}
		// Hack to set linux-only field without build tags.
		deathSigField := reflect.ValueOf(attrs).Elem().FieldByName("Pdeathsig")
		if deathSigField.IsValid() {
			deathSigField.Set(reflect.ValueOf(syscall.SIGKILL))
		} else {
			cmdErrorChan <- fmt.Errorf("Failed to set Pdeathsig field (non-linux build)")
			return
		}
		cmd.Cmd.SysProcAttr = attrs

		// Run the command
		err = cmd.Run()
		if err != nil {
			cmdErrorChan <- fmt.Errorf("%s Failed with error \"%v\".  Output written to: %s", cmd, err, outPath)
			return
		}
	}()

	endTime := time.Now().Add(*serverStartTimeout)
	for endTime.After(time.Now()) {
		select {
		case err := <-cmdErrorChan:
			return err
		case <-time.After(time.Second):
			resp, err := http.Get(cmd.HealthCheckUrl)
			if err == nil && resp.StatusCode == http.StatusOK {
				return nil
			}
		}
	}
	return fmt.Errorf("Timeout waiting for service %s", cmd)
}

func (es *e2eService) stopService(cmd *killCmd) error {
	return cmd.Kill()
}

// killCmd is a struct to kill a given cmd. The cmd member specifies a command
// to find the pid of and attempt to kill.
// If the override field is set, that will be used instead to kill the command.
// name is only used for logging
type killCmd struct {
	name     string
	cmd      *exec.Cmd
	override *exec.Cmd
}

func (k *killCmd) Kill() error {
	if k.override != nil {
		return k.override.Run()
	}
	name := k.name
	cmd := k.cmd

	if cmd == nil || cmd.Process == nil {
		glog.V(2).Infof("%s not running", name)
		return nil
	}
	pid := cmd.Process.Pid
	if pid <= 1 {
		return fmt.Errorf("invalid PID %d for %s", pid, name)
	}

	// Attempt to shut down the process in a friendly manner before forcing it.
	waitChan := make(chan error)
	go func() {
		_, err := cmd.Process.Wait()
		waitChan <- err
		close(waitChan)
	}()

	const timeout = 10 * time.Second
	for _, signal := range []string{"-TERM", "-KILL"} {
		glog.V(2).Infof("Killing process %d (%s) with %s", pid, name, signal)
		_, err := exec.Command("sudo", "kill", signal, strconv.Itoa(pid)).Output()
		if err != nil {
			glog.Errorf("Error signaling process %d (%s) with %s: %v", pid, name, signal, err)
			continue
		}

		select {
		case err := <-waitChan:
			if err != nil {
				return fmt.Errorf("error stopping %s: %v", name, err)
			}
			// Success!
			return nil
		case <-time.After(timeout):
			// Continue.
		}
	}

	return fmt.Errorf("unable to stop %s", name)
}

type healthCheckCommand struct {
	*exec.Cmd
	HealthCheckUrl string
	outputFilename string
}

func newHealthCheckCommand(healthCheckUrl string, cmd *exec.Cmd, filename string) *healthCheckCommand {
	return &healthCheckCommand{
		HealthCheckUrl: healthCheckUrl,
		Cmd:            cmd,
		outputFilename: filename,
	}
}

func (hcc *healthCheckCommand) String() string {
	return fmt.Sprintf("`%s %s` health-check: %s", hcc.Path, strings.Join(hcc.Args, " "), hcc.HealthCheckUrl)
}
