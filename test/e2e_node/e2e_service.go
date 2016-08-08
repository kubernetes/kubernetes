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
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/golang/glog"
	"github.com/kardianos/osext"

	"k8s.io/kubernetes/test/e2e/framework"
)

// TODO(random-liu): Move this file to a separate package.
var serverStartTimeout = flag.Duration("server-start-timeout", time.Second*120, "Time to wait for each server to become healthy.")

// E2EServices starts and stops e2e services. The test use it to start and
// stop all dependent e2e services.
type E2EServices struct {
	cmd    *exec.Cmd
	output *os.File
}

func NewE2eServices() *E2EServices {
	return &E2EServices{}
}

// services.log is the combined log of all services
const servicesLogFile = "services.log"

// StartE2eServices start the e2e services in another process, it returns
// when all e2e services are ready.
// We want to statically link e2e services into the test binary, but we don't
// want their glog to pollute the test result. So we run the binary in start-
// services-only mode to start e2e services in another process.
func (e *E2EServices) StartE2EServices() (string, error) {
	// Create the manifest path for kubelet.
	// TODO(random-liu): Remove related logic when we move kubelet starting logic out of the test.
	manifestPath, err := ioutil.TempDir("", "node-e2e-pod")
	if err != nil {
		return "", err
	}
	// Create the log file
	p := path.Join(framework.TestContext.ReportDir, servicesLogFile)
	e.output, err = os.Create(p)
	if err != nil {
		return "", err
	}
	// Start the e2e services in another process.
	testBin, err := osext.Executable()
	if err != nil {
		return manifestPath, fmt.Errorf("can't get current binary: %v", err)
	}
	cmd := exec.Command(testBin,
		"--start-services-only",
		"--server-start-timeout", serverStartTimeout.String(),
		"--report-dir", framework.TestContext.ReportDir,
		// TODO(random-liu): Remove the following flags after we move kubelet starting logic
		// out of the test.
		"--node-name", framework.TestContext.NodeName,
		"--disable-kubenet="+strconv.FormatBool(framework.TestContext.DisableKubenet),
		"--cgroups-per-qos="+strconv.FormatBool(framework.TestContext.CgroupsPerQOS),
		"--manifest-path", manifestPath,
		"--eviction-hard", framework.TestContext.EvictionHard,
	)
	// TODO(random-liu): Redirect output to log file after switching to static link solution.
	cmd.Stdout = e.output
	cmd.Stderr = e.output
	e.cmd = cmd
	if err := e.cmd.Start(); err != nil {
		return manifestPath, err
	}
	if err := readinessCheck(getHealthCheckURLs()); err != nil {
		return manifestPath, err
	}
	return manifestPath, nil
}

// StopE2EServices stop the e2e services.
func (e *E2EServices) StopE2EServices() error {
	defer func() {
		// Cleanup the manifest path for kubelet.
		manifestPath := framework.TestContext.ManifestPath
		if manifestPath != "" {
			err := os.RemoveAll(manifestPath)
			if err != nil {
				glog.Errorf("Failed to delete static pod manifest directory %s.\n%v", manifestPath, err)
			}
		}
		// Close the log file.
		defer e.output.Close()
		defer e.output.Sync()
	}()
	cmd := e.cmd
	if cmd == nil {
		return fmt.Errorf("can't stop e2e services, because `cmd` is nil")
	}
	if cmd.Process == nil {
		glog.V(2).Info("E2E services are not running")
		return nil
	}
	pid := cmd.Process.Pid

	// Attempt to shut down the process in a friendly manner before forcing it.
	waitChan := make(chan error)
	go func() {
		_, err := cmd.Process.Wait()
		waitChan <- err
		close(waitChan)
	}()

	const timeout = 10 * time.Second
	for _, signal := range []os.Signal{os.Interrupt, os.Kill} {
		glog.V(2).Infof("Stopping e2e services (pid=%d) with %s", pid, signal.String())
		err := cmd.Process.Signal(signal)
		if err != nil {
			glog.Errorf("Error signaling e2e services (pid=%d) with %s: %v", pid, signal.String(), err)
			continue
		}

		select {
		case err := <-waitChan:
			if err != nil {
				return fmt.Errorf("error stopping e2e services: %v", err)
			}
			// Success!
			return nil
		case <-time.After(timeout):
			// Continue.
		}
	}
	return nil
}

// RunE2EServices actually start the e2e services. This function is used to
// start e2e services in current process. This is used in start-services-only
// mode.
func RunE2EServices() {
	e := newE2eService()
	if err := e.run(); err != nil {
		glog.Fatalf("Failed to run e2e services: %v", err)
	}
}

// Ports of different e2e services.
const (
	apiserverPort       = "8080"
	kubeletPort         = "10250"
	kubeletReadOnlyPort = "10255"
)

// Health check urls of different e2e services.
var (
	apiserverHealthCheckURL = getEndpoint(apiserverPort) + "/healthz"
	kubeletHealthCheckURL   = getEndpoint(kubeletReadOnlyPort) + "/healthz"
)

// getEndpoint generates endpoint url from service port.
func getEndpoint(port string) string {
	return "http://127.0.0.1:" + port
}

func getHealthCheckURLs() []string {
	return []string{
		getEtcdHealthCheckURL(),
		apiserverHealthCheckURL,
		kubeletHealthCheckURL,
	}
}

// readinessCheck checks whether services are ready via the health check urls.
// TODO(random-liu): Move this to util
func readinessCheck(urls []string) error {
	endTime := time.Now().Add(*serverStartTimeout)
	for endTime.After(time.Now()) {
		select {
		case <-time.After(time.Second):
			ready := true
			for _, url := range urls {
				resp, err := http.Get(url)
				if err != nil || resp.StatusCode != http.StatusOK {
					ready = false
					break
				}
			}
			if ready {
				return nil
			}
		}
	}
	return fmt.Errorf("e2e service readiness check timeout %v", *serverStartTimeout)
}

// e2eService is used internally in this file to start e2e services in current process.
type e2eService struct {
	killCmds []*killCmd
	rmDirs   []string

	etcdDataDir string
	logFiles    map[string]logFileData

	// All e2e services
	etcdServer   *EtcdServer
	nsController *NamespaceController
}

type logFileData struct {
	files             []string
	journalctlCommand []string
}

const (
	// This is consistent with the level used in a cluster e2e test.
	LOG_VERBOSITY_LEVEL = "4"
	// Etcd binary is expected to either be available via PATH, or at this location.
	defaultEtcdPath = "/tmp/etcd"
)

func newE2eService() *e2eService {
	// Special log files that need to be collected for additional debugging.
	var logFiles = map[string]logFileData{
		"kern.log":   {[]string{"/var/log/kern.log"}, []string{"-k"}},
		"docker.log": {[]string{"/var/log/docker.log", "/var/log/upstart/docker.log"}, []string{"-u", "docker"}},
	}

	return &e2eService{logFiles: logFiles}
}

// run starts all e2e services and wait for the termination signal. Once receives the
// termination signal, it will stop the e2e services gracefully.
func (es *e2eService) run() error {
	defer es.stop()
	if err := es.start(); err != nil {
		glog.Fatalf("Unable to start node services.\n%v", err)
	}
	return es.wait()
}

// terminationSignals are signals that cause the program to exit in the
// supported platforms (linux, darwin, windows).
var terminationSignals = []os.Signal{syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT}

// wait waits until receiving a termination signal.
func (es *e2eService) wait() error {
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, terminationSignals...)
	<-sig
	return nil
}

func (es *e2eService) start() error {
	if _, err := getK8sBin("kubelet"); err != nil {
		return err
	}
	if _, err := getK8sBin("kube-apiserver"); err != nil {
		return err
	}

	err := es.startEtcd()
	if err != nil {
		return err
	}
	es.rmDirs = append(es.rmDirs, es.etcdDataDir)

	cmd, err := es.startApiServer()
	if err != nil {
		return err
	}
	es.killCmds = append(es.killCmds, cmd)

	cmd, err = es.startKubeletServer()
	if err != nil {
		return err
	}
	es.killCmds = append(es.killCmds, cmd)

	err = es.startNamespaceController()
	if err != nil {
		return nil
	}

	return nil
}

// Get logs of interest either via journalctl or by creating sym links.
// Since we scp files from the remote directory, symlinks will be treated as normal files and file contents will be copied over.
func (es *e2eService) getLogFiles() {
	// Nothing to do if report dir is not specified.
	if framework.TestContext.ReportDir == "" {
		return
	}
	journaldFound := isJournaldAvailable()
	for targetFileName, logFileData := range es.logFiles {
		targetLink := path.Join(framework.TestContext.ReportDir, targetFileName)
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
	es.getLogFiles()
	// TODO(random-liu): Use a loop to stop all services after introducing service interface.
	// Stop namespace controller
	if err := es.nsController.Stop(); err != nil {
		glog.Errorf("Failed to stop %q: %v", es.nsController.Name(), err)
	}
	for _, k := range es.killCmds {
		if err := k.Kill(); err != nil {
			glog.Errorf("Failed to stop %v: %v", k.name, err)
		}
	}
	// Stop etcd
	if err := es.etcdServer.Stop(); err != nil {
		glog.Errorf("Failed to stop %q: %v", es.etcdServer.Name(), err)
	}
	for _, d := range es.rmDirs {
		err := os.RemoveAll(d)
		if err != nil {
			glog.Errorf("Failed to delete directory %s.\n%v", d, err)
		}
	}
}

func (es *e2eService) startEtcd() error {
	dataDir, err := ioutil.TempDir("", "node-e2e")
	if err != nil {
		return err
	}
	es.etcdDataDir = dataDir
	es.etcdServer = NewEtcd(dataDir)
	return es.etcdServer.Start()
}

func (es *e2eService) startApiServer() (*killCmd, error) {
	cmd := exec.Command("sudo", getApiServerBin(),
		"--etcd-servers", getEtcdClientURL(),
		"--insecure-bind-address", "0.0.0.0",
		"--service-cluster-ip-range", "10.0.0.1/24",
		"--kubelet-port", kubeletPort,
		"--allow-privileged", "true",
		"--v", LOG_VERBOSITY_LEVEL, "--logtostderr",
	)
	hcc := newHealthCheckCommand(
		apiserverHealthCheckURL,
		cmd,
		"kube-apiserver.log")
	return &killCmd{name: "kube-apiserver", cmd: cmd}, es.startServer(hcc)
}

func (es *e2eService) startKubeletServer() (*killCmd, error) {
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
		cmdArgs = append(cmdArgs,
			"--runtime-cgroups=/docker-daemon",
			"--kubelet-cgroups=/kubelet",
		)
	}
	cmdArgs = append(cmdArgs,
		"--api-servers", getEndpoint(apiserverPort),
		"--address", "0.0.0.0",
		"--port", kubeletPort,
		"--read-only-port", kubeletReadOnlyPort,
		"--hostname-override", framework.TestContext.NodeName, // Required because hostname is inconsistent across hosts
		"--volume-stats-agg-period", "10s", // Aggregate volumes frequently so tests don't need to wait as long
		"--allow-privileged", "true",
		"--serialize-image-pulls", "false",
		"--config", framework.TestContext.ManifestPath,
		"--file-check-frequency", "10s", // Check file frequently so tests won't wait too long
		"--v", LOG_VERBOSITY_LEVEL, "--logtostderr",
		"--pod-cidr=10.180.0.0/24", // Assign a fixed CIDR to the node because there is no node controller.
		"--eviction-hard", framework.TestContext.EvictionHard,
		"--eviction-pressure-transition-period", "30s",
	)
	if framework.TestContext.CgroupsPerQOS {
		cmdArgs = append(cmdArgs,
			"--cgroups-per-qos", "true",
			"--cgroup-root", "/",
		)
	}
	if !framework.TestContext.DisableKubenet {
		cwd, err := os.Getwd()
		if err != nil {
			return nil, err
		}
		cmdArgs = append(cmdArgs,
			"--network-plugin=kubenet",
			"--network-plugin-dir", filepath.Join(cwd, CNIDirectory, "bin")) // Enable kubenet
	}

	cmd := exec.Command("sudo", cmdArgs...)
	hcc := newHealthCheckCommand(
		kubeletHealthCheckURL,
		cmd,
		"kubelet.log")
	return &killCmd{name: "kubelet", cmd: cmd, override: killOverride}, es.startServer(hcc)
}

func (es *e2eService) startNamespaceController() error {
	es.nsController = NewNamespaceController()
	return es.nsController.Start()
}

func (es *e2eService) startServer(cmd *healthCheckCommand) error {
	cmdErrorChan := make(chan error)
	go func() {
		defer close(cmdErrorChan)

		// Create the output filename
		outPath := path.Join(framework.TestContext.ReportDir, cmd.outputFilename)
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
	name := k.name
	cmd := k.cmd

	if k.override != nil {
		return k.override.Run()
	}

	if cmd == nil {
		return fmt.Errorf("Could not kill %s because both `override` and `cmd` are nil", name)
	}

	if cmd.Process == nil {
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
	return fmt.Sprintf("`%s` health-check: %s", strings.Join(append([]string{hcc.Path}, hcc.Args[1:]...), " "), hcc.HealthCheckUrl)
}
