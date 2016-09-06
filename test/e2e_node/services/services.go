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

package services

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
	"k8s.io/kubernetes/test/e2e_node/build"
)

// TODO(random-liu): Move this file to a separate package.
var serverStartTimeout = flag.Duration("server-start-timeout", time.Second*120, "Time to wait for each server to become healthy.")

// E2EServices starts and stops e2e services in a separate process. The test uses it to start and
// stop all e2e services.
type E2EServices struct {
	services *server
}

func NewE2EServices() *E2EServices {
	return &E2EServices{}
}

// services.log is the combined log of all services
const servicesLogFile = "services.log"

// Start starts the e2e services in another process, it returns when all e2e
// services are ready.
// We want to statically link e2e services into the test binary, but we don't
// want their glog to pollute the test result. So we run the binary in run-
// services-mode to start e2e services in another process.
func (e *E2EServices) Start() error {
	var err error
	// Create the manifest path for kubelet.
	// TODO(random-liu): Remove related logic when we move kubelet starting logic out of the test.
	framework.TestContext.ManifestPath, err = ioutil.TempDir("", "node-e2e-pod")
	if err != nil {
		return fmt.Errorf("failed to create static pod manifest directory: %v", err)
	}
	testBin, err := osext.Executable()
	if err != nil {
		return fmt.Errorf("can't get current binary: %v", err)
	}
	// TODO(random-liu): Add sudo after we statically link apiserver and etcd, because apiserver needs
	// sudo. We can't add sudo now, because etcd may not be in PATH of root.
	startCmd := exec.Command(testBin,
		"--run-services-mode",
		"--server-start-timeout", serverStartTimeout.String(),
		"--report-dir", framework.TestContext.ReportDir,
		// TODO(random-liu): Remove the following flags after we move kubelet starting logic
		// out of the test.
		"--node-name", framework.TestContext.NodeName,
		"--disable-kubenet="+strconv.FormatBool(framework.TestContext.DisableKubenet),
		// TODO: enable when flag is introduced in 1.5
		// "--cgroups-per-qos="+strconv.FormatBool(framework.TestContext.CgroupsPerQOS),
		"--manifest-path", framework.TestContext.ManifestPath,
		"--eviction-hard", framework.TestContext.EvictionHard,
		"--logtostderr",
		"--vmodule=*=4",
	)
	e.services = newServer("services", startCmd, nil, nil, getHealthCheckURLs(), servicesLogFile, false)
	return e.services.start()
}

// Stop stops the e2e services.
func (e *E2EServices) Stop() error {
	defer func() {
		// Cleanup the manifest path for kubelet.
		manifestPath := framework.TestContext.ManifestPath
		if manifestPath != "" {
			err := os.RemoveAll(manifestPath)
			if err != nil {
				glog.Errorf("Failed to delete static pod manifest directory %s: %v", manifestPath, err)
			}
		}
	}()
	if e.services == nil {
		glog.Errorf("can't stop e2e services, because `services` is nil")
	}
	return e.services.kill()
}

// RunE2EServices actually start the e2e services. This function is used to
// start e2e services in current process. This is only used in run-services-mode.
func RunE2EServices() {
	e := newE2EService()
	if err := e.run(); err != nil {
		glog.Fatalf("Failed to run e2e services: %v", err)
	}
}

// Ports of different e2e services.
const (
	kubeletPort         = "10250"
	kubeletReadOnlyPort = "10255"
)

// Health check urls of different e2e services.
var (
	kubeletHealthCheckURL = getEndpoint(kubeletReadOnlyPort) + "/healthz"
)

// getEndpoint generates endpoint url from service port.
func getEndpoint(port string) string {
	return "http://127.0.0.1:" + port
}

func getHealthCheckURLs() []string {
	return []string{
		getEtcdHealthCheckURL(),
		getAPIServerHealthCheckURL(),
		kubeletHealthCheckURL,
	}
}

// e2eService is used internally in this file to start e2e services in current process.
type e2eService struct {
	services []*server
	rmDirs   []string
	logFiles map[string]logFileData

	// All statically linked e2e services
	etcdServer   *EtcdServer
	apiServer    *APIServer
	nsController *NamespaceController
}

type logFileData struct {
	files             []string
	journalctlCommand []string
}

const (
	// This is consistent with the level used in a cluster e2e test.
	LOG_VERBOSITY_LEVEL = "4"
)

func newE2EService() *e2eService {
	// Special log files that need to be collected for additional debugging.
	var logFiles = map[string]logFileData{
		"kern.log":       {[]string{"/var/log/kern.log"}, []string{"-k"}},
		"docker.log":     {[]string{"/var/log/docker.log", "/var/log/upstart/docker.log"}, []string{"-u", "docker"}},
		"cloud-init.log": {[]string{"/var/log/cloud-init.log"}, []string{"-u", "cloud*"}},
	}

	return &e2eService{logFiles: logFiles}
}

// terminationSignals are signals that cause the program to exit in the
// supported platforms (linux, darwin, windows).
var terminationSignals = []os.Signal{syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT}

// run starts all e2e services and wait for the termination signal. Once receives the
// termination signal, it will stop the e2e services gracefully.
func (es *e2eService) run() error {
	defer es.stop()
	if err := es.start(); err != nil {
		return err
	}
	// Wait until receiving a termination signal.
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, terminationSignals...)
	<-sig
	return nil
}

func (es *e2eService) start() error {
	glog.Info("Starting e2e services...")
	err := es.startEtcd()
	if err != nil {
		return err
	}

	err = es.startApiServer()
	if err != nil {
		return err
	}

	s, err := es.startKubeletServer()
	if err != nil {
		return err
	}
	es.services = append(es.services, s)

	err = es.startNamespaceController()
	if err != nil {
		return nil
	}
	glog.Info("E2E services started.")
	return nil
}

// Get logs of interest either via journalctl or by creating sym links.
// Since we scp files from the remote directory, symlinks will be treated as normal files and file contents will be copied over.
func (es *e2eService) getLogFiles() {
	// Nothing to do if report dir is not specified.
	if framework.TestContext.ReportDir == "" {
		return
	}
	glog.Info("Fetching log files...")
	journaldFound := isJournaldAvailable()
	for targetFileName, logFileData := range es.logFiles {
		targetLink := path.Join(framework.TestContext.ReportDir, targetFileName)
		if journaldFound {
			// Skip log files that do not have an equivalent in journald based machines.
			if len(logFileData.journalctlCommand) == 0 {
				continue
			}
			glog.Infof("Get log file %q with journalctl command %v.", targetFileName, logFileData.journalctlCommand)
			out, err := exec.Command("sudo", append([]string{"journalctl"}, logFileData.journalctlCommand...)...).CombinedOutput()
			if err != nil {
				glog.Errorf("failed to get %q from journald: %v, %v", targetFileName, string(out), err)
			} else {
				if err = ioutil.WriteFile(targetLink, out, 0644); err != nil {
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
	glog.Info("Stopping e2e services...")
	es.getLogFiles()
	// TODO(random-liu): Use a loop to stop all services after introducing service interface.
	// Stop namespace controller
	if es.nsController != nil {
		if err := es.nsController.Stop(); err != nil {
			glog.Errorf("Failed to stop %q: %v", es.nsController.Name(), err)
		}
	}
	// Stop apiserver
	if es.apiServer != nil {
		if err := es.apiServer.Stop(); err != nil {
			glog.Errorf("Failed to stop %q: %v", es.apiServer.Name(), err)
		}
	}
	for _, s := range es.services {
		if err := s.kill(); err != nil {
			glog.Errorf("Failed to stop %v: %v", s.name, err)
		}
	}
	// Stop etcd
	if es.etcdServer != nil {
		if err := es.etcdServer.Stop(); err != nil {
			glog.Errorf("Failed to stop %q: %v", es.etcdServer.Name(), err)
		}
	}
	for _, d := range es.rmDirs {
		err := os.RemoveAll(d)
		if err != nil {
			glog.Errorf("Failed to delete directory %s.\n%v", d, err)
		}
	}
	glog.Info("E2E services stopped.")
}

func (es *e2eService) startEtcd() error {
	dataDir, err := ioutil.TempDir("", "node-e2e")
	if err != nil {
		return err
	}
	// Mark the dataDir as directories to remove.
	es.rmDirs = append(es.rmDirs, dataDir)
	es.etcdServer = NewEtcd(dataDir)
	return es.etcdServer.Start()
}

func (es *e2eService) startApiServer() error {
	es.apiServer = NewAPIServer()
	return es.apiServer.Start()
}

func (es *e2eService) startNamespaceController() error {
	es.nsController = NewNamespaceController()
	return es.nsController.Start()
}

func (es *e2eService) startKubeletServer() (*server, error) {
	var killCommand, restartCommand *exec.Cmd
	cmdArgs := []string{}
	if systemdRun, err := exec.LookPath("systemd-run"); err == nil {
		// On systemd services, detection of a service / unit works reliably while
		// detection of a process started from an ssh session does not work.
		// Since kubelet will typically be run as a service it also makes more
		// sense to test it that way
		unitName := fmt.Sprintf("kubelet-%d.service", rand.Int31())
		cmdArgs = append(cmdArgs, systemdRun, "--unit="+unitName, "--remain-after-exit", build.GetKubeletServerBin())
		killCommand = exec.Command("sudo", "systemctl", "kill", unitName)
		restartCommand = exec.Command("sudo", "systemctl", "restart", unitName)
		es.logFiles["kubelet.log"] = logFileData{
			journalctlCommand: []string{"-u", unitName},
		}
		framework.TestContext.EvictionHard = adjustConfigForSystemd(framework.TestContext.EvictionHard)
	} else {
		cmdArgs = append(cmdArgs, build.GetKubeletServerBin())
		cmdArgs = append(cmdArgs,
			"--runtime-cgroups=/docker-daemon",
			"--kubelet-cgroups=/kubelet",
			"--cgroup-root=/",
			"--system-cgroups=/system",
		)
	}
	cmdArgs = append(cmdArgs,
		"--api-servers", getAPIServerClientURL(),
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
		"--feature-gates", "DynamicKubeletConfig=true", // TODO(mtaufen): Eventually replace with a value from the framework.TestContext
	)
	if framework.TestContext.CgroupsPerQOS {
		// TODO: enable this when the flag is stable and available in kubelet.
		// cmdArgs = append(cmdArgs,
		// 	"--cgroups-per-qos", "true",
		// )
	}
	if !framework.TestContext.DisableKubenet {
		cwd, err := os.Getwd()
		if err != nil {
			return nil, err
		}
		cmdArgs = append(cmdArgs,
			"--network-plugin=kubenet",
			// TODO(random-liu): Make sure the cni directory name is the same with that in remote/remote.go
			"--network-plugin-dir", filepath.Join(cwd, "cni", "bin")) // Enable kubenet
	}

	cmd := exec.Command("sudo", cmdArgs...)
	server := newServer(
		"kubelet",
		cmd,
		killCommand,
		restartCommand,
		[]string{kubeletHealthCheckURL},
		"kubelet.log",
		true)
	return server, server.start()
}

// server manages a server started and killed with commands.
type server struct {
	// name is the name of the server, it is only used for logging.
	name string
	// startCommand is the command used to start the server
	startCommand *exec.Cmd
	// killCommand is the command used to stop the server. It is not required. If it
	// is not specified, `sudo kill` will be used to stop the server.
	killCommand *exec.Cmd
	// restartCommand is the command used to restart the server. If provided, it will be used
	// instead of startCommand when restarting the server.
	restartCommand *exec.Cmd
	// healthCheckUrls is the urls used to check whether the server is ready.
	healthCheckUrls []string
	// outFilename is the name of the log file. The stdout and stderr of the server
	// will be redirected to this file.
	outFilename string
	// restartOnExit determines whether a restart loop is launched with the server
	restartOnExit bool
	// Writing to this channel, if it is not nil, stops the restart loop.
	// When tearing down a server, you should check for this channel and write to it if it exists.
	stopRestartingCh chan<- bool
	// Read from this to confirm that the restart loop has stopped.
	ackStopRestartingCh <-chan bool
}

func newServer(name string, start, kill, restart *exec.Cmd, urls []string, filename string, restartOnExit bool) *server {
	return &server{
		name:            name,
		startCommand:    start,
		killCommand:     kill,
		restartCommand:  restart,
		healthCheckUrls: urls,
		outFilename:     filename,
		restartOnExit:   restartOnExit,
	}
}

// commandToString format command to string.
func commandToString(c *exec.Cmd) string {
	if c == nil {
		return ""
	}
	return strings.Join(append([]string{c.Path}, c.Args[1:]...), " ")
}

func (s *server) String() string {
	return fmt.Sprintf("server %q start-command: `%s`, kill-command: `%s`, restart-command: `%s`, health-check: %v, output-file: %q", s.name,
		commandToString(s.startCommand), commandToString(s.killCommand), commandToString(s.restartCommand), s.healthCheckUrls, s.outFilename)
}

// readinessCheck checks whether services are ready via the health check urls. Once there is
// an error in errCh, the function will stop waiting and return the error.
// TODO(random-liu): Move this to util
func readinessCheck(urls []string, errCh <-chan error) error {
	endTime := time.Now().Add(*serverStartTimeout)
	blockCh := make(chan error)
	defer close(blockCh)
	for endTime.After(time.Now()) {
		select {
		// We *always* want to run the health check if there is no error on the channel.
		// With systemd, reads from errCh report nil because cmd.Run() waits
		// on systemd-run, rather than the service process. systemd-run quickly
		// exits with status 0, causing the channel to be closed with no error. In
		// this case, you want to wait for the health check to complete, rather
		// than returning from readinessCheck as soon as the channel is closed.
		case err, ok := <-errCh:
			if ok { // The channel is not closed, this is a real error
				if err != nil { // If there is an error, return it
					return err
				}
				// If not, keep checking readiness.
			} else { // The channel is closed, this is only a zero value.
				// Replace the errCh with blockCh to avoid busy loop,
				// and keep checking readiness.
				errCh = blockCh
			}
		case <-time.After(time.Second):
			ready := true
			for _, url := range urls {
				resp, err := http.Head(url)
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

// Note: restartOnExit == true requires len(s.healthCheckUrls) > 0 to work properly.
func (s *server) start() error {
	glog.Infof("Start server %q with command %q", s.name, commandToString(s.startCommand))
	errCh := make(chan error)

	var stopRestartingCh, ackStopRestartingCh chan bool
	if s.restartOnExit {
		if len(s.healthCheckUrls) == 0 {
			return fmt.Errorf("Tried to start %s which has s.restartOnExit == true, but no health check urls provided.", s)
		}

		stopRestartingCh = make(chan bool)
		ackStopRestartingCh = make(chan bool)

		s.stopRestartingCh = stopRestartingCh
		s.ackStopRestartingCh = ackStopRestartingCh
	}

	go func() {
		defer close(errCh)

		// Create the output filename
		outPath := path.Join(framework.TestContext.ReportDir, s.outFilename)
		outfile, err := os.Create(outPath)
		if err != nil {
			errCh <- fmt.Errorf("failed to create file %q for `%s` %v.", outPath, s, err)
			return
		}
		defer outfile.Close()
		defer outfile.Sync()

		// Set the command to write the output file
		s.startCommand.Stdout = outfile
		s.startCommand.Stderr = outfile

		// Death of this test process should kill the server as well.
		attrs := &syscall.SysProcAttr{}
		// Hack to set linux-only field without build tags.
		deathSigField := reflect.ValueOf(attrs).Elem().FieldByName("Pdeathsig")
		if deathSigField.IsValid() {
			deathSigField.Set(reflect.ValueOf(syscall.SIGTERM))
		} else {
			errCh <- fmt.Errorf("failed to set Pdeathsig field (non-linux build)")
			return
		}
		s.startCommand.SysProcAttr = attrs

		// Start the command
		err = s.startCommand.Start()
		if err != nil {
			errCh <- fmt.Errorf("failed to run %s: %v", s, err)
			return
		}
		if !s.restartOnExit {
			// If we aren't planning on restarting, ok to Wait() here to release resources.
			// Otherwise, we Wait() in the restart loop.
			err = s.startCommand.Wait()
			if err != nil {
				errCh <- fmt.Errorf("failed to run %s: %v", s, err)
				return
			}
		} else {
			// New stuff
			usedStartCmd := true
			for {
				// Wait for an initial health check to pass, so that we are sure the server started.
				err := readinessCheck(s.healthCheckUrls, nil)
				if err != nil {
					if usedStartCmd {
						s.startCommand.Wait() // Release resources if necessary.
					}
					// This should not happen, immediately stop the e2eService process.
					glog.Fatalf("restart loop readinessCheck failed for %s", s)
				}

				// Initial health check passed, wait until a health check fails again.
			stillAlive:
				for {
					select {
					case <-stopRestartingCh:
						ackStopRestartingCh <- true
						return
					case <-time.After(time.Second):
						for _, url := range s.healthCheckUrls {
							resp, err := http.Head(url)
							if err != nil || resp.StatusCode != http.StatusOK {
								break stillAlive
							}
						}
					}
				}

				if usedStartCmd {
					s.startCommand.Wait() // Release resources from last cmd
					usedStartCmd = false
				}
				if s.restartCommand != nil {
					// Always make a fresh copy of restartCommand before running, we may have to restart multiple times
					s.restartCommand = &exec.Cmd{
						Path:        s.restartCommand.Path,
						Args:        s.restartCommand.Args,
						Env:         s.restartCommand.Env,
						Dir:         s.restartCommand.Dir,
						Stdin:       s.restartCommand.Stdin,
						Stdout:      s.restartCommand.Stdout,
						Stderr:      s.restartCommand.Stderr,
						ExtraFiles:  s.restartCommand.ExtraFiles,
						SysProcAttr: s.restartCommand.SysProcAttr,
					}
					err = s.restartCommand.Run() // Run and wait for exit. This command is assumed to have short duration, e.g. systemctl restart
					if err != nil {
						// This should not happen, immediately stop the e2eService process.
						glog.Fatalf("restarting %s with restartCommand failed. Error: %v.", s, err)
					}
				} else {
					s.startCommand = &exec.Cmd{
						Path:        s.startCommand.Path,
						Args:        s.startCommand.Args,
						Env:         s.startCommand.Env,
						Dir:         s.startCommand.Dir,
						Stdin:       s.startCommand.Stdin,
						Stdout:      s.startCommand.Stdout,
						Stderr:      s.startCommand.Stderr,
						ExtraFiles:  s.startCommand.ExtraFiles,
						SysProcAttr: s.startCommand.SysProcAttr,
					}
					err = s.startCommand.Start()
					usedStartCmd = true
					if err != nil {
						// This should not happen, immediately stop the e2eService process.
						glog.Fatalf("restarting %s with startCommand failed. Error: %v.", s, err)
					}
				}
			}
		}
	}()

	return readinessCheck(s.healthCheckUrls, errCh)
}

func (s *server) kill() error {
	glog.Infof("Kill server %q", s.name)
	name := s.name
	cmd := s.startCommand

	// If s has a restart loop, turn it off.
	if s.restartOnExit {
		s.stopRestartingCh <- true
		<-s.ackStopRestartingCh
	}

	if s.killCommand != nil {
		return s.killCommand.Run()
	}

	if cmd == nil {
		return fmt.Errorf("could not kill %q because both `killCommand` and `startCommand` are nil", name)
	}

	if cmd.Process == nil {
		glog.V(2).Infof("%q not running", name)
		return nil
	}
	pid := cmd.Process.Pid
	if pid <= 1 {
		return fmt.Errorf("invalid PID %d for %q", pid, name)
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
		cmd := exec.Command("sudo", "kill", signal, strconv.Itoa(pid))

		// Run the 'kill' command in a separate process group so sudo doesn't ignore it
		attrs := &syscall.SysProcAttr{}
		// Hack to set unix-only field without build tags.
		setpgidField := reflect.ValueOf(attrs).Elem().FieldByName("Setpgid")
		if setpgidField.IsValid() {
			setpgidField.Set(reflect.ValueOf(true))
		} else {
			return fmt.Errorf("Failed to set Setpgid field (non-unix build)")
		}
		cmd.SysProcAttr = attrs

		_, err := cmd.Output()
		if err != nil {
			glog.Errorf("Error signaling process %d (%s) with %s: %v", pid, name, signal, err)
			continue
		}

		select {
		case err := <-waitChan:
			if err != nil {
				return fmt.Errorf("error stopping %q: %v", name, err)
			}
			// Success!
			return nil
		case <-time.After(timeout):
			// Continue.
		}
	}

	return fmt.Errorf("unable to stop %q", name)
}

func adjustConfigForSystemd(config string) string {
	return strings.Replace(config, "%", "%%", -1)
}
