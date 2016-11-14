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
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"

	"github.com/golang/glog"
	"github.com/kardianos/osext"

	utilconfig "k8s.io/kubernetes/pkg/util/config"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e_node/builder"
)

// E2EServices starts and stops e2e services in a separate process. The test
// uses it to start and stop all e2e services.
type E2EServices struct {
	// monitorParent determines whether the sub-processes should watch and die with the current
	// process.
	monitorParent bool
	services      *server
	kubelet       *server
	logFiles      map[string]logFileData
}

// logFileData holds data about logfiles to fetch with a journalctl command or
// symlink from a node's file system.
type logFileData struct {
	files             []string
	journalctlCommand []string
}

// NewE2EServices returns a new E2EServices instance.
func NewE2EServices(monitorParent bool) *E2EServices {
	return &E2EServices{
		monitorParent: monitorParent,
		// Special log files that need to be collected for additional debugging.
		logFiles: map[string]logFileData{
			"kern.log":       {[]string{"/var/log/kern.log"}, []string{"-k"}},
			"docker.log":     {[]string{"/var/log/docker.log", "/var/log/upstart/docker.log"}, []string{"-u", "docker"}},
			"cloud-init.log": {[]string{"/var/log/cloud-init.log"}, []string{"-u", "cloud*"}},
		},
	}
}

// Start starts the e2e services in another process by calling back into the
// test binary.  Returns when all e2e services are ready or an error.
//
// We want to statically link e2e services into the test binary, but we don't
// want their glog output to pollute the test result. So we run the binary in
// run-services-mode to start e2e services in another process.
// The function starts 2 processes:
// * internal e2e services: services which statically linked in the test binary - apiserver, etcd and
// namespace controller.
// * kubelet: kubelet binary is outside. (We plan to move main kubelet start logic out when we have
// standard kubelet launcher)
func (e *E2EServices) Start() error {
	var err error
	if !framework.TestContext.NodeConformance {
		// Start kubelet
		// Create the manifest path for kubelet.
		// TODO(random-liu): Remove related logic when we move kubelet starting logic out of the test.
		cwd, err := os.Getwd()
		if err != nil {
			return fmt.Errorf("failed to get current working directory: %v", err)
		}
		framework.TestContext.ManifestPath, err = ioutil.TempDir(cwd, "pod-manifest")
		if err != nil {
			return fmt.Errorf("failed to create static pod manifest directory: %v", err)
		}
		e.kubelet, err = e.startKubelet()
		if err != nil {
			return fmt.Errorf("failed to start kubelet: %v", err)
		}
	}
	e.services, err = e.startInternalServices()
	return err
}

// Stop stops the e2e services.
func (e *E2EServices) Stop() {
	defer func() {
		if !framework.TestContext.NodeConformance {
			// Collect log files.
			e.getLogFiles()
			// Cleanup the manifest path for kubelet.
			manifestPath := framework.TestContext.ManifestPath
			if manifestPath != "" {
				err := os.RemoveAll(manifestPath)
				if err != nil {
					glog.Errorf("Failed to delete static pod manifest directory %s: %v", manifestPath, err)
				}
			}
		}
	}()
	if e.services != nil {
		if err := e.services.kill(); err != nil {
			glog.Errorf("Failed to stop services: %v", err)
		}
	}
	if e.kubelet != nil {
		if err := e.kubelet.kill(); err != nil {
			glog.Errorf("Failed to stop kubelet: %v", err)
		}
	}
}

// RunE2EServices actually start the e2e services. This function is used to
// start e2e services in current process. This is only used in run-services-mode.
func RunE2EServices() {
	// Populate global DefaultFeatureGate with value from TestContext.FeatureGates.
	// This way, statically-linked components see the same feature gate config as the test context.
	utilconfig.DefaultFeatureGate.Set(framework.TestContext.FeatureGates)
	e := newE2EServices()
	if err := e.run(); err != nil {
		glog.Fatalf("Failed to run e2e services: %v", err)
	}
}

const (
	// services.log is the combined log of all services
	servicesLogFile = "services.log"
	// LOG_VERBOSITY_LEVEL is consistent with the level used in a cluster e2e test.
	LOG_VERBOSITY_LEVEL = "4"
)

// startInternalServices starts the internal services in a separate process.
func (e *E2EServices) startInternalServices() (*server, error) {
	testBin, err := osext.Executable()
	if err != nil {
		return nil, fmt.Errorf("can't get current binary: %v", err)
	}
	// Pass all flags into the child process, so that it will see the same flag set.
	startCmd := exec.Command(testBin, append([]string{"--run-services-mode"}, os.Args[1:]...)...)
	server := newServer("services", startCmd, nil, nil, getServicesHealthCheckURLs(), servicesLogFile, e.monitorParent, false)
	return server, server.start()
}

const (
	// Ports of different e2e services.
	kubeletPort         = "10250"
	kubeletReadOnlyPort = "10255"
	// Health check url of kubelet
	kubeletHealthCheckURL = "http://127.0.0.1:" + kubeletReadOnlyPort + "/healthz"
)

// startKubelet starts the Kubelet in a separate process or returns an error
// if the Kubelet fails to start.
func (e *E2EServices) startKubelet() (*server, error) {
	glog.Info("Starting kubelet")
	var killCommand, restartCommand *exec.Cmd
	cmdArgs := []string{}
	if systemdRun, err := exec.LookPath("systemd-run"); err == nil {
		// On systemd services, detection of a service / unit works reliably while
		// detection of a process started from an ssh session does not work.
		// Since kubelet will typically be run as a service it also makes more
		// sense to test it that way
		unitName := fmt.Sprintf("kubelet-%d.service", rand.Int31())
		cmdArgs = append(cmdArgs, systemdRun, "--unit="+unitName, "--remain-after-exit", builder.GetKubeletServerBin())
		killCommand = exec.Command("systemctl", "kill", unitName)
		restartCommand = exec.Command("systemctl", "restart", unitName)
		e.logFiles["kubelet.log"] = logFileData{
			journalctlCommand: []string{"-u", unitName},
		}
		framework.TestContext.EvictionHard = adjustConfigForSystemd(framework.TestContext.EvictionHard)
	} else {
		cmdArgs = append(cmdArgs, builder.GetKubeletServerBin())
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
		"--volume-stats-agg-period", "10s", // Aggregate volumes frequently so tests don't need to wait as long
		"--allow-privileged", "true",
		"--serialize-image-pulls", "false",
		"--config", framework.TestContext.ManifestPath,
		"--file-check-frequency", "10s", // Check file frequently so tests won't wait too long
		"--pod-cidr=10.180.0.0/24", // Assign a fixed CIDR to the node because there is no node controller.
		"--eviction-hard", framework.TestContext.EvictionHard,
		"--eviction-pressure-transition-period", "30s",
		"--feature-gates", framework.TestContext.FeatureGates,
		"--v", LOG_VERBOSITY_LEVEL, "--logtostderr",

		"--experimental-mounter-path", framework.TestContext.MounterPath,
	)
	if framework.TestContext.NodeName != "" { // If node name is specified, set hostname override.
		cmdArgs = append(cmdArgs, "--hostname-override", framework.TestContext.NodeName)
	}
	if framework.TestContext.EnableCRI {
		cmdArgs = append(cmdArgs, "--experimental-cri", "true") // Whether to use experimental cri integration.
	}
	if framework.TestContext.ContainerRuntimeEndpoint != "" {
		cmdArgs = append(cmdArgs, "--container-runtime-endpoint", framework.TestContext.ContainerRuntimeEndpoint)
	}
	if framework.TestContext.CgroupsPerQOS {
		cmdArgs = append(cmdArgs,
			"--experimental-cgroups-per-qos", "true",
			"--cgroup-root", "/",
		)
	}
	if framework.TestContext.CgroupDriver != "" {
		cmdArgs = append(cmdArgs,
			"--cgroup-driver", framework.TestContext.CgroupDriver,
		)
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

	cmd := exec.Command(cmdArgs[0], cmdArgs[1:]...)
	server := newServer(
		"kubelet",
		cmd,
		killCommand,
		restartCommand,
		[]string{kubeletHealthCheckURL},
		"kubelet.log",
		e.monitorParent,
		true /* restartOnExit */)
	return server, server.start()
}

func adjustConfigForSystemd(config string) string {
	return strings.Replace(config, "%", "%%", -1)
}

// getLogFiles gets logs of interest either via journalctl or by creating sym
// links. Since we scp files from the remote directory, symlinks will be
// treated as normal files and file contents will be copied over.
func (e *E2EServices) getLogFiles() {
	// Nothing to do if report dir is not specified.
	if framework.TestContext.ReportDir == "" {
		return
	}
	glog.Info("Fetching log files...")
	journaldFound := isJournaldAvailable()
	for targetFileName, logFileData := range e.logFiles {
		targetLink := path.Join(framework.TestContext.ReportDir, targetFileName)
		if journaldFound {
			// Skip log files that do not have an equivalent in journald-based machines.
			if len(logFileData.journalctlCommand) == 0 {
				continue
			}
			glog.Infof("Get log file %q with journalctl command %v.", targetFileName, logFileData.journalctlCommand)
			out, err := exec.Command("journalctl", logFileData.journalctlCommand...).CombinedOutput()
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

// isJournaldAvailable returns whether the system executing the tests uses
// journald.
func isJournaldAvailable() bool {
	_, err := exec.LookPath("journalctl")
	return err == nil
}

func copyLogFile(src, target string) error {
	// If not a journald based distro, then just symlink files.
	if out, err := exec.Command("cp", src, target).CombinedOutput(); err != nil {
		return fmt.Errorf("failed to copy %q to %q: %v, %v", src, target, out, err)
	}
	if out, err := exec.Command("chmod", "a+r", target).CombinedOutput(); err != nil {
		return fmt.Errorf("failed to make log file %q world readable: %v, %v", target, out, err)
	}
	return nil
}
