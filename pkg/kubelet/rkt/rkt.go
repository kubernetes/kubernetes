/*
Copyright 2015 Google Inc. All rights reserved.

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

package rkt

import (
	"fmt"
	"os/exec"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/credentialprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/coreos/go-systemd/dbus"
	"github.com/golang/glog"
)

const (
	acversion             = "0.5.1"
	rktMinimumVersion     = "0.5.4"
	systemdMinimumVersion = "215"

	systemdServiceDir      = "/run/systemd/system"
	rktDataDir             = "/var/lib/rkt"
	rktLocalConfigDir      = "/etc/rkt"
	rktMetadataServiceFile = "rkt-metadata.service"
	rktMetadataSocketFile  = "rkt-metadata.socket"

	kubernetesUnitPrefix  = "k8s"
	unitKubernetesSection = "X-Kubernetes"
	unitPodName           = "POD"
	unitRktID             = "RktID"

	dockerPrefix = "docker://"
)

const (
	rktBinName = "rkt"

	Embryo         = "embryo"
	Preparing      = "preparing"
	AbortedPrepare = "aborted prepare"
	Prepared       = "prepared"
	Running        = "running"
	Deleting       = "deleting" // This covers pod.isExitedDeleting and pod.isDeleting.
	Exited         = "exited"   // This covers pod.isExited and pod.isExitedGarbage.
	Garbage        = "garbage"
)

// Runtime implements the ContainerRuntime for rkt. The implementation
// uses systemd, so in order to run this runtime, systemd must be installed
// on the machine.
type Runtime struct {
	systemd *dbus.Conn
	absPath string
	config  *Config
	// TODO(yifan): Refactor this to be generic keyring.
	dockerKeyring credentialprovider.DockerKeyring
}

// New creates the rkt container runtime which implements the container runtime interface.
// It will test if the rkt binary is in the $PATH, and whether we can get the
// version of it. If so, creates the rkt container runtime, otherwise returns an error.
func New(config *Config) (*Runtime, error) {
	systemdVersion, err := getSystemdVersion()
	if err != nil {
		return nil, err
	}
	result, err := systemdVersion.Compare(systemdMinimumVersion)
	if err != nil {
		return nil, err
	}
	if result < 0 {
		return nil, fmt.Errorf("rkt: systemd version is too old, requires at least %v", systemdMinimumVersion)
	}

	systemd, err := dbus.New()
	if err != nil {
		glog.Errorf("rkt: Cannot connect to dbus: %v", err)
		return nil, err
	}

	// Test if rkt binary is in $PATH.
	absPath, err := exec.LookPath(rktBinName)
	if err != nil {
		glog.Errorf("rkt: Cannot find rkt binary: %v", err)
		return nil, err
	}

	rkt := &Runtime{
		systemd:       systemd,
		absPath:       absPath,
		config:        config,
		dockerKeyring: credentialprovider.NewDockerKeyring(),
	}

	// Test the rkt version.
	version, err := rkt.Version()
	if err != nil {
		return nil, err
	}
	result, err = version.Compare(rktMinimumVersion)
	if err != nil {
		return nil, err
	}
	if result < 0 {
		return nil, fmt.Errorf("rkt: Version is too old, requires at least %v", rktMinimumVersion)
	}
	return rkt, nil
}

func (r *Runtime) buildCommand(args ...string) *exec.Cmd {
	cmd := exec.Command(rktBinName)
	cmd.Args = append(cmd.Args, r.config.buildGlobalOptions()...)
	cmd.Args = append(cmd.Args, args...)
	return cmd
}

// runCommand invokes rkt binary with arguments and returns the result
// from stdout in a list of strings.
func (r *Runtime) runCommand(args ...string) ([]string, error) {
	glog.V(4).Info("rkt: Run command:", args)

	output, err := r.buildCommand(args...).Output()
	if err != nil {
		return nil, err
	}
	return strings.Split(strings.TrimSpace(string(output)), "\n"), nil
}

// makePodServiceFileName constructs the unit file name for a pod using its UID.
func makePodServiceFileName(uid types.UID) string {
	// TODO(yifan): Revisit this later, decide whether we want to use UID.
	return fmt.Sprintf("%s_%s.service", kubernetesUnitPrefix, uid)
}
