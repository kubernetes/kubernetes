/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"strconv"
	"strings"

	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/golang/glog"
)

type rktVersion []int

func parseVersion(input string) (rktVersion, error) {
	tail := strings.Index(input, "+")
	if tail > 0 {
		input = input[:tail]
	}
	var result rktVersion
	tuples := strings.Split(input, ".")
	for _, t := range tuples {
		n, err := strconv.Atoi(t)
		if err != nil {
			return nil, err
		}
		result = append(result, n)
	}
	return result, nil
}

func (r rktVersion) Compare(other string) (int, error) {
	v, err := parseVersion(other)
	if err != nil {
		return -1, err
	}

	for i := range r {
		if i > len(v)-1 {
			return 1, nil
		}
		if r[i] < v[i] {
			return -1, nil
		}
		if r[i] > v[i] {
			return 1, nil
		}
	}

	// When loop ends, len(r) is <= len(v).
	if len(r) < len(v) {
		return -1, nil
	}
	return 0, nil
}

func (r rktVersion) String() string {
	var version []string
	for _, v := range r {
		version = append(version, fmt.Sprintf("%d", v))
	}
	return strings.Join(version, ".")
}

// Version invokes 'rkt version' to get the version information of the rkt
// runtime on the machine.
// The return values are an int array containers the version number.
//
// Example:
// rkt:0.3.2+git --> []int{0, 3, 2}.
//
func (r *Runtime) Version() (kubecontainer.Version, error) {
	output, err := r.runCommand("version")
	if err != nil {
		return nil, err
	}

	// Example output for 'rkt version':
	// rkt version 0.3.2+git
	// appc version 0.3.0+git
	for _, line := range output {
		tuples := strings.Split(strings.TrimSpace(line), " ")
		if len(tuples) != 3 {
			glog.Warningf("rkt: cannot parse the output: %q.", line)
			continue
		}
		if tuples[0] == "rkt" {
			return parseVersion(tuples[2])
		}
	}
	return nil, fmt.Errorf("rkt: cannot determine the version")
}

type systemdVersion int

func (s systemdVersion) String() string {
	return fmt.Sprintf("%d", s)
}

func (s systemdVersion) Compare(other string) (int, error) {
	v, err := strconv.Atoi(other)
	if err != nil {
		return -1, err
	}
	if int(s) < v {
		return -1, nil
	} else if int(s) > v {
		return 1, nil
	}
	return 0, nil
}

func getSystemdVersion() (systemdVersion, error) {
	output, err := exec.Command("systemctl", "--version").Output()
	if err != nil {
		return -1, err
	}
	// Example output of 'systemctl --version':
	//
	// systemd 215
	// +PAM +AUDIT +SELINUX +IMA +SYSVINIT +LIBCRYPTSETUP +GCRYPT +ACL +XZ -SECCOMP -APPARMOR
	//
	lines := strings.Split(string(output), "\n")
	tuples := strings.Split(lines[0], " ")
	if len(tuples) != 2 {
		return -1, fmt.Errorf("rkt: Failed to parse version %v", lines)
	}
	result, err := strconv.Atoi(string(tuples[1]))
	if err != nil {
		return -1, err
	}
	return systemdVersion(result), nil
}
