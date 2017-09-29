// +build linux

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

package host_path

import (
	"fmt"

	"k8s.io/utils/exec"
)

const (
	hostProcMountsNamespace = "/rootfs/proc/1/ns/mnt"
	nsenterCmd              = "nsenter"
	statCmd                 = "stat"
	touchCmd                = "touch"
	mkdirCmd                = "mkdir"
)

// nsenterFileTypeChecker is part of experimental support for running the kubelet
// in a container. nsenterFileTypeChecker works by executing "nsenter" to run commands in
// the host's mount namespace.
//
// nsenterFileTypeChecker requires:
//
// 1.  The host's root filesystem must be available at "/rootfs";
// 2.  The "nsenter" binary must be on the Kubelet process' PATH in the container's
//     filesystem;
// 3.  The Kubelet process must have CAP_SYS_ADMIN (required by "nsenter"); at
//     the present, this effectively means that the kubelet is running in a
//     privileged container;
// 4.  The host image must have "stat", "touch", "mkdir" binaries in "/bin", "/usr/sbin", or "/usr/bin";

type nsenterFileTypeChecker struct {
	path   string
	exists bool
}

func newNsenterFileTypeChecker(path string) (hostPathTypeChecker, error) {
	ftc := &nsenterFileTypeChecker{path: path}
	ftc.Exists()
	return ftc, nil
}

func (ftc *nsenterFileTypeChecker) Exists() bool {
	args := []string{
		fmt.Sprintf("--mount=%s", hostProcMountsNamespace),
		"--",
		"ls",
		ftc.path,
	}
	exec := exec.New()
	_, err := exec.Command(nsenterCmd, args...).CombinedOutput()
	if err == nil {
		ftc.exists = true
	}
	return ftc.exists
}

func (ftc *nsenterFileTypeChecker) IsFile() bool {
	if !ftc.Exists() {
		return false
	}
	return !ftc.IsDir()
}

func (ftc *nsenterFileTypeChecker) MakeFile() error {
	args := []string{
		fmt.Sprintf("--mount=%s", hostProcMountsNamespace),
		"--",
		touchCmd,
		ftc.path,
	}
	exec := exec.New()
	if _, err := exec.Command(nsenterCmd, args...).CombinedOutput(); err != nil {
		return err
	}
	return nil
}

func (ftc *nsenterFileTypeChecker) IsDir() bool {
	return ftc.checkMimetype("directory")
}

func (ftc *nsenterFileTypeChecker) MakeDir() error {
	args := []string{
		fmt.Sprintf("--mount=%s", hostProcMountsNamespace),
		"--",
		mkdirCmd,
		"-p",
		ftc.path,
	}
	exec := exec.New()
	if _, err := exec.Command(nsenterCmd, args...).CombinedOutput(); err != nil {
		return err
	}
	return nil
}

func (ftc *nsenterFileTypeChecker) IsBlock() bool {
	return ftc.checkMimetype("block special file")
}

func (ftc *nsenterFileTypeChecker) IsChar() bool {
	return ftc.checkMimetype("character special file")
}

func (ftc *nsenterFileTypeChecker) IsSocket() bool {
	return ftc.checkMimetype("socket")
}

func (ftc *nsenterFileTypeChecker) GetPath() string {
	return ftc.path
}

func (ftc *nsenterFileTypeChecker) checkMimetype(checkedType string) bool {
	if !ftc.Exists() {
		return false
	}

	args := []string{
		fmt.Sprintf("--mount=%s", hostProcMountsNamespace),
		"--",
		statCmd,
		"-L",
		`--printf "%F"`,
		ftc.path,
	}
	exec := exec.New()
	outputBytes, err := exec.Command(nsenterCmd, args...).CombinedOutput()
	if err != nil {
		return false
	}
	return string(outputBytes) == checkedType
}
