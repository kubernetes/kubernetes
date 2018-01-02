// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ssh.go file provides ssh communication with pod for kvm flavor

package ssh

import (
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"syscall"

	"github.com/hashicorp/errwrap"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/networking/netinfo"
	"github.com/coreos/rkt/pkg/lock"
)

const (
	kvmSettingsDir        = "/var/lib/rkt-stage1-kvm"
	kvmPrivateKeyFilename = "ssh_kvm_key"
	// TODO: overwrite below default by environment value + generate .socket unit just before pod start
	KvmSSHPort = "122" // hardcoded value in .socket file
)

// fileAccessible checks if the given path exists and is a regular file
func fileAccessible(path string) bool {
	if info, err := os.Stat(path); err == nil {
		return info.Mode().IsRegular()
	}
	return false
}

func sshPrivateKeyPath() string {
	return filepath.Join(kvmSettingsDir, kvmPrivateKeyFilename)
}

func sshPublicKeyPath() string {
	return sshPrivateKeyPath() + ".pub"
}

// generateKeyPair calls ssh-keygen with private key location for key generation purpose
func generateKeyPair(private string) error {
	out, err := exec.Command(
		"ssh-keygen",
		"-q",        // silence
		"-t", "dsa", // type
		"-b", "1024", // length in bits
		"-f", private, // output file
		"-N", "", // no passphrase
	).Output()
	if err != nil {
		// out is in form of bytes buffer and we have to turn it into slice ending on first \0 occurrence
		return fmt.Errorf("error in keygen time. ret_val: %v, output: %v", err, string(out[:]))
	}
	return nil
}

func ensureKeysExistOnHost() error {
	private, public := sshPrivateKeyPath(), sshPublicKeyPath()
	if !fileAccessible(private) || !fileAccessible(public) {
		if err := os.MkdirAll(kvmSettingsDir, 0700); err != nil {
			return err
		}

		if err := generateKeyPair(private); err != nil {
			return err
		}
	}
	return nil
}

func ensureAuthorizedKeysExist(keyDirPath string) error {
	fout, err := os.OpenFile(
		filepath.Join(keyDirPath, "/authorized_keys"),
		os.O_CREATE|os.O_TRUNC|os.O_WRONLY,
		0600,
	)
	if err != nil {
		return err
	}
	defer fout.Close()

	fin, err := os.Open(sshPublicKeyPath())
	if err != nil {
		return err
	}
	defer fin.Close()

	if _, err := io.Copy(fout, fin); err != nil {
		return err
	}
	return fout.Sync()
}

func ensureKeysExistInPod(workDir string) error {
	u, _ := user.Current()
	destRootfs := common.Stage1RootfsPath(workDir)
	keyDirPath := filepath.Join(destRootfs, u.HomeDir, ".ssh")
	if err := os.MkdirAll(keyDirPath, 0700); err != nil {
		return err
	}
	return ensureAuthorizedKeysExist(keyDirPath)
}

func kvmCheckSSHSetup(workDir string) error {
	if err := ensureKeysExistOnHost(); err != nil {
		return err
	}
	return ensureKeysExistInPod(workDir)
}

func getPodDefaultIP(workDir string) (string, error) {
	// get pod lock
	l, err := lock.NewLock(workDir, lock.Dir)
	if err != nil {
		return "", err
	}

	// get file descriptor for lock
	fd, err := l.Fd()
	if err != nil {
		return "", err
	}

	// use this descriptor as method of reading pod network configuration
	nets, err := netinfo.LoadAt(fd)
	if err != nil {
		return "", err
	}
	// kvm flavored container must have at first position default vm<->host network
	if len(nets) == 0 {
		return "", fmt.Errorf("pod has no configured networks")
	}

	for _, net := range nets {
		if net.NetName == "default" || net.NetName == "default-restricted" {
			return net.IP.String(), nil
		}
	}

	return "", fmt.Errorf("pod has no default network!")
}

func ExecSSH(execArgs []string) error {
	u, _ := user.Current()

	sshPath, err := exec.LookPath("ssh")
	if err != nil {
		return errwrap.Wrap(errors.New("cannot find 'ssh' binary"), err)
	}

	workDir, err := os.Getwd()
	if err != nil {
		return errwrap.Wrap(errors.New("cannot get working directory"), err)
	}

	podDefaultIP, err := getPodDefaultIP(workDir)
	if err != nil {
		return errwrap.Wrap(errors.New("cannot load networking configuration"), err)
	}

	// escape from running pod directory into base directory
	if err = os.Chdir("../../.."); err != nil {
		return errwrap.Wrap(errors.New("cannot change directory to rkt work directory"), err)
	}

	if err := kvmCheckSSHSetup(workDir); err != nil {
		return errwrap.Wrap(errors.New("error setting up ssh keys"), err)
	}

	// prepare args for ssh invocation
	keyFile := sshPrivateKeyPath()
	args := []string{
		"ssh",
		"-t",          // use tty
		"-i", keyFile, // use keyfile
		"-l", u.Username, // login as user
		"-p", KvmSSHPort, // port to connect
		"-o", "StrictHostKeyChecking=no", // do not check changing host keys
		"-o", "UserKnownHostsFile=/dev/null", // do not add host key to default knownhosts file
		"-o", "LogLevel=quiet", // do not log minor informations
		podDefaultIP,
	}

	args = append(args, execArgs...)

	// this should not return in case of success
	err = syscall.Exec(sshPath, args, os.Environ())
	return errwrap.Wrap(errors.New("cannot exec to ssh"), err)
}
