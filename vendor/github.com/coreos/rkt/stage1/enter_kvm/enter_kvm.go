// Copyright 2015 The rkt Authors
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

package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"syscall"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/networking/netinfo"
	"github.com/coreos/rkt/pkg/lock"
	rktlog "github.com/coreos/rkt/pkg/log"
	"github.com/hashicorp/errwrap"
)

const (
	kvmSettingsDir        = "/var/lib/rkt-stage1-kvm"
	kvmPrivateKeyFilename = "ssh_kvm_key"
	// TODO: overwrite below default by environment value + generate .socket unit just before pod start
	kvmSSHPort = "122" // hardcoded value in .socket file
)

var (
	debug   bool
	podPid  string
	appName string
	sshPath string
	u, _    = user.Current()
	log     *rktlog.Logger
	diag    *rktlog.Logger
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

func init() {
	flag.BoolVar(&debug, "debug", false, "Run in debug mode")
	flag.StringVar(&podPid, "pid", "", "podPID")
	flag.StringVar(&appName, "appname", "", "application to use")

	log, diag, _ = rktlog.NewLogSet("kvm", false)

	var err error
	if sshPath, err = exec.LookPath("ssh"); err != nil {
		log.FatalE("cannot find 'ssh' binary in PATH", err)
	}
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

func getAppexecArgs() []string {
	// Documentation/devel/stage1-implementors-guide.md#arguments-1
	// also from ../enter/enter.c
	args := []string{
		"/appexec",
		fmt.Sprintf("/opt/stage2/%s/rootfs", appName),
		"/", // as in ../enter/enter.c - this should be app.WorkingDirectory
		fmt.Sprintf("/rkt/env/%s", appName),
		u.Uid,
		u.Gid,
		"-e", /* entering phase */
		"--",
	}
	return append(args, flag.Args()...)
}

func execSSH() error {
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
		"-p", kvmSSHPort, // port to connect
		"-o", "StrictHostKeyChecking=no", // do not check changing host keys
		"-o", "UserKnownHostsFile=/dev/null", // do not add host key to default knownhosts file
		"-o", "LogLevel=quiet", // do not log minor informations
		podDefaultIP,
	}
	args = append(args, getAppexecArgs()...)

	// this should not return in case of success
	err = syscall.Exec(sshPath, args, os.Environ())
	return errwrap.Wrap(errors.New("cannot exec to ssh"), err)
}

func main() {
	flag.Parse()

	log.SetDebug(debug)
	diag.SetDebug(debug)

	if !debug {
		diag.SetOutput(ioutil.Discard)
	}

	if appName == "" {
		log.Fatal("--appname not set to correct value")
	}

	// execSSH should return only with error
	log.Error(execSSH())
	os.Exit(2)
}
