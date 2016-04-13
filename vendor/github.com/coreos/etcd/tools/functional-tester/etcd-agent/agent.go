// Copyright 2015 CoreOS, Inc.
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
	"fmt"
	"net"
	"os"
	"os/exec"
	"path"
	"syscall"
	"time"

	"github.com/coreos/etcd/pkg/netutil"
	"github.com/coreos/etcd/tools/functional-tester/etcd-agent/client"
)

const (
	stateUninitialized = "uninitialized"
	stateStarted       = "started"
	stateStopped       = "stopped"
	stateTerminated    = "terminated"
)

type Agent struct {
	state string // the state of etcd process

	cmd         *exec.Cmd
	logfile     *os.File
	etcdLogPath string

	l net.Listener
}

func newAgent(etcd, etcdLogPath string) (*Agent, error) {
	// check if the file exists
	_, err := os.Stat(etcd)
	if err != nil {
		return nil, err
	}

	c := exec.Command(etcd)

	f, err := os.Create(etcdLogPath)
	if err != nil {
		return nil, err
	}

	return &Agent{state: stateUninitialized, cmd: c, logfile: f, etcdLogPath: etcdLogPath}, nil
}

// start starts a new etcd process with the given args.
func (a *Agent) start(args ...string) error {
	a.cmd = exec.Command(a.cmd.Path, args...)
	a.cmd.Stdout = a.logfile
	a.cmd.Stderr = a.logfile
	err := a.cmd.Start()
	if err != nil {
		return err
	}

	a.state = stateStarted
	return nil
}

// stop stops the existing etcd process the agent started.
func (a *Agent) stop() error {
	if a.state != stateStarted {
		return nil
	}

	err := sigtermAndWait(a.cmd)
	if err != nil {
		return err
	}

	a.state = stateStopped
	return nil
}

func sigtermAndWait(cmd *exec.Cmd) error {
	err := cmd.Process.Signal(syscall.SIGTERM)
	if err != nil {
		return err
	}

	errc := make(chan error)
	go func() {
		_, err := cmd.Process.Wait()
		errc <- err
		close(errc)
	}()

	select {
	case <-time.After(5 * time.Second):
		cmd.Process.Kill()
	case err := <-errc:
		return err
	}
	err = <-errc
	return err
}

// restart restarts the stopped etcd process.
func (a *Agent) restart() error {
	a.cmd = exec.Command(a.cmd.Path, a.cmd.Args[1:]...)
	a.cmd.Stdout = a.logfile
	a.cmd.Stderr = a.logfile
	err := a.cmd.Start()
	if err != nil {
		return err
	}

	a.state = stateStarted
	return nil
}

func (a *Agent) cleanup() error {
	if err := a.stop(); err != nil {
		return err
	}
	a.state = stateUninitialized

	a.logfile.Close()
	if err := archiveLogAndDataDir(a.etcdLogPath, a.dataDir()); err != nil {
		return err
	}
	f, err := os.Create(a.etcdLogPath)
	a.logfile = f
	if err != nil {
		return err
	}

	// https://www.kernel.org/doc/Documentation/sysctl/vm.txt
	// https://github.com/torvalds/linux/blob/master/fs/drop_caches.c
	cmd := exec.Command("/bin/sh", "-c", `echo "echo 1 > /proc/sys/vm/drop_caches" | sudo sh`)
	if err := cmd.Run(); err != nil {
		plog.Printf("error when cleaning page cache (%v)", err)
	}
	return nil
}

// terminate stops the exiting etcd process the agent started
// and removes the data dir.
func (a *Agent) terminate() error {
	err := a.stop()
	if err != nil {
		return err
	}
	err = os.RemoveAll(a.dataDir())
	if err != nil {
		return err
	}
	a.state = stateTerminated
	return nil
}

func (a *Agent) dropPort(port int) error {
	return netutil.DropPort(port)
}

func (a *Agent) recoverPort(port int) error {
	return netutil.RecoverPort(port)
}

func (a *Agent) status() client.Status {
	return client.Status{State: a.state}
}

func (a *Agent) dataDir() string {
	datadir := path.Join(a.cmd.Path, "*.etcd")
	args := a.cmd.Args
	// only parse the simple case like "--data-dir /var/lib/etcd"
	for i, arg := range args {
		if arg == "--data-dir" {
			datadir = args[i+1]
			break
		}
	}
	return datadir
}

func existDir(fpath string) bool {
	st, err := os.Stat(fpath)
	if err != nil {
		if os.IsNotExist(err) {
			return false
		}
	} else {
		return st.IsDir()
	}
	return false
}

func archiveLogAndDataDir(log string, datadir string) error {
	dir := path.Join("failure_archive", fmt.Sprint(time.Now().Format(time.RFC3339)))
	if existDir(dir) {
		dir = path.Join("failure_archive", fmt.Sprint(time.Now().Add(time.Second).Format(time.RFC3339)))
	}
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	if err := os.Rename(log, path.Join(dir, path.Base(log))); err != nil {
		if !os.IsNotExist(err) {
			return err
		}
	}
	if err := os.Rename(datadir, path.Join(dir, path.Base(datadir))); err != nil {
		if !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}
