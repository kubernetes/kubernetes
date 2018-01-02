// Copyright 2015 The etcd Authors
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
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"time"

	"github.com/coreos/etcd/pkg/fileutil"
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

	cmd     *exec.Cmd
	logfile *os.File

	cfg AgentConfig
}

type AgentConfig struct {
	EtcdPath      string
	LogDir        string
	FailpointAddr string
	UseRoot       bool
}

func newAgent(cfg AgentConfig) (*Agent, error) {
	// check if the file exists
	_, err := os.Stat(cfg.EtcdPath)
	if err != nil {
		return nil, err
	}

	c := exec.Command(cfg.EtcdPath)

	err = fileutil.TouchDirAll(cfg.LogDir)
	if err != nil {
		return nil, err
	}

	var f *os.File
	f, err = os.Create(filepath.Join(cfg.LogDir, "etcd.log"))
	if err != nil {
		return nil, err
	}

	return &Agent{state: stateUninitialized, cmd: c, logfile: f, cfg: cfg}, nil
}

// start starts a new etcd process with the given args.
func (a *Agent) start(args ...string) error {
	a.cmd = exec.Command(a.cmd.Path, args...)
	a.cmd.Env = []string{"GOFAIL_HTTP=" + a.cfg.FailpointAddr}
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
func (a *Agent) stopWithSig(sig os.Signal) error {
	if a.state != stateStarted {
		return nil
	}

	err := stopWithSig(a.cmd, sig)
	if err != nil {
		return err
	}

	a.state = stateStopped
	return nil
}

func stopWithSig(cmd *exec.Cmd, sig os.Signal) error {
	err := cmd.Process.Signal(sig)
	if err != nil {
		return err
	}

	errc := make(chan error)
	go func() {
		_, ew := cmd.Process.Wait()
		errc <- ew
		close(errc)
	}()

	select {
	case <-time.After(5 * time.Second):
		cmd.Process.Kill()
	case e := <-errc:
		return e
	}
	err = <-errc
	return err
}

// restart restarts the stopped etcd process.
func (a *Agent) restart() error {
	return a.start(a.cmd.Args[1:]...)
}

func (a *Agent) cleanup() error {
	// exit with stackstrace
	if err := a.stopWithSig(syscall.SIGQUIT); err != nil {
		return err
	}
	a.state = stateUninitialized

	a.logfile.Close()
	if err := archiveLogAndDataDir(a.cfg.LogDir, a.dataDir()); err != nil {
		return err
	}

	if err := fileutil.TouchDirAll(a.cfg.LogDir); err != nil {
		return err
	}

	f, err := os.Create(filepath.Join(a.cfg.LogDir, "etcd.log"))
	if err != nil {
		return err
	}
	a.logfile = f

	// https://www.kernel.org/doc/Documentation/sysctl/vm.txt
	// https://github.com/torvalds/linux/blob/master/fs/drop_caches.c
	cmd := exec.Command("/bin/sh", "-c", `echo "echo 1 > /proc/sys/vm/drop_caches" | sudo sh`)
	if err := cmd.Run(); err != nil {
		plog.Infof("error when cleaning page cache (%v)", err)
	}
	return nil
}

// terminate stops the exiting etcd process the agent started
// and removes the data dir.
func (a *Agent) terminate() error {
	err := a.stopWithSig(syscall.SIGTERM)
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
	if !a.cfg.UseRoot {
		return nil
	}
	return netutil.DropPort(port)
}

func (a *Agent) recoverPort(port int) error {
	if !a.cfg.UseRoot {
		return nil
	}
	return netutil.RecoverPort(port)
}

func (a *Agent) setLatency(ms, rv int) error {
	if !a.cfg.UseRoot {
		return nil
	}
	if ms == 0 {
		return netutil.RemoveLatency()
	}
	return netutil.SetLatency(ms, rv)
}

func (a *Agent) status() client.Status {
	return client.Status{State: a.state}
}

func (a *Agent) dataDir() string {
	datadir := filepath.Join(a.cmd.Path, "*.etcd")
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

func archiveLogAndDataDir(logDir string, datadir string) error {
	dir := filepath.Join("failure_archive", fmt.Sprint(time.Now().Format(time.RFC3339)))
	if existDir(dir) {
		dir = filepath.Join("failure_archive", fmt.Sprint(time.Now().Add(time.Second).Format(time.RFC3339)))
	}
	if err := fileutil.TouchDirAll(dir); err != nil {
		return err
	}
	if err := os.Rename(logDir, filepath.Join(dir, filepath.Base(logDir))); err != nil {
		if !os.IsNotExist(err) {
			return err
		}
	}
	if err := os.Rename(datadir, filepath.Join(dir, filepath.Base(datadir))); err != nil {
		if !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}
