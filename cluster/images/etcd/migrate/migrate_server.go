/*
Copyright 2018 The Kubernetes Authors.

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

package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/golang/glog"
)

// EtcdMigrateServer manages starting and stopping a versioned etcd server binary.
type EtcdMigrateServer struct {
	cfg    *EtcdMigrateCfg
	client EtcdMigrateClient
	cmd    *exec.Cmd
}

// NewEtcdMigrateServer creates a EtcdMigrateServer for starting and stopping a etcd server at the given version.
func NewEtcdMigrateServer(cfg *EtcdMigrateCfg, client EtcdMigrateClient) *EtcdMigrateServer {
	return &EtcdMigrateServer{cfg: cfg, client: client}
}

// Start starts an etcd server as a separate process, waits until it has started, and returns a exec.Cmd.
func (r *EtcdMigrateServer) Start(version *EtcdVersion) error {
	etcdCmd := exec.Command(
		fmt.Sprintf("%s/etcd-%s", r.cfg.binPath, version),
		"--name", r.cfg.name,
		"--initial-cluster", r.cfg.initialCluster,
		"--debug",
		"--data-dir", r.cfg.dataDirectory,
		"--listen-client-urls", fmt.Sprintf("http://127.0.0.1:%d", r.cfg.port),
		"--advertise-client-urls", fmt.Sprintf("http://127.0.0.1:%d", r.cfg.port),
		"--listen-peer-urls", r.cfg.peerListenUrls,
		"--initial-advertise-peer-urls", r.cfg.peerAdvertiseUrls,
	)
	if r.cfg.etcdServerArgs != "" {
		extraArgs := strings.Fields(r.cfg.etcdServerArgs)
		etcdCmd.Args = append(etcdCmd.Args, extraArgs...)
	}
	fmt.Printf("Starting server %s: %+v\n", r.cfg.name, etcdCmd.Args)

	etcdCmd.Stdout = os.Stdout
	etcdCmd.Stderr = os.Stderr
	err := etcdCmd.Start()
	if err != nil {
		return err
	}
	interval := time.NewTicker(time.Millisecond * 500)
	defer interval.Stop()
	done := make(chan bool)
	go func() {
		time.Sleep(time.Minute * 2)
		done <- true
	}()
	for {
		select {
		case <-interval.C:
			err := r.client.SetEtcdVersionKeyValue(version)
			if err != nil {
				glog.Infof("Still waiting for etcd to start, current error: %v", err)
				// keep waiting
			} else {
				glog.Infof("Etcd on port %d is up.", r.cfg.port)
				r.cmd = etcdCmd
				return nil
			}
		case <-done:
			err = etcdCmd.Process.Kill()
			if err != nil {
				return fmt.Errorf("error killing etcd: %v", err)
			}
			return fmt.Errorf("Timed out waiting for etcd on port %d", r.cfg.port)
		}
	}
}

// Stop terminates the etcd server process. If the etcd server process has not been started
// or is not still running, this returns an error.
func (r *EtcdMigrateServer) Stop() error {
	if r.cmd == nil {
		return fmt.Errorf("cannot stop EtcdMigrateServer that has not been started")
	}
	err := r.cmd.Process.Signal(os.Interrupt)
	if err != nil {
		return fmt.Errorf("error sending SIGINT to etcd for graceful shutdown: %v", err)
	}
	gracefulWait := time.Minute * 2
	stopped := make(chan bool)
	timedout := make(chan bool)
	go func() {
		time.Sleep(gracefulWait)
		timedout <- true
	}()
	go func() {
		select {
		case <-stopped:
			return
		case <-timedout:
			glog.Infof("etcd server has not terminated gracefully after %s, killing it.", gracefulWait)
			r.cmd.Process.Kill()
			return
		}
	}()
	err = r.cmd.Wait()
	stopped <- true
	if exiterr, ok := err.(*exec.ExitError); ok {
		glog.Infof("etcd server stopped (signal: %s)", exiterr.Error())
		// stopped
	} else if err != nil {
		return fmt.Errorf("error waiting for etcd to stop: %v", err)
	}
	glog.Infof("Stopped etcd server %s", r.cfg.name)
	return nil
}
