/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package e2e_node

import (
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/golang/glog"
)

var serverStartTimeout = flag.Duration("server-start-timeout", time.Second*30, "Time to wait for each server to become healthy.")

type e2eService struct {
	etcdCmd              *exec.Cmd
	etcdCombinedOut      bytes.Buffer
	etcdDataDir          string
	apiServerCmd         *exec.Cmd
	apiServerCombinedOut bytes.Buffer
	kubeletCmd           *exec.Cmd
	kubeletCombinedOut   bytes.Buffer
	nodeName             string
}

func newE2eService(nodeName string) *e2eService {
	return &e2eService{nodeName: nodeName}
}

func (es *e2eService) start() error {
	if _, err := getK8sBin("kubelet"); err != nil {
		return err
	}
	if _, err := getK8sBin("kube-apiserver"); err != nil {
		return err
	}

	cmd, err := es.startEtcd()
	if err != nil {
		return err
	}
	es.etcdCmd = cmd

	cmd, err = es.startApiServer()
	if err != nil {
		return err
	}
	es.apiServerCmd = cmd

	cmd, err = es.startKubeletServer()
	if err != nil {
		return err
	}
	es.kubeletCmd = cmd

	return nil
}

func (es *e2eService) stop() {
	if es.kubeletCmd != nil {
		err := es.kubeletCmd.Process.Kill()
		if err != nil {
			glog.Errorf("Failed to stop kubelet.\n%v", err)
		}
	}
	if es.apiServerCmd != nil {
		err := es.apiServerCmd.Process.Kill()
		if err != nil {
			glog.Errorf("Failed to stop be-apiserver.\n%v", err)
		}
	}
	if es.etcdCmd != nil {
		err := es.etcdCmd.Process.Kill()
		if err != nil {
			glog.Errorf("Failed to stop etcd.\n%v", err)
		}
	}
	if es.etcdDataDir != "" {
		err := os.RemoveAll(es.etcdDataDir)
		if err != nil {
			glog.Errorf("Failed to delete etcd data directory %s.\n%v", es.etcdDataDir, err)
		}
	}
}

func (es *e2eService) startEtcd() (*exec.Cmd, error) {
	dataDir, err := ioutil.TempDir("", "node-e2e")
	if err != nil {
		return nil, err
	}
	es.etcdDataDir = dataDir
	return es.startServer(healthCheckCommand{
		combinedOut:    &es.etcdCombinedOut,
		healthCheckUrl: "http://127.0.0.1:4001/v2/keys",
		command:        "etcd",
		args:           []string{"--data-dir", dataDir, "--name", "e2e-node"},
	})
}

func (es *e2eService) startApiServer() (*exec.Cmd, error) {
	return es.startServer(
		healthCheckCommand{
			combinedOut:    &es.apiServerCombinedOut,
			healthCheckUrl: "http://127.0.0.1:8080/healthz",
			command:        "sudo",
			args: []string{getApiServerBin(),
				"--v", "2", "--logtostderr", "--log_dir", "./",
				"--etcd-servers", "http://127.0.0.1:4001",
				"--insecure-bind-address", "0.0.0.0",
				"--service-cluster-ip-range", "10.0.0.1/24",
				"--kubelet-port", "10250"},
		})
}

func (es *e2eService) startKubeletServer() (*exec.Cmd, error) {
	return es.startServer(
		healthCheckCommand{
			combinedOut:    &es.kubeletCombinedOut,
			healthCheckUrl: "http://127.0.0.1:10255/healthz",
			command:        "sudo",
			args: []string{getKubeletServerBin(),
				"--v", "2", "--logtostderr", "--log_dir", "./",
				"--api-servers", "http://127.0.0.1:8080",
				"--address", "0.0.0.0",
				"--port", "10250",
				"--hostname-override", es.nodeName, // Required because hostname is inconsistent across hosts
			},
		})
}

func (es *e2eService) startServer(hcc healthCheckCommand) (*exec.Cmd, error) {
	cmdErrorChan := make(chan error)
	cmd := exec.Command(hcc.command, hcc.args...)
	cmd.Stdout = hcc.combinedOut
	cmd.Stderr = hcc.combinedOut
	go func() {
		err := cmd.Run()
		if err != nil {
			cmdErrorChan <- fmt.Errorf("%v Exited with status %v.  Output:\n%s", hcc, err, *hcc.combinedOut)
		}
		close(cmdErrorChan)
	}()

	endTime := time.Now().Add(*serverStartTimeout)
	for endTime.After(time.Now()) {
		select {
		case err := <-cmdErrorChan:
			return nil, err
		case <-time.After(time.Second):
			resp, err := http.Get(hcc.healthCheckUrl)
			if err == nil && resp.StatusCode == http.StatusOK {
				return cmd, nil
			}
		}
	}
	return nil, fmt.Errorf("Timeout waiting for service %v", hcc)
}

type healthCheckCommand struct {
	healthCheckUrl string
	command        string
	args           []string
	combinedOut    *bytes.Buffer
}

func (hcc *healthCheckCommand) String() string {
	return fmt.Sprintf("`%s %s` %s", hcc.command, strings.Join(hcc.args, " "), hcc.healthCheckUrl)
}
