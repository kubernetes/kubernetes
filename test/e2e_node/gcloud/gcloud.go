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

package gcloud

import (
	"bytes"
	"errors"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/golang/glog"
)

var freePortRegexp = regexp.MustCompile(".+:([0-9]+)")

type TearDown func() *RunResult

type GCloudClient interface {
	RunAndWaitTillHealthy(
		sudo bool, copyBin bool, remotePort string,
		timeout time.Duration, healthUrl string, bin string, args ...string) (*CmdHandle, error)
}

type gCloudClientImpl struct {
	host string
	zone string
}

type RunResult struct {
	out []byte
	err error
	cmd string
}

type CmdHandle struct {
	TearDown       TearDown
	CombinedOutput bytes.Buffer
	Output         chan RunResult
	LPort          string
}

func NewGCloudClient(host string, zone string) GCloudClient {
	return &gCloudClientImpl{host, zone}
}

func (gc *gCloudClientImpl) Command(cmd string, moreargs ...string) ([]byte, error) {
	args := append([]string{"compute", "ssh"})
	if gc.zone != "" {
		args = append(args, "--zone", gc.zone)
	}
	args = append(args, gc.host, "--", cmd)
	args = append(args, moreargs...)
	glog.V(2).Infof("Command gcloud %s", strings.Join(args, " "))
	return exec.Command("gcloud", args...).CombinedOutput()
}

func (gc *gCloudClientImpl) TunnelCommand(sudo bool, lPort string, rPort string, dir string, cmd string, moreargs ...string) *exec.Cmd {
	tunnelStr := fmt.Sprintf("-L %s:localhost:%s", lPort, rPort)
	args := []string{"compute", "ssh"}
	if gc.zone != "" {
		args = append(args, "--zone", gc.zone)
	}
	args = append(args, "--ssh-flag", tunnelStr, gc.host, "--")
	args = append(args, "cd", dir, ";")
	if sudo {
		args = append(args, "sudo")
	}
	args = append(args, cmd)
	args = append(args, moreargs...)
	glog.V(2).Infof("Command gcloud %s", strings.Join(args, " "))
	return exec.Command("gcloud", args...)
}

func (gc *gCloudClientImpl) CopyToHost(from string, to string) ([]byte, error) {
	rto := fmt.Sprintf("%s:%s", gc.host, to)
	args := []string{"compute", "copy-files"}
	if gc.zone != "" {
		args = append(args, "--zone", gc.zone)
	}
	args = append(args, from, rto)
	glog.V(2).Infof("Command gcloud %s", strings.Join(args, " "))
	return exec.Command("gcloud", args...).CombinedOutput()
}

func (gc *gCloudClientImpl) Run(
	sudo bool, copyBin bool, remotePort string, bin string, args ...string) *CmdHandle {

	h := &CmdHandle{}
	h.Output = make(chan RunResult)

	rand.Seed(time.Now().UnixNano())

	// Define where we will copy the temp binary
	tDir := fmt.Sprintf("/tmp/gcloud-e2e-%d", rand.Int31())
	_, f := filepath.Split(bin)
	cmd := f
	if copyBin {
		cmd = filepath.Join(tDir, f)
	}
	h.LPort = getLocalPort()

	h.TearDown = func() *RunResult {
		out, err := gc.Command("sudo", "pkill", f)
		if err != nil {
			return &RunResult{out, err, fmt.Sprintf("pkill %s", f)}
		}
		out, err = gc.Command("rm", "-rf", tDir)
		if err != nil {
			return &RunResult{out, err, fmt.Sprintf("rm -rf %s", tDir)}
		}
		return &RunResult{}
	}

	// Run the commands in a Go fn so that this method doesn't block when writing to a channel
	// to report an error
	go func() {
		// Create the tmp directory
		out, err := gc.Command("mkdir", "-p", tDir)

		// Work around for gcloud flakiness - TODO: debug why gcloud sometimes cannot find credentials for some hosts
		// If there was an error about credentials, retry making the directory 6 times to see if it can be resolved
		// This is to help debug if the credential issues are persistent for a given host on a given run, or transient
		// And if downstream gcloud commands are also impacted
		for i := 0; i < 6 && err != nil && strings.Contains(string(out), "does not have any valid credentials"); i++ {
			glog.Warningf("mkdir failed on host %s due to credential issues, retrying in 5 seconds %v %s", gc.host, err, out)
			time.Sleep(5 * time.Second)
			out, err = gc.Command("mkdir", "-p", tDir)
		}
		if err != nil {
			glog.Errorf("mkdir failed %v %s", err, out)
			h.Output <- RunResult{out, err, fmt.Sprintf("mkdir -p %s", tDir)}
			return
		}

		// Copy the binary
		if copyBin {
			out, err = gc.CopyToHost(bin, tDir)
			if err != nil {
				glog.Errorf("copy-files failed %v %s", err, out)
				h.Output <- RunResult{out, err, fmt.Sprintf("copy-files %s %s", bin, tDir)}
				return
			}
		}

		c := gc.TunnelCommand(sudo, h.LPort, remotePort, tDir, cmd, args...)
		c.Stdout = &h.CombinedOutput
		c.Stderr = &h.CombinedOutput
		go func() {
			// Start the process
			err = c.Run()
			if err != nil {
				glog.Errorf("command failed %v %s", err, h.CombinedOutput.Bytes())
				h.Output <- RunResult{h.CombinedOutput.Bytes(), err, fmt.Sprintf("%s %s", cmd, strings.Join(args, " "))}
				return
			}
		}()
	}()
	return h
}

func (gc *gCloudClientImpl) RunAndWaitTillHealthy(
	sudo bool, copyBin bool,
	remotePort string, timeout time.Duration, healthUrl string, bin string, args ...string) (*CmdHandle, error) {
	h := gc.Run(sudo, copyBin, remotePort, bin, args...)
	eTime := time.Now().Add(timeout)
	done := false
	for eTime.After(time.Now()) && !done {
		select {
		case r := <-h.Output:
			glog.V(2).Infof("Error running %s Output:\n%s Error:\n%v", r.cmd, r.out, r.err)
			return h, r.err
		case <-time.After(2 * time.Second):
			resp, err := http.Get(fmt.Sprintf("http://localhost:%s/%s", h.LPort, healthUrl))
			if err == nil && resp.StatusCode == http.StatusOK {
				done = true
				break
			}
		}
	}
	if !done {
		return h, errors.New(fmt.Sprintf("Timeout waiting for service to be healthy at http://localhost:%s/%s", h.LPort, healthUrl))
	}
	glog.Info("Healthz Success")
	return h, nil
}

// GetLocalPort returns a free local port that can be used for ssh tunneling
func getLocalPort() string {
	l, _ := net.Listen("tcp", ":0")
	defer l.Close()
	return freePortRegexp.FindStringSubmatch(l.Addr().String())[1]
}
