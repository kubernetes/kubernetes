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

package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"runtime"

	"github.com/golang/glog"
	"k8s.io/kubernetes/test/e2e_node/gcloud"
	"path/filepath"
)

type RunFunc func(host string, port string) ([]byte, error)

type Result struct {
	host   string
	output []byte
	err    error
}

var u = sync.WaitGroup{}
var zone = flag.String("zone", "", "gce zone the hosts live in")
var hosts = flag.String("hosts", "", "hosts to test")
var wait = flag.Bool("wait", false, "if true, wait for input before running tests")
var kubeOutputRelPath = flag.String("k8s-build-output", "_output/local/bin/linux/amd64", "Where k8s binary files are written")

var kubeRoot = ""

const buildScriptRelPath = "hack/build-go.sh"
const ginkoTestRelPath = "test/e2e_node"
const healthyTimeoutDuration = time.Minute * 3

func main() {
	flag.Parse()
	if *hosts == "" {
		glog.Fatalf("Must specific --hosts flag")
	}

	// Figure out the kube root
	_, path, _, _ := runtime.Caller(0)
	kubeRoot, _ = filepath.Split(path)
	kubeRoot = strings.Split(kubeRoot, "/test/e2e_node")[0]

	// Build the go code
	out, err := exec.Command(filepath.Join(kubeRoot, buildScriptRelPath)).CombinedOutput()
	if err != nil {
		glog.Fatalf("Failed to build go packages %s: %v", out, err)
	}

	// Copy kubelet to each host and run test
	if *wait {
		u.Add(1)
	}

	w := sync.WaitGroup{}
	for _, h := range strings.Split(*hosts, ",") {
		w.Add(1)
		go func(host string) {
			out, err := runTests(host)
			if err != nil {
				glog.Infof("Failure Finished Test Suite %s %v", out, err)
			} else {
				glog.Infof("Success Finished Test Suite %s", out)
			}
			w.Done()
		}(h)
	}

	// Maybe wait for user input before running tests
	if *wait {
		WaitForUser()
	}

	// Wait for the tests to finish
	w.Wait()
	glog.Infof("All hosts finished")
}

func WaitForUser() {
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Printf("Enter \"y\" to run tests\n")
	for scanner.Scan() {
		if strings.ToUpper(scanner.Text()) != "Y\n" {
			break
		}
		fmt.Printf("Enter \"y\" to run tests\n")
	}
	u.Done()
}

func runTests(host string) ([]byte, error) {
	c := gcloud.NewGCloudClient(host, *zone)
	// TODO(pwittrock): Come up with something better for bootstrapping the environment.
	etcdBin := filepath.Join(kubeRoot, "third_party/etcd/etcd")
	eh, err := c.CopyAndWaitTillHealthy(false, "4001", healthyTimeoutDuration, "v2/keys/", etcdBin)
	defer func() { eh.TearDown() }()
	if err != nil {
		return nil, fmt.Errorf("Host %s failed to run command %v", host, err)
	}

	apiBin := filepath.Join(kubeRoot, *kubeOutputRelPath, "kube-apiserver")
	ah, err := c.CopyAndWaitTillHealthy(
		true, "8080", healthyTimeoutDuration, "healthz", apiBin, "--service-cluster-ip-range",
		"10.0.0.1/24", "--insecure-bind-address", "0.0.0.0", "--etcd-servers", "http://localhost:4001",
		"--cluster-name", "kubernetes", "--v", "2", "--kubelet-port", "10250")
	defer func() { ah.TearDown() }()
	if err != nil {
		return nil, fmt.Errorf("Host %s failed to run command %v", host, err)
	}

	kubeletBin := filepath.Join(kubeRoot, *kubeOutputRelPath, "kubelet")
	kh, err := c.CopyAndWaitTillHealthy(
		true, "4194", healthyTimeoutDuration, "healthz", kubeletBin, "--api-servers", "http://localhost:8080",
		"--logtostderr", "--address", "0.0.0.0", "--port", "10250")
	defer func() { kh.TearDown() }()
	if err != nil {
		return nil, fmt.Errorf("Host %s failed to run command %v", host, err)
	}

	// Run the tests
	glog.Infof("Kubelet healthy on host %s", host)
	glog.Infof("Kubelet host %s tunnel running on port %s", host, ah.LPort)
	u.Wait()
	glog.Infof("Running ginkgo tests against host %s", host)
	ginkoTests := filepath.Join(kubeRoot, ginkoTestRelPath)
	return exec.Command(
		"ginkgo", ginkoTests, "--",
		"--kubelet-host", "localhost", "--kubelet-port", kh.LPort,
		"--api-server-host", "localhost", "--api-server-port", kh.LPort,
		"-logtostderr").CombinedOutput()
}
