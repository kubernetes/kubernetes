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

// To run the e2e tests against one or more hosts on gce: $ go run run_e2e.go --hosts <comma separated hosts>
// Requires gcloud compute ssh access to the hosts
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

const gray = "\033[1;30m"
const blue = "\033[0;34m"
const noColour = "\033[0m"

var u = sync.WaitGroup{}
var zone = flag.String("zone", "", "gce zone the hosts live in")
var hosts = flag.String("hosts", "", "hosts to test")
var wait = flag.Bool("wait", false, "if true, wait for input before running tests")
var kubeOutputRelPath = flag.String("k8s-build-output", "_output/local/bin/linux/amd64", "Where k8s binary files are written")

var kubeRoot = ""

const buildScriptRelPath = "hack/build-go.sh"
const ginkgoTestRelPath = "test/e2e_node"
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

	results := make(chan *TestResult)
	hs := strings.Split(*hosts, ",")
	for _, h := range hs {
		go func(host string) { results <- runTests(host) }(h)
	}

	// Maybe wait for user input before running tests
	if *wait {
		WaitForUser()
	}

	// Wait for all tests to complete and emit the results
	errCount := 0
	for i := 0; i < len(hs); i++ {
		tr := <-results
		host := tr.fullhost
		if tr.err != nil {
			errCount++
			glog.Infof("%s================================================================%s", blue, noColour)
			glog.Infof("Failure Finished Host %s Test Suite %s %v", host, tr.testCombinedOutput, tr.err)
			glog.V(2).Infof("----------------------------------------------------------------")
			glog.V(5).Infof("Host %s Etcd Logs\n%s%s%s", host, gray, tr.etcdCombinedOutput, noColour)
			glog.V(5).Infof("----------------------------------------------------------------")
			glog.V(5).Infof("Host %s Apiserver Logs\n%s%s%s", host, gray, tr.apiServerCombinedOutput, noColour)
			glog.V(5).Infof("----------------------------------------------------------------")
			glog.V(2).Infof("Host %s Kubelet Logs\n%s%s%s", host, gray, tr.kubeletCombinedOutput, noColour)
			glog.Infof("%s================================================================%s", blue, noColour)
		} else {
			glog.Infof("================================================================")
			glog.Infof("Success Finished Host %s Test Suite %s", host, tr.testCombinedOutput)
			glog.Infof("================================================================")
		}
	}

	// Set the exit code if there were failures
	if errCount > 0 {
		glog.Errorf("Failure: %d errors encountered.", errCount)
		os.Exit(1)
	}
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

type TestResult struct {
	fullhost                string
	err                     error
	testCombinedOutput      string
	etcdCombinedOutput      string
	apiServerCombinedOutput string
	kubeletCombinedOutput   string
}

func runTests(fullhost string) *TestResult {
	result := &TestResult{fullhost: fullhost}

	host := strings.Split(fullhost, ".")[0]
	c := gcloud.NewGCloudClient(host, *zone)
	// TODO(pwittrock): Come up with something better for bootstrapping the environment.
	eh, err := c.RunAndWaitTillHealthy(
		false, false, "4001", healthyTimeoutDuration, "v2/keys/", "etcd", "--data-dir", "./", "--name", "e2e-node")
	defer func() {
		eh.TearDown()
		result.etcdCombinedOutput = fmt.Sprintf("%s", eh.CombinedOutput.Bytes())
	}()
	if err != nil {
		result.err = fmt.Errorf("Host %s failed to run command %v", host, err)
		return result
	}

	apiBin := filepath.Join(kubeRoot, *kubeOutputRelPath, "kube-apiserver")
	ah, err := c.RunAndWaitTillHealthy(
		true, true, "8080", healthyTimeoutDuration, "healthz", apiBin, "--service-cluster-ip-range",
		"10.0.0.1/24", "--insecure-bind-address", "0.0.0.0", "--etcd-servers", "http://127.0.0.1:4001",
		"--v", "2", "--alsologtostderr", "--kubelet-port", "10250")
	defer func() {
		ah.TearDown()
		result.apiServerCombinedOutput = fmt.Sprintf("%s", ah.CombinedOutput.Bytes())
	}()
	if err != nil {
		result.err = fmt.Errorf("Host %s failed to run command %v", host, err)
		return result
	}

	kubeletBin := filepath.Join(kubeRoot, *kubeOutputRelPath, "kubelet")
	// TODO: Used --v 4 or higher and upload to gcs instead of printing to the console
	// TODO: Copy /var/log/messages and upload to GCS for failed tests
	kh, err := c.RunAndWaitTillHealthy(
		true, true, "10255", healthyTimeoutDuration, "healthz", kubeletBin, "--api-servers", "http://127.0.0.1:8080",
		"--v", "2", "--alsologtostderr", "--address", "0.0.0.0", "--port", "10250")
	defer func() {
		kh.TearDown()
		result.kubeletCombinedOutput = fmt.Sprintf("%s", kh.CombinedOutput.Bytes())
	}()
	if err != nil {
		result.err = fmt.Errorf("Host %s failed to run command %v", host, err)
	}

	// Run the tests
	glog.Infof("Kubelet healthy on host %s", host)
	glog.Infof("Kubelet host %s tunnel running on port %s", host, ah.LPort)
	u.Wait()
	glog.Infof("Running ginkgo tests against host %s", host)
	ginkgoTests := filepath.Join(kubeRoot, ginkgoTestRelPath)
	out, err := exec.Command(
		"ginkgo", ginkgoTests, "--",
		"--kubelet-address", fmt.Sprintf("http://127.0.0.1:%s", kh.LPort),
		"--api-server-address", fmt.Sprintf("http://127.0.0.1:%s", ah.LPort),
		"--node-name", fullhost,
		"--v", "2", "--alsologtostderr").CombinedOutput()

	result.err = err
	result.testCombinedOutput = fmt.Sprintf("%s", out)
	return result
}
