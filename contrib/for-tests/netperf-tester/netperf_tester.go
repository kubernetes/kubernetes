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

/*
	NOTE: This is a tool for running netperf tests on a cluster.
	The cluster should have two worker nodes.
	Call this with run.sh or set the environment variable KUBECTL to have
	the path to the kubectl binary you want it to use or set --kubectl=
	in the cli flags.
*/

package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"time"
)

var (
	kubectl      string
	outputPath   string
	outputFile   *os.File
	numNetperf   int
	printResults bool
	debug        bool
	logger       *log.Logger
	deletePods   bool
)

func init() {
	flag.StringVar(&kubectl, "kubectl", "", "the full path to the kubectl binary")
	flag.StringVar(&outputPath, "output", "", "the full path to the csv file to output")
	flag.IntVar(&numNetperf, "number", 1000, "the number of times to run netperf")
	flag.BoolVar(&printResults, "print", true, "print results to standard out")
	flag.BoolVar(&debug, "debug", true, "print debug to log")
	flag.BoolVar(&deletePods, "cleanup", false, "delete test pods when done")
}

func main() {
	flag.Parse()
	if kubectl == "" {
		log.Fatal("kubectl path not set! please set --kubectl=/path/to/kubectl")
	}
	if debug {
		logger = log.New(os.Stderr, "", log.LstdFlags)
	} else {
		f, _ := os.Open(os.DevNull)
		logger = log.New(f, "", 0)
	}
	if outputPath != "" {
		err := error(nil)
		outputFile, err = os.Create(outputPath)
		if err != nil {
			log.Fatalf("Failed to open output file for path %s Error: %v", outputPath, err)
		}
	}
	logger.Printf("kubectl path := %s\n", kubectl)
	// add the test host and client pods
	addServices()
	// wait for pods to come online
	waitForServicesToBeRunning()
	// get and display test pods
	args := []string{"get", "pods", "-o=wide"}
	err := runCommandInShell(kubectl, args)
	if err != nil {
		logger.Printf("Error running command: %v", err)
	}
	// run the tests
	if err := runTests(); err != nil {
		logger.Printf("Error running tests: %v", err)
	}
	// cleanup services
	if deletePods {
		removeServices()
	}
}

func runCommandInShell(command string, args []string) error {
	logger.Printf("Running %s with args := %v\n", command, args)
	cmd := exec.Command(command, args...)
	cmd.Stderr = os.Stderr
	cmd.Stdout = os.Stdout
	cmd.Stdin = os.Stdin
	return cmd.Run()
}

func addServices() {
	// setup pod with server to access from test pod
	addService("netperf-tester-host", "paultiplady/netserver:ubuntu.2", 12865)
	// setup test pod
	addService("netperf-tester-client", "paultiplady/netserver:ubuntu.2", 12865)
}

func addService(serviceName, image string, port int) {
	args := []string{"run", serviceName, "--image=" + image, fmt.Sprintf("--port=%d", port), "--hostport=65530"}
	logger.Printf("Running %s with args := %v\n", kubectl, args)
	bytes, err := exec.Command(kubectl, args...).CombinedOutput()
	if err != nil {
		if strings.HasSuffix(string(bytes), "already exists\n") {
			logger.Printf("Service: %s already exists.", serviceName)
		} else {
			log.Fatalf("Error adding service: %v\nOutput: %s", err, bytes)
		}
	}
}

func waitForServicesToBeRunning() {
	logger.Println("Waiting for services to be Running...")
	waitTime := time.Second
	done := false
	for !done {
		template := `{{range .items}}{{.status.phase}}
{{end}}`
		args := []string{"get", "pods", "-o=template", "--template=" + template}
		bytes, err := exec.Command(kubectl, args...).CombinedOutput()
		if err != nil {
			logger.Printf("Error running command: %v", err)
		}
		lines := strings.Split(string(bytes), "\n")
		if len(lines) < 2 {
			logger.Printf("Service status output too short. Waiting %v then checking again.", waitTime)
			time.Sleep(waitTime)
			waitTime *= 2
			continue
		}
		if lines[0] != "Running" || lines[1] != "Running" {
			logger.Printf("Services not running. Waiting %v then checking again.", waitTime)
			time.Sleep(waitTime)
			waitTime *= 2
		} else {
			done = true
		}
	}
}

func removeServices() {
	removeService("netperf-tester-host")
	removeService("netperf-tester-client")
}

func removeService(serviceName string) {
	args := []string{"delete", "rc/" + serviceName}
	err := runCommandInShell(kubectl, args)
	if err != nil {
		logger.Printf("Error running command: %v", err)
	}
}

func getPodName(serviceName string) (string, error) {
	template := `{{range .items}}{{.metadata.name}}
{{end}}`
	args := []string{"get", "pods", "-l", "run=" + serviceName, "-o=template", "--template=" + template}
	cmd := exec.Command(kubectl, args...)
	bytes, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimRight(string(bytes), "\n"), nil
}

func getPodIp(serviceName string) (string, error) {
	template := `{{range .items}}{{.status.podIP}}
{{end}}`
	args := []string{"get", "pods", "-l", "run=" + serviceName, "-o=template", "--template=" + template}
	cmd := exec.Command(kubectl, args...)
	bytes, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.Trim(string(bytes), " \n"), nil
}

func runTest(clientName, hostIP string, testNumber int) error {
	args := []string{"exec", "-t", clientName, "--", "netperf", "-H", hostIP, "-j", "-c", "-l", "-1000", "-t", "TCP_RR"}
	if testNumber != 0 {
		args = append(args, "-P", "0")
	}
	args = append(args, "--", "-D", "-O", "THROUGHPUT_UNITS,THROUGHPUT,MEAN_LATENCY,MIN_LATENCY,MAX_LATENCY,P50_LATENCY,P90_LATENCY,P99_LATENCY,STDDEV_LATENCY,LOCAL_CPU_UTIL")
	if testNumber == 0 {
		logger.Printf("Running %s with args := %v\n", kubectl, args)
	}
	bytes, err := exec.Command(kubectl, args...).CombinedOutput()
	if err != nil {
		logger.Printf("Error running command: %v", err)
		return err
	}
	if printResults {
		fmt.Printf("%s", bytes)
	}
	if outputFile != nil {
		outputFile.WriteString(resultsToCSV(bytes, testNumber))
	}
	return nil
}

var spacesRegex *regexp.Regexp = regexp.MustCompile("[ ]+")

func resultsToCSV(rawResults []byte, testNumber int) string {
	ret := ""
	line := ""
	if testNumber == 0 {
		ret = "Test #,Throughput Units,Throughput,Mean Latency Microseconds,Minimum Latency Microseconds,Maximum Latency Microseconds,50th Percentile Latency Microseconds,90th Percentile Latency Microseconds,99th Percentile Latency Microseconds,Stddev Latency Microseconds,Local CPU Util %\n"
		lines := strings.SplitN(string(rawResults), "\n", -1)
		line = lines[len(lines)-2] + "\n"
	} else {
		line = string(rawResults)
	}
	csvLine := spacesRegex.ReplaceAllLiteralString(line, ",")
	csvLine = strings.Replace(csvLine, "\r", "", -1)
	ret += fmt.Sprintf("%d, ", testNumber+1)
	ret += strings.TrimSuffix(csvLine, ",\n") + "\n"
	return ret
}

func runTests() error {
	// get client pod name
	clientName, err := getPodName("netperf-tester-client")
	if err != nil {
		return err
	}
	// get ip of the host pod
	hostIP, err := getPodIp("netperf-tester-host")
	if err != nil {
		return err
	}
	// run tests
	logger.Printf("Running netperf tests %d times.", numNetperf)
	for i := 0; i < numNetperf; i++ {
		runTest(clientName, hostIP, i)
		//time.Sleep(time.Millisecond * 1000)
	}
	return nil
}
