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
	"flag"
	"fmt"
	"os"
	"strings"

	"k8s.io/kubernetes/test/e2e_node"
)

var hosts = flag.String("hosts", "", "hosts to test")

func main() {
	// Setup coloring
	stat, _ := os.Stdout.Stat()
	useColor := (stat.Mode() & os.ModeCharDevice) != 0
	blue := ""
	noColour := ""
	if useColor {
		blue = "\033[0;34m"
		noColour = "\033[0m"
	}

	flag.Parse()
	if *hosts == "" {
		fmt.Printf("Must specific --hosts flag")
	}
	archive := e2e_node.CreateTestArchive()
	defer os.Remove(archive)

	results := make(chan *TestResult)
	hs := strings.Split(*hosts, ",")
	for _, h := range hs {
		fmt.Printf("Starting tests on host %s.", h)
		go func(host string) {
			output, err := e2e_node.RunRemote(archive, host)
			results <- &TestResult{
				output: output,
				err:    err,
				host:   host,
			}
		}(h)
	}

	// Wait for all tests to complete and emit the results
	errCount := 0
	for i := 0; i < len(hs); i++ {
		tr := <-results
		host := tr.host
		fmt.Printf("%s================================================================%s\n", blue, noColour)
		if tr.err != nil {
			errCount++
			fmt.Printf("Failure Finished Host %s Test Suite %s %v\n", host, tr.output, tr.err)
		} else {
			fmt.Printf("Success Finished Host %s Test Suite %s\n", host, tr.output)
		}
		fmt.Printf("%s================================================================%s\n", blue, noColour)
	}

	// Set the exit code if there were failures
	if errCount > 0 {
		fmt.Printf("Failure: %d errors encountered.", errCount)
		os.Exit(1)
	}
}

type TestResult struct {
	output string
	err    error
	host   string
}
