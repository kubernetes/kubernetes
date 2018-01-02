// Copyright 2015 Google Inc. All Rights Reserved.
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
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"path"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/golang/glog"
	cadvisorApi "github.com/google/cadvisor/info/v2"
)

// must be able to ssh into hosts without password
// go run ./integration/runner/runner.go --logtostderr --v 2 --ssh-config <.ssh/config file> <list of hosts>

const (
	cadvisorBinary = "cadvisor"
	testTimeout    = 15 * time.Minute
)

var cadvisorTimeout = flag.Duration("cadvisor_timeout", 15*time.Second, "Time to wait for cAdvisor to come up on the remote host")
var port = flag.Int("port", 8080, "Port in which to start cAdvisor in the remote host")
var testRetryCount = flag.Int("test-retry-count", 3, "Number of times to retry failed tests before failing.")
var testRetryWhitelist = flag.String("test-retry-whitelist", "", "Path to newline separated list of regexexp for test failures that should be retried.  If empty, no tests are retried.")
var sshOptions = flag.String("ssh-options", "", "Commandline options passed to ssh.")
var retryRegex *regexp.Regexp

func getAttributes(ipAddress, portStr string) (*cadvisorApi.Attributes, error) {
	// Get host attributes and log attributes if the tests fail.
	var attributes cadvisorApi.Attributes
	resp, err := http.Get(fmt.Sprintf("http://%s:%s/api/v2.1/attributes", ipAddress, portStr))
	if err != nil {
		return nil, fmt.Errorf("failed to get attributes - %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get attributes. Status code - %v", resp.StatusCode)
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("unable to read attributes response body - %v", err)
	}
	if err := json.Unmarshal(body, &attributes); err != nil {
		return nil, fmt.Errorf("failed to unmarshal attributes - %v", err)
	}
	return &attributes, nil
}

func RunCommand(cmd string, args ...string) error {
	output, err := exec.Command(cmd, args...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("command %q %q failed with error: %v and output: %s", cmd, args, err, output)
	}

	return nil
}

func RunSshCommand(cmd string, args ...string) error {
	if *sshOptions != "" {
		args = append(strings.Split(*sshOptions, " "), args...)
	}
	return RunCommand(cmd, args...)
}

func PushAndRunTests(host, testDir string) (result error) {
	// Push binary.
	glog.Infof("Pushing cAdvisor binary to %q...", host)

	err := RunSshCommand("ssh", host, "--", "mkdir", "-p", testDir)
	if err != nil {
		return fmt.Errorf("failed to make remote testing directory: %v", err)
	}
	defer func() {
		err = RunSshCommand("ssh", host, "--", "rm", "-rf", testDir)
		if err != nil {
			glog.Errorf("Failed to cleanup test directory: %v", err)
		}
	}()

	err = RunSshCommand("scp", "-r", cadvisorBinary, fmt.Sprintf("%s:%s", host, testDir))
	if err != nil {
		return fmt.Errorf("failed to copy binary: %v", err)
	}

	// Start cAdvisor.
	glog.Infof("Running cAdvisor on %q...", host)
	portStr := strconv.Itoa(*port)
	errChan := make(chan error)
	go func() {
		err = RunSshCommand("ssh", host, "--", fmt.Sprintf("sudo GORACE='halt_on_error=1' %s --port %s --logtostderr --docker_env_metadata_whitelist=TEST_VAR  &> %s/log.txt", path.Join(testDir, cadvisorBinary), portStr, testDir))
		if err != nil {
			errChan <- fmt.Errorf("error running cAdvisor: %v", err)
		}
	}()
	defer func() {
		err = RunSshCommand("ssh", host, "--", "sudo", "pkill", cadvisorBinary)
		if err != nil {
			glog.Errorf("Failed to cleanup: %v", err)
		}
	}()
	defer func() {
		if result != nil {
			// Copy logs from the host
			err := RunSshCommand("scp", fmt.Sprintf("%s:%s/log.txt", host, testDir), "./")
			if err != nil {
				result = fmt.Errorf("error fetching logs: %v for %v", err, result)
				return
			}
			defer os.Remove("./log.txt")
			logs, err := ioutil.ReadFile("./log.txt")
			if err != nil {
				result = fmt.Errorf("error reading local log file: %v for %v", err, result)
				return
			}
			glog.Errorf("----------------------\nLogs from Host: %q\n%v\n", host, string(logs))

			// Get attributes for debugging purposes.
			attributes, err := getAttributes(host, portStr)
			if err != nil {
				glog.Errorf("Failed to read host attributes: %v", err)
			}
			result = fmt.Errorf("error on host %s: %v\n%+v", host, result, attributes)
		}
	}()

	// Wait for cAdvisor to come up.
	endTime := time.Now().Add(*cadvisorTimeout)
	done := false
	for endTime.After(time.Now()) && !done {
		select {
		case err := <-errChan:
			// Quit early if there was an error.
			return err
		case <-time.After(500 * time.Millisecond):
			// Stop waiting when cAdvisor is healthy..
			resp, err := http.Get(fmt.Sprintf("http://%s:%s/healthz", host, portStr))
			if err == nil && resp.StatusCode == http.StatusOK {
				done = true
				break
			}
		}
	}
	if !done {
		return fmt.Errorf("timed out waiting for cAdvisor to come up at host %q", host)
	}

	// Run the tests in a retry loop.
	glog.Infof("Running integration tests targeting %q...", host)
	for i := 0; i <= *testRetryCount; i++ {
		// Check if this is a retry
		if i > 0 {
			time.Sleep(time.Second * 15) // Wait 15 seconds before retrying
			glog.Warningf("Retrying (%d of %d) tests on host %s due to error %v", i, *testRetryCount, host, err)
		}
		// Run the command

		err = RunCommand("go", "test", "--timeout", testTimeout.String(), "github.com/google/cadvisor/integration/tests/...", "--host", host, "--port", portStr, "--ssh-options", *sshOptions)
		if err == nil {
			// On success, break out of retry loop
			break
		}

		// Only retry on test failures caused by these known flaky failure conditions
		if retryRegex == nil || !retryRegex.Match([]byte(err.Error())) {
			glog.Warningf("Skipping retry for tests on host %s because error is not whitelisted", host)
			break
		}
	}
	return err
}

func Run() error {
	start := time.Now()
	defer func() {
		glog.Infof("Execution time %v", time.Since(start))
	}()
	defer glog.Flush()

	hosts := flag.Args()
	testDir := fmt.Sprintf("/tmp/cadvisor-%d", os.Getpid())
	glog.Infof("Running integration tests on host(s) %q", strings.Join(hosts, ","))

	// Build cAdvisor.
	glog.Infof("Building cAdvisor...")
	err := RunCommand("build/build.sh")
	if err != nil {
		return err
	}
	defer func() {
		err := RunCommand("rm", cadvisorBinary)
		if err != nil {
			glog.Error(err)
		}
	}()

	// Run test on all hosts in parallel.
	var wg sync.WaitGroup
	allErrors := make([]error, 0)
	var allErrorsLock sync.Mutex
	for _, host := range hosts {
		wg.Add(1)
		go func(host string) {
			defer wg.Done()
			err := PushAndRunTests(host, testDir)
			if err != nil {
				func() {
					allErrorsLock.Lock()
					defer allErrorsLock.Unlock()
					allErrors = append(allErrors, err)
				}()
			}
		}(host)
	}
	wg.Wait()

	if len(allErrors) != 0 {
		var buffer bytes.Buffer
		for i, err := range allErrors {
			buffer.WriteString(fmt.Sprintf("Error %d: ", i))
			buffer.WriteString(err.Error())
			buffer.WriteString("\n")
		}
		return errors.New(buffer.String())
	}

	glog.Infof("All tests pass!")
	return nil
}

// initRetryWhitelist initializes the whitelist of test failures that can be retried.
func initRetryWhitelist() {
	if *testRetryWhitelist == "" {
		return
	}

	file, err := os.Open(*testRetryWhitelist)
	if err != nil {
		glog.Fatal(err)
	}
	defer file.Close()

	retryStrings := []string{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		text := scanner.Text()
		if text != "" {
			retryStrings = append(retryStrings, text)
		}
	}
	if err := scanner.Err(); err != nil {
		glog.Fatal(err)
	}
	retryRegex = regexp.MustCompile(strings.Join(retryStrings, "|"))
}

func main() {
	flag.Parse()

	// Check usage.
	if len(flag.Args()) == 0 {
		glog.Fatalf("USAGE: runner <hosts to test>")
	}
	initRetryWhitelist()

	// Run the tests.
	err := Run()
	if err != nil {
		glog.Fatal(err)
	}
}
