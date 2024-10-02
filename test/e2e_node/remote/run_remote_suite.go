/*
Copyright 2016 The Kubernetes Authors.

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

package remote

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"sync"

	"k8s.io/klog/v2"
)

var mode = flag.String("mode", "gce", "Mode to operate in. One of gce|ssh. Defaults to gce")
var testArgs = flag.String("test_args", "", "Space-separated list of arguments to pass to Ginkgo test runner.")
var instanceNamePrefix = flag.String("instance-name-prefix", "", "prefix for instance names")
var imageConfigFile = flag.String("image-config-file", "", "yaml file describing images to run")
var imageConfigDir = flag.String("image-config-dir", "", "(optional) path to image config files")
var images = flag.String("images", "", "images to test")
var hosts = flag.String("hosts", "", "hosts to test")
var cleanup = flag.Bool("cleanup", true, "If true remove files from remote hosts and delete temporary instances")
var deleteInstances = flag.Bool("delete-instances", true, "If true, delete any instances created")
var buildOnly = flag.Bool("build-only", false, "If true, build e2e_node_test.tar.gz and exit.")
var gubernator = flag.Bool("gubernator", false, "If true, output Gubernator link to view logs")
var ginkgoFlags = flag.String("ginkgo-flags", "", "Passed to ginkgo to specify additional flags such as --skip=.")
var (
	arc Archive
)

// Archive contains path info in the archive.
type Archive struct {
	sync.Once
	path string
	err  error
}

func getFlag(name string) string {
	lookup := flag.Lookup(name)
	if lookup == nil {
		return ""
	}
	return lookup.Value.String()
}

func RunRemoteTestSuite(testSuite TestSuite) {
	// Listen for SIGINT and ignore the first one. In case SIGINT is sent to this
	// process and all its children, we ignore it here, while our children ssh connections
	// are stopped. This allows us to gather artifacts and print out test state before
	// being killed.
	c := make(chan os.Signal, 2)
	signal.Notify(c, os.Interrupt)
	go func() {
		<-c
		fmt.Printf("Received SIGINT. Will exit on next SIGINT.\n")
		<-c
		fmt.Printf("Received another SIGINT. Will exit.\n")
		os.Exit(1)
	}()

	if *buildOnly {
		// Build the archive and exit
		CreateTestArchive(testSuite,
			getFlag("system-spec-name"),
			getFlag("kubelet-config-file"))
		return
	}

	// Append some default ginkgo flags. We use similar defaults here as hack/ginkgo-e2e.sh
	allGinkgoFlags := fmt.Sprintf("%s --no-color -v", *ginkgoFlags)
	fmt.Printf("Will use ginkgo flags as: %s", allGinkgoFlags)

	var runner Runner
	cfg := Config{
		InstanceNamePrefix: *instanceNamePrefix,
		ImageConfigFile:    *imageConfigFile,
		ImageConfigDir:     *imageConfigDir,
		Images:             splitCommaList(*images),
		Hosts:              parseHostsList(*hosts),
		GinkgoFlags:        allGinkgoFlags,
		DeleteInstances:    *deleteInstances,
		Cleanup:            *cleanup,
		TestArgs:           *testArgs,
		ExtraEnvs:          getFlag("extra-envs"),
		RuntimeConfig:      getFlag("runtime-config"),
		SystemSpecName:     getFlag("system-spec-name"),
	}

	var sshRunner Runner

	if *mode == "ssh" {
		runner = NewSSHRunner(cfg)
	} else {
		getRunner, err := GetRunner(*mode)
		if err != nil {
			klog.Fatalf("getting runner mode %q : %v", *mode, err)
		}
		runner = getRunner(cfg)
		sshRunner = NewSSHRunner(cfg)
	}

	if err := runner.Validate(); err != nil {
		klog.Fatalf("validating remote config, %s", err)
	}

	// Setup coloring
	stat, _ := os.Stdout.Stat()
	useColor := (stat.Mode() & os.ModeCharDevice) != 0
	blue := ""
	noColour := ""
	if useColor {
		blue = "\033[0;34m"
		noColour = "\033[0m"
	}

	results := make(chan *TestResult)

	path, err := arc.getArchive(testSuite)
	if err != nil {
		log.Fatalf("unable to create test archive: %s", err)
	}
	defer arc.deleteArchive()

	running := runner.StartTests(testSuite, path, results)
	// You can potentially run SSH based tests while running image based test as well.  The GCE provider does this, see
	// test-e2e-node.sh.
	if sshRunner != nil && len(cfg.Hosts) > 0 {
		running += sshRunner.StartTests(testSuite, path, results)
	}

	// Wait for all tests to complete and emit the results
	errCount := 0
	exitOk := true
	for i := 0; i < running; i++ {
		tr := <-results
		host := tr.Host
		fmt.Println() // Print an empty line
		fmt.Printf("%s>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>%s\n", blue, noColour)
		fmt.Printf("%s>                              START TEST                                >%s\n", blue, noColour)
		fmt.Printf("%s>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>%s\n", blue, noColour)
		fmt.Printf("Start Test Suite on Host %s\n", host)
		fmt.Printf("%s\n", tr.Output)
		if tr.Err != nil {
			errCount++
			fmt.Printf("Failure Finished Test Suite on Host %s. Refer to artifacts directory for ginkgo log for this host.\n%v\n", host, tr.Err)
		} else {
			fmt.Printf("Success Finished Test Suite on Host %s. Refer to artifacts directory for ginkgo log for this host.\n", host)
		}
		exitOk = exitOk && tr.ExitOK
		fmt.Printf("%s<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<%s\n", blue, noColour)
		fmt.Printf("%s<                              FINISH TEST                               <%s\n", blue, noColour)
		fmt.Printf("%s<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<%s\n", blue, noColour)
		fmt.Println() // Print an empty line
	}
	// Set the exit code if there were failures
	if !exitOk {
		fmt.Printf("Failure: %d errors encountered.\n", errCount)
		callGubernator(*gubernator)
		arc.deleteArchive()
		os.Exit(1)
	}
	callGubernator(*gubernator)
}

func splitCommaList(s string) []string {
	if len(s) == 0 {
		return nil
	}
	return strings.Split(s, ",")
}

func callGubernator(gubernator bool) {
	if gubernator {
		fmt.Println("Running gubernator.sh")
		output, err := exec.Command("./test/e2e_node/gubernator.sh", "y").Output()

		if err != nil {
			fmt.Println("gubernator.sh Failed")
			fmt.Println(err)
			return
		}
		fmt.Printf("%s", output)
	}
	return
}

func (a *Archive) getArchive(suite TestSuite) (string, error) {
	a.Do(func() {
		a.path, a.err = CreateTestArchive(suite,
			getFlag("system-spec-name"),
			getFlag("kubelet-config-file"))
	})
	return a.path, a.err
}

func (a *Archive) deleteArchive() {
	path, err := a.getArchive(nil)
	if err != nil {
		return
	}
	os.Remove(path)
}

// parseHostsList splits a host list of the form a=1.2.3.4,b=5.6.7.8 into the list of hosts [a,b] while registering the
// given addresses
func parseHostsList(hostList string) []string {
	if len(hostList) == 0 {
		return nil
	}
	hosts := strings.Split(hostList, ",")
	var hostsOnly []string
	for _, host := range hosts {
		segs := strings.Split(host, "=")
		if len(segs) == 2 {
			AddHostnameIP(segs[0], segs[1])
		} else if len(segs) > 2 {
			klog.Fatalf("invalid format of host %q", hostList)
		}
		hostsOnly = append(hostsOnly, segs[0])
	}
	return hostsOnly
}
