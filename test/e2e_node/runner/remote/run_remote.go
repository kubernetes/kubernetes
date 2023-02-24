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

// To run the node e2e tests remotely against one or more hosts on gce:
// $ go run run_remote.go --v 2 --ssh-env gce --hosts <comma separated hosts>
// To run the node e2e tests remotely against one or more images on gce and provision them:
// $ go run run_remote.go --v 2 --project <project> --zone <zone> --ssh-env gce --images <comma separated images>
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"sync"
	"time"

	"k8s.io/kubernetes/test/e2e_node/remote"
	"k8s.io/kubernetes/test/e2e_node/system"

	"k8s.io/klog/v2"
)

var mode = flag.String("mode", "gce", "Mode to operate in. One of gce|ssh. Defaults to gce")
var testArgs = flag.String("test_args", "", "Space-separated list of arguments to pass to Ginkgo test runner.")
var testSuite = flag.String("test-suite", "default", "Test suite the runner initializes with. Currently support default|cadvisor|conformance")
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
var systemSpecName = flag.String("system-spec-name", "", fmt.Sprintf("The name of the system spec used for validating the image in the node conformance test. The specs are at %s. If unspecified, the default built-in spec (system.DefaultSpec) will be used.", system.SystemSpecPath))
var extraEnvs = flag.String("extra-envs", "", "The extra environment variables needed for node e2e tests. Format: a list of key=value pairs, e.g., env1=val1,env2=val2")
var runtimeConfig = flag.String("runtime-config", "", "The runtime configuration for the API server on the node e2e tests.. Format: a list of key=value pairs, e.g., env1=val1,env2=val2")
var kubeletConfigFile = flag.String("kubelet-config-file", "", "The KubeletConfiguration file that should be applied to the kubelet")
var (
	arc Archive
)

// Archive contains path info in the archive.
type Archive struct {
	sync.Once
	path string
	err  error
}

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	var suite remote.TestSuite
	switch *testSuite {
	case "conformance":
		suite = remote.InitConformanceRemote()
	case "cadvisor":
		suite = remote.InitCAdvisorE2ERemote()
	// TODO: Add subcommand for node soaking, node conformance, cri validation.
	case "default":
		// Use node e2e suite by default if no subcommand is specified.
		suite = remote.InitNodeE2ERemote()
	default:
		klog.Fatalf("--test-suite must be one of default, cadvisor, or conformance")
	}

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

	rand.Seed(time.Now().UnixNano())
	if *buildOnly {
		// Build the archive and exit
		remote.CreateTestArchive(suite, *systemSpecName, *kubeletConfigFile)
		return
	}

	// Append some default ginkgo flags. We use similar defaults here as hack/ginkgo-e2e.sh
	allGinkgoFlags := fmt.Sprintf("%s --no-color -v", *ginkgoFlags)
	fmt.Printf("Will use ginkgo flags as: %s", allGinkgoFlags)

	var runner remote.Runner
	cfg := remote.Config{
		InstanceNamePrefix: *instanceNamePrefix,
		ImageConfigFile:    *imageConfigFile,
		ImageConfigDir:     *imageConfigDir,
		Images:             splitCommaList(*images),
		Hosts:              parseHostsList(*hosts),
		GinkgoFlags:        allGinkgoFlags,
		DeleteInstances:    *deleteInstances,
		Cleanup:            *cleanup,
		TestArgs:           *testArgs,
		ExtraEnvs:          *extraEnvs,
		RuntimeConfig:      *runtimeConfig,
		SystemSpecName:     *systemSpecName,
	}

	var sshRunner remote.Runner
	switch *mode {
	case "gce":
		runner = remote.NewGCERunner(cfg)
		sshRunner = remote.NewSSHRunner(cfg)
	case "ssh":
		runner = remote.NewSSHRunner(cfg)
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

	results := make(chan *remote.TestResult)

	path, err := arc.getArchive(suite)
	if err != nil {
		log.Fatalf("unable to create test archive: %s", err)
	}
	defer arc.deleteArchive()

	running := runner.StartTests(suite, path, results)
	// You can potentially run SSH based tests while running image based test as well.  The GCE provider does this, see
	// test-e2e-node.sh.
	if sshRunner != nil && len(cfg.Hosts) > 0 {
		running += sshRunner.StartTests(suite, path, results)
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

func (a *Archive) getArchive(suite remote.TestSuite) (string, error) {
	a.Do(func() { a.path, a.err = remote.CreateTestArchive(suite, *systemSpecName, *kubeletConfigFile) })
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
			remote.AddHostnameIP(segs[0], segs[1])
		} else if len(segs) > 2 {
			klog.Fatalf("invalid format of host %q", hostList)
		}
		hostsOnly = append(hostsOnly, segs[0])
	}
	return hostsOnly
}
