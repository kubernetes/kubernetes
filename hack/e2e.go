/*
Copyright 2014 The Kubernetes Authors.

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

// e2e.go runs the e2e test suite. No non-standard package dependencies; call with "go run".
package main

import (
	"encoding/xml"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

var (
	// TODO(fejta): change all these _ flags to -
	build            = flag.Bool("build", false, "If true, build a new release. Otherwise, use whatever is there.")
	checkVersionSkew = flag.Bool("check_version_skew", true, ""+
		"By default, verify that client and server have exact version match. "+
		"You can explicitly set to false if you're, e.g., testing client changes "+
		"for which the server version doesn't make a difference.")
	checkLeakedResources = flag.Bool("check_leaked_resources", false, "Ensure project ends with the same resources")
	down                 = flag.Bool("down", false, "If true, tear down the cluster before exiting.")
	dump                 = flag.String("dump", "", "If set, dump cluster logs to this location on test or cluster-up failure")
	kubemark             = flag.Bool("kubemark", false, "If true, run kubemark tests.")
	skewTests            = flag.Bool("skew", false, "If true, run tests in another version at ../kubernetes/hack/e2e.go")
	testArgs             = flag.String("test_args", "", "Space-separated list of arguments to pass to Ginkgo test runner.")
	test                 = flag.Bool("test", false, "Run Ginkgo tests.")
	up                   = flag.Bool("up", false, "If true, start the the e2e cluster. If cluster is already up, recreate it.")
	upgradeArgs          = flag.String("upgrade_args", "", "If set, run upgrade tests before other tests")
	verbose              = flag.Bool("v", false, "If true, print all command output.")

	deprecatedPush   = flag.Bool("push", false, "Deprecated. Does nothing.")
	deprecatedPushup = flag.Bool("pushup", false, "Deprecated. Does nothing.")
	deprecatedCtlCmd = flag.String("ctl", "", "Deprecated. Does nothing.")
)

func appendError(errs []error, err error) []error {
	if err != nil {
		return append(errs, err)
	}
	return errs
}

func validWorkingDirectory() error {
	cwd, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("could not get pwd: %v", err)
	}
	acwd, err := filepath.Abs(cwd)
	if err != nil {
		return fmt.Errorf("failed to convert %s to an absolute path: %v", cwd, err)
	}
	// This also matches "kubernetes_skew" for upgrades.
	if !strings.Contains(filepath.Base(acwd), "kubernetes") {
		return fmt.Errorf("must run from kubernetes directory root: %v", acwd)
	}
	return nil
}

type TestCase struct {
	XMLName   xml.Name `xml:"testcase"`
	ClassName string   `xml:"classname,attr"`
	Name      string   `xml:"name,attr"`
	Time      float64  `xml:"time,attr"`
	Failure   string   `xml:"failure,omitempty"`
}

type TestSuite struct {
	XMLName  xml.Name `xml:"testsuite"`
	Failures int      `xml:"failures,attr"`
	Tests    int      `xml:"tests,attr"`
	Time     float64  `xml:"time,attr"`
	Cases    []TestCase
}

var suite TestSuite

func xmlWrap(name string, f func() error) error {
	start := time.Now()
	err := f()
	duration := time.Since(start)
	c := TestCase{
		Name:      name,
		ClassName: "e2e.go",
		Time:      duration.Seconds(),
	}
	if err != nil {
		c.Failure = err.Error()
		suite.Failures++
	}
	suite.Cases = append(suite.Cases, c)
	suite.Tests++
	return err
}

func writeXML(start time.Time) {
	suite.Time = time.Since(start).Seconds()
	out, err := xml.MarshalIndent(&suite, "", "    ")
	if err != nil {
		log.Fatalf("Could not marshal XML: %s", err)
	}
	path := filepath.Join(*dump, "junit_runner.xml")
	f, err := os.Create(path)
	if err != nil {
		log.Fatalf("Could not create file: %s", err)
	}
	defer f.Close()
	if _, err := f.WriteString(xml.Header); err != nil {
		log.Fatalf("Error writing XML header: %s", err)
	}
	if _, err := f.Write(out); err != nil {
		log.Fatalf("Error writing XML data: %s", err)
	}
	log.Printf("Saved XML output to %s.", path)
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	flag.Parse()

	if err := validWorkingDirectory(); err != nil {
		log.Fatalf("Called from invalid working directory: %v", err)
	}

	if err := run(); err != nil {
		log.Fatalf("Something went wrong: %s", err)
	}
}

func run() error {
	if *dump != "" {
		defer writeXML(time.Now())
	}

	if *build {
		if err := xmlWrap("Build", Build); err != nil {
			return fmt.Errorf("error building: %s", err)
		}
	}

	if *checkVersionSkew {
		os.Setenv("KUBECTL", "./cluster/kubectl.sh --match-server-version")
	} else {
		os.Setenv("KUBECTL", "./cluster/kubectl.sh")
	}
	os.Setenv("KUBE_CONFIG_FILE", "config-test.sh")
	// force having batch/v2alpha1 always on for e2e tests
	os.Setenv("KUBE_RUNTIME_CONFIG", "batch/v2alpha1=true")

	if *up {
		if err := xmlWrap("TearDown", TearDown); err != nil {
			return fmt.Errorf("error tearing down previous cluster: %s", err)
		}
	}

	var err error
	var errs []error

	var (
		beforeResources []byte
		upResources     []byte
		afterResources  []byte
	)

	if *checkLeakedResources {
		errs = appendError(errs, xmlWrap("ListResources Before", func() error {
			beforeResources, err = ListResources()
			return err
		}))
	}

	if *up {
		// Start the cluster using this version.
		if err := xmlWrap("Up", Up); err != nil {
			return fmt.Errorf("starting e2e cluster: %s", err)
		}
	}

	if *checkLeakedResources {
		errs = appendError(errs, xmlWrap("ListResources Up", func() error {
			upResources, err = ListResources()
			return err
		}))
	}

	if *upgradeArgs != "" {
		errs = appendError(errs, xmlWrap("UpgradeTest", func() error {
			return UpgradeTest(*upgradeArgs)
		}))
	}

	if *test {
		errs = appendError(errs, xmlWrap("kubectl version", func() error {
			return finishRunning("kubectl version", exec.Command("./cluster/kubectl.sh", "version", "--match-server-version=false"))
		}))
		// Individual tests will create their own JUnit, so don't xmlWrap.
		if *skewTests {
			errs = appendError(errs, SkewTest())
		} else {
			if err := xmlWrap("IsUp", IsUp); err != nil {
				errs = appendError(errs, err)
			} else {
				errs = appendError(errs, Test())
			}
		}
	}

	if *kubemark {
		errs = appendError(errs, xmlWrap("KubemarkTest", KubemarkTest))
	}

	if len(errs) > 0 && *dump != "" {
		errs = appendError(errs, xmlWrap("DumpClusterLogs", func() error {
			return DumpClusterLogs(*dump)
		}))
	}

	if *down {
		errs = appendError(errs, xmlWrap("TearDown", TearDown))
	}

	if *checkLeakedResources {
		log.Print("Sleeping for 30 seconds...") // Wait for eventually consistent listing
		time.Sleep(30 * time.Second)
		if err := xmlWrap("ListResources After", func() error {
			afterResources, err = ListResources()
			return err
		}); err != nil {
			errs = append(errs, err)
		} else {
			errs = appendError(errs, xmlWrap("DiffResources", func() error {
				return DiffResources(beforeResources, upResources, afterResources, *dump)
			}))
		}
	}

	if len(errs) != 0 {
		return fmt.Errorf("encountered %d errors: %v", len(errs), errs)
	}
	return nil
}

func DiffResources(before, clusterUp, after []byte, location string) error {
	if location == "" {
		var err error
		location, err = ioutil.TempDir("", "e2e-check-resources")
		if err != nil {
			return fmt.Errorf("Could not create e2e-check-resources temp dir: %s", err)
		}
	}

	var mode os.FileMode = 0664
	bp := filepath.Join(location, "gcp-resources-before.txt")
	up := filepath.Join(location, "gcp-resources-cluster-up.txt")
	ap := filepath.Join(location, "gcp-resources-after.txt")
	dp := filepath.Join(location, "gcp-resources-diff.txt")

	if err := ioutil.WriteFile(bp, before, mode); err != nil {
		return err
	}
	if err := ioutil.WriteFile(up, clusterUp, mode); err != nil {
		return err
	}
	if err := ioutil.WriteFile(ap, after, mode); err != nil {
		return err
	}

	cmd := exec.Command("diff", "-sw", "-U0", "-F^\\[.*\\]$", bp, ap)
	if *verbose {
		cmd.Stderr = os.Stderr
	}
	stdout, cerr := cmd.Output()
	if err := ioutil.WriteFile(dp, stdout, mode); err != nil {
		return err
	}
	if cerr == nil { // No diffs
		return nil
	}
	lines := strings.Split(string(stdout), "\n")
	if len(lines) < 3 { // Ignore the +++ and --- header lines
		return nil
	}
	lines = lines[2:]

	var added []string
	for _, l := range lines {
		if strings.HasPrefix(l, "+") && len(strings.TrimPrefix(l, "+")) > 0 {
			added = append(added, l)
		}
	}
	if len(added) > 0 {
		return fmt.Errorf("Error: %d leaked resources\n%v", len(added), strings.Join(added, "\n"))
	}
	return nil
}

func ListResources() ([]byte, error) {
	log.Printf("Listing resources...")
	cmd := exec.Command("./cluster/gce/list-resources.sh")
	if *verbose {
		cmd.Stderr = os.Stderr
	}
	stdout, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("Failed to list resources (%s):\n%s", err, string(stdout))
	}
	return stdout, nil
}

func Build() error {
	// The build-release script needs stdin to ask the user whether
	// it's OK to download the docker image.
	cmd := exec.Command("make", "quick-release")
	cmd.Stdin = os.Stdin
	if err := finishRunning("build-release", cmd); err != nil {
		return fmt.Errorf("error building kubernetes: %v", err)
	}
	return nil
}

func TearDown() error {
	return finishRunning("teardown", exec.Command("./hack/e2e-internal/e2e-down.sh"))
}

func Up() error {
	return finishRunning("up", exec.Command("./hack/e2e-internal/e2e-up.sh"))
}

// Is the e2e cluster up?
func IsUp() error {
	return finishRunning("get status", exec.Command("./hack/e2e-internal/e2e-status.sh"))
}

func DumpClusterLogs(location string) error {
	log.Printf("Dumping cluster logs to: %v", location)
	return finishRunning("dump cluster logs", exec.Command("./cluster/log-dump.sh", location))
}

func KubemarkTest() error {
	// Stop previous run
	err := finishRunning("Stop kubemark", exec.Command("./test/kubemark/stop-kubemark.sh"))
	if err != nil {
		return err
	}

	// Start new run
	backups := []string{"NUM_NODES", "MASTER_SIZE"}
	for _, item := range backups {
		old, present := os.LookupEnv(item)
		if present {
			defer os.Setenv(item, old)
		} else {
			defer os.Unsetenv(item)
		}
	}
	os.Setenv("NUM_NODES", os.Getenv("KUBEMARK_NUM_NODES"))
	os.Setenv("MASTER_SIZE", os.Getenv("KUBEMARK_MASTER_SIZE"))
	err = finishRunning("Start Kubemark", exec.Command("./test/kubemark/start-kubemark.sh"))
	if err != nil {
		return err
	}

	// Run kubemark tests
	focus, present := os.LookupEnv("KUBEMARK_TESTS")
	if !present {
		focus = "starting\\s30\\pods"
	}
	test_args := os.Getenv("KUBEMARK_TEST_ARGS")

	err = finishRunning("Run kubemark tests", exec.Command("./test/kubemark/run-e2e-tests.sh", "--ginkgo.focus="+focus, test_args))
	if err != nil {
		return err
	}

	err = finishRunning("Stop kubemark", exec.Command("./test/kubemark/stop-kubemark.sh"))
	if err != nil {
		return err
	}
	return nil
}

func chdirSkew() (string, error) {
	old, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("failed to os.Getwd(): %v", err)
	}
	err = os.Chdir("../kubernetes_skew")
	if err != nil {
		return "", fmt.Errorf("failed to cd ../kubernetes_skew: %v", err)
	}
	return old, nil
}

func UpgradeTest(args string) error {
	old, err := chdirSkew()
	if err != nil {
		return err
	}
	defer os.Chdir(old)
	previous, present := os.LookupEnv("E2E_REPORT_PREFIX")
	if present {
		defer os.Setenv("E2E_REPORT_PREFIX", previous)
	} else {
		defer os.Unsetenv("E2E_REPORT_PREFIX")
	}
	os.Setenv("E2E_REPORT_PREFIX", "upgrade")
	return finishRunning("Upgrade Ginkgo tests",
		exec.Command(
			"go", "run", "./hack/e2e.go",
			"--test",
			"--test_args="+args,
			fmt.Sprintf("--v=%t", *verbose),
			fmt.Sprintf("--check_version_skew=%t", *checkVersionSkew)))
}

func SkewTest() error {
	old, err := chdirSkew()
	if err != nil {
		return err
	}
	defer os.Chdir(old)
	return finishRunning("Skewed Ginkgo tests",
		exec.Command(
			"go", "run", "./hack/e2e.go",
			"--test",
			"--test_args="+*testArgs,
			fmt.Sprintf("--v=%t", *verbose),
			fmt.Sprintf("--check_version_skew=%t", *checkVersionSkew)))
}

func Test() error {
	// TODO(fejta): add a --federated or something similar
	if os.Getenv("FEDERATION") != "true" {
		return finishRunning("Ginkgo tests", exec.Command("./hack/ginkgo-e2e.sh", strings.Fields(*testArgs)...))
	}

	if *testArgs == "" {
		*testArgs = "--ginkgo.focus=\\[Feature:Federation\\]"
	}
	return finishRunning("Federated Ginkgo tests", exec.Command("./hack/federated-ginkgo-e2e.sh", strings.Fields(*testArgs)...))
}

func finishRunning(stepName string, cmd *exec.Cmd) error {
	if *verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	log.Printf("Running: %v", stepName)
	defer func(start time.Time) {
		log.Printf("Step '%s' finished in %s", stepName, time.Since(start))
	}(time.Now())

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("error running %v: %v", stepName, err)
	}
	return nil
}
