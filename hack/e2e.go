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
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

var (
	build          = flag.Bool("build", false, "If true, build a new release. Otherwise, use whatever is there.")
	checkNodeCount = flag.Bool("check_node_count", true, ""+
		"By default, verify that the cluster has at least two nodes."+
		"You can explicitly set to false if you're, e.g., testing single-node clusters "+
		"for which the node count is supposed to be one.")
	checkVersionSkew = flag.Bool("check_version_skew", true, ""+
		"By default, verify that client and server have exact version match. "+
		"You can explicitly set to false if you're, e.g., testing client changes "+
		"for which the server version doesn't make a difference.")
	checkLeakedResources = flag.Bool("check_leaked_resources", false, "Ensure project ends with the same resources")
	ctlCmd               = flag.String("ctl", "", "If nonempty, pass this as an argument, and call kubectl. Implies -v.")
	down                 = flag.Bool("down", false, "If true, tear down the cluster before exiting.")
	dump                 = flag.String("dump", "", "If set, dump cluster logs to this location on test or cluster-up failure")
	kubemark             = flag.Bool("kubemark", false, "If true, run kubemark tests.")
	isup                 = flag.Bool("isup", false, "Check to see if the e2e cluster is up, then exit.")
	push                 = flag.Bool("push", false, "If true, push to e2e cluster. Has no effect if -up is true.")
	pushup               = flag.Bool("pushup", false, "If true, push to e2e cluster if it's up, otherwise start the e2e cluster.")
	skewTests            = flag.Bool("skew", false, "If true, run tests in another version at ../kubernetes/hack/e2e.go")
	testArgs             = flag.String("test_args", "", "Space-separated list of arguments to pass to Ginkgo test runner.")
	test                 = flag.Bool("test", false, "Run Ginkgo tests.")
	up                   = flag.Bool("up", false, "If true, start the the e2e cluster. If cluster is already up, recreate it.")
	upgradeArgs          = flag.String("upgrade_args", "", "If set, run upgrade tests before other tests")
	verbose              = flag.Bool("v", false, "If true, print all command output.")
)

const (
	minNodeCount = 2
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	flag.Parse()

	cwd, err := os.Getwd()
	if err != nil {
		log.Fatalf("Could not get pwd: %v", err)
	}
	acwd, err := filepath.Abs(cwd)
	if err != nil {
		log.Fatalf("Failed to convert to an absolute path: %v", err)
	}
	if !strings.Contains(filepath.Base(acwd), "kubernetes") {
		// TODO(fejta): cd up into  the kubernetes directory
		log.Fatalf("Must run from kubernetes directory: %v", cwd)
	}

	if *isup {
		status := 1
		if IsUp() {
			status = 0
			log.Printf("Cluster is UP")
		} else {
			log.Printf("Cluster is DOWN")
		}
		os.Exit(status)
	}

	if *build {
		// The build-release script needs stdin to ask the user whether
		// it's OK to download the docker image.
		cmd := exec.Command("make", "quick-release")
		cmd.Stdin = os.Stdin
		if !finishRunning("build-release", cmd) {
			log.Fatal("Error building. Aborting.")
		}
	}

	if *up && !TearDown() {
		log.Fatal("Could not tear down previous cluster")
	}

	beforeResources := ""
	if *checkLeakedResources {
		beforeResources = ListResources()
	}

	os.Setenv("KUBECTL", strings.Join(append([]string{"./cluster/kubectl.sh"}, kubectlArgs()...), " "))

	if *upgradeArgs != "" { // Start the cluster using a previous version.
		if !UpgradeUp() {
			log.Fatal("Failed to start cluster to upgrade. Aborting.")
		}
	} else { // Start the cluster using this version.
		if *pushup {
			if IsUp() {
				log.Printf("e2e cluster is up, pushing.")
				*up = false
				*push = true
			} else {
				log.Printf("e2e cluster is down, creating.")
				*up = true
				*push = false
			}
		}
		if *up {
			if !Up() {
				DumpClusterLogs(*dump)
				log.Fatal("Error starting e2e cluster. Aborting.")
			}
		} else if *push {
			if !finishRunning("push", exec.Command("./hack/e2e-internal/e2e-push.sh")) {
				DumpClusterLogs(*dump)
				log.Fatal("Error pushing e2e cluster. Aborting.")
			}
		}
	}

	upResources := ""
	if *checkLeakedResources {
		upResources = ListResources()
	}

	success := true

	if *ctlCmd != "" {
		ctlArgs := strings.Fields(*ctlCmd)
		os.Setenv("KUBE_CONFIG_FILE", "config-test.sh")
		ctlSuccess := finishRunning("'kubectl "+*ctlCmd+"'", exec.Command("./cluster/kubectl.sh", ctlArgs...))
		success = success && ctlSuccess
	}

	if *upgradeArgs != "" {
		upgradeSuccess := UpgradeTest(*upgradeArgs)
		success = success && upgradeSuccess
	}

	if *test {
		if *skewTests {
			skewSuccess := SkewTest()
			success = success && skewSuccess
		} else {
			testSuccess := Test()
			success = success && testSuccess
		}
	}

	if *kubemark {
		kubeSuccess := KubemarkTest()
		success = success && kubeSuccess
	}

	if !success {
		DumpClusterLogs(*dump)
	}

	if *down {
		tearSuccess := TearDown()
		success = success && tearSuccess
	}

	if *checkLeakedResources {
		log.Print("Sleeping for 30 seconds...") // Wait for eventually consistent listing
		time.Sleep(30 * time.Second)
		DiffResources(beforeResources, upResources, ListResources(), *dump)
	}

	if !success {
		os.Exit(1)
	}
}

func writeOrDie(dir, name, data string) string {
	f, err := os.Create(filepath.Join(dir, name))
	if err != nil {
		log.Fatal(err)
	}
	if _, err := f.WriteString(data); err != nil {
		log.Fatal(err)
	}
	if err := f.Close(); err != nil {
		log.Fatal(err)
	}
	log.Printf("Created file: %s", f.Name())
	return f.Name()
}

func DiffResources(before, clusterUp, after, location string) {
	if location == "" {
		var err error
		location, err = ioutil.TempDir("", "e2e-check-resources")
		if err != nil {
			log.Fatal(err)
		}
	}
	bp := writeOrDie(location, "gcp-resources-before.txt", before)
	writeOrDie(location, "gcp-resources-cluster-up.txt", clusterUp)
	ap := writeOrDie(location, "gcp-resources-after.txt", after)

	cmd := exec.Command("diff", "-sw", "-U0", "-F^\\[.*\\]$", bp, ap)
	if *verbose {
		cmd.Stderr = os.Stderr
	}
	o, err := cmd.Output()
	stdout := string(o)
	writeOrDie(location, "gcp-resources-diff.txt", stdout)
	if err == nil {
		return
	}
	lines := strings.Split(stdout, "\n")
	if len(lines) < 3 { // Ignore the +++ and --- header lines
		return
	}
	lines = lines[2:]

	var added []string
	for _, l := range lines {
		if strings.HasPrefix(l, "+") && len(strings.TrimPrefix(l, "+")) > 0 {
			added = append(added, l)
		}
	}
	if len(added) > 0 {
		log.Printf("Error: %d leaked resources", len(added))
		log.Fatal(strings.Join(added, "\n"))
	}
}

func ListResources() string {
	log.Printf("Listing resources...")
	cmd := exec.Command("./cluster/gce/list-resources.sh")
	if *verbose {
		cmd.Stderr = os.Stderr
	}
	stdout, err := cmd.Output()
	if err != nil {
		log.Fatalf("Failed to list resources (%s):\n%s", err, stdout)
	}
	return string(stdout)
}

func TearDown() bool {
	return finishRunning("teardown", exec.Command("./hack/e2e-internal/e2e-down.sh"))
}

// Up brings an e2e cluster up, recreating it if one is already running.
func Up() bool {
	return finishRunning("up", exec.Command("./hack/e2e-internal/e2e-up.sh"))
}

// Ensure that the cluster is large engough to run the e2e tests.
func ValidateClusterSize() {
	// Check that there are at least minNodeCount nodes running
	cmd := exec.Command("./hack/e2e-internal/e2e-cluster-size.sh")
	if *verbose {
		cmd.Stderr = os.Stderr
	}
	stdout, err := cmd.Output()
	if err != nil {
		log.Fatalf("Could not get nodes to validate cluster size (%s)", err)
	}

	numNodes, err := strconv.Atoi(strings.TrimSpace(string(stdout)))
	if err != nil {
		log.Fatalf("Could not count number of nodes to validate cluster size (%s)", err)
	}

	if numNodes < minNodeCount {
		log.Fatalf("Cluster size (%d) is too small to run e2e tests.  %d Nodes are required.", numNodes, minNodeCount)
	}
}

// Is the e2e cluster up?
func IsUp() bool {
	return finishRunning("get status", exec.Command("./hack/e2e-internal/e2e-status.sh"))
}

func DumpClusterLogs(location string) {
	if location == "" {
		return
	}
	log.Printf("Dumping cluster logs to: %v", location)
	finishRunning("dump cluster logs", exec.Command("./cluster/log-dump.sh", location))
}

func KubemarkTest() bool {
	// Stop previous run
	if !finishRunning("Stop kubemark", exec.Command("./test/kubemark/stop-kubemark.sh")) {
		log.Print("stop kubemark failed")
		return false
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
	if !finishRunning("Start Kubemark", exec.Command("./test/kubemark/start-kubemark.sh")) {
		log.Print("Error: start kubemark failed")
		return false
	}

	// Run kubemark tests
	focus, present := os.LookupEnv("KUBEMARK_TESTS")
	if !present {
		focus = "starting\\s30\\pods"
	}
	test_args := os.Getenv("KUBEMARK_TEST_ARGS")

	if !finishRunning("Run kubemark tests", exec.Command("./test/kubemark/run-e2e-tests.sh", "--ginkgo.focus="+focus, test_args)) {
		log.Print("Error: run kubemark tests failed")
		return false
	}

	// Stop kubemark
	if !finishRunning("Stop kubemark", exec.Command("./test/kubemark/stop-kubemark.sh")) {
		log.Print("Error: stop kubemark failed")
		return false
	}
	return true
}

func UpgradeUp() bool {
	old, err := os.Getwd()
	if err != nil {
		log.Printf("Failed to os.Getwd(): %v", err)
		return false
	}
	defer os.Chdir(old)
	err = os.Chdir("../kubernetes_skew")
	if err != nil {
		log.Printf("Failed to cd ../kubernetes_skew: %v", err)
		return false
	}
	return finishRunning("UpgradeUp",
		exec.Command(
			"go", "run", "./hack/e2e.go",
			fmt.Sprintf("--check_version_skew=%t", *checkVersionSkew),
			fmt.Sprintf("--push=%t", *push),
			fmt.Sprintf("--pushup=%t", *pushup),
			fmt.Sprintf("--up=%t", *up),
			fmt.Sprintf("--v=%t", *verbose),
		))
}

func UpgradeTest(args string) bool {
	old, err := os.Getwd()
	if err != nil {
		log.Printf("Failed to os.Getwd(): %v", err)
		return false
	}
	defer os.Chdir(old)
	err = os.Chdir("../kubernetes_skew")
	if err != nil {
		log.Printf("Failed to cd ../kubernetes_skew: %v", err)
		return false
	}
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

func SkewTest() bool {
	old, err := os.Getwd()
	if err != nil {
		log.Printf("Failed to Getwd: %v", err)
		return false
	}
	defer os.Chdir(old)
	err = os.Chdir("../kubernetes_skew")
	if err != nil {
		log.Printf("Failed to cd ../kubernetes_skew: %v", err)
		return false
	}
	return finishRunning("Skewed Ginkgo tests",
		exec.Command(
			"go", "run", "./hack/e2e.go",
			"--test",
			"--test_args="+*testArgs,
			fmt.Sprintf("--v=%t", *verbose),
			fmt.Sprintf("--check_version_skew=%t", *checkVersionSkew)))
}

func Test() bool {
	if !IsUp() {
		log.Fatal("Testing requested, but e2e cluster not up!")
	}

	// TODO(fejta): add a --federated or something similar
	if os.Getenv("FEDERATION") == "" {
		if *checkNodeCount {
			ValidateClusterSize()
		}
		return finishRunning("Ginkgo tests", exec.Command("./hack/ginkgo-e2e.sh", strings.Fields(*testArgs)...))
	}

	if *testArgs == "" {
		*testArgs = "--ginkgo.focus=\\[Feature:Federation\\]"
	}
	return finishRunning("Federated Ginkgo tests", exec.Command("./hack/federated-ginkgo-e2e.sh", strings.Fields(*testArgs)...))
}

func finishRunning(stepName string, cmd *exec.Cmd) bool {
	if *verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	log.Printf("Running: %v", stepName)
	defer func(start time.Time) {
		log.Printf("Step '%s' finished in %s", stepName, time.Since(start))
	}(time.Now())

	if err := cmd.Run(); err != nil {
		log.Printf("Error running %v: %v", stepName, err)
		return false
	}
	return true
}

// returns either "", or a list of args intended for appending with the
// kubectl command (beginning with a space).
func kubectlArgs() []string {
	if !*checkVersionSkew {
		return []string{}
	}
	return []string{"--match-server-version"}
}
