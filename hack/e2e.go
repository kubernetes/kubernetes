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
	"os/signal"
	"os/user"
	"path/filepath"
	"strconv"
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
	deployment           = flag.String("deployment", "bash", "up/down mechanism (defaults to cluster/kube-{up,down}.sh) (choices: bash/kops)")
	down                 = flag.Bool("down", false, "If true, tear down the cluster before exiting.")
	dump                 = flag.String("dump", "", "If set, dump cluster logs to this location on test or cluster-up failure")
	kubemark             = flag.Bool("kubemark", false, "If true, run kubemark tests.")
	skewTests            = flag.Bool("skew", false, "If true, run tests in another version at ../kubernetes/hack/e2e.go")
	testArgs             = flag.String("test_args", "", "Space-separated list of arguments to pass to Ginkgo test runner.")
	test                 = flag.Bool("test", false, "Run Ginkgo tests.")
	up                   = flag.Bool("up", false, "If true, start the the e2e cluster. If cluster is already up, recreate it.")
	upgradeArgs          = flag.String("upgrade_args", "", "If set, run upgrade tests before other tests")
	verbose              = flag.Bool("v", false, "If true, print all command output.")

	// kops specific flags.
	kopsPath        = flag.String("kops", "", "(kops only) Path to the kops binary. Must be set for kops.")
	kopsCluster     = flag.String("kops-cluster", "", "(kops only) Cluster name. Must be set for kops.")
	kopsState       = flag.String("kops-state", os.Getenv("KOPS_STATE_STORE"), "(kops only) s3:// path to kops state store. Must be set. (This flag defaults to $KOPS_STATE_STORE, and overrides it if set.)")
	kopsSSHKey      = flag.String("kops-ssh-key", os.Getenv("AWS_SSH_KEY"), "(kops only) Path to ssh key-pair for each node. (Defaults to $AWS_SSH_KEY or '~/.ssh/kube_aws_rsa'.)")
	kopsKubeVersion = flag.String("kops-kubernetes-version", "", "(kops only) If set, the version of Kubernetes to deploy (can be a URL to a GCS path where the release is stored) (Defaults to kops default, latest stable release.).")
	kopsZones       = flag.String("kops-zones", "us-west-2a", "(kops AWS only) AWS zones for kops deployment, comma delimited.")
	kopsNodes       = flag.Int("kops-nodes", 2, "(kops only) Number of nodes to create.")
	kopsUpTimeout   = flag.Duration("kops-up-timeout", 20*time.Minute, "(kops only) Time limit between 'kops config / kops update' and a response from the Kubernetes API.")

	// Deprecated flags.
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

	deploy, err := getDeployer()
	if err != nil {
		log.Fatalf("Error creating deployer: %v", err)
	}

	if *down {
		// listen for signals such as ^C and gracefully attempt to clean up
		c := make(chan os.Signal, 1)
		signal.Notify(c, os.Interrupt)
		go func() {
			for range c {
				log.Print("Captured ^C, gracefully attempting to cleanup resources..")
				if err := deploy.Down(); err != nil {
					log.Printf("Tearing down deployment failed: %v", err)
					os.Exit(1)
				}
			}
		}()
	}

	if err := run(deploy); err != nil {
		log.Fatalf("Something went wrong: %s", err)
	}
}

func run(deploy deployer) error {
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
		if err := xmlWrap("TearDown", deploy.Down); err != nil {
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
		// If we tried to bring the cluster up, make a courtesy
		// attempt to bring it down so we're not leaving resources around.
		//
		// TODO: We should try calling deploy.Down exactly once. Though to
		// stop the leaking resources for now, we want to be on the safe side
		// and call it explictly in defer if the other one is not called.
		if *down {
			defer xmlWrap("Deferred TearDown", deploy.Down)
		}
		// Start the cluster using this version.
		if err := xmlWrap("Up", deploy.Up); err != nil {
			return fmt.Errorf("starting e2e cluster: %s", err)
		}
		if *dump != "" {
			cmd := exec.Command("./cluster/kubectl.sh", "--match-server-version=false", "get", "nodes", "-oyaml")
			b, err := cmd.CombinedOutput()
			if *verbose {
				log.Printf("kubectl get nodes:\n%s", string(b))
			}
			if err == nil {
				if err := ioutil.WriteFile(filepath.Join(*dump, "nodes.yaml"), b, 0644); err != nil {
					errs = appendError(errs, fmt.Errorf("error writing nodes.yaml: %v", err))
				}
			} else {
				errs = appendError(errs, fmt.Errorf("error running get nodes: %v", err))
			}
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
		errs = appendError(errs, xmlWrap("get kubeconfig", deploy.SetupKubecfg))
		errs = appendError(errs, xmlWrap("kubectl version", func() error {
			return finishRunning("kubectl version", exec.Command("./cluster/kubectl.sh", "version", "--match-server-version=false"))
		}))
		// Individual tests will create their own JUnit, so don't xmlWrap.
		if *skewTests {
			errs = appendError(errs, SkewTest())
		} else {
			if err := xmlWrap("IsUp", deploy.IsUp); err != nil {
				errs = appendError(errs, err)
			} else {
				errs = appendError(errs, Test())
			}
		}
	}

	if *kubemark {
		errs = appendError(errs, KubemarkTest())
	}

	if len(errs) > 0 && *dump != "" {
		errs = appendError(errs, xmlWrap("DumpClusterLogs", func() error {
			return DumpClusterLogs(*dump)
		}))
	}

	if *down {
		errs = appendError(errs, xmlWrap("TearDown", deploy.Down))
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

type deployer interface {
	Up() error
	IsUp() error
	SetupKubecfg() error
	Down() error
}

func getDeployer() (deployer, error) {
	switch *deployment {
	case "bash":
		return bash{}, nil
	case "kops":
		return NewKops()
	default:
		return nil, fmt.Errorf("Unknown deployment strategy %q", *deployment)
	}
}

type bash struct{}

func (b bash) Up() error {
	return finishRunning("up", exec.Command("./hack/e2e-internal/e2e-up.sh"))
}

func (b bash) IsUp() error {
	return finishRunning("get status", exec.Command("./hack/e2e-internal/e2e-status.sh"))
}

func (b bash) SetupKubecfg() error {
	return nil
}

func (b bash) Down() error {
	return finishRunning("teardown", exec.Command("./hack/e2e-internal/e2e-down.sh"))
}

type kops struct {
	path        string
	kubeVersion string
	sshKey      string
	zones       []string
	nodes       int
	cluster     string
	kubecfg     string
}

func NewKops() (*kops, error) {
	if *kopsPath == "" {
		return nil, fmt.Errorf("--kops must be set to a valid binary path for kops deployment.")
	}
	if *kopsCluster == "" {
		return nil, fmt.Errorf("--kops-cluster must be set to a valid cluster name for kops deployment.")
	}
	if *kopsState == "" {
		return nil, fmt.Errorf("--kops-state must be set to a valid S3 path for kops deployment.")
	}
	sshKey := *kopsSSHKey
	if sshKey == "" {
		usr, err := user.Current()
		if err != nil {
			return nil, err
		}
		sshKey = filepath.Join(usr.HomeDir, ".ssh/kube_aws_rsa")
	}
	if err := os.Setenv("KOPS_STATE_STORE", *kopsState); err != nil {
		return nil, err
	}
	f, err := ioutil.TempFile("", "kops-kubecfg")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	kubecfg := f.Name()
	if err := f.Chmod(0600); err != nil {
		return nil, err
	}
	if err := os.Setenv("KUBECONFIG", kubecfg); err != nil {
		return nil, err
	}
	// Set KUBERNETES_CONFORMANCE_TEST so the auth info is picked up
	// from kubectl instead of bash inference.
	if err := os.Setenv("KUBERNETES_CONFORMANCE_TEST", "yes"); err != nil {
		return nil, err
	}
	// Set KUBERNETES_CONFORMANCE_PROVIDER to override the
	// cloudprovider for KUBERNETES_CONFORMANCE_TEST.
	if err := os.Setenv("KUBERNETES_CONFORMANCE_PROVIDER", "aws"); err != nil {
		return nil, err
	}
	// AWS_SSH_KEY is required by the AWS e2e tests.
	if err := os.Setenv("AWS_SSH_KEY", sshKey); err != nil {
		return nil, err
	}
	// ZONE is required by the AWS e2e tests.
	zones := strings.Split(*kopsZones, ",")
	if err := os.Setenv("ZONE", zones[0]); err != nil {
		return nil, err
	}
	return &kops{
		path:        *kopsPath,
		kubeVersion: *kopsKubeVersion,
		sshKey:      sshKey + ".pub", // kops only needs the public key, e2es need the private key.
		zones:       zones,
		nodes:       *kopsNodes,
		cluster:     *kopsCluster,
		kubecfg:     kubecfg,
	}, nil
}

func (k kops) Up() error {
	createArgs := []string{
		"create", "cluster",
		"--name", k.cluster,
		"--ssh-public-key", k.sshKey,
		"--node-count", strconv.Itoa(k.nodes),
		"--zones", strings.Join(k.zones, ","),
	}
	if k.kubeVersion != "" {
		createArgs = append(createArgs, "--kubernetes-version", k.kubeVersion)
	}
	if err := finishRunning("kops config", exec.Command(k.path, createArgs...)); err != nil {
		return fmt.Errorf("kops configuration failed: %v", err)
	}
	if err := finishRunning("kops update", exec.Command(k.path, "update", "cluster", k.cluster, "--yes")); err != nil {
		return fmt.Errorf("kops bringup failed: %v", err)
	}
	// TODO(zmerlynn): More cluster validation. This should perhaps be
	// added to kops and not here, but this is a fine place to loop
	// for now.
	for stop := time.Now().Add(*kopsUpTimeout); time.Now().Before(stop); time.Sleep(30 * time.Second) {
		n, err := clusterSize(k)
		if err != nil {
			log.Printf("Can't get cluster size, sleeping: %v", err)
			continue
		}
		if n < k.nodes+1 {
			log.Printf("%d (current nodes) < %d (requested instances), sleeping", n, k.nodes+1)
			continue
		}
		return nil
	}
	return fmt.Errorf("kops bringup timed out")
}

func (k kops) IsUp() error {
	n, err := clusterSize(k)
	if err != nil {
		return err
	}
	if n <= 0 {
		return fmt.Errorf("kops cluster found, but %d nodes reported", n)
	}
	return nil
}

func (k kops) SetupKubecfg() error {
	info, err := os.Stat(k.kubecfg)
	if err != nil {
		return err
	}
	if info.Size() > 0 {
		// Assume that if we already have it, it's good.
		return nil
	}
	if err := finishRunning("kops export", exec.Command(k.path, "export", "kubecfg", k.cluster)); err != nil {
		return fmt.Errorf("Failure exporting kops kubecfg: %v", err)
	}
	return nil
}

func (k kops) Down() error {
	// We do a "kops get" first so the exit status of "kops delete" is
	// more sensical in the case of a non-existant cluster. ("kops
	// delete" will exit with status 1 on a non-existant cluster)
	err := finishRunning("kops get", exec.Command(k.path, "get", "clusters", k.cluster))
	if err != nil {
		// This is expected if the cluster doesn't exist.
		return nil
	}
	return finishRunning("kops delete", exec.Command(k.path, "delete", "cluster", k.cluster, "--yes"))
}

func clusterSize(deploy deployer) (int, error) {
	if err := deploy.SetupKubecfg(); err != nil {
		return -1, err
	}
	o, err := exec.Command("kubectl", "get", "nodes", "--no-headers").Output()
	if err != nil {
		log.Printf("kubectl get nodes failed: %v", err)
		return -1, err
	}
	stdout := strings.TrimSpace(string(o))
	log.Printf("Cluster nodes:\n%s", stdout)
	return len(strings.Split(stdout, "\n")), nil
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
	// If we tried to bring the Kubemark cluster up, make a courtesy
	// attempt to bring it down so we're not leaving resources around.
	//
	// TODO: We should try calling stop-kubemark exactly once. Though to
	// stop the leaking resources for now, we want to be on the safe side
	// and call it explictly in defer if the other one is not called.
	defer xmlWrap("Deferred Stop kubemark", func() error {
		return finishRunning("Stop kubemark", exec.Command("./test/kubemark/stop-kubemark.sh"))
	})

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
	err = xmlWrap("Start kubemark", func() error {
		return finishRunning("Start kubemark", exec.Command("./test/kubemark/start-kubemark.sh"))
	})
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

	err = xmlWrap("Stop kubemark", func() error {
		return finishRunning("Stop kubemark", exec.Command("./test/kubemark/stop-kubemark.sh"))
	})
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
