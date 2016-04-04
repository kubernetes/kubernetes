/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

var (
	isup             = flag.Bool("isup", false, "Check to see if the e2e cluster is up, then exit.")
	build            = flag.Bool("build", false, "If true, build a new release. Otherwise, use whatever is there.")
	up               = flag.Bool("up", false, "If true, start the the e2e cluster. If cluster is already up, recreate it.")
	push             = flag.Bool("push", false, "If true, push to e2e cluster. Has no effect if -up is true.")
	pushup           = flag.Bool("pushup", false, "If true, push to e2e cluster if it's up, otherwise start the e2e cluster.")
	down             = flag.Bool("down", false, "If true, tear down the cluster before exiting.")
	test             = flag.Bool("test", false, "Run Ginkgo tests.")
	testArgs         = flag.String("test_args", "", "Space-separated list of arguments to pass to Ginkgo test runner.")
	root             = flag.String("root", absOrDie(filepath.Clean(filepath.Join(path.Base(os.Args[0]), ".."))), "Root directory of kubernetes repository.")
	verbose          = flag.Bool("v", false, "If true, print all command output.")
	checkVersionSkew = flag.Bool("check_version_skew", true, ""+
		"By default, verify that client and server have exact version match. "+
		"You can explicitly set to false if you're, e.g., testing client changes "+
		"for which the server version doesn't make a difference.")

	ctlCmd = flag.String("ctl", "", "If nonempty, pass this as an argument, and call kubectl. Implies -v. (-test, -cfg, -ctl are mutually exclusive)")
)

const (
	minNodeCount = 2
)

func absOrDie(path string) string {
	out, err := filepath.Abs(path)
	if err != nil {
		panic(err)
	}
	return out
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	flag.Parse()

	setEnvIfUnset("KUBECTL", path.Join(*root, "/cluster/kubectl.sh")+kubectlArgs())
	setEnvIfUnset("KUBE_CONFIG_FILE", "config-test.sh")

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
		if !finishRunning("build-release", utilCommand("test-build-release")) {
			log.Fatal("Error building. Aborting.")
		}
	}

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
			log.Fatal("Error starting e2e cluster. Aborting.")
		}
	} else if *push {
		if !finishRunning("push", utilCommand("\"${KUBE_ROOT}/cluster/kube-push.sh\"")) {
			log.Fatal("Error pushing e2e cluster. Aborting.")
		}
	}

	success := true
	switch {
	case *ctlCmd != "":
		ctlArgs := strings.Fields(*ctlCmd)
		success = finishRunning("'kubectl "+*ctlCmd+"'", exec.Command(path.Join(*root, "cluster/kubectl.sh"), ctlArgs...))
	case *test:
		success = Test()
	}

	if *down {
		TearDown()
	}

	if !success {
		os.Exit(1)
	}
}

func TearDown() bool {
	return finishRunning("teardown", utilCommand("test-teardown"))
}

// Up brings an e2e cluster up, recreating it if one is already running.
func Up() bool {
	if IsUp() {
		log.Printf("e2e cluster already running; will teardown")
		if res := TearDown(); !res {
			return false
		}
	}
	return finishRunning("up", utilCommand("test-setup"))
}

// Ensure that the cluster is large engough to run the e2e tests.
func ValidateClusterSize() {
	// Check that there are at least minNodeCount nodes running
	cmd := utilCommand("${KUBECTL} get nodes --no-headers | wc -l")
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
	return finishRunning("get status", utilCommand("${KUBECTL} version"))
}

func Test() bool {
	if !IsUp() {
		log.Fatal("Testing requested, but e2e cluster not up!")
	}

	ValidateClusterSize()

	return finishRunning("Ginkgo tests", exec.Command(filepath.Join(*root, "hack/ginkgo-e2e.sh"), strings.Fields(*testArgs)...))
}

// Sets environment variable env to value if it is unset.
func setEnvIfUnset(env, value string) {
	if oldVal := os.Getenv(env); oldVal == "" {
		os.Setenv(env, value)
	}
}

// Returns an exec.Cmd that sources cluster/kube-util.sh, runs prepare-e2e,
// and then runs command.
func utilCommand(command string) *exec.Cmd {
	expandedCommand := os.ExpandEnv(command)
	cmd := exec.Command("bash", "-s")
	cmd.Stdin = strings.NewReader("source \"" + path.Join(*root, "cluster/kube-util.sh") + "\" && prepare-e2e && " + expandedCommand)
	return cmd
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
func kubectlArgs() string {
	if *checkVersionSkew {
		return " --match-server-version"
	}
	return ""
}
