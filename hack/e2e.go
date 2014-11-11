/*
Copyright 2014 Google Inc. All rights reserved.

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
	"bytes"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"path"
	"path/filepath"
	"strings"
)

var (
	isup    = flag.Bool("isup", false, "Check to see if the e2e cluster is up, then exit.")
	build   = flag.Bool("build", false, "If true, build a new release. Otherwise, use whatever is there.")
	up      = flag.Bool("up", false, "If true, start the the e2e cluster. If cluster is already up, recreate it.")
	push    = flag.Bool("push", false, "If true, push to e2e cluster. Has no effect if -up is true.")
	down    = flag.Bool("down", false, "If true, tear down the cluster before exiting.")
	test    = flag.Bool("test", false, "Run all tests in hack/e2e-suite.")
	tests   = flag.String("tests", "", "Run only tests in hack/e2e-suite matching this glob. Ignored if -test is set.")
	root    = flag.String("root", absOrDie(filepath.Clean(filepath.Join(path.Base(os.Args[0]), ".."))), "Root directory of kubernetes repository.")
	verbose = flag.Bool("v", false, "If true, print all command output.")

	cfgCmd = flag.String("cfg", "", "If nonempty, pass this as an argument, and call kubecfg. Implies -v.")
	ctlCmd = flag.String("ctl", "", "If nonempty, pass this as an argument, and call kubectl. Implies -v. (-test, -cfg, -ctl are mutually exclusive)")
)

var signals = make(chan os.Signal, 100)

func absOrDie(path string) string {
	out, err := filepath.Abs(path)
	if err != nil {
		panic(err)
	}
	return out
}

func main() {
	flag.Parse()
	signal.Notify(signals, os.Interrupt)

	if *test {
		*tests = "*"
	}

	if *isup {
		status := 1
		if runBash("get status", `$KUBECFG -server_version`) {
			status = 0
			log.Printf("Cluster is UP")
		} else {
			log.Printf("Cluster is DOWN")
		}
		os.Exit(status)
	}

	if *build {
		if !runBash("build-release", `test-build-release`) {
			log.Fatal("Error building. Aborting.")
		}
	}

	if *up {
		if !Up() {
			log.Fatal("Error starting e2e cluster. Aborting.")
		}
	} else if *push {
		if !runBash("push", path.Join(*root, "/cluster/kube-push.sh")) {
			log.Fatal("Error pushing e2e cluster. Aborting.")
		}
	}

	failure := false
	switch {
	case *cfgCmd != "":
		failure = !runBash("'kubecfg "+*cfgCmd+"'", "$KUBECFG "+*cfgCmd)
	case *ctlCmd != "":
		failure = !runBash("'kubectl "+*ctlCmd+"'", "$KUBECFG "+*ctlCmd)
	case *tests != "":
		failed, passed := Test()
		log.Printf("Passed tests: %v", passed)
		log.Printf("Failed tests: %v", failed)
		failure = len(failed) > 0
	}

	if *down {
		TearDown()
	}

	if failure {
		os.Exit(1)
	}
}

func TearDown() {
	runBash("teardown", "test-teardown")
}

func Up() bool {
	if !tryUp() {
		log.Printf("kube-up failed; will tear down and retry. (Possibly your cluster was in some partially created state?)")
		TearDown()
		return tryUp()
	}
	return true
}

func tryUp() bool {
	return runBash("up", path.Join(*root, "/cluster/kube-up.sh; test-setup;"))
}

func Test() (failed, passed []string) {
	// run tests!
	dir, err := os.Open(filepath.Join(*root, "hack", "e2e-suite"))
	if err != nil {
		log.Fatal("Couldn't open e2e-suite dir")
	}
	defer dir.Close()
	names, err := dir.Readdirnames(0)
	if err != nil {
		log.Fatal("Couldn't read names in e2e-suite dir")
	}

	for i := range names {
		name := names[i]
		if name == "." || name == ".." {
			continue
		}
		if match, err := path.Match(*tests, name); !match && err == nil {
			continue
		}
		absName := filepath.Join(*root, "hack", "e2e-suite", name)
		log.Printf("%v matches %v. Starting test.", name, *tests)
		if runBash(name, absName) {
			log.Printf("%v passed", name)
			passed = append(passed, name)
		} else {
			log.Printf("%v failed", name)
			failed = append(failed, name)
		}
	}

	return
}

// All nonsense below is temporary until we have go versions of these things.

func runBash(stepName, bashFragment string) bool {
	cmd := exec.Command("bash", "-s")
	cmd.Stdin = strings.NewReader(bashWrap(bashFragment))
	return finishRunning(stepName, cmd)
}

func run(stepName, cmdPath string) bool {
	return finishRunning(stepName, exec.Command(filepath.Join(*root, cmdPath)))
}

func finishRunning(stepName string, cmd *exec.Cmd) bool {
	log.Printf("Running: %v", stepName)
	stdout, stderr := bytes.NewBuffer(nil), bytes.NewBuffer(nil)
	if *verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	} else {
		cmd.Stdout = stdout
		cmd.Stderr = stderr
	}

	done := make(chan struct{})
	defer close(done)
	go func() {
		for {
			select {
			case <-done:
				return
			case s := <-signals:
				cmd.Process.Signal(s)
			}
		}
	}()

	if err := cmd.Run(); err != nil {
		log.Printf("Error running %v: %v", stepName, err)
		if !*verbose {
			fmt.Printf("stdout:\n------\n%v\n------\n", string(stdout.Bytes()))
			fmt.Printf("stderr:\n------\n%v\n------\n", string(stderr.Bytes()))
		}
		return false
	}
	return true
}

var bashCommandPrefix = `
set -o errexit
set -o nounset
set -o pipefail

export KUBE_CONFIG_FILE="config-test.sh"

# TODO(jbeda): This will break on usage if there is a space in
# ${KUBE_ROOT}.  Covert to an array?  Or an exported function?
export KUBECFG="` + *root + `/cluster/kubecfg.sh -expect_version_match"

source "` + *root + `/cluster/kube-env.sh"
source "` + *root + `/cluster/${KUBERNETES_PROVIDER}/util.sh"

prepare-e2e

`

var bashCommandSuffix = `

`

func bashWrap(cmd string) string {
	return bashCommandPrefix + cmd + bashCommandSuffix
}
