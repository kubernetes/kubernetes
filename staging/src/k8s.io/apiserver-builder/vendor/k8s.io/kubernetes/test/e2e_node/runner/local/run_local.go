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

package main

import (
	"flag"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"k8s.io/kubernetes/test/e2e_node/builder"

	"github.com/golang/glog"
)

var buildDependencies = flag.Bool("build-dependencies", true, "If true, build all dependencies.")
var ginkgoFlags = flag.String("ginkgo-flags", "", "Space-separated list of arguments to pass to Ginkgo test runner.")
var testFlags = flag.String("test-flags", "", "Space-separated list of arguments to pass to node e2e test.")

func main() {
	flag.Parse()

	// Build dependencies - ginkgo, kubelet and apiserver.
	if *buildDependencies {
		if err := builder.BuildGo(); err != nil {
			glog.Fatalf("Failed to build the dependencies: %v", err)
		}
	}

	// Run node e2e test
	outputDir, err := builder.GetK8sBuildOutputDir()
	if err != nil {
		glog.Fatalf("Failed to get build output directory: %v", err)
	}
	glog.Infof("Got build output dir: %v", outputDir)
	ginkgo := filepath.Join(outputDir, "ginkgo")
	test := filepath.Join(outputDir, "e2e_node.test")
	runCommand(ginkgo, *ginkgoFlags, test, "--", *testFlags)
	return
}

func runCommand(name string, args ...string) error {
	glog.Infof("Running command: %v %v", name, strings.Join(args, " "))
	cmd := exec.Command("sudo", "sh", "-c", strings.Join(append([]string{name}, args...), " "))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
