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
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"k8s.io/kubernetes/test/e2e_node/builder"
	"k8s.io/kubernetes/test/utils"

	"github.com/golang/glog"
)

var buildDependencies = flag.Bool("build-dependencies", true, "If true, build all dependencies.")
var ginkgoFlags = flag.String("ginkgo-flags", "", "Space-separated list of arguments to pass to Ginkgo test runner.")
var testFlags = flag.String("test-flags", "", "Space-separated list of arguments to pass to node e2e test.")
var systemSpecName = flag.String("system-spec-name", "", "The name of the system spec used for validating the image in the node conformance test. The specs are at k8s.io/kubernetes/cmd/kubeadm/app/util/system/specs/. If unspecified, the default built-in spec (system.DefaultSpec) will be used.")

const (
	systemSpecPath = "k8s.io/kubernetes/cmd/kubeadm/app/util/system/specs"
)

func main() {
	flag.Parse()

	// Build dependencies - ginkgo, kubelet and apiserver.
	if *buildDependencies {
		if err := builder.BuildGo(); err != nil {
			glog.Fatalf("Failed to build the dependencies: %v", err)
		}
	}

	// Run node e2e test
	outputDir, err := utils.GetK8sBuildOutputDir()
	if err != nil {
		glog.Fatalf("Failed to get build output directory: %v", err)
	}
	glog.Infof("Got build output dir: %v", outputDir)
	ginkgo := filepath.Join(outputDir, "ginkgo")
	test := filepath.Join(outputDir, "e2e_node.test")

	args := []string{*ginkgoFlags, test, "--", *testFlags}
	if *systemSpecName != "" {
		rootDir, err := utils.GetK8sRootDir()
		if err != nil {
			glog.Fatalf("Failed to get k8s root directory: %v", err)
		}
		systemSpecFile := filepath.Join(rootDir, systemSpecPath, *systemSpecName+".yaml")
		args = append(args, fmt.Sprintf("--system-spec-name=%s --system-spec-file=%s", *systemSpecName, systemSpecFile))
	}
	if err := runCommand(ginkgo, args...); err != nil {
		glog.Exitf("Test failed: %v", err)
	}
	return
}

func runCommand(name string, args ...string) error {
	glog.Infof("Running command: %v %v", name, strings.Join(args, " "))
	cmd := exec.Command("sudo", "sh", "-c", strings.Join(append([]string{name}, args...), " "))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
