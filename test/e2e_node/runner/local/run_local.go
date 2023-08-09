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
	"k8s.io/kubernetes/test/e2e_node/system"
	"k8s.io/kubernetes/test/utils"

	"k8s.io/klog/v2"
)

var buildDependencies = flag.Bool("build-dependencies", true, "If true, build all dependencies.")
var ginkgoFlags = flag.String("ginkgo-flags", "", "Space-separated list of arguments to pass to Ginkgo test runner.")
var testFlags = flag.String("test-flags", "", "Space-separated list of arguments to pass to node e2e test.")
var systemSpecName = flag.String("system-spec-name", "", fmt.Sprintf("The name of the system spec used for validating the image in the node conformance test. The specs are at %s. If unspecified, the default built-in spec (system.DefaultSpec) will be used.", system.SystemSpecPath))
var extraEnvs = flag.String("extra-envs", "", "The extra environment variables needed for node e2e tests. Format: a list of key=value pairs, e.g., env1=val1,env2=val2")
var runtimeConfig = flag.String("runtime-config", "", "The runtime configuration for the API server on the node e2e tests. Format: a list of key=value pairs, e.g., env1=val1,env2=val2")
var kubeletConfigFile = flag.String("kubelet-config-file", "", "The KubeletConfiguration file that should be applied to the kubelet")

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	// Build dependencies - ginkgo, kubelet, e2e_node.test, and mounter.
	if *buildDependencies {
		if err := builder.BuildGo(); err != nil {
			klog.Fatalf("Failed to build the dependencies: %v", err)
		}
	}

	// Run node e2e test
	outputDir, err := utils.GetK8sBuildOutputDir(builder.IsDockerizedBuild(), builder.GetTargetBuildArch())
	if err != nil {
		klog.Fatalf("Failed to get build output directory: %v", err)
	}
	klog.Infof("Got build output dir: %v", outputDir)
	ginkgo := filepath.Join(outputDir, "ginkgo")
	test := filepath.Join(outputDir, "e2e_node.test")

	args := []string{*ginkgoFlags, test, "--", *testFlags, fmt.Sprintf("--runtime-config=%s", *runtimeConfig)}
	if *systemSpecName != "" {
		rootDir, err := utils.GetK8sRootDir()
		if err != nil {
			klog.Fatalf("Failed to get k8s root directory: %v", err)
		}
		systemSpecFile := filepath.Join(rootDir, system.SystemSpecPath, *systemSpecName+".yaml")
		args = append(args, fmt.Sprintf("--system-spec-name=%s --system-spec-file=%s --extra-envs=%s", *systemSpecName, systemSpecFile, *extraEnvs))
	}
	if *kubeletConfigFile != "" {
		args = append(args, fmt.Sprintf("--kubelet-config-file=\"%s\"", *kubeletConfigFile))
	}
	if err := runCommand(ginkgo, args...); err != nil {
		klog.Exitf("Test failed: %v", err)
	}
	return
}

func runCommand(name string, args ...string) error {
	klog.Infof("Running command: %v %v", name, strings.Join(args, " "))
	cmd := exec.Command("sudo", "sh", "-c", strings.Join(append([]string{name}, args...), " "))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
