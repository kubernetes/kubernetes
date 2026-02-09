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
	"regexp"
	"strings"

	"k8s.io/kubernetes/test/e2e_node/builder"
	"k8s.io/kubernetes/test/e2e_node/system"
	"k8s.io/kubernetes/test/utils"

	"k8s.io/klog/v2"
)

var buildDependencies = flag.Bool("build-dependencies", true, "If true, build all dependencies.")
var ginkgoFlags = flag.String("ginkgo-flags", "", "Space-separated list of arguments to pass to Ginkgo test runner, with shell-style quoting and escaping.")
var testFlags = flag.String("test-flags", "", "Space-separated list of arguments to pass to node e2e test, with shell-style quoting and escaping.")
var systemSpecName = flag.String("system-spec-name", "", fmt.Sprintf("The name of the system spec used for validating the image in the node conformance test. The specs are at %s. If unspecified, the default built-in spec (system.DefaultSpec) will be used.", system.SystemSpecPath))
var extraEnvs = flag.String("extra-envs", "", "The extra environment variables needed for node e2e tests. Format: a list of key=value pairs, e.g., env1=val1,env2=val2")
var runtimeConfig = flag.String("runtime-config", "", "The runtime configuration for the API server on the node e2e tests. Format: a list of key=value pairs, e.g., env1=val1,env2=val2")
var kubeletConfigFile = flag.String("kubelet-config-file", "", "The KubeletConfiguration file that should be applied to the kubelet")
var debugTool = flag.String("debug-tool", "", "'delve', 'dlv' or 'gdb': run e2e_node.test under that debugger")

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
	test := filepath.Join(outputDir, "e2e_node.test")
	interactive := false
	var cmd string
	var args []string
	switch *debugTool {
	case "":
		// No debugger, run gingko directly.
		cmd = filepath.Join(outputDir, "ginkgo")
		args = []string{*ginkgoFlags, test, "--"}
	case "delve", "dlv":
		dlv, err := exec.LookPath("dlv")
		if err != nil {
			klog.Fatalf("'dlv' not found: %v", err)
		}
		interactive = true
		cmd = dlv
		args = []string{"exec", test, "--", addGinkgoArgPrefix(*ginkgoFlags)}
	case "gdb":
		gdb, err := exec.LookPath("gdb")
		if err != nil {
			klog.Fatalf("'gdb' not found: %v", err)
		}
		interactive = true
		cmd = gdb
		args = []string{test, "--", addGinkgoArgPrefix(*ginkgoFlags)}
	}

	args = append(args, *testFlags, fmt.Sprintf("--runtime-config=%s", *runtimeConfig))
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
	if err := runCommand(interactive, cmd, args...); err != nil {
		klog.Exitf("Test failed: %v", err)
	}
	return
}

func addGinkgoArgPrefix(ginkgoFlags string) string {
	// Warning, hack! This simplistic search/replace assumes that hyphens do not appear
	// inside argument values.
	//
	// The right solution would be to use github.com/anmitsu/go-shlex to split
	// the -ginkgo-flags and -test-flags strings into individual arguments, then invoke
	// exec.Command with the resulting string slice instead of passing a single string
	// to sh. But github.com/anmitsu/go-shlex is not a Kubernetes dependency and not
	// worth adding.

	ginkgoFlags = regexp.MustCompile(`(^| )--?`).ReplaceAllString(ginkgoFlags, `$1--ginkgo.`)
	return ginkgoFlags
}

func runCommand(interactive bool, name string, args ...string) error {
	klog.Infof("Running command: %v %v", name, strings.Join(args, " "))
	// Using sh is necessary because the args are using POSIX quoting.
	// sh has to parse that.
	cmd := exec.Command("sudo", "sh", "-c", strings.Join(append([]string{name}, args...), " "))
	if interactive {
		// stdin must be a console.
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdin
		cmd.Stderr = os.Stdin
	} else {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}

	return cmd.Run()
}
