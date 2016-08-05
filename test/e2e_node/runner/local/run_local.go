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
	"strings"

	"k8s.io/kubernetes/test/e2e_node"

	"github.com/golang/glog"
)

var buildDependencies = flag.Bool("build-dependencies", true, "If true, build all dependencies.")
var buildImages = flag.Bool("build-images", true, "If true, build all container images.")
var ginkgoFlags = flag.String("ginkgo-flags", "", "Space-separated list of arguments to pass to Ginkgo test runner.")
var testFlags = flag.String("test-flags", "", "Space-separated list of arguments to pass to node e2e test.")

const apiserver = "kube-apiserver"

func main() {
	flag.Parse()

	// Build dependencies - ginkgo, kubelet and apiserver.
	if *buildDependencies {
		e2e_node.BuildGo()
	}

	// Build image for test infra and apiserver container.
	if *buildImages {
		images := e2e_node.BuildContainerImage()
		// Cleanup the images
		defer cleanupContainerImages(images)
		loadContainerImages(images)
		// Must not specify apiserver-image if buildImages is enabled.
		flags := *testFlags + " --apiserver-image=" + images[apiserver].Tag
		// Overwrite the flag.
		testFlags = &flags
	}

	// Run node e2e test
	ginkgo := e2e_node.GetGinkgoBin()
	test, err := e2e_node.GetK8sNodeTestDir()
	if err != nil {
		glog.Fatalf("Failed to get node test directory: %v", err)
	}
	runCommand(ginkgo, *ginkgoFlags, test, "--", *testFlags)
	return
}

func loadContainerImages(images map[string]e2e_node.ImageInfo) {
	glog.Info("Loading k8s container images...")
	for _, image := range images {
		err := runCommand("sudo", "docker", "load", "-i", image.Tar)
		if err != nil {
			glog.Fatalf("Failed to load container image %v: %v\n", image.Tar, err)
		}
	}
}

func cleanupContainerImages(images map[string]e2e_node.ImageInfo) {
	glog.Info("Cleanup k8s container images...")
	for _, image := range images {
		err := runCommand("sudo", "docker", "rmi", image.Tag)
		if err != nil {
			glog.Fatalf("Failed to remove container image %v: %v\n", image.Tag, err)
		}
	}
}

func runCommand(name string, args ...string) error {
	cmd := exec.Command("sh", "-c", strings.Join(append([]string{name}, args...), " "))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
