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

package remote

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"k8s.io/kubernetes/test/e2e_node/builder"
)

type NodeE2ERemote struct{}

func InitNodeE2ERemote() TestSuite {
	// TODO: Register flags.
	return &NodeE2ERemote{}
}

const (
	localGCIMounterPath = "cluster/gce/gci/mounter/mounter"
	CNIRelease          = "07a8a28637e97b22eb8dfe710eeae1344f69d16e"
	CNIDirectory        = "cni"
	CNIURL              = "https://storage.googleapis.com/kubernetes-release/network-plugins/cni-" + CNIRelease + ".tar.gz"
)

// SetupTestPackage sets up the test package with binaries k8s required for node e2e tests
func (n *NodeE2ERemote) SetupTestPackage(tardir string) error {
	// Build the executables
	if err := builder.BuildGo(); err != nil {
		return fmt.Errorf("failed to build the depedencies: %v", err)
	}

	// Make sure we can find the newly built binaries
	buildOutputDir, err := builder.GetK8sBuildOutputDir()
	if err != nil {
		return fmt.Errorf("failed to locate kubernetes build output directory %v", err)
	}

	// Copy binaries
	requiredBins := []string{"kubelet", "e2e_node.test", "ginkgo"}
	for _, bin := range requiredBins {
		source := filepath.Join(buildOutputDir, bin)
		if _, err := os.Stat(source); err != nil {
			return fmt.Errorf("failed to locate test binary %s: %v", bin, err)
		}
		out, err := exec.Command("cp", source, filepath.Join(tardir, bin)).CombinedOutput()
		if err != nil {
			return fmt.Errorf("failed to copy %q: %v Output: %q", bin, err, out)
		}
	}

	// Include the GCI mounter artifacts in the deployed tarball
	k8sDir, err := builder.GetK8sRootDir()
	if err != nil {
		return fmt.Errorf("Could not find K8s root dir! Err: %v", err)
	}
	localSource := "cluster/gce/gci/mounter/mounter"
	source := filepath.Join(k8sDir, localSource)

	// Require the GCI mounter script, we want to make sure the remote test runner stays up to date if the mounter file moves
	if _, err := os.Stat(source); err != nil {
		return fmt.Errorf("Could not find GCI mounter script at %q! If this script has been (re)moved, please update the e2e node remote test runner accordingly! Err: %v", source, err)
	}

	bindir := "cluster/gce/gci/mounter"
	bin := "mounter"
	destdir := filepath.Join(tardir, bindir)
	dest := filepath.Join(destdir, bin)
	out, err := exec.Command("mkdir", "-p", filepath.Join(tardir, bindir)).CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to create directory %q for GCI mounter script. Err: %v. Output:\n%s", destdir, err, out)
	}
	out, err = exec.Command("cp", source, dest).CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to copy GCI mounter script to the archive bin. Err: %v. Output:\n%s", err, out)
	}
	return nil
}

// RunTest runs test on the node.
func (n *NodeE2ERemote) RunTest(host, workspace, results, junitFilePrefix, testArgs, ginkgoFlags string, timeout time.Duration) (string, error) {
	return "", fmt.Errorf("not implemented")
}
