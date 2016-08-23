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

package build

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/golang/glog"
)

var k8sBinDir = flag.String("k8s-bin-dir", "", "Directory containing k8s kubelet binaries.")

var buildTargets = []string{
	"cmd/kubelet",
	"test/e2e_node/e2e_node.test",
	"vendor/github.com/onsi/ginkgo/ginkgo",
}

func BuildGo() error {
	glog.Infof("Building k8s binaries...")
	k8sRoot, err := getK8sRootDir()
	if err != nil {
		return fmt.Errorf("failed to locate kubernetes root directory %v.", err)
	}
	targets := strings.Join(buildTargets, " ")
	cmd := exec.Command("make", "-C", k8sRoot, fmt.Sprintf("WHAT=%s", targets))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err = cmd.Run()
	if err != nil {
		return fmt.Errorf("failed to build go packages %v\n", err)
	}
	return nil
}

func getK8sBin(bin string) (string, error) {
	// Use commandline specified path
	if *k8sBinDir != "" {
		absPath, err := filepath.Abs(*k8sBinDir)
		if err != nil {
			return "", err
		}
		if _, err := os.Stat(filepath.Join(*k8sBinDir, bin)); err != nil {
			return "", fmt.Errorf("Could not find %s under directory %s.", bin, absPath)
		}
		return filepath.Join(absPath, bin), nil
	}

	path, err := filepath.Abs(filepath.Dir(os.Args[0]))
	if err != nil {
		return "", fmt.Errorf("Could not find absolute path of directory containing the tests %s.", filepath.Dir(os.Args[0]))
	}
	if _, err := os.Stat(filepath.Join(path, bin)); err == nil {
		return filepath.Join(path, bin), nil
	}

	buildOutputDir, err := GetK8sBuildOutputDir()
	if err != nil {
		return "", err
	}
	if _, err := os.Stat(filepath.Join(buildOutputDir, bin)); err == nil {
		return filepath.Join(buildOutputDir, bin), nil
	}

	// Give up with error
	return "", fmt.Errorf("Unable to locate %s.  Can be defined using --k8s-path.", bin)
}

// TODO: Dedup / merge this with comparable utilities in e2e/util.go
func getK8sRootDir() (string, error) {
	// Get the directory of the current executable
	_, testExec, _, _ := runtime.Caller(0)
	path := filepath.Dir(testExec)

	// Look for the kubernetes source root directory
	if strings.Contains(path, "k8s.io/kubernetes") {
		splitPath := strings.Split(path, "k8s.io/kubernetes")
		return filepath.Join(splitPath[0], "k8s.io/kubernetes/"), nil
	}

	return "", fmt.Errorf("Could not find kubernetes source root directory.")
}

func GetK8sBuildOutputDir() (string, error) {
	k8sRoot, err := getK8sRootDir()
	if err != nil {
		return "", err
	}
	buildOutputDir := filepath.Join(k8sRoot, "_output/local/go/bin")
	if _, err := os.Stat(buildOutputDir); err != nil {
		return "", err
	}
	return buildOutputDir, nil
}

func GetKubeletServerBin() string {
	bin, err := getK8sBin("kubelet")
	if err != nil {
		glog.Fatalf("Could not locate kubelet binary %v.", err)
	}
	return bin
}
