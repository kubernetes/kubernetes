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
	"test/e2e_cri/e2e_cri.test",
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
