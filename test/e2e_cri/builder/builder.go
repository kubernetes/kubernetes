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

package builder

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
	"k8s.io/kubernetes/test/e2e_cri",
	"k8s.io/kubernetes/vendor/github.com/onsi/ginkgo/ginkgo",
}

func BuildGo() error {
	glog.Infof("Building k8s binaries...")
	outputDir, err := GetK8sBuildOutputDir()
	if err != nil {
		glog.Fatalf("Failed to get build output directory: %v", err)
	}

	err = RunCommand("go", "test", "-c", "-v", "-o", filepath.Join(outputDir, "e2e_cri.test"), buildTargets[0])
	if err != nil {
		return fmt.Errorf("failed to build e2e_cri.test %v\n", err)
	}

	err = RunCommand("go", "build", "-o", filepath.Join(outputDir, "ginkgo"), buildTargets[1])
	if err != nil {
		return fmt.Errorf("failed to build go ginkgo %v\n", err)
	}
	return nil
}

func RunCommand(name string, args ...string) error {
	glog.Infof("Building command: %v %v", name, strings.Join(args, " "))
	cmd := exec.Command("sh", "-c", strings.Join(append([]string{name}, args...), " "))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
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
	if err := os.MkdirAll(buildOutputDir, 0731); err != nil {
		return "", err
	}
	return buildOutputDir, nil
}
