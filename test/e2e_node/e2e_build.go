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

package e2e_node

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/golang/glog"
)

// TODO(random-liu): Move this to build directory.
var k8sBinDir = flag.String("k8s-bin-dir", "", "Directory containing k8s kubelet and kube-apiserver binaries.")

var buildTargets = []string{
	"cmd/kubelet",
	"cmd/kube-apiserver",
	"test/e2e_node/e2e_node.test",
	"vendor/github.com/onsi/ginkgo/ginkgo",
}

const outputDir = "_output/local/go/bin"

func BuildGo() {
	glog.Infof("Building k8s binaries...")
	k8sRoot, err := getK8sRootDir()
	if err != nil {
		glog.Fatalf("Failed to locate kubernetes root directory %v.", err)
	}
	targets := strings.Join(buildTargets, " ")
	cmd := exec.Command("make", "-C", k8sRoot, fmt.Sprintf("WHAT=%s", targets))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err = cmd.Run()
	if err != nil {
		glog.Fatalf("Failed to build go packages %v\n", err)
	}
}

const (
	buildImageCommon = "build/common.sh"
	buildImageFunc   = "kube::release::create_docker_images_for_server"
	imageRegistery   = "gcr.io/google_containers"
)

// Currently we only build container image for apiserver.
var buildImageTargets = []string{
	"kube-apiserver",
}

type ImageInfo struct {
	Tag string
	Tar string
}

// BuildContainerImage builds all target images, saves to tarballs,
// returns the tags and tarball paths of the images.
func BuildContainerImage() map[string]ImageInfo {
	glog.Info("Building k8s container images...")
	k8sRoot, err := getK8sRootDir()
	if err != nil {
		glog.Fatalf("Failed to locate kubernetes root directory %v.", err)
	}
	buildScript := fmt.Sprintf("source %s;  %s %s %s %s",
		filepath.Join(k8sRoot, buildImageCommon), buildImageFunc, filepath.Join(k8sRoot, outputDir),
		runtime.GOARCH, strings.Join(buildImageTargets, " "))
	cmd := exec.Command("bash", "-c", buildScript)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err = cmd.Run()
	if err != nil {
		glog.Fatalf("Failed to build container images %v\n", err)
	}
	tags := map[string]ImageInfo{}
	for _, target := range buildImageTargets {
		tagFile := filepath.Join(k8sRoot, outputDir, target+".docker_tag")
		tag, err := ioutil.ReadFile(tagFile)
		if err != nil {
			glog.Fatalf("Failed to read tag file %q: %v", tagFile, tag)
		}
		tags[target] = ImageInfo{
			Tag: fmt.Sprintf("%s/%s:%s", imageRegistery, target, strings.TrimSpace(string(tag))),
			Tar: filepath.Join(k8sRoot, outputDir, target+".tar"),
		}
	}
	return tags
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
	buildOutputDir := filepath.Join(k8sRoot, outputDir)
	if _, err := os.Stat(buildOutputDir); err != nil {
		return "", err
	}
	return buildOutputDir, nil
}

func GetK8sNodeTestDir() (string, error) {
	k8sRoot, err := getK8sRootDir()
	if err != nil {
		return "", err
	}
	buildOutputDir := filepath.Join(k8sRoot, "test/e2e_node")
	if _, err := os.Stat(buildOutputDir); err != nil {
		return "", err
	}
	return buildOutputDir, nil
}

func getKubeletServerBin() string {
	bin, err := getK8sBin("kubelet")
	if err != nil {
		glog.Fatalf("Could not locate kubelet binary %v.", err)
	}
	return bin
}

func getApiServerBin() string {
	bin, err := getK8sBin("kube-apiserver")
	if err != nil {
		glog.Fatalf("Could not locate kube-apiserver binary %v.", err)
	}
	return bin
}

func GetGinkgoBin() string {
	bin, err := getK8sBin("ginkgo")
	if err != nil {
		glog.Fatalf("Could not locate ginkgo binary %v.", err)
	}
	return bin
}
