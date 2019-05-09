/*
Copyright 2018 The Kubernetes Authors.

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

package utils

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// GetK8sRootDir returns the root directory for kubernetes, if present in the gopath.
func GetK8sRootDir() (string, error) {
	dir, err := RootDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, fmt.Sprintf("%s/", "k8s.io/kubernetes")), nil
}

// GetCAdvisorRootDir returns the root directory for cAdvisor, if present in the gopath.
func GetCAdvisorRootDir() (string, error) {
	dir, err := RootDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, fmt.Sprintf("%s/", "github.com/google/cadvisor")), nil
}

// RootDir gets the on-disk kubernetes source directory, returning an error is none is found
func RootDir() (string, error) {
	// Get the directory of the current executable
	_, testExec, _, _ := runtime.Caller(0)
	path := filepath.Dir(testExec)

	// Look for the kubernetes source root directory
	if strings.Contains(path, "k8s.io/kubernetes") {
		splitPath := strings.Split(path, "k8s.io/kubernetes")
		return splitPath[0], nil
	}

	return "", errors.New("could not find kubernetes source root directory")
}

// GetK8sBuildOutputDir returns the build output directory for k8s
func GetK8sBuildOutputDir() (string, error) {
	k8sRoot, err := GetK8sRootDir()
	if err != nil {
		return "", err
	}
	buildOutputDir := filepath.Join(k8sRoot, "_output/local/go/bin")
	if _, err := os.Stat(buildOutputDir); err != nil {
		return "", err
	}
	return buildOutputDir, nil
}
