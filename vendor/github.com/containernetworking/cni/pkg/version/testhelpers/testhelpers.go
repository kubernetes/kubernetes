// Copyright 2016 CNI authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package testhelpers supports testing of CNI components of different versions
//
// For example, to build a plugin against an old version of the CNI library,
// we can pass the plugin's source and the old git commit reference to BuildAt.
// We could then test how the built binary responds when called by the latest
// version of this library.
package testhelpers

import (
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const packageBaseName = "github.com/containernetworking/cni"

func run(cmd *exec.Cmd) error {
	out, err := cmd.CombinedOutput()
	if err != nil {
		command := strings.Join(cmd.Args, " ")
		return fmt.Errorf("running %q: %s", command, out)
	}
	return nil
}

func goBuildEnviron(gopath string) []string {
	environ := os.Environ()
	for i, kvp := range environ {
		if strings.HasPrefix(kvp, "GOPATH=") {
			environ[i] = "GOPATH=" + gopath
			return environ
		}
	}
	environ = append(environ, "GOPATH="+gopath)
	return environ
}

func buildGoProgram(gopath, packageName, outputFilePath string) error {
	cmd := exec.Command("go", "build", "-o", outputFilePath, packageName)
	cmd.Env = goBuildEnviron(gopath)
	return run(cmd)
}

func createSingleFilePackage(gopath, packageName string, fileContents []byte) error {
	dirName := filepath.Join(gopath, "src", packageName)
	err := os.MkdirAll(dirName, 0700)
	if err != nil {
		return err
	}

	return ioutil.WriteFile(filepath.Join(dirName, "main.go"), fileContents, 0600)
}

func removePackage(gopath, packageName string) error {
	dirName := filepath.Join(gopath, "src", packageName)
	return os.RemoveAll(dirName)
}

func isRepoRoot(path string) bool {
	_, err := ioutil.ReadDir(filepath.Join(path, ".git"))
	return (err == nil) && (filepath.Base(path) == "cni")
}

func LocateCurrentGitRepo() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}

	for i := 0; i < 5; i++ {
		if isRepoRoot(dir) {
			return dir, nil
		}

		dir, err = filepath.Abs(filepath.Dir(dir))
		if err != nil {
			return "", fmt.Errorf("abs(dir(%q)): %s", dir, err)
		}
	}

	return "", fmt.Errorf("unable to find cni repo root, landed at %q", dir)
}

func gitCloneThisRepo(cloneDestination string) error {
	err := os.MkdirAll(cloneDestination, 0700)
	if err != nil {
		return err
	}

	currentGitRepo, err := LocateCurrentGitRepo()
	if err != nil {
		return err
	}

	return run(exec.Command("git", "clone", currentGitRepo, cloneDestination))
}

func gitCheckout(localRepo string, gitRef string) error {
	return run(exec.Command("git", "-C", localRepo, "checkout", gitRef))
}

// BuildAt builds the go programSource using the version of the CNI library
// at gitRef, and saves the resulting binary file at outputFilePath
func BuildAt(programSource []byte, gitRef string, outputFilePath string) error {
	tempGoPath, err := ioutil.TempDir("", "cni-git-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tempGoPath)

	cloneDestination := filepath.Join(tempGoPath, "src", packageBaseName)
	err = gitCloneThisRepo(cloneDestination)
	if err != nil {
		return err
	}

	err = gitCheckout(cloneDestination, gitRef)
	if err != nil {
		return err
	}

	rand.Seed(time.Now().UnixNano())
	testPackageName := fmt.Sprintf("test-package-%x", rand.Int31())

	err = createSingleFilePackage(tempGoPath, testPackageName, programSource)
	if err != nil {
		return err
	}
	defer removePackage(tempGoPath, testPackageName)

	err = buildGoProgram(tempGoPath, testPackageName, outputFilePath)
	if err != nil {
		return err
	}

	return nil
}
