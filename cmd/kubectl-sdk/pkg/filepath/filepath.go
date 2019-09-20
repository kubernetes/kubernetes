/*
Copyright 2019 The Kubernetes Authors.

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

package filepath

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"k8s.io/apimachinery/pkg/version"
	"k8s.io/kubernetes/cmd/kubectl-sdk/pkg/util"
)

const windowsOS = "windows"

// DirectoryGetter wraps a few operating sytem specific methods.
type DirectoryGetter interface {
	CurrentDirectory() (string, error)
	GetOS() string
}

// ExeDirGetter implements the DirectoryGetter interface.
type ExeDirGetter struct{}

// CurrentDirectory returns the absolute full directory path of the
// currently running executable.
func (e *ExeDirGetter) CurrentDirectory() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	abs, err := filepath.EvalSymlinks(exe)
	if err != nil {
		return "", err
	}
	return filepath.Dir(abs), nil
}

// Get OS returns the current operating system as a string. There should
// be three: "linux", "darwin", and "windows".
func (e *ExeDirGetter) GetOS() string {
	return runtime.GOOS
}

// FilepathBuilder encapsulates the data and functionality to build the full
// versioned kubectl filepath from the server version.
type FilepathBuilder struct {
	dirGetter DirectoryGetter
	// Function to call to check if a file exists.
	filestatFunc func(string) (os.FileInfo, error)
}

// NewFilepathBuilder encapsulates information necessary to build the full
// file path of the versioned kubectl binary to execute.
func NewFilepathBuilder(dirGetter DirectoryGetter, filestat func(string) (os.FileInfo, error)) *FilepathBuilder {
	return &FilepathBuilder{
		dirGetter:    dirGetter,
		filestatFunc: filestat,
	}
}

const kubectlBinaryName = "kubectl"

// VersionedFilePath returns the full absolute file path to the versioned kubectl
// binary to dispatch to. On error, empty string is returned.
func (c *FilepathBuilder) VersionedFilePath(version version.Info) (string, error) {
	// Initial parameter validation.
	if c.dirGetter == nil {
		return "", fmt.Errorf("VersionedFilePath: directory getter is nil")
	}
	// Get the major and minor versions.
	major, err := util.GetMajorVersion(version)
	if err != nil {
		return "", err
	}
	minor, err := util.GetMinorVersion(version)
	if err != nil {
		return "", err
	}
	// Example: major: "1", minor: "12" -> "kubectl.1.12"
	kubectlFilename := fmt.Sprintf("%s.%d.%d", kubectlBinaryName, major, minor)
	if c.dirGetter.GetOS() == windowsOS {
		kubectlFilename += ".exe"
	}
	currentDir, err := c.dirGetter.CurrentDirectory()
	if err != nil {
		return "", err
	}
	return filepath.Join(currentDir, kubectlFilename), nil
}

func (c *FilepathBuilder) ValidateFilepath(filepath string) error {
	if _, err := c.filestatFunc(filepath); err != nil {
		return err
	}
	return nil
}
