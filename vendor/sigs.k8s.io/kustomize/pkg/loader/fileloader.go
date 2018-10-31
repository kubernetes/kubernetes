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

package loader

import (
	"fmt"
	"os"
	"path/filepath"

	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/ifc"
)

const currentDir = "."

// fileLoader loads files from a file system.
type fileLoader struct {
	root string
	fSys fs.FileSystem
}

// NewFileLoader returns a new fileLoader.
func NewFileLoader(fSys fs.FileSystem) *fileLoader {
	return newFileLoaderAtRoot("", fSys)
}

// newFileLoaderAtRoot returns a new fileLoader with given root.
func newFileLoaderAtRoot(root string, fs fs.FileSystem) *fileLoader {
	return &fileLoader{root: root, fSys: fs}
}

// Root returns the root location for this Loader.
func (l *fileLoader) Root() string {
	return l.root
}

// Returns a new Loader rooted at newRoot. "newRoot" MUST be
// a directory (not a file). The directory can have a trailing
// slash or not.
// Example: "/home/seans/project" or "/home/seans/project/"
// NOT "/home/seans/project/file.yaml".
func (l *fileLoader) New(newRoot string) (ifc.Loader, error) {
	return NewLoader(newRoot, l.root, l.fSys)
}

// IsAbsPath return true if the location calculated with the root
// and location params a full file path.
func (l *fileLoader) IsAbsPath(root string, location string) bool {
	fullFilePath, err := l.fullLocation(root, location)
	if err != nil {
		return false
	}
	return filepath.IsAbs(fullFilePath)
}

// fullLocation returns some notion of a full path.
// If location is a full file path, then ignore root. If location is relative, then
// join the root path with the location path. Either root or location can be empty,
// but not both. Special case for ".": Expands to current working directory.
// Example: "/home/seans/project", "subdir/bar" -> "/home/seans/project/subdir/bar".
func (l *fileLoader) fullLocation(root string, location string) (string, error) {
	// First, validate the parameters
	if len(root) == 0 && len(location) == 0 {
		return "", fmt.Errorf("unable to calculate full location: root and location empty")
	}
	// Special case current directory, expanding to full file path.
	if location == currentDir {
		currentDir, err := os.Getwd()
		if err != nil {
			return "", err
		}
		location = currentDir
	}
	// Assume the location is a full file path. If not, then join root with location.
	fullLocation := location
	if !filepath.IsAbs(location) {
		fullLocation = filepath.Join(root, location)
	}
	return fullLocation, nil
}

// Load returns the bytes from reading a file at fullFilePath.
// Implements the Loader interface.
func (l *fileLoader) Load(location string) ([]byte, error) {
	fullLocation, err := l.fullLocation(l.root, location)
	if err != nil {
		fmt.Printf("Trouble in fulllocation: %v\n", err)
		return nil, err
	}
	return l.fSys.ReadFile(fullLocation)
}

// Cleanup does nothing
func (l *fileLoader) Cleanup() error {
	return nil
}
