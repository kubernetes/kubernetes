/*
Copyright 2016 Google Inc. All Rights Reserved.

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

// Package wspace provides a method to find the root of the bazel tree.
package wspace

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/bazelbuild/buildtools/build"
)

const workspaceFile = "WORKSPACE"

func alwaysTrue(fi os.FileInfo) bool {
	return true
}

var repoRootFiles = map[string]func(os.FileInfo) bool{
	workspaceFile: alwaysTrue,
	".buckconfig": alwaysTrue,
	"pants": func(fi os.FileInfo) bool {
		return fi.Mode()&os.ModeType == 0 && fi.Mode()&0100 == 0100
	},
}

// findContextPath finds the context path inside of a WORKSPACE-rooted source tree.
func findContextPath(rootDir string) (string, error) {
	if rootDir == "" {
		return os.Getwd()
	}
	return rootDir, nil
}

// FindWorkspaceRoot splits the current code context (the rootDir if present,
// the working directory if not.) It returns the path of the directory
// containing the WORKSPACE file, and the rest.
func FindWorkspaceRoot(rootDir string) (root string, rest string) {
	wd, err := findContextPath(rootDir)
	if err != nil {
		return "", ""
	}
	if root, err = Find(wd); err != nil {
		return "", ""
	}
	if len(wd) == len(root) {
		return root, ""
	}
	return root, wd[len(root)+1:]
}

// Find searches from the given dir and up for the WORKSPACE file
// returning the directory containing it, or an error if none found in the tree.
func Find(dir string) (string, error) {
	if dir == "" || dir == "/" || dir == "." || (len(dir) == 3 && strings.HasSuffix(dir, ":\\")) {
		return "", os.ErrNotExist
	}
	for repoRootFile, fiFunc := range repoRootFiles {
		if fi, err := os.Stat(filepath.Join(dir, repoRootFile)); err == nil && fiFunc(fi) {
			return dir, nil
		} else if !os.IsNotExist(err) {
			return "", err
		}
	}
	return Find(filepath.Dir(dir))
}

// FindRepoBuildFiles parses the WORKSPACE to find BUILD files for non-Bazel
// external repositories, specifically those defined by one of these rules:
//   new_local_repository(), new_git_repository(), new_http_archive()
func FindRepoBuildFiles(root string) (map[string]string, error) {
	ws := filepath.Join(root, workspaceFile)
	kinds := []string{
		"new_local_repository",
		"new_git_repository",
		"new_http_archive",
	}
	data, err := ioutil.ReadFile(ws)
	if err != nil {
		return nil, err
	}
	ast, err := build.Parse(ws, data)
	if err != nil {
		return nil, err
	}
	files := make(map[string]string)
	for _, kind := range kinds {
		for _, r := range ast.Rules(kind) {
			buildFile := r.AttrString("build_file")
			if buildFile == "" {
				continue
			}
			buildFile = strings.Replace(buildFile, ":", "/", -1)
			files[r.Name()] = filepath.Join(root, buildFile)
		}
	}
	return files, nil
}
