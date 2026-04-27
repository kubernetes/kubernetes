/*
Copyright 2021 The Kubernetes Authors.

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

package cp

import (
	"path"
	"path/filepath"
	"strings"
)

type fileSpec struct {
	PodName      string
	PodNamespace string
	File         pathSpec
}

type pathSpec interface {
	String() string
}

// localPath represents a client-native path, which will differ based
// on the client OS, its methods will use path/filepath package which
// is OS dependant
type localPath struct {
	file string
}

func newLocalPath(fileName string) localPath {
	file := stripTrailingSlash(fileName)
	return localPath{file: file}
}

func (p localPath) String() string {
	return p.file
}

func (p localPath) Dir() localPath {
	return newLocalPath(filepath.Dir(p.file))
}

func (p localPath) Base() localPath {
	return newLocalPath(filepath.Base(p.file))
}

func (p localPath) Clean() localPath {
	return newLocalPath(filepath.Clean(p.file))
}

func (p localPath) Join(elem pathSpec) localPath {
	return newLocalPath(filepath.Join(p.file, elem.String()))
}

func (p localPath) Glob() (matches []string, err error) {
	return filepath.Glob(p.file)
}

func (p localPath) StripSlashes() localPath {
	return newLocalPath(stripLeadingSlash(p.file))
}

func isRelative(base, target localPath) bool {
	relative, err := filepath.Rel(base.String(), target.String())
	if err != nil {
		return false
	}
	return relative == "." || relative == stripPathShortcuts(relative)
}

// remotePath represents always UNIX path, its methods will use path
// package which is always using `/`
type remotePath struct {
	file string
}

func newRemotePath(fileName string) remotePath {
	// we assume remote file is a linux container but we need to convert
	// windows path separators to unix style for consistent processing
	file := strings.ReplaceAll(stripTrailingSlash(fileName), `\`, "/")
	return remotePath{file: file}
}

func (p remotePath) String() string {
	return p.file
}

func (p remotePath) Dir() remotePath {
	return newRemotePath(path.Dir(p.file))
}

func (p remotePath) Base() remotePath {
	return newRemotePath(path.Base(p.file))
}

func (p remotePath) Clean() remotePath {
	return newRemotePath(path.Clean(p.file))
}

func (p remotePath) Join(elem pathSpec) remotePath {
	return newRemotePath(path.Join(p.file, elem.String()))
}

func (p remotePath) StripShortcuts() remotePath {
	p = p.Clean()
	return newRemotePath(stripPathShortcuts(p.file))
}

func (p remotePath) StripSlashes() remotePath {
	return newRemotePath(stripLeadingSlash(p.file))
}

// strips trailing slash (if any) both unix and windows style
func stripTrailingSlash(file string) string {
	if len(file) == 0 {
		return file
	}
	if file != "/" && strings.HasSuffix(string(file[len(file)-1]), "/") {
		return file[:len(file)-1]
	}
	return file
}

func stripLeadingSlash(file string) string {
	// tar strips the leading '/' and '\' if it's there, so we will too
	return strings.TrimLeft(file, `/\`)
}

// stripPathShortcuts removes any leading or trailing "../" from a given path
func stripPathShortcuts(p string) string {
	newPath := p
	trimmed := strings.TrimPrefix(newPath, "../")

	for trimmed != newPath {
		newPath = trimmed
		trimmed = strings.TrimPrefix(newPath, "../")
	}

	// trim leftover {".", ".."}
	if newPath == "." || newPath == ".." {
		newPath = ""
	}

	if len(newPath) > 0 && string(newPath[0]) == "/" {
		return newPath[1:]
	}

	return newPath
}
