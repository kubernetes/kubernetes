// Copyright 2018 Microsoft Corporation
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

package util

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// PackageSet is a collection of Go packages.
// Key is the full package import path, value indicates if the package has been indexed.
type PackageSet map[string]bool

// GetIndexedPackages returns the set of packages that have already been indexed.
// It finds all entries matching the regex `github\.com/Azure/azure-sdk-for-go/services/.*?"`.
func GetIndexedPackages(content io.Reader) (PackageSet, error) {
	body, err := ioutil.ReadAll(content)
	if err != nil {
		return nil, err
	}

	if len(body) < 1 {
		return nil, errors.New("did't receive a response body when lookinig for indexed packages")
	}

	// scrape the content to create the package list
	pkgs := PackageSet{}
	regex := regexp.MustCompile(`github\.com/Azure/azure-sdk-for-go/services/.*?"`)
	finds := regex.FindAllString(string(body), -1)

	for _, find := range finds {
		// strip of the trailing "
		pkg := find[:len(find)-1]
		pkgs[pkg] = true
	}
	return pkgs, nil
}

// GetPackagesForIndexing returns the set of packages, calculated from the specified directory, to be indexed.
// Each directory entry is converted to a complete package path, e.g. "github.com/Azure/azure-sdk-for-go/services/foo/...".
func GetPackagesForIndexing(dir string) (PackageSet, error) {
	leafDirs := []string{}
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			// check if leaf dir
			fi, err := ioutil.ReadDir(path)
			if err != nil {
				return err
			}
			hasSubDirs := false
			for _, f := range fi {
				if f.IsDir() {
					hasSubDirs = true
					break
				}
			}
			if !hasSubDirs {
				leafDirs = append(leafDirs, path)
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	// dirs will look something like "D:\work\src\github.com\Azure\azure-sdk-for-go\services\..."
	// strip off the stuff before the github.com and change the whacks so it looks like a package import

	pkgs := PackageSet{}
	for _, dir := range leafDirs {
		i := strings.Index(dir, "github.com")
		if i < 0 {
			return nil, fmt.Errorf("didn't find github.com in directory '%s'", dir)
		}
		pkg := strings.Replace(dir[i:], "\\", "/", -1)
		pkgs[pkg] = false
	}
	return pkgs, nil
}
