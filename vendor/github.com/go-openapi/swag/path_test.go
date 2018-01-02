// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package swag

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"path"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
)

func makeDirStructure(t *testing.T, tgt string) (string, string, error) {
	if tgt == "" {
		tgt = "pkgpaths"
	}
	td, err := ioutil.TempDir("", tgt)
	if err != nil {
		return "", "", err
	}
	td2, err := ioutil.TempDir("", tgt+"-2")
	if err != nil {
		return "", "", err
	}
	realPath := filepath.Join(td, "src", "foo", "bar")
	if err := os.MkdirAll(realPath, os.ModePerm); err != nil {
		return "", "", err
	}
	linkPathBase := filepath.Join(td, "src", "baz")
	if err := os.MkdirAll(linkPathBase, os.ModePerm); err != nil {
		return "", "", err
	}
	linkPath := filepath.Join(linkPathBase, "das")
	if err := os.Symlink(realPath, linkPath); err != nil {
		return "", "", err
	}

	realPath = filepath.Join(td2, "src", "fuu", "bir")
	if err := os.MkdirAll(realPath, os.ModePerm); err != nil {
		return "", "", err
	}
	linkPathBase = filepath.Join(td2, "src", "biz")
	if err := os.MkdirAll(linkPathBase, os.ModePerm); err != nil {
		return "", "", err
	}
	linkPath = filepath.Join(linkPathBase, "dis")
	if err := os.Symlink(realPath, linkPath); err != nil {
		return "", "", err
	}
	return td, td2, nil
}

func TestFindPackage(t *testing.T) {
	pth, pth2, err := makeDirStructure(t, "")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		os.RemoveAll(pth)
		os.RemoveAll(pth2)
	}()

	searchPath := pth + string(filepath.ListSeparator)  + pth2
	// finds package when real name mentioned
	pkg := FindInSearchPath(searchPath, "foo/bar")
	assert.NotEmpty(t, pkg)
	assertPath(t, path.Join(pth, "src", "foo", "bar"), pkg)
	// finds package when real name is mentioned in secondary
	pkg = FindInSearchPath(searchPath, "fuu/bir")
	assert.NotEmpty(t, pkg)
	assertPath(t, path.Join(pth2, "src", "fuu", "bir"), pkg)
	// finds package when symlinked
	pkg = FindInSearchPath(searchPath, "baz/das")
	assert.NotEmpty(t, pkg)
	assertPath(t, path.Join(pth, "src", "foo", "bar"), pkg)
	// finds package when symlinked in secondary
	pkg = FindInSearchPath(searchPath, "biz/dis")
	assert.NotEmpty(t, pkg)
	assertPath(t, path.Join(pth2, "src", "fuu", "bir"), pkg)
	// return empty string when nothing is found
	pkg = FindInSearchPath(searchPath, "not/there")
	assert.Empty(t, pkg)
}

func assertPath(t testing.TB, expected, actual string) bool {
	fp, err := filepath.EvalSymlinks(expected)
	if assert.NoError(t, err) {
		return assert.Equal(t, fp, actual)
	}
	return true
}

func TestFullGOPATH(t *testing.T) {
	os.Unsetenv(GOPATHKey)
	ngp := "/some/where:/other/place"
	os.Setenv(GOPATHKey, ngp)

	ogp := os.Getenv(GOPATHKey)
	defer os.Setenv(GOPATHKey, ogp)

	expected := ngp + ":" + runtime.GOROOT()
	assert.Equal(t, expected, FullGoSearchPath())
}
