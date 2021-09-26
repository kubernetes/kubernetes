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

package cmd

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/Azure/azure-sdk-for-go/tools/apidiff/report"
	"github.com/Azure/azure-sdk-for-go/tools/internal/modinfo"
)

var newline = "\n"

func init() {
	if runtime.GOOS == "windows" {
		newline = "\r\n"
	}

}
func Test_getTags(t *testing.T) {
	if os.Getenv("TRAVIS") == "true" {
		// travis does a shallow clone so tag count is not consistent
		t.SkipNow()
	}
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("failed to get cwd: %v", err)
	}
	tags, err := getTags(cwd, "v10")
	if err != nil {
		t.Fatalf("failed to get tags: %v", err)
	}
	if l := len(tags); l != 11 {
		t.Fatalf("expected 11 tags, got %d", l)
	}
	found := false
	for _, tag := range tags {
		if tag == "v10.1.0-beta" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("didn't find tag v10.1.0-beta")
	}
}

func Test_getTagPrefix(t *testing.T) {
	p, err := getTagPrefix(filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "redis", "mgmt", "2018-03-01", "redis"))
	if err != nil {
		t.Fatal("failed to get tag prefix")
	}
	if p != "services/redis/mgmt/2018-03-01/redis" {
		t.Fatalf("wrong value '%s' for tag prefix", p)
	}
	p, err = getTagPrefix("/work/src/github.com/something/else")
	if err == nil {
		t.Fatal("unexpected nil error")
	}
	if p != "" {
		t.Fatalf("unexpected tag '%s'", p)
	}
}

type mockModInfo struct {
	dir     string
	exports bool
	breaks  bool
}

func (mock mockModInfo) DestDir() string {
	return mock.dir
}

func (mock mockModInfo) NewExports() bool {
	return mock.exports
}

func (mock mockModInfo) BreakingChanges() bool {
	return mock.breaks
}

func (mock mockModInfo) VersionSuffix() bool {
	return modinfo.HasVersionSuffix(mock.dir)
}

func (mock mockModInfo) NewModule() bool {
	// not needed by tests
	return false
}

func (mock mockModInfo) GenerateReport() report.Package {
	// not needed by tests
	return report.Package{}
}

func Test_calculateModuleTagMajorV1(t *testing.T) {
	pkg := mockModInfo{
		dir: filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "foo"),
	}
	tag, err := calculateModuleTag([]string{}, pkg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tag != "services/foo/v1.0.0" {
		t.Fatalf("bad tag '%s", tag)
	}
}

func Test_calculateModuleTagMajorV2(t *testing.T) {
	tags := []string{
		"services/foo/v1.0.0",
		"services/foo/v1.1.0",
	}
	pkg := mockModInfo{
		dir:    filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "foo", "v2"),
		breaks: true,
	}
	tag, err := calculateModuleTag(tags, pkg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tag != "services/foo/v2.0.0" {
		t.Fatalf("bad tag '%s", tag)
	}
}

func Test_calculateModuleTagMajorV2Invalid(t *testing.T) {
	tags := []string{
		"services/foo/v1.0.0",
		"services/foo/v1.1.0",
	}
	pkg := mockModInfo{
		dir:    filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "foo"), // missing /v2 suffix
		breaks: true,
	}
	tag, err := calculateModuleTag(tags, pkg)
	if err == nil {
		t.Fatal("expected non-nil error")
	}
	if tag != "" {
		t.Fatal("expected no tag")
	}
}

func Test_calculateModuleTagMinorV1(t *testing.T) {
	tags := []string{
		"services/foo/v1.0.0",
		"services/foo/v1.0.1",
	}
	pkg := mockModInfo{
		dir:     filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "foo"),
		exports: true,
	}
	tag, err := calculateModuleTag(tags, pkg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tag != "services/foo/v1.1.0" {
		t.Fatalf("bad tag '%s", tag)
	}
}

func Test_calculateModuleTagMinorV2(t *testing.T) {
	tags := []string{
		"services/foo/v1.0.0",
		"services/foo/v1.0.1",
		"services/foo/v2.0.0",
	}
	pkg := mockModInfo{
		dir: filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "foo", "v2"),
	}
	tag, err := calculateModuleTag(tags, pkg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tag != "services/foo/v2.0.1" {
		t.Fatalf("bad tag '%s", tag)
	}
}

func Test_findLatestMajorVersion(t *testing.T) {
	ver, err := findLatestMajorVersion("../../testdata/scenarioa/foo/stage")
	if err != nil {
		t.Fatalf("failed to find LMV: %v", err)
	}
	if ver != filepath.Join("..", "..", "testdata", "scenarioa", "foo") {
		t.Fatalf("bad LMV %s", ver)
	}
	ver, err = findLatestMajorVersion("../../testdata/scenariod/foo/stage")
	if err != nil {
		t.Fatalf("failed to find LMV: %v", err)
	}
	if ver != filepath.Join("..", "..", "testdata", "scenariod", "foo", "v2") {
		t.Fatalf("bad LMV %s", ver)
	}
	ver, err = findLatestMajorVersion("../../testdata/scenariof/foo/stage")
	if err != nil {
		t.Fatalf("failed to find LMV: %v", err)
	}
	if ver != filepath.Join("..", "..", "testdata", "scenariof", "foo") {
		t.Fatalf("bad LMV %s", ver)
	}
}

type byteBufferSeeker struct {
	buf *bytes.Buffer
}

func (b byteBufferSeeker) Read(p []byte) (int, error) {
	return b.buf.Read(p)
}

func (b byteBufferSeeker) Write(p []byte) (int, error) {
	return b.buf.Write(p)
}

func (b byteBufferSeeker) Seek(offset int64, whence int) (int64, error) {
	if offset != 0 && whence != 0 {
		panic("seek only supports 0, 0")
	}
	b.buf.Reset()
	return 0, nil
}

func Test_updateGoModVerA(t *testing.T) {
	// updates from v1 to v2
	const before = "module github.com/Azure/azure-sdk-for-go/services/foo/mgmt/2019-05-01/foo\n\ngo 1.12\n"
	buf := byteBufferSeeker{
		buf: bytes.NewBuffer([]byte(before)),
	}
	err := updateGoModVer(buf, "v2")
	if err != nil {
		t.Fatalf("updateGoModVerA failed: %v", err)
	}
	const after = "module github.com/Azure/azure-sdk-for-go/services/foo/mgmt/2019-05-01/foo/v2\n\ngo 1.12\n"
	if !strings.EqualFold(buf.buf.String(), after) {
		t.Fatalf("bad go.mod update, epected %s got %s", after, buf.buf.String())
	}
}

func Test_updateGoModVerB(t *testing.T) {
	// updates from v2 to v3
	const before = "module github.com/Azure/azure-sdk-for-go/services/foo/mgmt/2019-05-01/foo/v2\n\ngo 1.12\n"
	buf := byteBufferSeeker{
		buf: bytes.NewBuffer([]byte(before)),
	}
	err := updateGoModVer(buf, "v3")
	if err != nil {
		t.Fatalf("updateGoModVerA failed: %v", err)
	}
	const after = "module github.com/Azure/azure-sdk-for-go/services/foo/mgmt/2019-05-01/foo/v3\n\ngo 1.12\n"
	if !strings.EqualFold(buf.buf.String(), after) {
		t.Fatalf("bad go.mod update, epected %s got %s", after, buf.buf.String())
	}
}

func cleanTestData() {
	cmd := exec.Command("git", "clean", "-xdf", "../../testdata")
	output, err := cmd.CombinedOutput()
	if err != nil {
		panic(string(output))
	}
	cmd = exec.Command("git", "checkout", "--", "../../testdata")
	output, err = cmd.CombinedOutput()
	if err != nil {
		panic(string(output))
	}
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	if err == nil {
		return true
	} else if os.IsNotExist(err) {
		return false
	}
	panic(err)
}

func Test_theCommandImplMajor(t *testing.T) {
	cleanTestData()
	defer cleanTestData()
	getTagsHook = func(root, prefix string) ([]string, error) {
		// root doesn't matter
		if !strings.HasSuffix(prefix, "/testdata/scenariob/foo") {
			return nil, fmt.Errorf("bad prefix '%s'", prefix)
		}
		return []string{
			"tools/testdata/scenariob/foo/v1.0.0",
			"tools/testdata/scenariob/foo/v1.1.0",
		}, nil
	}
	pkg, err := filepath.Abs("../../testdata/scenariob/foo/stage")
	if err != nil {
		t.Fatalf("failed to get absolute path: %v", err)
	}
	tag, err := theCommandImpl([]string{pkg})
	if err != nil {
		t.Fatalf("failed: %v", err)
	}
	const expected = "tools/testdata/scenariob/foo/v2.0.0"
	if tag != expected {
		t.Fatalf("bad tag, expected '%s' got '%s'", expected, tag)
	}
	b, err := ioutil.ReadFile("../../testdata/scenariob/foo/v2/go.mod")
	if err != nil {
		t.Fatalf("failed to read go.mod file: %v", err)
	}
	const after = "module github.com/Azure/azure-sdk-for-go/tools/testdata/scenariob/foo/v2\n\ngo 1.12\n"
	if !strings.EqualFold(string(b), after) {
		t.Fatalf("bad go.mod file, expected '%s' got '%s'", after, string(b))
	}
	if !fileExists(filepath.Join("../../testdata/scenariob/foo/v2", "CHANGELOG.md")) {
		t.Fatal("expected changelog in scenariob")
	}
}

func Test_theCommandImplMinor(t *testing.T) {
	cleanTestData()
	defer cleanTestData()
	getTagsHook = func(root, prefix string) ([]string, error) {
		// root doesn't matter
		if !strings.HasSuffix(prefix, "/testdata/scenarioa/foo") {
			return nil, fmt.Errorf("bad prefix '%s'", prefix)
		}
		return []string{
			"tools/testdata/scenarioa/foo/v1.0.0",
		}, nil
	}
	pkg, err := filepath.Abs("../../testdata/scenarioa/foo/stage")
	if err != nil {
		t.Fatalf("failed to get absolute path: %v", err)
	}
	tag, err := theCommandImpl([]string{pkg})
	if err != nil {
		t.Fatalf("failed: %v", err)
	}
	const expected = "tools/testdata/scenarioa/foo/v1.1.0"
	if tag != expected {
		t.Fatalf("bad tag, expected '%s' got '%s'", expected, tag)
	}
	b, err := ioutil.ReadFile("../../testdata/scenarioa/foo/go.mod")
	if err != nil {
		t.Fatalf("failed to read go.mod file: %v", err)
	}
	after := "module github.com/Azure/azure-sdk-for-go/tools/testdata/scenarioa/foo" + newline + newline + "go 1.12" + newline
	if !strings.EqualFold(string(b), after) {
		t.Fatalf("bad go.mod file, expected '%s' got '%s'", after, string(b))
	}
	if !fileExists(filepath.Join("../../testdata/scenarioa/foo", "CHANGELOG.md")) {
		t.Fatal("expected changelog in scenarioa")
	}
}

func Test_theCommandImplPatch(t *testing.T) {
	cleanTestData()
	defer cleanTestData()
	getTagsHook = func(root, prefix string) ([]string, error) {
		// root doesn't matter
		if !strings.HasSuffix(prefix, "/testdata/scenarioc/foo") {
			return nil, fmt.Errorf("bad prefix '%s'", prefix)
		}
		return []string{
			"tools/testdata/scenarioc/foo/v1.0.0",
		}, nil
	}
	pkg, err := filepath.Abs("../../testdata/scenarioc/foo/stage")
	if err != nil {
		t.Fatalf("failed to get absolute path: %v", err)
	}
	tag, err := theCommandImpl([]string{pkg})
	if err != nil {
		t.Fatalf("failed: %v", err)
	}
	const expected = "tools/testdata/scenarioc/foo/v1.0.1"
	if tag != expected {
		t.Fatalf("bad tag, expected '%s' got '%s'", expected, tag)
	}
	b, err := ioutil.ReadFile("../../testdata/scenarioc/foo/go.mod")
	if err != nil {
		t.Fatalf("failed to read go.mod file: %v", err)
	}
	after := "module github.com/Azure/azure-sdk-for-go/tools/testdata/scenarioc/foo" + newline + newline + "go 1.12" + newline
	if !strings.EqualFold(string(b), after) {
		t.Fatalf("bad go.mod file, expected '%s' got '%s'", after, string(b))
	}
	if !fileExists(filepath.Join("../../testdata/scenarioc/foo", "CHANGELOG.md")) {
		t.Fatal("expected changelog in scenarioc")
	}
}

func Test_theCommandImplNewMod(t *testing.T) {
	cleanTestData()
	defer cleanTestData()
	getTagsHook = func(root, prefix string) ([]string, error) {
		// root doesn't matter
		if !strings.HasSuffix(prefix, "/testdata/scenariof/foo") {
			return nil, fmt.Errorf("bad prefix '%s'", prefix)
		}
		return []string{}, nil
	}
	pkg, err := filepath.Abs("../../testdata/scenariof/foo/stage")
	if err != nil {
		t.Fatalf("failed to get absolute path: %v", err)
	}
	tag, err := theCommandImpl([]string{pkg})
	if err != nil {
		t.Fatalf("failed: %v", err)
	}
	const expected = "tools/testdata/scenariof/foo/v1.0.0"
	if tag != expected {
		t.Fatalf("bad tag, expected '%s' got '%s'", expected, tag)
	}
	b, err := ioutil.ReadFile("../../testdata/scenariof/foo/go.mod")
	if err != nil {
		t.Fatalf("failed to read go.mod file: %v", err)
	}
	after := "module github.com/Azure/azure-sdk-for-go/tools/testdata/scenariof/foo" + newline + newline + "go 1.12" + newline
	if !strings.EqualFold(string(b), after) {
		t.Fatalf("bad go.mod file, expected '%s' got '%s'", after, string(b))
	}
	if fileExists(filepath.Join("../../testdata/scenariof/foo", "CHANGELOG.md")) {
		t.Fatal("unexpected changelog in scenariof")
	}
}
