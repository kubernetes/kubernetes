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

package modinfo

import (
	"path/filepath"
	"reflect"
	"regexp"
	"testing"
)

func Test_ScenarioA(t *testing.T) {
	// scenario A has no breaking changes, additive only
	mod, err := GetModuleInfo("../../testdata/scenarioa/foo", "../../testdata/scenarioa/foo/stage")
	if err != nil {
		t.Fatalf("failed to get module info: %v", err)
	}
	if mod.BreakingChanges() {
		t.Fatal("no breaking changes in scenario A")
	}
	if !mod.NewExports() {
		t.Fatal("expected new exports in scenario A")
	}
	if mod.VersionSuffix() {
		t.Fatalf("unexpected version suffix in scenario A")
	}
	regex := regexp.MustCompile(`testdata/scenarioa/foo$`)
	if !regex.MatchString(mod.DestDir()) {
		t.Fatalf("bad destination dir: %s", mod.DestDir())
	}
}

func Test_ScenarioB(t *testing.T) {
	// scenario B has a breaking change
	mod, err := GetModuleInfo("../../testdata/scenariob/foo", "../../testdata/scenariob/foo/stage")
	if err != nil {
		t.Fatalf("failed to get module info: %v", err)
	}
	if !mod.BreakingChanges() {
		t.Fatal("expected breaking changes in scenario B")
	}
	if !mod.NewExports() {
		t.Fatal("expected new exports in scenario B")
	}
	if !mod.VersionSuffix() {
		t.Fatalf("expected version suffix in scenario B")
	}
	regex := regexp.MustCompile(`testdata[/\\]scenariob[/\\]foo[/\\]v2$`)
	if !regex.MatchString(mod.DestDir()) {
		t.Fatalf("bad destination dir: %s", mod.DestDir())
	}
}

func Test_ScenarioC(t *testing.T) {
	// scenario C has no new exports or breaking changes (function body/doc changes only)
	mod, err := GetModuleInfo("../../testdata/scenarioc/foo", "../../testdata/scenarioc/foo/stage")
	if err != nil {
		t.Fatalf("failed to get module info: %v", err)
	}
	if mod.BreakingChanges() {
		t.Fatal("unexpected breaking changes in scenario C")
	}
	if mod.NewExports() {
		t.Fatal("unexpected new exports in scenario C")
	}
	if mod.VersionSuffix() {
		t.Fatalf("unexpected version suffix in scenario C")
	}
	regex := regexp.MustCompile(`testdata/scenarioc/foo$`)
	if !regex.MatchString(mod.DestDir()) {
		t.Fatalf("bad destination dir: %s", mod.DestDir())
	}
}

func Test_ScenarioD(t *testing.T) {
	// scenario D has a breaking change on top of a v2 release
	mod, err := GetModuleInfo("../../testdata/scenariod/foo/v2", "../../testdata/scenariod/foo/stage")
	if err != nil {
		t.Fatalf("failed to get module info: %v", err)
	}
	if !mod.BreakingChanges() {
		t.Fatal("expected breaking changes in scenario D")
	}
	if mod.NewExports() {
		t.Fatal("unexpected new exports in scenario D")
	}
	if !mod.VersionSuffix() {
		t.Fatalf("expected version suffix in scenario D")
	}
	regex := regexp.MustCompile(`testdata[/\\]scenariod[/\\]foo[/\\]v3$`)
	if !regex.MatchString(mod.DestDir()) {
		t.Fatalf("bad destination dir: %s", mod.DestDir())
	}
}

func Test_ScenarioE(t *testing.T) {
	// scenario E has a new export on top of a v2 release
	mod, err := GetModuleInfo("../../testdata/scenarioe/foo/v2", "../../testdata/scenarioe/foo/stage")
	if err != nil {
		t.Fatalf("failed to get module info: %v", err)
	}
	if mod.BreakingChanges() {
		t.Fatal("unexpected breaking changes in scenario E")
	}
	if !mod.NewExports() {
		t.Fatal("expected new exports in scenario E")
	}
	if !mod.VersionSuffix() {
		t.Fatalf("expected version suffix in scenario E")
	}
	regex := regexp.MustCompile(`testdata/scenarioe/foo/v2$`)
	if !regex.MatchString(mod.DestDir()) {
		t.Fatalf("bad destination dir: %s", mod.DestDir())
	}
}

func Test_ScenarioF(t *testing.T) {
	// scenario F is a new module
	mod, err := GetModuleInfo("../../testdata/scenariof/foo", "../../testdata/scenariof/foo/stage")
	if err != nil {
		t.Fatalf("failed to get module info: %v", err)
	}
	if mod.BreakingChanges() {
		t.Fatal("unexpected breaking changes in scenario F")
	}
	if !mod.NewExports() {
		t.Fatal("expected new exports in scenario F")
	}
	if mod.VersionSuffix() {
		t.Fatalf("unexpected version suffix in scenario F")
	}
	if !mod.NewModule() {
		t.Fatal("expected new module in scenario F")
	}
	regex := regexp.MustCompile(`testdata/scenariof/foo$`)
	if !regex.MatchString(mod.DestDir()) {
		t.Fatalf("bad destination dir: %s", mod.DestDir())
	}
}

func Test_sortModuleTagsBySemver(t *testing.T) {
	before := []string{
		"v1.0.0",
		"v1.0.1",
		"v1.1.0",
		"v10.0.0",
		"v11.1.1",
		"v2.0.0",
		"v20.2.3",
		"v3.1.0",
	}
	sortModuleTagsBySemver(before)
	after := []string{
		"v1.0.0",
		"v1.0.1",
		"v1.1.0",
		"v2.0.0",
		"v3.1.0",
		"v10.0.0",
		"v11.1.1",
		"v20.2.3",
	}
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("sort order doesn't match, expected '%v' got '%v'", after, before)
	}
}

func TestIncrementModuleVersion(t *testing.T) {
	v := IncrementModuleVersion("")
	if v != "v2" {
		t.Fatalf("expected v2 got %s", v)
	}
	v = IncrementModuleVersion("v2")
	if v != "v3" {
		t.Fatalf("expected v3 got %s", v)
	}
	v = IncrementModuleVersion("v10")
	if v != "v11" {
		t.Fatalf("expected v11 got %s", v)
	}
}

func TestCreateModuleNameFromPath(t *testing.T) {
	n, err := CreateModuleNameFromPath(filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "foo", "apiver", "foo"))
	if err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	const expected = "github.com/Azure/azure-sdk-for-go/services/foo/apiver/foo"
	if n != expected {
		t.Fatalf("expected '%s' got '%s'", expected, n)
	}
}

func TestCreateModuleNameFromPathFail(t *testing.T) {
	n, err := CreateModuleNameFromPath(filepath.Join("work", "src", "github.com", "other", "project", "foo", "bar"))
	if err == nil {
		t.Fatal("expected non-nil error")
	}
	if n != "" {
		t.Fatalf("expected empty module name, got %s", n)
	}
}

func TestIsValidModuleVersion(t *testing.T) {
	if !IsValidModuleVersion("v10.21.23") {
		t.Fatal("unexpected invalid module version")
	}
	if IsValidModuleVersion("1.2.3") {
		t.Fatal("unexpected valid module version, missing v")
	}
	if IsValidModuleVersion("v11.563") {
		t.Fatal("unexpected valid module version, missing patch")
	}
}
