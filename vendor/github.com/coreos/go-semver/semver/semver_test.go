// Copyright 2013-2015 CoreOS, Inc.
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

package semver

import (
	"bytes"
	"encoding/json"
	"errors"
	"math/rand"
	"reflect"
	"testing"
	"time"
)

type fixture struct {
	GreaterVersion string
	LesserVersion  string
}

var fixtures = []fixture{
	fixture{"0.0.0", "0.0.0-foo"},
	fixture{"0.0.1", "0.0.0"},
	fixture{"1.0.0", "0.9.9"},
	fixture{"0.10.0", "0.9.0"},
	fixture{"0.99.0", "0.10.0"},
	fixture{"2.0.0", "1.2.3"},
	fixture{"0.0.0", "0.0.0-foo"},
	fixture{"0.0.1", "0.0.0"},
	fixture{"1.0.0", "0.9.9"},
	fixture{"0.10.0", "0.9.0"},
	fixture{"0.99.0", "0.10.0"},
	fixture{"2.0.0", "1.2.3"},
	fixture{"0.0.0", "0.0.0-foo"},
	fixture{"0.0.1", "0.0.0"},
	fixture{"1.0.0", "0.9.9"},
	fixture{"0.10.0", "0.9.0"},
	fixture{"0.99.0", "0.10.0"},
	fixture{"2.0.0", "1.2.3"},
	fixture{"1.2.3", "1.2.3-asdf"},
	fixture{"1.2.3", "1.2.3-4"},
	fixture{"1.2.3", "1.2.3-4-foo"},
	fixture{"1.2.3-5-foo", "1.2.3-5"},
	fixture{"1.2.3-5", "1.2.3-4"},
	fixture{"1.2.3-5-foo", "1.2.3-5-Foo"},
	fixture{"3.0.0", "2.7.2+asdf"},
	fixture{"3.0.0+foobar", "2.7.2"},
	fixture{"1.2.3-a.10", "1.2.3-a.5"},
	fixture{"1.2.3-a.b", "1.2.3-a.5"},
	fixture{"1.2.3-a.b", "1.2.3-a"},
	fixture{"1.2.3-a.b.c.10.d.5", "1.2.3-a.b.c.5.d.100"},
	fixture{"1.0.0", "1.0.0-rc.1"},
	fixture{"1.0.0-rc.2", "1.0.0-rc.1"},
	fixture{"1.0.0-rc.1", "1.0.0-beta.11"},
	fixture{"1.0.0-beta.11", "1.0.0-beta.2"},
	fixture{"1.0.0-beta.2", "1.0.0-beta"},
	fixture{"1.0.0-beta", "1.0.0-alpha.beta"},
	fixture{"1.0.0-alpha.beta", "1.0.0-alpha.1"},
	fixture{"1.0.0-alpha.1", "1.0.0-alpha"},
}

func TestCompare(t *testing.T) {
	for _, v := range fixtures {
		gt, err := NewVersion(v.GreaterVersion)
		if err != nil {
			t.Error(err)
		}

		lt, err := NewVersion(v.LesserVersion)
		if err != nil {
			t.Error(err)
		}

		if gt.LessThan(*lt) == true {
			t.Errorf("%s should not be less than %s", gt, lt)
		}
	}
}

func testString(t *testing.T, orig string, version *Version) {
	if orig != version.String() {
		t.Errorf("%s != %s", orig, version)
	}
}

func TestString(t *testing.T) {
	for _, v := range fixtures {
		gt, err := NewVersion(v.GreaterVersion)
		if err != nil {
			t.Error(err)
		}
		testString(t, v.GreaterVersion, gt)

		lt, err := NewVersion(v.LesserVersion)
		if err != nil {
			t.Error(err)
		}
		testString(t, v.LesserVersion, lt)
	}
}

func shuffleStringSlice(src []string) []string {
	dest := make([]string, len(src))
	rand.Seed(time.Now().Unix())
	perm := rand.Perm(len(src))
	for i, v := range perm {
		dest[v] = src[i]
	}
	return dest
}

func TestSort(t *testing.T) {
	sortedVersions := []string{"1.0.0", "1.0.2", "1.2.0", "3.1.1"}
	unsortedVersions := shuffleStringSlice(sortedVersions)

	semvers := []*Version{}
	for _, v := range unsortedVersions {
		sv, err := NewVersion(v)
		if err != nil {
			t.Fatal(err)
		}
		semvers = append(semvers, sv)
	}

	Sort(semvers)

	for idx, sv := range semvers {
		if sv.String() != sortedVersions[idx] {
			t.Fatalf("incorrect sort at index %v", idx)
		}
	}
}

func TestBumpMajor(t *testing.T) {
	version, _ := NewVersion("1.0.0")
	version.BumpMajor()
	if version.Major != 2 {
		t.Fatalf("bumping major on 1.0.0 resulted in %v", version)
	}

	version, _ = NewVersion("1.5.2")
	version.BumpMajor()
	if version.Minor != 0 && version.Patch != 0 {
		t.Fatalf("bumping major on 1.5.2 resulted in %v", version)
	}

	version, _ = NewVersion("1.0.0+build.1-alpha.1")
	version.BumpMajor()
	if version.PreRelease != "" && version.PreRelease != "" {
		t.Fatalf("bumping major on 1.0.0+build.1-alpha.1 resulted in %v", version)
	}
}

func TestBumpMinor(t *testing.T) {
	version, _ := NewVersion("1.0.0")
	version.BumpMinor()

	if version.Major != 1 {
		t.Fatalf("bumping minor on 1.0.0 resulted in %v", version)
	}

	if version.Minor != 1 {
		t.Fatalf("bumping major on 1.0.0 resulted in %v", version)
	}

	version, _ = NewVersion("1.0.0+build.1-alpha.1")
	version.BumpMinor()
	if version.PreRelease != "" && version.PreRelease != "" {
		t.Fatalf("bumping major on 1.0.0+build.1-alpha.1 resulted in %v", version)
	}
}

func TestBumpPatch(t *testing.T) {
	version, _ := NewVersion("1.0.0")
	version.BumpPatch()

	if version.Major != 1 {
		t.Fatalf("bumping minor on 1.0.0 resulted in %v", version)
	}

	if version.Minor != 0 {
		t.Fatalf("bumping major on 1.0.0 resulted in %v", version)
	}

	if version.Patch != 1 {
		t.Fatalf("bumping major on 1.0.0 resulted in %v", version)
	}

	version, _ = NewVersion("1.0.0+build.1-alpha.1")
	version.BumpPatch()
	if version.PreRelease != "" && version.PreRelease != "" {
		t.Fatalf("bumping major on 1.0.0+build.1-alpha.1 resulted in %v", version)
	}
}

func TestMust(t *testing.T) {
	tests := []struct {
		versionStr string

		version *Version
		recov   interface{}
	}{
		{
			versionStr: "1.0.0",
			version:    &Version{Major: 1},
		},
		{
			versionStr: "version number",
			recov:      errors.New("version number is not in dotted-tri format"),
		},
	}

	for _, tt := range tests {
		func() {
			defer func() {
				recov := recover()
				if !reflect.DeepEqual(tt.recov, recov) {
					t.Fatalf("incorrect panic for %q: want %v, got %v", tt.versionStr, tt.recov, recov)
				}
			}()

			version := Must(NewVersion(tt.versionStr))
			if !reflect.DeepEqual(tt.version, version) {
				t.Fatalf("incorrect version for %q: want %+v, got %+v", tt.versionStr, tt.version, version)
			}
		}()
	}
}

type fixtureJSON struct {
	GreaterVersion *Version
	LesserVersion  *Version
}

func TestJSON(t *testing.T) {
	fj := make([]fixtureJSON, len(fixtures))
	for i, v := range fixtures {
		var err error
		fj[i].GreaterVersion, err = NewVersion(v.GreaterVersion)
		if err != nil {
			t.Fatal(err)
		}
		fj[i].LesserVersion, err = NewVersion(v.LesserVersion)
		if err != nil {
			t.Fatal(err)
		}
	}

	fromStrings, err := json.Marshal(fixtures)
	if err != nil {
		t.Fatal(err)
	}
	fromVersions, err := json.Marshal(fj)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(fromStrings, fromVersions) {
		t.Errorf("Expected:   %s", fromStrings)
		t.Errorf("Unexpected: %s", fromVersions)
	}

	fromJson := make([]fixtureJSON, 0, len(fj))
	err = json.Unmarshal(fromStrings, &fromJson)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(fromJson, fj) {
		t.Error("Expected:   ", fj)
		t.Error("Unexpected: ", fromJson)
	}
}

func TestBadInput(t *testing.T) {
	bad := []string{
		"1.2",
		"1.2.3x",
		"0x1.3.4",
		"-1.2.3",
		"1.2.3.4",
	}
	for _, b := range bad {
		if _, err := NewVersion(b); err == nil {
			t.Error("Improperly accepted value: ", b)
		}
	}
}
