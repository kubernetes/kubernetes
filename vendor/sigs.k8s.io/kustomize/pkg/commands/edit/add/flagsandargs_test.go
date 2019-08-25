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

package add

import (
	"reflect"
	"testing"

	"sigs.k8s.io/kustomize/pkg/fs"
)

func TestDataValidation_NoName(t *testing.T) {
	fa := flagsAndArgs{}

	if fa.Validate([]string{}) == nil {
		t.Fatal("Validation should fail if no name is specified")
	}
}

func TestDataValidation_MoreThanOneName(t *testing.T) {
	fa := flagsAndArgs{}

	if fa.Validate([]string{"name", "othername"}) == nil {
		t.Fatal("Validation should fail if more than one name is specified")
	}
}

func TestDataConfigValidation_Flags(t *testing.T) {
	tests := []struct {
		name       string
		fa         flagsAndArgs
		shouldFail bool
	}{
		{
			name: "env-file-source and literal are both set",
			fa: flagsAndArgs{
				LiteralSources: []string{"one", "two"},
				EnvFileSource:  "three",
			},
			shouldFail: true,
		},
		{
			name: "env-file-source and from-file are both set",
			fa: flagsAndArgs{
				FileSources:   []string{"one", "two"},
				EnvFileSource: "three",
			},
			shouldFail: true,
		},
		{
			name:       "we don't have any option set",
			fa:         flagsAndArgs{},
			shouldFail: true,
		},
		{
			name: "we have from-file and literal ",
			fa: flagsAndArgs{
				LiteralSources: []string{"one", "two"},
				FileSources:    []string{"three", "four"},
			},
			shouldFail: false,
		},
	}

	for _, test := range tests {
		if test.fa.Validate([]string{"name"}) == nil && test.shouldFail {
			t.Fatalf("Validation should fail if %s", test.name)
		} else if test.fa.Validate([]string{"name"}) != nil && !test.shouldFail {
			t.Fatalf("Validation should succeed if %s", test.name)
		}
	}
}

func TestExpandFileSource(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.Create("dir/fa1")
	fakeFS.Create("dir/fa2")
	fakeFS.Create("dir/readme")
	fa := flagsAndArgs{
		FileSources: []string{"dir/fa*"},
	}
	fa.ExpandFileSource(fakeFS)
	expected := []string{
		"dir/fa1",
		"dir/fa2",
	}
	if !reflect.DeepEqual(fa.FileSources, expected) {
		t.Fatalf("FileSources is not correctly expanded: %v", fa.FileSources)
	}
}

func TestExpandFileSourceWithKey(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.Create("dir/faaaaaaaaaabbbbbbbbbccccccccccccccccc")
	fakeFS.Create("dir/foobar")
	fakeFS.Create("dir/simplebar")
	fakeFS.Create("dir/readme")
	fa := flagsAndArgs{
		FileSources: []string{"foo-key=dir/fa*", "bar-key=dir/foobar", "dir/simplebar"},
	}
	fa.ExpandFileSource(fakeFS)
	expected := []string{
		"foo-key=dir/faaaaaaaaaabbbbbbbbbccccccccccccccccc",
		"bar-key=dir/foobar",
		"dir/simplebar",
	}
	if !reflect.DeepEqual(fa.FileSources, expected) {
		t.Fatalf("FileSources is not correctly expanded: %v", fa.FileSources)
	}
}

func TestExpandFileSourceWithKeyAndError(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.Create("dir/fa1")
	fakeFS.Create("dir/fa2")
	fakeFS.Create("dir/readme")
	fa := flagsAndArgs{
		FileSources: []string{"foo-key=dir/fa*"},
	}
	err := fa.ExpandFileSource(fakeFS)
	if err == nil {
		t.Fatalf("FileSources should not be correctly expanded: %v", fa.FileSources)
	}
}
