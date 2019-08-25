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

package misc

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"sigs.k8s.io/kustomize/pkg/fs"
)

func TestValidate(t *testing.T) {
	o := saveOptions{
		saveDirectory: "",
	}
	err := o.Validate()
	if !strings.Contains(err.Error(), "must specify one local directory") {
		t.Fatalf("Incorrect error %v", err)
	}

	o.saveDirectory = "/some/dir"
	err = o.Validate()
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
}

func TestComplete(t *testing.T) {
	fsys := fs.MakeFakeFS()
	fsys.Mkdir("/some/dir")
	fsys.WriteFile("/some/file", []byte(`some file`))

	type testcase struct {
		dir    string
		expect error
	}
	testcases := []testcase{
		{
			dir:    "/some/dir",
			expect: nil,
		},
		{
			dir:    "/some/dir/not/existing",
			expect: nil,
		},
		{
			dir:    "/some/file",
			expect: fmt.Errorf("%s is not a directory", "/some/file"),
		},
	}

	for _, tcase := range testcases {
		o := saveOptions{saveDirectory: tcase.dir}
		actual := o.Complete(fsys)
		if !reflect.DeepEqual(actual, tcase.expect) {
			t.Fatalf("Expected %v\n but bot %v\n", tcase.expect, actual)
		}
	}
}

func TestRunSave(t *testing.T) {
	fsys := fs.MakeFakeFS()
	o := saveOptions{saveDirectory: "/some/dir"}
	err := o.RunSave(fsys)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if !fsys.Exists("/some/dir/nameprefix.yaml") {
		t.Fatal("default configurations are not successfully save.")
	}
}
