/*
Copyright 2015 The Kubernetes Authors.

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

package util

import (
	"bytes"
	"io/ioutil"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestArgs(t *testing.T) {
	if e, a := []string{"/bin/bash", "-c \"test\""}, (CmdTool{Args: []string{"/bin/bash", "-c"}, Shell: true}).args("test"); !reflect.DeepEqual(e, a) {
		t.Errorf("unexpected args: %v", a)
	}
	if e, a := []string{"/bin/bash", "-c", "test"}, (CmdTool{Args: []string{"/bin/bash", "-c"}, Shell: false}).args("test"); !reflect.DeepEqual(e, a) {
		t.Errorf("unexpected args: %v", a)
	}
	if e, a := []string{"/bin/bash", "-i -c \"test\""}, (CmdTool{Args: []string{"/bin/bash", "-i -c"}, Shell: true}).args("test"); !reflect.DeepEqual(e, a) {
		t.Errorf("unexpected args: %v", a)
	}
	if e, a := []string{"/test", "test"}, (CmdTool{Args: []string{"/test"}}).args("test"); !reflect.DeepEqual(e, a) {
		t.Errorf("unexpected args: %v", a)
	}
}

func TestLaunchTempFile(t *testing.T) {
	edit := CmdTool{Args: []string{"cat"}}
	testStr := "test something\n"

	contents, path, err := edit.LaunchTempFile(TempFile{Prefix: "", Suffix: "someprefix", Buffer: bytes.NewBufferString(testStr)})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("no temp file: %s", path)
	}
	defer os.Remove(path)
	if disk, err := ioutil.ReadFile(path); err != nil || !bytes.Equal(contents, disk) {
		t.Errorf("unexpected file on disk: %v %s", err, string(disk))
	}
	if !bytes.Equal(contents, []byte(testStr)) {
		t.Errorf("unexpected contents: %s", string(contents))
	}
	if !strings.Contains(path, "someprefix") {
		t.Errorf("path not expected: %s", path)
	}
}

func TestLaunchTempFiles(t *testing.T) {
	diff := CmdTool{Args: []string{"cat"}}
	testStr1 := "test something\n"
	testStr2 := "test something\n"

	tmpFileSlice := []TempFile{
		{Prefix: "", Suffix: "someprefix", Buffer: bytes.NewBufferString(testStr1)},
		{Prefix: "", Suffix: "someprefix", Buffer: bytes.NewBufferString(testStr2)},
	}
	paths, err := diff.LaunchTempFiles(tmpFileSlice...)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, path := range paths {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("no temp file: %s", path)
		}
		defer os.Remove(path)

		if !strings.Contains(path, "someprefix") {
			t.Errorf("path not expected: %s", path)
		}
	}
}
