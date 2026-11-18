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

package editor

import (
	"bytes"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestArgs(t *testing.T) {
	if e, a := []string{"/bin/bash", "-c \"test\""}, (Editor{Args: []string{"/bin/bash", "-c"}, Shell: true}).args("test"); !reflect.DeepEqual(e, a) {
		t.Errorf("unexpected args: %v", a)
	}
	if e, a := []string{"/bin/bash", "-c", "test"}, (Editor{Args: []string{"/bin/bash", "-c"}, Shell: false}).args("test"); !reflect.DeepEqual(e, a) {
		t.Errorf("unexpected args: %v", a)
	}
	if e, a := []string{"/bin/bash", "-i -c \"test\""}, (Editor{Args: []string{"/bin/bash", "-i -c"}, Shell: true}).args("test"); !reflect.DeepEqual(e, a) {
		t.Errorf("unexpected args: %v", a)
	}
	if e, a := []string{"/test", "test"}, (Editor{Args: []string{"/test"}}).args("test"); !reflect.DeepEqual(e, a) {
		t.Errorf("unexpected args: %v", a)
	}
}

func TestDefaultEnvEditor(t *testing.T) {
	t.Setenv("SHELL", "test-shell")
	tests := []struct {
		name   string
		editor string
		args   []string
		shell  bool
	}{
		{
			name:   "bare command",
			editor: "code",
			args:   []string{"code"},
		},
		{
			name:   "single space",
			editor: "code --wait",
			args:   []string{"code", "--wait"},
		},
		{
			name:   "repeated spaces",
			editor: "code  --wait",
			args:   []string{"code", "--wait"},
		},
		{
			name:   "tab separator",
			editor: "code\t--wait",
			args:   []string{"code", "--wait"},
		},
		{
			name:   "leading and trailing whitespace",
			editor: " \tcode --wait\t ",
			args:   []string{"code", "--wait"},
		},
		{
			name:   "multiple arguments",
			editor: "code  --wait\t--new-window",
			args:   []string{"code", "--wait", "--new-window"},
		},
		{
			name:   "quoted argument",
			editor: `code --profile "my profile" --wait`,
			args:   []string{"test-shell", "-c", `code --profile "my profile" --wait`},
			shell:  true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("KUBE_EDITOR", tt.editor)
			args, shell := defaultEnvEditor([]string{"KUBE_EDITOR"})
			if !reflect.DeepEqual(args, tt.args) || shell != tt.shell {
				t.Errorf("defaultEnvEditor() = (%q, %t), want (%q, %t)", args, shell, tt.args, tt.shell)
			}
		})
	}
}

func TestEditor(t *testing.T) {
	edit := Editor{Args: []string{"cat"}}
	testStr := "test something\n"
	contents, path, err := edit.LaunchTempFile("", "someprefix", bytes.NewBufferString(testStr))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("no temp file: %s", path)
	}
	defer os.Remove(path)
	if disk, err := os.ReadFile(path); err != nil || !bytes.Equal(contents, disk) {
		t.Errorf("unexpected file on disk: %v %s", err, string(disk))
	}
	if !bytes.Equal(contents, []byte(testStr)) {
		t.Errorf("unexpected contents: %s", string(contents))
	}
	if !strings.Contains(path, "someprefix") {
		t.Errorf("path not expected: %s", path)
	}
}
