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

package fs

import (
	"path/filepath"
	"testing"
)

func TestJoin(t *testing.T) {
	fSys := MakeFakeFS()
	err := fSys.Mkdir("/foo")
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	d, f, err := fSys.CleanedAbs("/foo")
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if f != "" {
		t.Fatalf("unexpected file: %v", f)
	}
	if d.Join("bar") != "/foo/bar" {
		t.Fatalf("expected join %s", d.Join("bar"))
	}
}

func TestHasPrefix_Slash(t *testing.T) {
	d, f, err := MakeFakeFS().CleanedAbs("/")
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if f != "" {
		t.Fatalf("unexpected file: %v", f)
	}
	if d.HasPrefix("/hey") {
		t.Fatalf("should be false")
	}
	if !d.HasPrefix("/") {
		t.Fatalf("/ should have the prefix /")
	}
}

func TestHasPrefix_SlashFoo(t *testing.T) {
	fSys := MakeFakeFS()
	err := fSys.Mkdir("/foo")
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	d, _, err := fSys.CleanedAbs("/foo")
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if d.HasPrefix("/fo") {
		t.Fatalf("/foo does not have path prefix /fo")
	}
	if d.HasPrefix("/fod") {
		t.Fatalf("/foo does not have path prefix /fod")
	}
	if !d.HasPrefix("/foo") {
		t.Fatalf("/foo should have prefix /foo")
	}
}

func TestHasPrefix_SlashFooBar(t *testing.T) {
	fSys := MakeFakeFS()
	err := fSys.MkdirAll("/foo/bar")
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	d, _, err := fSys.CleanedAbs("/foo/bar")
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if d.HasPrefix("/fo") {
		t.Fatalf("/foo/bar does not have path prefix /fo")
	}
	if d.HasPrefix("/foobar") {
		t.Fatalf("/foo/bar does not have path prefix /foobar")
	}
	if !d.HasPrefix("/foo/bar") {
		t.Fatalf("/foo/bar should have prefix /foo/bar")
	}
	if !d.HasPrefix("/foo") {
		t.Fatalf("/foo/bar should have prefix /foo")
	}
	if !d.HasPrefix("/") {
		t.Fatalf("/foo/bar should have prefix /")
	}
}

func TestNewTempConfirmDir(t *testing.T) {
	tmp, err := NewTmpConfirmedDir()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	delinked, err := filepath.EvalSymlinks(string(tmp))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(tmp) != delinked {
		t.Fatalf("unexpected path containing symlinks")
	}
}
