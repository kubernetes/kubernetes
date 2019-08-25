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

package loader

import (
	"strings"
	"testing"

	"sigs.k8s.io/kustomize/pkg/fs"
)

func TestRestrictionNone(t *testing.T) {
	fSys := fs.MakeFakeFS()
	root := fs.ConfirmedDir("irrelevant")
	path := "whatever"
	p, err := RestrictionNone(fSys, root, path)
	if err != nil {
		t.Fatal(err)
	}
	if p != path {
		t.Fatalf("expected '%s', got '%s'", path, p)
	}
}

func TestRestrictionRootOnly(t *testing.T) {
	fSys := fs.MakeFakeFS()
	root := fs.ConfirmedDir("/tmp/foo")

	path := "/tmp/foo/whatever/beans"
	p, err := RestrictionRootOnly(fSys, root, path)
	if err != nil {
		t.Fatal(err)
	}
	if p != path {
		t.Fatalf("expected '%s', got '%s'", path, p)
	}

	// Legal.
	path = "/tmp/foo/whatever/../../foo/whatever"
	p, err = RestrictionRootOnly(fSys, root, path)
	if err != nil {
		t.Fatal(err)
	}
	path = "/tmp/foo/whatever"
	if p != path {
		t.Fatalf("expected '%s', got '%s'", path, p)
	}

	// Illegal.
	path = "/tmp/illegal"
	_, err = RestrictionRootOnly(fSys, root, path)
	if err == nil {
		t.Fatal("should have an error")
	}
	if !strings.Contains(
		err.Error(),
		"file '/tmp/illegal' is not in or below '/tmp/foo'") {
		t.Fatalf("unexpected err: %s", err)
	}
}
