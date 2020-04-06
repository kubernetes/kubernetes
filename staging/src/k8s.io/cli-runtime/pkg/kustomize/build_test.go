// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package kustomize

import (
	"testing"

	"sigs.k8s.io/kustomize/api/filesys"
	"sigs.k8s.io/kustomize/api/konfig"
)

func TestNewOptionsToSilenceCodeInspectionError(t *testing.T) {
	if NewOptions("foo", "bar") == nil {
		t.Fatal("could not make new options")
	}
}

func TestBuildValidate(t *testing.T) {
	var cases = []struct {
		name  string
		args  []string
		path  string
		erMsg string
	}{
		{"noargs", []string{}, filesys.SelfDir, ""},
		{"file", []string{"beans"}, "beans", ""},
		{"path", []string{"a/b/c"}, "a/b/c", ""},
		{"path", []string{"too", "many"},
			"",
			"specify one path to " +
				konfig.DefaultKustomizationFileName()},
	}
	for _, mycase := range cases {
		opts := Options{}
		e := opts.Validate(mycase.args)
		if len(mycase.erMsg) > 0 {
			if e == nil {
				t.Errorf("%s: Expected an error %v", mycase.name, mycase.erMsg)
			}
			if e.Error() != mycase.erMsg {
				t.Errorf("%s: Expected error %s, but got %v", mycase.name, mycase.erMsg, e)
			}
			continue
		}
		if e != nil {
			t.Errorf("%s: unknown error: %v", mycase.name, e)
			continue
		}
		if opts.kustomizationPath != mycase.path {
			t.Errorf("%s: expected path '%s', got '%s'", mycase.name, mycase.path, opts.kustomizationPath)
		}
	}
}
