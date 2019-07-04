/*
Copyright 2019 The Kubernetes Authors.

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

package kustomize

import (
	"testing"
)

func TestValidate(t *testing.T) {
	var cases = []struct {
		name  string
		args  []string
		path  string
		erMsg string
	}{
		{"noargs", []string{}, "./", ""},
		{"file", []string{"beans"}, "beans", ""},
		{"path", []string{"a/b/c"}, "a/b/c", ""},
		{"path", []string{"too", "many"},
			"", "specify one path to a kustomization directory"},
	}
	for _, mycase := range cases {
		opts := kustomizeOptions{}
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
			t.Errorf("%s: unknown error %v", mycase.name, e)
			continue
		}
		if opts.kustomizationDir != mycase.path {
			t.Errorf("%s: expected path '%s', got '%s'", mycase.name, mycase.path, opts.kustomizationDir)
		}
	}
}
