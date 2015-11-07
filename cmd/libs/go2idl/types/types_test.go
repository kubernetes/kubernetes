/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package types

import (
	"testing"
)

func TestGetBuiltin(t *testing.T) {
	u := Universe{}
	if builtinPkg := u.Package(""); builtinPkg.Has("string") {
		t.Errorf("Expected builtin package to not have builtins until they're asked for explicitly. %#v", builtinPkg)
	}
	s := u.Get(Name{"", "string"})
	if s != String {
		t.Errorf("Expected canonical string type.")
	}
	if builtinPkg := u.Package(""); !builtinPkg.Has("string") {
		t.Errorf("Expected builtin package to exist and have builtins by default. %#v", builtinPkg)
	}
	if builtinPkg := u.Package(""); len(builtinPkg.Types) != 1 {
		t.Errorf("Expected builtin package to not have builtins until they're asked for explicitly. %#v", builtinPkg)
	}
}

func TestGetMarker(t *testing.T) {
	u := Universe{}
	n := Name{"path/to/package", "Foo"}
	f := u.Get(n)
	if f == nil || f.Name != n {
		t.Errorf("Expected marker type.")
	}
}
